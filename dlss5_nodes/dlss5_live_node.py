"""DLSS 5 Live Preview node.

Loops a clip through the *native* DLSS 5 worker in live mode and shows the result in the
node while you move the sliders: tone / structure / skin / mask / style take effect on the
next frame (a few ms), upscale / temporal / model preset hot-swap the worker (~1.5 s) while
the old one keeps rendering. A Bake button (or running the node in a flow) renders the whole
clip with the current settings into the project's output folder.

Completely separate from ``DLSS5NeuralRenderNode``: that node keeps using the DLSS 5 Visual
Enhancer worker and none of its code paths are touched here. This node reuses its ffmpeg
helpers (colour probe, writer, audio mux) for the bake.
"""

from __future__ import annotations

import contextlib
import queue
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

try:
    import dlss5_worker_bridge  # noqa: F401
except ImportError:  # ensure sibling modules resolve regardless of loader
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

from dlss5_live_bridge import NativeLiveWorker, NativeRuntime, find_native_runtime  # noqa: E402
from dlss5_live_server import DEFAULT_PREVIEW_WIDTH, LiveSession  # noqa: E402
from dlss5_video_node import (  # noqa: E402  (helpers only; the render node itself is untouched)
    PROXY_MAX_HEIGHT,
    PROXY_MAX_WIDTH,
    DLSS5NeuralRenderNode,
    _resolve_video_value,
    _to_rgb,
)
from dlss5_worker_bridge import (  # noqa: E402
    DIS_PRESETS,
    MODEL_PRESETS,
    MV_MODE_AUTO_DIS,
    MV_MODE_NONE,
    NR_STYLES,
    UPSCALE_MODES,
    DLSS5Settings,
    DLSS5WorkerError,
)

LIBRARY_NAME = "GTN DLSS 5 Neural Rendering"
WIDGET_NAME = "DLSS5LivePreview"

# Look presets. Intensity is deliberately not part of them: below 1.0 the current runtime only
# darkens the frame (measured), so the live node pins it at 1.0.
LOOK_ULTRA = "Ultra (max realism)"
LOOK_HIGH = "High"
LOOK_MEDIUM = "Medium"
LOOK_LOW = "Low (subtle)"
LOOK_CUSTOM = "Custom (use the sliders)"
LOOK_CHOICES = [LOOK_ULTRA, LOOK_HIGH, LOOK_MEDIUM, LOOK_LOW, LOOK_CUSTOM]
LOOK_PRESETS: dict[str, dict[str, Any]] = {
    LOOK_ULTRA: {"local_tone": 1.0, "local_structure": 2.0, "skin_structure": 2.0, "auto_mask": True},
    LOOK_HIGH: {"local_tone": 1.0, "local_structure": 1.5, "skin_structure": 1.5, "auto_mask": True},
    LOOK_MEDIUM: {"local_tone": 0.8, "local_structure": 1.0, "skin_structure": 1.0, "auto_mask": True},
    LOOK_LOW: {"local_tone": 0.5, "local_structure": 0.5, "skin_structure": 0.5, "auto_mask": True},
}
LOOK_CONTROLS = ("local_tone", "local_structure", "skin_structure", "auto_mask")
LIVE_CONTROLS = LOOK_CONTROLS + ("nr_style", "upscale_mode", "temporal", "model_preset", "dis_preset")

TEMPORAL_SINGLE = "Single Frame (no temporal history)"
TEMPORAL_SEQUENCE = "Sequence (temporal, optical flow in worker)"
TEMPORAL_CHOICES = [TEMPORAL_SINGLE, TEMPORAL_SEQUENCE]

DEFAULT_OUTPUT_FILENAME = "dlss5_live.mp4"
BAKE_PIPELINE_DEPTH = 3

# Live sessions keyed by node object id. Module-level so the weakref finaliser can stop a
# session after its node object is gone (workflow reload, engine shutdown).
_sessions: dict[int, LiveSession] = {}
_sessions_lock = threading.Lock()
_bake_lock = threading.Lock()  # one bake at a time per engine (GPU + worker are exclusive)
_ALIVE_MISSES = 5  # consecutive "node not registered" checks before the session gives up


def _stop_session(key: int) -> None:
    with _sessions_lock:
        session = _sessions.pop(key, None)
    if session is not None:
        session.stop()
        with contextlib.suppress(OSError):
            Path(session.video_path).unlink()


class _NodeAlive:
    """Session -> node liveness probe. True while the node object exists and is still the node
    registered under its name (deleted nodes are removed from the ObjectManager)."""

    def __init__(self, node: ControlNode) -> None:
        self._ref = weakref.ref(node)
        self._misses = 0

    def __call__(self) -> bool:
        node = self._ref()
        if node is None:
            return False
        try:
            manager = GriptapeNodes.ObjectManager()
            current = manager.attempt_get_object_by_name(node.name)
        except Exception:  # noqa: BLE001 - outside a running engine there is no object manager
            return True
        if current is node or (current is None and not getattr(manager, "_name_to_objects", None)):
            self._misses = 0  # registered, or an empty manager (running outside the editor)
            return True
        self._misses += 1
        return self._misses < _ALIVE_MISSES


class DLSS5LivePreviewNode(ControlNode):
    """Real-time DLSS 5 preview with live sliders, plus a bake to the project's outputs."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self._applying_preset = False
        self._bake_thread: threading.Thread | None = None
        self._session_key = id(self)
        self._finalizer = weakref.finalize(self, _stop_session, self._session_key)

        # -- video in / out -----------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoArtifact", "VideoUrlArtifact", "dict"],
                type="VideoArtifact",
                output_type="VideoUrlArtifact",
                tooltip="Clip to preview. The right-hand port passes the original through untouched.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Baked video (after Bake, or after running the node in a flow).",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )

        # -- live preview ----------------------------------------------------------------
        self.add_parameter(
            ParameterDict(
                name="live_preview",
                default_value={"status": "stopped", "url": "", "message": "Press  Start live preview."},
                tooltip=(
                    "Live before/after view. Drag on the image to move the wipe; use the transport to pause, step and "
                    "scrub. Slider changes show up on the next frame."
                ),
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name=WIDGET_NAME, library=LIBRARY_NAME)},
                hide_label=True,
            )
        )
        self.add_node_element(
            ParameterButton(
                name="start_live",
                label="Start live preview",
                icon="play",
                variant="default",
                full_width=True,
                tooltip="Start (or restart) the resident DLSS 5 worker and loop the clip in the preview above.",
                on_click=self._on_start_clicked,
            )
        )
        self.add_node_element(
            ParameterButton(
                name="stop_live",
                label="Stop live preview",
                icon="square",
                variant="secondary",
                full_width=True,
                tooltip="Stop the loop and release the GPU worker.",
                on_click=self._on_stop_clicked,
            )
        )

        # -- look controls (live) --------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="look",
                input_types=["str"],
                type="str",
                default_value=LOOK_ULTRA,
                traits={Options(choices=LOOK_CHOICES)},
                tooltip="One-click look preset; sets the four sliders below. Touching a slider switches this to Custom.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(self._slider("local_tone", 1.0, 0.0, 2.0, "Low-frequency tone / lighting response. Live."))
        self.add_parameter(
            self._slider("local_structure", 2.0, 0.0, 2.0, "High-frequency detail: AO, reflections, materials. Live.")
        )
        self.add_parameter(
            self._slider(
                "skin_structure", 2.0, -1.0, 2.0,
                "Skin / pore reconstruction (needs auto_mask). -1 leaves it to the model. Live.",
            )
        )
        self.add_parameter(
            Parameter(
                name="auto_mask",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Let the model detect skin regions (gates skin_structure). Live.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="nr_style",
                input_types=["str"],
                type="str",
                default_value="Default",
                traits={Options(choices=list(NR_STYLES))},
                tooltip="Neural Rendering style: Natural stays closer to the source, Cinematic pushes contrast. Live.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="upscale_mode",
                input_types=["str"],
                type="str",
                default_value="1.0x (DLAA / native)",
                traits={Options(choices=list(UPSCALE_MODES))},
                tooltip="1.0x = neural rendering only; above adds DLSS Super Resolution. Hot-swaps the worker (~1.5 s).",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="temporal",
                input_types=["str"],
                type="str",
                default_value=TEMPORAL_SINGLE,
                traits={Options(choices=TEMPORAL_CHOICES)},
                tooltip=(
                    "Single Frame: each frame independent (no ghosting; recommended for generated video). "
                    "Sequence: temporal history + optical flow estimated inside the worker. Hot-swaps the worker."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # -- bake ------------------------------------------------------------------------
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=DEFAULT_OUTPUT_FILENAME)
        self._output_file.add_parameter()
        self.add_node_element(
            ParameterButton(
                name="bake",
                label="Bake full clip with these settings",
                icon="film",
                variant="default",
                full_width=True,
                tooltip="Render every frame with the current settings to output_file (project outputs) -> output_video.",
                on_click=self._on_bake_clicked,
            )
        )

        # -- advanced --------------------------------------------------------------------
        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced:
            Parameter(
                name="model_preset",
                input_types=["str"],
                type="str",
                default_value="M",
                traits={Options(choices=list(MODEL_PRESETS))},
                tooltip=(
                    "DLSS model preset. Measured to change nothing in the neural-rendering pass; it steers the "
                    "Super Resolution pass when upscaling. Hot-swaps the worker."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="dis_preset",
                input_types=["str"],
                type="str",
                default_value="Balanced (640p)",
                traits={Options(choices=list(DIS_PRESETS))},
                tooltip="Optical-flow quality for Sequence mode (higher = slower per frame). Hot-swaps the worker.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="keep_audio",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Bake: copy the source audio track into the output.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="browser_proxy",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip=(
                    f"Bake: when the render is larger than {PROXY_MAX_WIDTH}x{PROXY_MAX_HEIGHT}, also save a UHD H.264 "
                    "proxy (<name>_proxy.mp4) and put THAT on output_video so Display Video plays it smoothly. Browsers "
                    "cannot hardware-decode H.264 above 4096 px. output_file always keeps the full-resolution master."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="max_cache_gb",
                input_types=["float"],
                type="float",
                default_value=8.0,
                traits={Slider(min_val=0.25, max_val=64.0)},
                tooltip=(
                    "RAM budget for decoded frames. Longer clips loop only their first part when the budget is hit "
                    "(1080p ~ 6 MB/frame: 8 GB = ~55 s at 24 fps; 4K ~ 25 MB/frame: 8 GB = ~13 s). Applies on the next Start."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="preview_width",
                input_types=["int"],
                type="int",
                default_value=DEFAULT_PREVIEW_WIDTH,
                traits={Options(choices=[960, 1280, 1920, 2560, 3840])},
                tooltip=(
                    "Width the preview stream is encoded at (frames wider than this are downscaled for the widget only; "
                    "the worker still runs at full resolution and the bake is untouched). Lower = faster loop at 4K. "
                    "Applies on the next Start."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="preview_quality",
                input_types=["int"],
                type="int",
                default_value=85,
                traits={Slider(min_val=50, max_val=100)},
                tooltip="JPEG quality of the preview stream (bandwidth to the editor only; the bake is untouched).",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="runtime_dir",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip=(
                    "Optional: path to a 'DLSS 5 Visual Enhancer' install (its nvngx_dlssnr.dll / nvngx_dlss.dll are "
                    "loaded by the native worker - nothing is copied). Empty = library setting dlss5.runtime_dir, "
                    "then DLSS5_RUNTIME_DIR, then DLLs next to the worker."
                ),
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        self.add_node_element(advanced)

        # -- results ---------------------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="report",
                output_type="str",
                tooltip="Summary of the last bake.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        with ParameterGroup(name="Logs", ui_options={"collapsed": True}) as logs_group:
            Parameter(
                name="logs",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Worker / preview / bake log.",
                ui_options={"multiline": True, "placeholder_text": ""},
            )
        self.add_node_element(logs_group)

    @staticmethod
    def _slider(name: str, default: float, lo: float, hi: float, tooltip: str) -> Parameter:
        return Parameter(
            name=name,
            input_types=["float"],
            type="float",
            default_value=default,
            tooltip=tooltip,
            traits={Slider(min_val=lo, max_val=hi)},
            allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
        )

    # ------------------------------------------------------------------ helpers

    def _log(self, message: str) -> None:
        with contextlib.suppress(Exception):
            self.log_params.append_to_logs(message)

    def _float(self, name: str, default: float) -> float:
        try:
            return float(self.get_parameter_value(name))
        except (TypeError, ValueError):
            return default

    def _runtime_dir(self) -> str:
        runtime_dir = str(self.get_parameter_value("runtime_dir") or "").strip()
        if not runtime_dir:
            try:
                runtime_dir = str(GriptapeNodes.ConfigManager().get_config_value("dlss5.runtime_dir", default="") or "").strip()
            except Exception:  # noqa: BLE001 - settings are optional
                runtime_dir = ""
        return runtime_dir

    def _runtime(self) -> NativeRuntime:
        return find_native_runtime(self._runtime_dir())

    def _sequence(self) -> bool:
        return str(self.get_parameter_value("temporal") or TEMPORAL_SINGLE) == TEMPORAL_SEQUENCE

    def _settings(self) -> DLSS5Settings:
        clamp = lambda v, lo, hi: min(hi, max(lo, v))  # noqa: E731
        return DLSS5Settings(
            upscale_mode=str(self.get_parameter_value("upscale_mode") or "1.0x (DLAA / native)"),
            model_preset=str(self.get_parameter_value("model_preset") or "M"),
            nr_style=str(self.get_parameter_value("nr_style") or "Default"),
            nr_preset="Default",
            auto_mask=bool(self.get_parameter_value("auto_mask")),
            intensity=1.0,
            local_tone=clamp(self._float("local_tone", 1.0), 0.0, 2.0),
            local_structure=clamp(self._float("local_structure", 2.0), 0.0, 2.0),
            skin_structure=clamp(self._float("skin_structure", 2.0), -1.0, 2.0),
            mv_mode=MV_MODE_AUTO_DIS if self._sequence() else MV_MODE_NONE,
            dis_preset=str(self.get_parameter_value("dis_preset") or "Balanced (640p)"),
            warmup_frames=0,
        )

    def _session(self) -> LiveSession | None:
        with _sessions_lock:
            return _sessions.get(self._session_key)

    def _set_preview_value(self, status: str, url: str = "", message: str = "") -> None:
        value = {"status": status, "url": url, "message": message}
        self.set_parameter_value("live_preview", value)
        with contextlib.suppress(Exception):
            self.publish_update_to_parameter("live_preview", value)

    def _video_bytes(self) -> tuple[bytes, str]:
        video_input = self.parameter_values.get("video")
        video_value = _resolve_video_value(video_input)
        if not video_value:
            raise ValueError("Connect a video to the node first.")
        suffix = Path(video_value.split("?")[0]).suffix or ".mp4"
        return File(video_value).read_bytes(), suffix

    def _save_sibling(self, suffix: str, data: bytes) -> File:
        """Write ``<output_file stem><suffix>`` through the same project situation as output_file."""
        value = self.get_parameter_value("output_file")
        stem = Path(value if isinstance(value, str) and value else DEFAULT_OUTPUT_FILENAME).stem or "dlss5_live"
        dest = ProjectFileDestination.from_situation(f"{stem}{suffix}", ProjectFileParameter.DEFAULT_SITUATION, node_name=self.name)
        return dest.write_bytes(data)

    # ------------------------------------------------------------------ presets / live updates

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if self._applying_preset:
            return
        if parameter.name == "look":
            preset = LOOK_PRESETS.get(str(value))
            if preset is not None:
                self._applying_preset = True
                try:
                    for name, val in preset.items():
                        self.set_parameter_value(name, val)
                        self.publish_update_to_parameter(name, val)
                finally:
                    self._applying_preset = False
        elif parameter.name in LOOK_CONTROLS:
            current = str(self.get_parameter_value("look") or LOOK_CUSTOM)
            preset = LOOK_PRESETS.get(current)
            if preset is not None and preset.get(parameter.name) != value:
                self._applying_preset = True
                try:
                    self.set_parameter_value("look", LOOK_CUSTOM)
                    self.publish_update_to_parameter("look", LOOK_CUSTOM)
                finally:
                    self._applying_preset = False
        if parameter.name in LIVE_CONTROLS or parameter.name == "look":
            session = self._session()
            if session is not None and session.running:
                session.update_settings(self._settings(), sequence=self._sequence())
        if parameter.name == "video":
            session = self._session()
            if session is not None and session.running:
                self._log("Video changed - restarting the live preview on the new clip.\n")
                try:
                    self._start_live()
                except Exception as exc:  # noqa: BLE001 - surfaced in the widget + logs
                    self._set_preview_value("error", message=str(exc))
                    self._log(f"Live preview could not restart: {exc}\n")

    # ------------------------------------------------------------------ live

    def _on_start_clicked(self, *_args: Any) -> None:
        self.log_params.clear_logs()
        try:
            self._start_live()
        except Exception as exc:  # noqa: BLE001 - surfaced in the widget + logs
            self._set_preview_value("error", message=str(exc))
            self._log(f"Live preview could not start: {exc}\n")

    def _on_stop_clicked(self, *_args: Any) -> None:
        self._stop_live("Live preview stopped.")

    def _stop_live(self, message: str) -> None:
        _stop_session(self._session_key)
        self._set_preview_value("stopped", message=message)

    def _start_live(self) -> None:
        _stop_session(self._session_key)
        runtime = self._runtime()
        data, suffix = self._video_bytes()
        tmp = tempfile.NamedTemporaryFile(prefix="dlss5_live_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(data)
        cache_bytes = int(max(0.25, self._float("max_cache_gb", 8.0)) * 1024**3)
        quality = int(max(50, min(100, self._float("preview_quality", 85))))
        preview_width = int(self._float("preview_width", DEFAULT_PREVIEW_WIDTH))
        node_ref = weakref.ref(self)
        self._log(runtime.describe() + "\n")

        session = LiveSession(
            Path(tmp.name),
            runtime,
            self._settings(),
            sequence=self._sequence(),
            cache_bytes=cache_bytes,
            jpeg_quality=quality,
            preview_max_width=preview_width,
            log=self._log,
            on_state=lambda state, ref=node_ref: _on_session_state(ref, state),
            alive=_NodeAlive(self),
        )
        with _sessions_lock:
            _sessions[self._session_key] = session
        url = session.start()
        self._set_preview_value("running", url=url, message="starting worker...")
        self._log(f"Live preview at {url}\n")

    # ------------------------------------------------------------------ bake

    def _on_bake_clicked(self, *_args: Any) -> None:
        if self._bake_thread is not None and self._bake_thread.is_alive():
            self._log("A bake is already running.\n")
            return
        self.log_params.clear_logs()

        def run() -> None:
            try:
                self._bake()
            except Exception as exc:  # noqa: BLE001 - surfaced in the logs / report
                self._log(f"Bake failed: {exc}\n")
                report = f"bake failed: {exc}"
                self.set_parameter_value("report", report)
                with contextlib.suppress(Exception):
                    self.publish_update_to_parameter("report", report)

        self._bake_thread = threading.Thread(target=run, name="dlss5-bake", daemon=True)
        self._bake_thread.start()

    def process(self) -> AsyncResult:
        """Running the node in a flow = bake with the current settings."""
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        video_input = self.parameter_values.get("video")
        video_value = _resolve_video_value(video_input)
        if video_value:
            self.parameter_output_values["video"] = (
                video_input if isinstance(video_input, VideoUrlArtifact) else VideoUrlArtifact(video_value)
            )
        try:
            self._bake()
        except DLSS5WorkerError as exc:
            raise RuntimeError(str(exc)) from None

    def _bake(self) -> None:  # noqa: C901, PLR0915
        runtime = self._runtime()
        settings = self._settings()
        keep_audio = bool(self.get_parameter_value("keep_audio"))
        data, suffix = self._video_bytes()
        helpers = DLSS5NeuralRenderNode  # static ffmpeg helpers, shared with the render node

        # Pause the live loop while baking so the two workers do not compete for the GPU.
        session = self._session()
        was_playing = bool(session and session.running and session.playing)
        if session is not None and session.running:
            session.command(play=0)

        with _bake_lock, tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            input_path = temp_dir / f"input{suffix}"
            input_path.write_bytes(data)
            color = helpers._probe_color_info(input_path)  # noqa: SLF001
            self._log(
                f"Bake: {settings.upscale_mode}, {'sequence' if settings.mv_mode == MV_MODE_AUTO_DIS else 'single-frame'}, "
                f"style {settings.nr_style}, tone {settings.local_tone:.2f}, structure {settings.local_structure:.2f}, "
                f"skin {settings.skin_structure:.2f}, mask {'on' if settings.auto_mask else 'off'}\n"
            )
            reader = imageio.get_reader(str(input_path), format="FFMPEG")
            try:
                meta = reader.get_meta_data()
                fps = float(meta.get("fps") or 24.0) or 24.0
                first = _to_rgb(np.asarray(reader.get_data(0)))
                height, width = first.shape[:2]
                t0 = time.perf_counter()
                worker = NativeLiveWorker(settings, width, height, runtime)
                setup = worker.start()
                self._log(
                    f"Worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms; "
                    f"{width}x{height} -> {setup.output_width}x{setup.output_height} @ {fps:.3f} fps\n"
                )
                output_path = temp_dir / f"dlss5_{uuid.uuid4().hex}.mp4"
                writer = helpers._open_writer(output_path, fps, color)  # noqa: SLF001
                count = 0
                t_start = time.perf_counter()
                # Decode + send on a helper thread, receive + encode here, a few frames in flight. Sending and
                # receiving must live on different threads: the worker blocks on its stdout once the pipe is
                # full, so a single thread that sends ahead deadlocks against it.
                sequence = settings.mv_mode == MV_MODE_AUTO_DIS
                in_flight = threading.Semaphore(BAKE_PIPELINE_DEPTH)
                sent: queue.Queue[bool | None] = queue.Queue()
                stop = threading.Event()
                errors: list[BaseException] = []

                def send_loop() -> None:
                    try:
                        for index, frame in enumerate(reader.iter_data()):
                            rgb = _to_rgb(np.asarray(frame))
                            if rgb.shape[:2] != (height, width):
                                raise ValueError(
                                    f"Frame {index} is {rgb.shape[1]}x{rgb.shape[0]}, expected {width}x{height}"
                                )
                            in_flight.acquire()
                            if stop.is_set():
                                break
                            worker.send(rgb, settings=settings, reset=(index == 0 or not sequence))
                            sent.put(True)
                    except BaseException as exc:  # noqa: BLE001 - re-raised on this thread's owner
                        errors.append(exc)
                    finally:
                        sent.put(None)

                sender = threading.Thread(target=send_loop, name="dlss5-bake-sender", daemon=True)
                sender.start()
                try:
                    while True:
                        if sent.get() is None:
                            break
                        writer.append_data(worker.receive())
                        in_flight.release()
                        count += 1
                        if count % 48 == 0:
                            elapsed = (time.perf_counter() - t_start) * 1000
                            self._log(f"  ...{count} frames ({elapsed / count:.1f} ms/frame avg)\n")
                finally:
                    stop.set()
                    in_flight.release()
                    sender.join(timeout=5.0)
                    writer.close()
                    worker.close()
                if errors:
                    raise errors[0]
                if count == 0:
                    raise ValueError("Video contains no readable frames.")
                total_ms = (time.perf_counter() - t_start) * 1000
                self._log(f"Baked {count} frames, {total_ms / count:.1f} ms/frame wall.\n")

                audio_copied = False
                if keep_audio:
                    muxed = helpers._mux_audio(input_path, output_path, temp_dir)  # noqa: SLF001
                    if muxed is not None:
                        output_path, audio_copied = muxed, True
                        self._log("Copied source audio track.\n")
                    else:
                        self._log("No audio track in source (or mux failed); output is video-only.\n")

                saved = self._output_file.build_file().write_bytes(output_path.read_bytes())
                self._log(f"Saved video to {saved.resolve()}\n")
                port_file, port_note = saved, ""
                if bool(self.get_parameter_value("browser_proxy")) and helpers._needs_proxy(  # noqa: SLF001
                    setup.output_width, setup.output_height
                ):
                    t_proxy = time.perf_counter()
                    proxy = helpers._make_proxy(output_path, temp_dir)  # noqa: SLF001
                    if proxy is not None:
                        proxy_saved = self._save_sibling("_proxy.mp4", proxy.read_bytes())
                        size = helpers._proxy_size(proxy) or (PROXY_MAX_WIDTH, PROXY_MAX_HEIGHT)  # noqa: SLF001
                        port_file, port_note = proxy_saved, f" | port: {size[0]}x{size[1]} proxy"
                        self._log(
                            f"Render is above {PROXY_MAX_WIDTH}x{PROXY_MAX_HEIGHT} (no browser hardware decode). Saved a "
                            f"{size[0]}x{size[1]} proxy for output_video in {time.perf_counter() - t_proxy:.1f} s: "
                            f"{proxy_saved.resolve()}\n"
                        )
                    else:
                        self._log("Proxy encode failed; output_video carries the full-resolution master.\n")
                artifact = VideoUrlArtifact(port_file.location)
                report = (
                    f"frames: {count} | fps: {fps:.3f} | out: {setup.output_width}x{setup.output_height} | "
                    f"{settings.upscale_mode} | {total_ms / count:.1f} ms/frame | audio: {'copied' if audio_copied else 'none'}"
                    f"{port_note}"
                )
            finally:
                reader.close()

        self.parameter_output_values["output_video"] = artifact
        self.parameter_output_values["report"] = report
        self.set_parameter_value("report", report)
        with contextlib.suppress(Exception):
            self.publish_update_to_parameter("output_video", artifact)
            self.publish_update_to_parameter("report", report)
        self._log("Done. " + report + "\n")
        if session is not None and session.running and was_playing:
            session.command(play=1)


def _on_session_state(node_ref: weakref.ReferenceType, state: dict[str, Any]) -> None:
    """Session -> node: reflect errors / stop in the widget value (throttled by the session)."""
    node = node_ref()
    if node is None:
        return
    if state.get("error"):
        node._set_preview_value("error", message=str(state["error"]))  # noqa: SLF001
    elif not state.get("running") and state.get("message") == "stopped":
        node._set_preview_value("stopped", message="Live preview stopped.")  # noqa: SLF001


__all__ = ["DLSS5LivePreviewNode"]
