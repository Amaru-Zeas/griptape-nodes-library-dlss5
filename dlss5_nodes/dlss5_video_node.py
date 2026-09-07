"""DLSS 5 Neural Rendering node for Griptape Nodes.

Runs a video (or a single preview frame from it) through NVIDIA DLSS Neural
Rendering (Feature 18) and, optionally, DLSS Super Resolution, using the
standalone worker process from the DLSS5-for-Nuke project. See README.md for
runtime requirements (the NVIDIA ``nvngx_dlssnr.dll`` is NOT bundled).

Two modes:

* **Preview single frame** - decodes one frame, evaluates it with temporal
  history reset, and emits an ImageUrlArtifact. Fast enough to iterate on the
  neural sliders. The worker process is kept alive between runs.
* **Render full video** - streams every frame through the worker and encodes a
  new mp4 (optionally keeping the source audio track).
"""

from __future__ import annotations

import contextlib
import io
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image

try:
    import dlss5_worker_bridge  # noqa: F401
except ImportError:  # ensure sibling module resolves regardless of loader
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

from dlss5_worker_bridge import (  # noqa: E402
    BACKEND_MERSERK,
    DIS_PRESETS,
    DLSS5Settings,
    DLSS5Worker,
    DLSS5WorkerError,
    MERSERK_ENV,
    MODEL_PRESETS,
    MV_MODE_AUTO_DIS,
    MV_MODE_NONE,
    MerserkWorker,
    NR_PRESETS,
    NR_STYLES,
    TemporalGuide,
    UPSCALE_MODES,
    cv_thread_budget,
    fit_frame_rgb,
    resolve_backend,
)

MODE_RENDER = "Render full video -> output_video"
MODE_PREVIEW = "Test one frame only -> preview_image"
MODE_CHOICES = [MODE_RENDER, MODE_PREVIEW]

# One-click quality presets. They only drive the look controls below; render speed is the
# same for all of them (the neural pass is a fixed cost - only upscale_mode changes speed).
QUALITY_ULTRA = "Ultra (max realism)"
QUALITY_HIGH = "High"
QUALITY_MEDIUM = "Medium"
QUALITY_LOW = "Low (subtle)"
QUALITY_CUSTOM = "Custom (use the sliders)"
QUALITY_CHOICES = [QUALITY_ULTRA, QUALITY_HIGH, QUALITY_MEDIUM, QUALITY_LOW, QUALITY_CUSTOM]
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    QUALITY_ULTRA: {
        "model_preset": "M", "intensity": 1.0, "local_tone": 1.0,
        "local_structure": 2.0, "skin_structure": 2.0, "auto_mask": True,
    },
    QUALITY_HIGH: {
        "model_preset": "L", "intensity": 1.0, "local_tone": 1.0,
        "local_structure": 1.5, "skin_structure": 1.5, "auto_mask": True,
    },
    QUALITY_MEDIUM: {
        "model_preset": "K", "intensity": 1.0, "local_tone": 0.8,
        "local_structure": 1.0, "skin_structure": 1.0, "auto_mask": True,
    },
    QUALITY_LOW: {
        "model_preset": "Default", "intensity": 0.7, "local_tone": 0.5,
        "local_structure": 0.5, "skin_structure": 0.5, "auto_mask": True,
    },
}
QUALITY_CONTROLS = ("model_preset", "intensity", "local_tone", "local_structure", "skin_structure", "auto_mask")

PIPELINE_SINGLE = "Single Frame (no temporal history)"
PIPELINE_SEQUENCE = "Sequence (temporal, auto optical flow)"
PIPELINE_CHOICES = [PIPELINE_SINGLE, PIPELINE_SEQUENCE]

TRANSFER_ENCODED = "As encoded (display-referred)"
TRANSFER_LINEAR = "Linearize sRGB before DLSS"
TRANSFER_CHOICES = [TRANSFER_ENCODED, TRANSFER_LINEAR]

DEFAULT_FPS = 24.0
DEFAULT_OUTPUT_FILENAME = "dlss5.mp4"
# Largest frame a browser (and NVDEC) will hardware-decode as H.264. Renders above this get a proxy for the port.
PROXY_MAX_WIDTH = 3840
PROXY_MAX_HEIGHT = 2160
# Frames allowed in flight between the sender thread and the worker. 2 is enough to hide our CPU work behind
# the GPU; more only adds memory (each slot holds a full RGBA frame + motion buffer in the pipe).
PIPELINE_DEPTH = 2
# Concurrent DIS optical-flow solves in Sequence mode (each capped to a few OpenCV threads; see TemporalGuide).
FLOW_WORKERS = 3

# ffmpeg stream-tag name -> scale filter in/out_color_matrix value
_SCALE_MATRIX_MAP = {
    "bt709": "bt709",
    "bt470bg": "bt470",
    "smpte170m": "smpte170m",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
    "fcc": "fcc",
}
_VALID_COLORSPACES = set(_SCALE_MATRIX_MAP)
_VALID_PRIMARIES = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "bt2020", "film", "smpte428", "smpte431", "smpte432"}
_VALID_TRCS = {
    "bt709", "gamma22", "gamma28", "smpte170m", "smpte240m", "linear",
    "iec61966-2-1", "iec61966-2-4", "bt1361e", "bt2020-10", "bt2020-12",
    "smpte2084", "arib-std-b67",
}

# One live worker shared across runs so preview iterations skip D3D12/NGX init.
_worker_lock = threading.Lock()
_cached_worker: DLSS5Worker | None = None
_cached_key: tuple[Any, ...] | None = None


def _resolve_video_value(video_input: Any) -> str:
    if video_input is None:
        return ""
    if isinstance(video_input, dict):
        value = video_input.get("value") or video_input.get("url")
        return value if isinstance(value, str) else ""
    value = getattr(video_input, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(video_input, str):
        return video_input
    return ""


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return np.ascontiguousarray(frame[:, :, :3])
    return frame


def _acquire_native_worker(settings: DLSS5Settings, width: int, height: int, linear: bool, exe: Path) -> DLSS5Worker:
    """Return a running native worker for these settings, reusing the cached one when it matches."""
    global _cached_worker, _cached_key
    key = (
        settings.upscale_mode, settings.model_preset, settings.nr_style, settings.nr_preset,
        settings.auto_mask, round(settings.intensity, 4), round(settings.local_tone, 4),
        round(settings.local_structure, 4), round(settings.skin_structure, 4),
        settings.mv_mode, settings.dis_preset, width, height, linear, str(exe),
    )
    if _cached_worker is not None and _cached_key == key and _cached_worker.running:
        return _cached_worker
    if _cached_worker is not None:
        _cached_worker.close()
        _cached_worker = None
        _cached_key = None
    worker = DLSS5Worker(settings, width, height, exe_path=exe, linear_transfer=linear)
    worker.start()
    _cached_worker = worker
    _cached_key = key
    return worker


def _drop_cached_worker() -> None:
    global _cached_worker, _cached_key
    if _cached_worker is not None:
        _cached_worker.close()
    _cached_worker = None
    _cached_key = None


@dataclass
class _Job:
    """A frame that is prepared for the worker but not sent yet."""

    rgb: np.ndarray
    reset: bool
    motion: Future[np.ndarray] | None  # DIS solve in flight on the flow pool (sequence mode only)


class _Session:
    """One run's frame pipeline over either backend.

    native  : cached long-lived worker; optical flow happens inside the worker (mv_mode).
    merserk : fresh worker per run (ReShade.log is only flushed on exit); optical flow is
              computed here with TemporalGuide; feature-18 evidence checked on close().
    """

    def __init__(
        self,
        backend: str,
        root: Path,
        settings: DLSS5Settings,
        width: int,
        height: int,
        *,
        linear: bool,
        sequence: bool,
        scene_threshold: float,
        frame_count: int,
    ) -> None:
        self.backend = backend
        self.sequence = sequence
        self.worker: DLSS5Worker | MerserkWorker
        self._guide: TemporalGuide | None = None
        self._pool: ThreadPoolExecutor | None = None
        if backend == BACKEND_MERSERK:
            self.worker = MerserkWorker(settings, width, height, root, frame_count=frame_count)
            self.worker.start()
            self._guide = TemporalGuide(
                self.worker.render_width,
                self.worker.render_height,
                scene_change_threshold=scene_threshold,
                enabled=sequence,
            )
        else:
            self.worker = _acquire_native_worker(settings, width, height, linear, root / "DLSS5Worker.exe")
        self.out_width, self.out_height = self.worker.out_width, self.worker.out_height

    @property
    def render_size(self) -> tuple[int, int]:
        if isinstance(self.worker, MerserkWorker):
            return self.worker.render_width, self.worker.render_height
        return self.worker.width, self.worker.height

    def prepare(self, frame: np.ndarray, first: bool) -> _Job:
        """Sequential prep for one frame (letterbox, scene-cut check); the DIS solve goes to the flow pool."""
        if isinstance(self.worker, MerserkWorker):
            rw, rh = self.worker.render_width, self.worker.render_height
            rgb = fit_frame_rgb(frame, rw, rh)
            if self.sequence and self._guide is not None:
                current, previous, reset, _score = self._guide.prepare(rgb)
                motion: Future[np.ndarray] | None = None
                if current is not None and previous is not None:
                    motion = self._flow_pool().submit(self._guide.flow, current, previous)
                return _Job(rgb, reset or first, motion)
            return _Job(rgb, True, None)  # single-frame: no temporal history
        return _Job(frame, first or not self.sequence, None)

    def submit(self, job: _Job) -> None:
        """Hand a prepared frame to the worker (waits for its motion field if one is being solved)."""
        motion = job.motion.result() if job.motion is not None else None
        self.worker.send(job.rgb, reset=job.reset, motion=motion)

    def send(self, frame: np.ndarray, first: bool) -> None:
        self.submit(self.prepare(frame, first))

    def receive(self) -> np.ndarray:
        return self.worker.receive()

    def run(self, frame: np.ndarray, first: bool) -> np.ndarray:
        self.send(frame, first)
        return self.receive()

    def _flow_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=FLOW_WORKERS, thread_name_prefix="dlss5-flow")
        return self._pool

    def shutdown_flow_pool(self, *, cancel: bool) -> None:
        """Stop the flow pool; always waits for solves already running (they hold OpenCV threads)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=cancel)
            self._pool = None

    def finish(self, verify: bool) -> dict[str, Any] | None:
        """Close the run. Returns the feature-18 report for merserk (or raises if unverifiable)."""
        self.shutdown_flow_pool(cancel=False)
        if isinstance(self.worker, MerserkWorker):
            self.worker.close()
            if verify:
                return self.worker.feature_report()
            return None
        return None  # native worker stays cached for the next run

    def abort(self) -> None:
        self.shutdown_flow_pool(cancel=True)
        if isinstance(self.worker, MerserkWorker):
            self.worker.abort()
        else:
            _drop_cached_worker()


class DLSS5NeuralRenderNode(ControlNode):
    """Apply NVIDIA DLSS 5 Neural Rendering (+ optional DLSS Super Resolution) to a video."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)

        # -- video in / out at the top ------------------------------------------------
        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoArtifact", "VideoUrlArtifact", "dict"],
                type="VideoArtifact",
                output_type="VideoUrlArtifact",
                tooltip=(
                    "Input video to neural-render. The right-hand port passes the same original video through "
                    "untouched (e.g. into a Compare Video node next to output_video)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},  # in-port left, pass-through out-port right
                hide_property=True,  # ports only, no embedded player
            )
        )
        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Neural-rendered video. Connect to Display Video / Compare Video / Save Video.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,  # port only, no embedded player
            )
        )

        # -- main controls -----------------------------------------------------------
        self._applying_preset = False
        self.add_parameter(
            Parameter(
                name="quality",
                input_types=["str"],
                type="str",
                default_value=QUALITY_ULTRA,
                traits={Options(choices=QUALITY_CHOICES)},
                tooltip=(
                    "One-click look preset. Sets model_preset, intensity, local_tone, local_structure, skin_structure "
                    "and auto_mask. Ultra = the strongest settings measured (most skin/hair/material detail); "
                    "Low = a subtle touch. Touching any of those controls switches this to Custom. "
                    "Speed is identical for all presets - only upscale_mode changes render time."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="mode",
                input_types=["str"],
                type="str",
                default_value=MODE_RENDER,
                traits={Options(choices=MODE_CHOICES)},
                tooltip=(
                    "Render full video: process every frame; the result is on output_video. "
                    "Test one frame only: process a single frame in a few seconds and show it on preview_image "
                    "(Advanced group; handy for trying slider values before a long render)."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        # Project-aware save location: the filename is expanded through the project's 'save_node_output'
        # situation ({outputs}/... by default), so renders land in the project's output folder. The cog button
        # spawns a FileOutputSettings node for picking another situation / macro.
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=DEFAULT_OUTPUT_FILENAME)
        self._output_file.add_parameter()
        self.add_parameter(
            Parameter(
                name="upscale_mode",
                input_types=["str"],
                type="str",
                default_value="1.0x (DLAA / native)",
                traits={Options(choices=list(UPSCALE_MODES))},
                tooltip="1.0x = Neural Rendering only. Above 1.0x adds a DLSS Super Resolution pass to the output size.",
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
                tooltip="Neural Rendering style. Natural stays closer to the source, Cinematic deepens shadows / pushes contrast.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            self._slider(
                "intensity", 1.0, 0.0, 2.0,
                "Strength of the neural pass. Current runtimes clamp at 1.0 (values above do nothing); below 1.0 blends back toward the source.",
            )
        )
        self.add_parameter(self._slider("local_tone", 1.0, 0.0, 2.0, "Low-frequency tone / lighting response (Tone Intensity)."))
        self.add_parameter(
            self._slider(
                "local_structure", 2.0, 0.0, 2.0,
                "High-frequency detail: AO, reflections, materials (Structure Intensity). 2.0 = most realistic faces.",
            )
        )
        self.add_parameter(
            self._slider(
                "skin_structure", 2.0, -1.0, 2.0,
                "Skin / pore reconstruction. Only active while auto_mask is on (the mask locates skin). -1 leaves it to the model.",
            )
        )
        self.add_parameter(
            Parameter(
                name="auto_mask",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Let the model detect the regions it treats as skin. Also the gate for skin_structure.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # -- advanced (expanded by default) ------------------------------------------
        with ParameterGroup(name="Advanced") as advanced:
            Parameter(
                name="pipeline",
                input_types=["str"],
                type="str",
                default_value=PIPELINE_SINGLE,
                traits={Options(choices=PIPELINE_CHOICES)},
                tooltip=(
                    "Single Frame: every frame evaluated independently (no ghosting, recommended for generated video). "
                    "Sequence: keep temporal history and feed estimated optical flow (real footage with coherent motion)."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="scene_change_threshold",
                input_types=["float"],
                type="float",
                default_value=0.24,
                traits={Slider(min_val=0.01, max_val=1.0)},
                tooltip=(
                    "Sequence mode: mean luminance change (0-1) between frames above which temporal history is reset "
                    "(scene cut)."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="dis_preset",
                input_types=["str"],
                type="str",
                default_value="Balanced (640p)",
                traits={Options(choices=list(DIS_PRESETS))},
                tooltip="Optical-flow quality used in Sequence mode with the bundled native worker (higher = slower).",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="model_preset",
                input_types=["str"],
                type="str",
                default_value="M",
                traits={Options(choices=list(MODEL_PRESETS))},
                tooltip=(
                    "DLSS model preset. Measured on generated footage: Default/J/K are softest, L and M reconstruct "
                    "markedly more skin and hair texture. If the worker reports a different applied preset, use Default."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="nr_preset",
                input_types=["str"],
                type="str",
                default_value="Default",
                traits={Options(choices=list(NR_PRESETS))},
                tooltip="Neural Rendering tuning preset. Measured to have no effect on current runtime builds.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="color_transfer",
                input_types=["str"],
                type="str",
                default_value=TRANSFER_ENCODED,
                traits={Options(choices=TRANSFER_CHOICES)},
                tooltip=(
                    "Bundled native worker only. 'As encoded' matches a game backbuffer. "
                    "'Linearize' converts sRGB->linear before and back after (the Nuke node's convention)."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="keep_audio",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Copy the audio track from the source video into the rendered output.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="browser_proxy",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip=(
                    f"When the render is larger than {PROXY_MAX_WIDTH}x{PROXY_MAX_HEIGHT}, also save a UHD H.264 proxy "
                    "(<name>_proxy.mp4) and put THAT on output_video so Display Video plays it smoothly. Browsers cannot "
                    "hardware-decode H.264 above 4096 px. output_file always keeps the full-resolution master."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="preview_frame",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Only for 'Test one frame' mode: which frame to process (0 = first).",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
            Parameter(
                name="preview_image",
                output_type="ImageUrlArtifact",
                tooltip="Neural-rendered frame (Test one frame mode only).",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,  # port only, no thumbnail
            )
            Parameter(
                name="runtime_dir",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip=(
                    "Optional override: path to a 'DLSS 5 Visual Enhancer' install (app folder or its bin\\runtime). "
                    "Leave empty to use the library setting (Settings -> dlss5 -> runtime_dir), then the "
                    f"{MERSERK_ENV} environment variable, then the bundled native worker."
                ),
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
            Parameter(
                name="verify_neural_rendering",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip=(
                    "DLSS 5 Visual Enhancer worker only: after the run, check ReShade.log for signed feature-18 "
                    "execution and fail if it cannot be proven (otherwise plain upscaling would look like success)."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
        self.add_node_element(advanced)

        # -- results -----------------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="report",
                output_type="str",
                tooltip="Summary of what was processed.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        with ParameterGroup(name="Logs", ui_options={"collapsed": True}) as logs_group:
            Parameter(
                name="logs",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Worker and verification log for the last run.",
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

    # ------------------------------------------------------------------ presets
    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if self._applying_preset:
            return
        if parameter.name == "quality":
            preset = QUALITY_PRESETS.get(str(value))
            if preset is not None:
                self._applying_preset = True
                try:
                    for name, val in preset.items():
                        self.set_parameter_value(name, val)
                        self.publish_update_to_parameter(name, val)
                finally:
                    self._applying_preset = False
        elif parameter.name in QUALITY_CONTROLS:
            # The user moved a governed control by hand -> the preset no longer describes the node.
            current = str(self.get_parameter_value("quality") or QUALITY_CUSTOM)
            preset = QUALITY_PRESETS.get(current)
            if preset is not None and preset.get(parameter.name) != value:
                self._applying_preset = True
                try:
                    self.set_parameter_value("quality", QUALITY_CUSTOM)
                    self.publish_update_to_parameter("quality", QUALITY_CUSTOM)
                finally:
                    self._applying_preset = False

    # ------------------------------------------------------------------ main
    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _float(self, name: str, default: float) -> float:
        try:
            return float(self.get_parameter_value(name))
        except (TypeError, ValueError):
            return default

    def _settings(self) -> DLSS5Settings:
        pipeline = str(self.get_parameter_value("pipeline") or PIPELINE_SINGLE)
        clamp = lambda v, lo, hi: min(hi, max(lo, v))  # noqa: E731
        return DLSS5Settings(
            upscale_mode=str(self.get_parameter_value("upscale_mode") or "1.0x (DLAA / native)"),
            model_preset=str(self.get_parameter_value("model_preset") or "M"),
            nr_style=str(self.get_parameter_value("nr_style") or "Default"),
            nr_preset=str(self.get_parameter_value("nr_preset") or "Default"),
            auto_mask=bool(self.get_parameter_value("auto_mask")),
            intensity=clamp(self._float("intensity", 1.0), 0.0, 2.0),
            local_tone=clamp(self._float("local_tone", 1.0), 0.0, 2.0),
            local_structure=clamp(self._float("local_structure", 2.0), 0.0, 2.0),
            skin_structure=clamp(self._float("skin_structure", 2.0), -1.0, 2.0),
            mv_mode=MV_MODE_AUTO_DIS if pipeline == PIPELINE_SEQUENCE else MV_MODE_NONE,
            dis_preset=str(self.get_parameter_value("dis_preset") or "Balanced (640p)"),
            warmup_frames=0,  # the ComfyUI pack's tested default; the native worker ignores it
        )

    def _process(self) -> None:
        self.log_params.clear_logs()

        runtime_dir = str(self.get_parameter_value("runtime_dir") or "").strip()
        if not runtime_dir:
            # Library setting (Griptape Nodes -> Settings -> dlss5 -> runtime_dir).
            try:
                runtime_dir = str(GriptapeNodes.ConfigManager().get_config_value("dlss5.runtime_dir", default="") or "").strip()
            except Exception:  # noqa: BLE001 - settings are optional
                runtime_dir = ""
        try:
            backend, root = resolve_backend(runtime_dir)
        except DLSS5WorkerError as exc:
            raise RuntimeError(str(exc)) from None

        video_input = self.parameter_values.get("video")
        video_value = _resolve_video_value(video_input)
        if not video_value:
            raise ValueError("A video input is required.")
        # Pass the untouched source out of the video row's right-hand port (for compare nodes).
        self.parameter_output_values["video"] = (
            video_input if isinstance(video_input, VideoUrlArtifact) else VideoUrlArtifact(video_value)
        )

        mode = str(self.get_parameter_value("mode") or MODE_RENDER)
        if mode not in MODE_CHOICES:  # stale value from an older library version -> render
            mode = MODE_RENDER
        settings = self._settings()
        sequence = settings.mv_mode == MV_MODE_AUTO_DIS
        linear = str(self.get_parameter_value("color_transfer") or TRANSFER_ENCODED) == TRANSFER_LINEAR
        keep_audio = bool(self.get_parameter_value("keep_audio"))
        verify = bool(self.get_parameter_value("verify_neural_rendering"))
        scene_threshold = min(1.0, max(0.01, self._float("scene_change_threshold", 0.24)))

        backend_label = (
            f"DLSS 5 Visual Enhancer worker at {root}" if backend == BACKEND_MERSERK else f"bundled native worker at {root}"
        )
        self.log_params.append_to_logs(f"Backend: {backend_label}\n")
        if backend == BACKEND_MERSERK and linear:
            self.log_params.append_to_logs("Note: color_transfer is ignored by the Visual Enhancer worker (RGBA8 as encoded).\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            input_path = temp_dir / "input.mp4"
            input_path.write_bytes(File(video_value).read_bytes())

            color = self._probe_color_info(input_path)
            self.log_params.append_to_logs(
                f"Source color: matrix={color['matrix']} range={color['range']} "
                f"primaries={color['primaries']} trc={color['trc']}"
                + (" (defaults; source untagged)" if not color["tagged"] else "")
                + "\n"
            )

            reader = imageio.get_reader(
                str(input_path),
                format="FFMPEG",
                output_params=[
                    "-vf",
                    f"scale=in_color_matrix={_SCALE_MATRIX_MAP[color['matrix']]}:in_range={color['range']}",
                ],
            )
            try:
                meta = reader.get_meta_data()
                fps = float(meta.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
                size = meta.get("size") or (0, 0)
                width, height = int(size[0]), int(size[1])
                if width <= 0 or height <= 0:
                    first = _to_rgb(np.asarray(reader.get_data(0)))
                    height, width = first.shape[:2]
                out_w, out_h = settings.output_size(width, height)
                # The merserk worker treats the frame count as an exact contract. imageio's nframes is only an
                # estimate (duration x fps), so renders always use streaming mode (None); preview sends exactly 1.
                frame_count: int | None = 1 if mode == MODE_PREVIEW else None
                self.log_params.append_to_logs(
                    f"Input {width}x{height} @ {fps:.3f} fps -> output {out_w}x{out_h} "
                    f"[{settings.upscale_mode}, {'sequence' if sequence else 'single-frame'}]\n"
                )

                with _worker_lock:
                    t0 = time.perf_counter()
                    session = _Session(
                        backend, root, settings, width, height,
                        linear=linear, sequence=sequence, scene_threshold=scene_threshold, frame_count=frame_count,
                    )
                    rw, rh = session.render_size
                    self.log_params.append_to_logs(
                        f"Worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms"
                        + (f" (render size {rw}x{rh})" if (rw, rh) != (width, height) else "")
                        + ".\n"
                    )
                    try:
                        if mode == MODE_PREVIEW:
                            self._run_preview(reader, session, settings, verify)
                        else:
                            self._run_render(reader, session, settings, fps, color, input_path, temp_dir, keep_audio, verify)
                    except BaseException:
                        session.abort()
                        raise
            except DLSS5WorkerError as exc:
                raise RuntimeError(str(exc)) from None
            finally:
                reader.close()

    def _report_verification(self, report: dict[str, Any] | None) -> str:
        if report is None:
            return ""
        for line in report.get("evidence", []):
            self.log_params.append_to_logs(f"  [ReShade] {line}\n")
        if report.get("native_fallback"):
            self.log_params.append_to_logs(
                "Note: 'NR upscaling fell back to native' - DLSS upscaled first, then the neural pass ran at output "
                "resolution. Frames are still upscaled and neurally rendered.\n"
            )
        self.log_params.append_to_logs("Feature 18 (DLSS Neural Rendering) execution verified from ReShade.log.\n")
        return " | NR verified"

    # --------------------------------------------------------------- preview

    def _run_preview(self, reader: Any, session: _Session, settings: DLSS5Settings, verify: bool) -> None:
        try:
            index = int(self.get_parameter_value("preview_frame") or 0)
        except (TypeError, ValueError):
            index = 0
        index = max(0, index)

        frame: np.ndarray | None = None
        last_frame: np.ndarray | None = None
        last_ok = -1
        # imageio's random access is unreliable for some containers; walk forward.
        for i, f in enumerate(reader):
            last_ok = i
            last_frame = f
            if i == index:
                frame = _to_rgb(np.asarray(f))
                break
        if frame is None:
            if last_frame is None:
                raise ValueError("Video contains no readable frames.")
            self.log_params.append_to_logs(f"Frame {index} out of range; using last frame {last_ok}.\n")
            index = last_ok
            frame = _to_rgb(np.asarray(last_frame))

        t0 = time.perf_counter()
        out = session.run(frame, first=True)
        ms = (time.perf_counter() - t0) * 1000
        self.log_params.append_to_logs(f"Frame {index} rendered in {ms:.1f} ms -> {out.shape[1]}x{out.shape[0]}\n")
        verified = self._report_verification(session.finish(verify))

        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG", compress_level=1)
        saved = self._save_preview(buf.getvalue(), index)
        self.log_params.append_to_logs(f"Saved preview to {saved.resolve()}\n")
        self.parameter_output_values["preview_image"] = ImageUrlArtifact(saved.location)
        self.parameter_output_values["output_video"] = None
        report = f"preview frame {index} | {out.shape[1]}x{out.shape[0]} | {ms:.1f} ms | {settings.upscale_mode}{verified}"
        self.parameter_output_values["report"] = report
        self.log_params.append_to_logs("Done. " + report + "\n")

    def _save_preview(self, png: bytes, frame_index: int) -> File:
        """Write the preview PNG next to where the video would go (same project situation)."""
        return self._save_sibling(f"_preview_f{frame_index:04d}.png", png)

    def _save_sibling(self, suffix: str, data: bytes) -> File:
        """Write ``<output_file stem><suffix>`` through the same project situation as output_file."""
        value = self.get_parameter_value("output_file")
        stem = Path(value if isinstance(value, str) and value else DEFAULT_OUTPUT_FILENAME).stem or "dlss5"
        dest = ProjectFileDestination.from_situation(f"{stem}{suffix}", ProjectFileParameter.DEFAULT_SITUATION, node_name=self.name)
        return dest.write_bytes(data)

    # ---------------------------------------------------------------- render

    def _run_render(
        self,
        reader: Any,
        session: _Session,
        settings: DLSS5Settings,
        fps: float,
        color: dict[str, Any],
        input_path: Path,
        temp_dir: Path,
        keep_audio: bool,
        verify: bool,
    ) -> None:
        output_path = temp_dir / f"dlss5_{uuid.uuid4().hex}.mp4"
        writer = self._open_writer(output_path, fps, color)
        count = 0
        t_start = time.perf_counter()

        # Three-stage pipeline so the worker's GPU time overlaps all of our CPU work instead of adding to it:
        #   producer : decode -> letterbox -> scene-cut check, DIS solve queued on the flow pool
        #   sender   : waits for the motion field, pushes frame N+1 into the worker while N renders
        #   here     : receives finished frames in order and hands them to the encoder
        jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=PIPELINE_DEPTH + FLOW_WORKERS)
        sent: queue.Queue[bool | None] = queue.Queue()
        in_flight = threading.Semaphore(PIPELINE_DEPTH)
        stop = threading.Event()
        errors: list[BaseException] = []

        def produce() -> None:
            try:
                for i, f in enumerate(reader):
                    if stop.is_set():
                        break
                    jobs.put(session.prepare(_to_rgb(np.asarray(f)), first=(i == 0)))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                errors.append(exc)
            finally:
                jobs.put(None)

        def send_loop() -> None:
            try:
                while True:
                    job = jobs.get()
                    if job is None:
                        break
                    in_flight.acquire()
                    if stop.is_set():
                        break
                    session.submit(job)
                    sent.put(True)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                errors.append(exc)
            finally:
                sent.put(None)

        producer = threading.Thread(target=produce, name="dlss5-producer", daemon=True)
        sender = threading.Thread(target=send_loop, name="dlss5-sender", daemon=True)
        # OpenCV's thread pool must not be resized while any thread is inside cv2, so the cap is taken once
        # here for the whole render (TemporalGuide.flow's own request then nests as a no-op).
        budget = cv_thread_budget(TemporalGuide.DIS_THREADS)
        budget.__enter__()
        producer.start()
        sender.start()
        try:
            while True:
                item = sent.get()
                if item is None:
                    break
                out = session.receive()
                in_flight.release()
                writer.append_data(out)
                count += 1
                if count % 24 == 0:
                    elapsed = (time.perf_counter() - t_start) * 1000
                    self.log_params.append_to_logs(f"  ...{count} frames ({elapsed / count:.1f} ms/frame avg)\n")
        finally:
            stop.set()
            in_flight.release()  # unblock a sender waiting for a slot
            while producer.is_alive() or sender.is_alive():
                # Keep the job queue moving: free space for a blocked producer, wake a sender waiting for a job.
                with contextlib.suppress(queue.Empty):
                    while True:
                        jobs.get_nowait()
                with contextlib.suppress(queue.Full):
                    jobs.put_nowait(None)
                producer.join(timeout=0.2)
                sender.join(timeout=0.2)
            session.shutdown_flow_pool(cancel=True)  # waits for running solves
            budget.__exit__(None, None, None)  # every thread is out of cv2 now
            writer.close()
        if errors:
            raise errors[0]
        if count == 0:
            raise ValueError("Video contains no readable frames.")

        total_ms = (time.perf_counter() - t_start) * 1000
        self.log_params.append_to_logs(
            f"Rendered {count} frames, {total_ms / count:.1f} ms/frame wall (decode + flow + worker + encode, pipelined).\n"
        )
        verified = self._report_verification(session.finish(verify))

        audio_copied = False
        if keep_audio:
            muxed = self._mux_audio(input_path, output_path, temp_dir)
            if muxed is not None:
                output_path = muxed
                audio_copied = True
                self.log_params.append_to_logs("Copied source audio track.\n")
            else:
                self.log_params.append_to_logs("No audio track in source (or mux failed); output is video-only.\n")

        saved = self._output_file.build_file().write_bytes(output_path.read_bytes())
        self.log_params.append_to_logs(f"Saved video to {saved.resolve()}\n")
        port_file, port_note = saved, ""
        if bool(self.get_parameter_value("browser_proxy")) and self._needs_proxy(session.out_width, session.out_height):
            t_proxy = time.perf_counter()
            proxy = self._make_proxy(output_path, temp_dir)
            if proxy is not None:
                proxy_saved = self._save_sibling("_proxy.mp4", proxy.read_bytes())
                size = self._proxy_size(proxy) or (PROXY_MAX_WIDTH, PROXY_MAX_HEIGHT)
                port_file, port_note = proxy_saved, f" | port: {size[0]}x{size[1]} proxy"
                self.log_params.append_to_logs(
                    f"Render is above {PROXY_MAX_WIDTH}x{PROXY_MAX_HEIGHT} (no browser hardware decode). Saved a "
                    f"{size[0]}x{size[1]} proxy for output_video in {time.perf_counter() - t_proxy:.1f} s: {proxy_saved.resolve()}\n"
                )
            else:
                self.log_params.append_to_logs("Proxy encode failed; output_video carries the full-resolution master.\n")
        self.parameter_output_values["output_video"] = VideoUrlArtifact(port_file.location)
        self.parameter_output_values["preview_image"] = None
        report = (
            f"frames: {count} | fps: {fps:.3f} | out: {session.out_width}x{session.out_height} | "
            f"{settings.upscale_mode} | {total_ms / count:.1f} ms/frame | audio: {'copied' if audio_copied else 'none'}"
            f"{port_note}{verified}"
        )
        self.parameter_output_values["report"] = report
        self.log_params.append_to_logs("Done. " + report + "\n")

    # ---------------------------------------------------------------- ffmpeg

    @staticmethod
    def _probe_color_info(input_path: Path) -> dict[str, Any]:
        """Read matrix/range/primaries/trc from ffmpeg's stream info (BT.709 limited defaults)."""
        info: dict[str, Any] = {"matrix": "bt709", "range": "tv", "primaries": "bt709", "trc": "bt709", "tagged": False}
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run([ffmpeg_exe, "-hide_banner", "-i", str(input_path)], capture_output=True, text=True, check=False)
            video_line = next((ln for ln in (proc.stderr or "").splitlines() if ": Video:" in ln), "")
            tokens: list[str] = []
            for group in re.findall(r"\(([^()]*)\)", video_line):
                if re.search(r"\b(tv|pc|bt\d|smpte|fcc)", group):
                    tokens = [t.strip() for t in group.split(",")]
                    break
            for token in tokens:
                if token in ("tv", "pc"):
                    info["range"] = token
                    info["tagged"] = True
                elif "/" in token:
                    parts = token.split("/")
                    if parts[0] in _VALID_COLORSPACES:
                        info["matrix"] = parts[0]
                        info["tagged"] = True
                    if len(parts) > 1 and parts[1] in _VALID_PRIMARIES:
                        info["primaries"] = parts[1]
                    if len(parts) > 2 and parts[2] in _VALID_TRCS:
                        info["trc"] = parts[2]
                elif token in _VALID_COLORSPACES:
                    info["matrix"] = token
                    if token in _VALID_PRIMARIES:
                        info["primaries"] = token
                    if token in _VALID_TRCS:
                        info["trc"] = token
                    info["tagged"] = True
        except Exception:
            pass
        return info

    @staticmethod
    def _open_writer(output_path: Path, fps: float, color: dict[str, Any]) -> Any:
        scale_matrix = _SCALE_MATRIX_MAP[color["matrix"]]
        return imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=None,
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=[
                "-crf", "15",
                "-preset", "medium",
                "-vf", f"scale=out_color_matrix={scale_matrix}:out_range={color['range']}",
                "-colorspace", color["matrix"],
                "-color_primaries", color["primaries"],
                "-color_trc", color["trc"],
                "-color_range", color["range"],
                "-movflags", "+faststart",  # index at the front: the browser starts playing before the download ends
            ],
        )

    @staticmethod
    def _mux_audio(source_path: Path, video_path: Path, temp_dir: Path) -> Path | None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        muxed_path = temp_dir / "muxed.mp4"
        command = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-i", str(source_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(muxed_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and muxed_path.exists() and muxed_path.stat().st_size > 0:
            return muxed_path
        return None

    @staticmethod
    def _needs_proxy(width: int, height: int) -> bool:
        """Browsers hardware-decode H.264 only up to 4096 px (NVDEC limit); above UHD everything is CPU-decoded."""
        return width > PROXY_MAX_WIDTH or height > PROXY_MAX_HEIGHT

    @staticmethod
    def _make_proxy(video_path: Path, temp_dir: Path) -> Path | None:
        """Re-encode the master into a UHD-or-smaller H.264 level 5.1 file that Display Video can hardware-decode.

        The master is untouched; this is only what goes on the ``output_video`` port. Audio is copied as-is.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proxy_path = temp_dir / "proxy.mp4"
        fit = (
            f"scale=w='min(iw,{PROXY_MAX_WIDTH})':h='min(ih,{PROXY_MAX_HEIGHT})':force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        command = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", fit,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "19",
            "-profile:v", "high",
            "-level:v", "5.1",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(proxy_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and proxy_path.exists() and proxy_path.stat().st_size > 0:
            return proxy_path
        return None

    @staticmethod
    def _proxy_size(path: Path) -> tuple[int, int] | None:
        try:
            reader = imageio.get_reader(str(path), format="FFMPEG")
            try:
                size = reader.get_meta_data().get("size")
            finally:
                reader.close()
            if size:
                return int(size[0]), int(size[1])
        except Exception:  # noqa: BLE001 - informational only
            pass
        return None
