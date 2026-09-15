"""DLSS 5 Live Image node.

Always-live still preview: connect an image and a landscape widget shows the neural
render on the left with look controls on the right. Slider changes hit the resident
native worker on the next evaluate. Separate from ``DLSS5NeuralRenderImageNode`` (the
one-shot bake node) which is left untouched.

Bake (button or running the node in a flow) writes ``output_image`` through the project's
outputs with the current widget settings.
"""

from __future__ import annotations

import contextlib
import io
import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.node_events import SetNodeMetadataRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.widget import Widget
from PIL import Image

try:
    import dlss5_worker_bridge  # noqa: F401
except ImportError:  # ensure sibling modules resolve regardless of loader
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

from dlss5_image_node import (  # noqa: E402
    FORMAT_CHOICES,
    FORMAT_JPEG,
    FORMAT_PNG,
    FORMAT_WEBP,
    _FORMAT_EXT,
    _image_bytes,
)
from dlss5_live_bridge import find_native_runtime  # noqa: E402
from dlss5_live_image_server import DEFAULT_PREVIEW_WIDTH, LiveImageSession  # noqa: E402
from dlss5_live_node import _NodeAlive  # noqa: E402
from dlss5_worker_bridge import (  # noqa: E402
    MERSERK_ENV,
    MODEL_PRESETS,
    MV_MODE_NONE,
    UPSCALE_MODES,
    DLSS5Settings,
)

LIBRARY_NAME = "GTN DLSS 5 Neural Rendering"
WIDGET_NAME = "DLSS5LiveImage"

LOOK_ULTRA = "Ultra (max realism)"
LOOK_HIGH = "High"
LOOK_MEDIUM = "Medium"
LOOK_LOW = "Low (subtle)"
LOOK_CUSTOM = "Custom (use the sliders)"

DEFAULT_LIVE = {
    "status": "stopped",
    "url": "",
    "message": "Connect an image — live preview starts automatically.",
    "look": LOOK_ULTRA,
    "local_tone": 1.0,
    "local_structure": 2.0,
    "skin_structure": 2.0,
    "auto_mask": True,
    "nr_style": "Default",
    "upscale_mode": "1.0x (DLAA / native)",
    "model_preset": "M",
    "view": "wipe",
    "wipe": 0.5,
}

DEFAULT_OUTPUT_FILENAME = "dlss5_live.png"
NODE_WIDTH_NORMAL = 860
NODE_WIDTH_SPLIT = 1340
# Title + ports + landscape widget (all look controls + Start/Stop/Full/Bake) + output_file + collapsed groups.
NODE_HEIGHT = 1100
WIDGET_HEIGHT = 720

# Transient flags the widget/node exchange; never persisted, never treated as settings.
_WIDGET_ONLY_KEYS = frozenset({"_fromWidget", "_action", "_fromNode"})

_sessions: dict[int, LiveImageSession] = {}
_sessions_lock = threading.Lock()
_bake_lock = threading.Lock()


def _stop_session(key: int) -> None:
    with _sessions_lock:
        session = _sessions.pop(key, None)
    if session is not None:
        session.stop()


class DLSS5LiveImageNode(ControlNode):
    """Real-time DLSS 5 still preview (landscape widget) plus bake to project outputs."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self._bake_thread: threading.Thread | None = None
        self._session_key = id(self)
        self._finalizer = weakref.finalize(self, _stop_session, self._session_key)
        self._applying_live = False
        self._alpha: Image.Image | None = None
        self._width_before_split: int | None = None
        self._last_pushed_state: tuple[str, str, str] | None = None
        self.set_initial_node_size(width=NODE_WIDTH_NORMAL, height=NODE_HEIGHT)
        # set_initial_node_size is a no-op once the canvas has stamped a size, so a node
        # dropped at the old 560/820 height would stay cropped. Floor it on every construct.
        self._ensure_min_size()

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "dict"],
                type="ImageArtifact",
                output_type="ImageUrlArtifact",
                tooltip="Still to preview. The right-hand port passes the original through untouched.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output_image",
                output_type="ImageUrlArtifact",
                tooltip="Baked image (after Bake, or after running the node in a flow).",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="live",
                default_value=dict(DEFAULT_LIVE),
                tooltip="Live before/after still. Look controls on the right apply on the next evaluate.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name=WIDGET_NAME, library=LIBRARY_NAME)},
                hide_label=True,
                ui_options={"height": WIDGET_HEIGHT, "is_full_width": True},
            )
        )

        self.add_node_element(
            ParameterButton(
                name="restart_live",
                label="Restart live preview",
                icon="refresh-cw",
                variant="secondary",
                full_width=True,
                tooltip="Restart the resident DLSS 5 worker on the current image (also happens automatically when the image changes).",
                on_click=self._on_restart_clicked,
                hide=True,
            )
        )
        self.add_node_element(
            ParameterButton(
                name="stop_live",
                label="Stop live preview",
                icon="square",
                variant="secondary",
                full_width=True,
                tooltip="Stop the worker and free the GPU. Connect/restart to start again.",
                on_click=self._on_stop_clicked,
                hide=True,
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=DEFAULT_OUTPUT_FILENAME)
        self._output_file.add_parameter()
        self.add_node_element(
            ParameterButton(
                name="bake",
                label="Bake image with these settings",
                icon="image",
                variant="secondary",
                icon_class="text-[#d9c6a4]",
                full_width=True,
                tooltip="Render the still with the current live settings to output_file -> output_image.",
                on_click=self._on_bake_clicked,
                hide=True,
            )
        )

        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced:
            Parameter(
                name="output_format",
                input_types=["str"],
                type="str",
                default_value=FORMAT_PNG,
                traits={Options(choices=FORMAT_CHOICES)},
                tooltip="Bake file format. PNG keeps alpha; JPEG drops it.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="preview_width",
                input_types=["int"],
                type="int",
                default_value=DEFAULT_PREVIEW_WIDTH,
                traits={Options(choices=[960, 1280, 1920, 2560])},
                tooltip="Max width of the live preview stream (worker + bake stay full-res). Applies on the next Restart.",
                allowed_modes={ParameterMode.PROPERTY},
            )
            Parameter(
                name="runtime_dir",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip=(
                    "Optional override: path to a 'DLSS 5 Visual Enhancer' install (app folder or its bin\\runtime). "
                    "Leave empty to use Settings -> dlss5 -> runtime_dir, then "
                    f"{MERSERK_ENV}, then the bundled native worker's sibling runtime."
                ),
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        self.add_node_element(advanced)

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
                tooltip="Live session / bake log.",
                ui_options={"multiline": True, "placeholder_text": ""},
            )
        self.add_node_element(logs_group)

    # ------------------------------------------------------------------ helpers

    def _log(self, message: str) -> None:
        self.log_params.append_to_logs(message)

    def _live_value(self) -> dict[str, Any]:
        merged = dict(DEFAULT_LIVE)
        value = self.get_parameter_value("live")
        if isinstance(value, dict):
            merged.update({k: v for k, v in value.items() if k not in _WIDGET_ONLY_KEYS})
        return merged

    def _float_from(self, data: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _settings(self) -> DLSS5Settings:
        live = self._live_value()
        clamp = lambda v, lo, hi: min(hi, max(lo, v))  # noqa: E731
        upscale = str(live.get("upscale_mode") or "1.0x (DLAA / native)")
        if upscale not in UPSCALE_MODES:
            upscale = "1.0x (DLAA / native)"
        model = str(live.get("model_preset") or "M")
        if model not in MODEL_PRESETS:
            model = "M"
        return DLSS5Settings(
            upscale_mode=upscale,
            model_preset=model,
            nr_style=str(live.get("nr_style") or "Default"),
            nr_preset="Default",
            auto_mask=bool(live.get("auto_mask", True)),
            intensity=1.0,
            local_tone=clamp(self._float_from(live, "local_tone", 1.0), 0.0, 2.0),
            local_structure=clamp(self._float_from(live, "local_structure", 2.0), 0.0, 2.0),
            skin_structure=clamp(self._float_from(live, "skin_structure", 2.0), -1.0, 2.0),
            mv_mode=MV_MODE_NONE,
            dis_preset="Balanced (640p)",
            warmup_frames=0,
        )

    def _runtime(self):
        runtime_dir = str(self.get_parameter_value("runtime_dir") or "").strip()
        if not runtime_dir:
            try:
                runtime_dir = str(
                    GriptapeNodes.ConfigManager().get_config_value("dlss5.runtime_dir", default="") or ""
                ).strip()
            except Exception:  # noqa: BLE001
                runtime_dir = ""
        return find_native_runtime(runtime_dir)

    def _session(self) -> LiveImageSession | None:
        with _sessions_lock:
            return _sessions.get(self._session_key)

    def _set_live_status(self, status: str, url: str = "", message: str = "") -> None:
        value = self._live_value()
        value["status"] = status
        if url or status != "running":
            value["url"] = url
        if message:
            value["message"] = message
        value["_fromNode"] = True
        self._last_pushed_state = (status, str(value["url"]), str(value["message"]))
        self._applying_live = True
        try:
            self.set_parameter_value("live", value)
            with contextlib.suppress(Exception):
                self.publish_update_to_parameter("live", value)
        finally:
            self._applying_live = False

    def _ensure_min_size(self) -> None:
        """Keep the canvas node at least tall/wide enough for the full landscape widget."""
        size = self.metadata.get("size") if isinstance(self.metadata.get("size"), dict) else {}

        def _int(raw: Any, default: int) -> int:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        want = {
            "width": max(_int(size.get("width"), NODE_WIDTH_NORMAL), NODE_WIDTH_NORMAL),
            "height": max(_int(size.get("height"), NODE_HEIGHT), NODE_HEIGHT),
        }
        if size.get("width") == want["width"] and size.get("height") == want["height"]:
            return
        self.metadata["size"] = want
        with contextlib.suppress(Exception):
            if getattr(self, "name", None):
                GriptapeNodes.handle_request(SetNodeMetadataRequest(node_name=self.name, metadata={"size": want}))

    def _sync_canvas_width(self, view: str) -> None:
        """Widen the node only for side-by-side so both stills fit; restore when leaving it.

        Never shrinks a node the user made wider, and leaves a manual resize alone.
        """
        size = self.metadata.get("size")
        current = size if isinstance(size, dict) else {}

        def _int(raw: Any, default: int) -> int:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        cur_w = _int(current.get("width"), NODE_WIDTH_NORMAL)
        cur_h = _int(current.get("height"), NODE_HEIGHT)
        if view == "split":
            if cur_w >= NODE_WIDTH_SPLIT:
                return
            if self._width_before_split is None:
                self._width_before_split = cur_w
            want_w = NODE_WIDTH_SPLIT
        else:
            if self._width_before_split is None:
                return
            want_w, self._width_before_split = self._width_before_split, None
            if cur_w != NODE_WIDTH_SPLIT:
                return  # user resized while in split; keep their size
        new_size = {"width": want_w, "height": max(cur_h, NODE_HEIGHT)}
        try:
            GriptapeNodes.handle_request(SetNodeMetadataRequest(node_name=self.name, metadata={"size": new_size}))
        except Exception:  # noqa: BLE001
            self.metadata["size"] = new_size

    def _load_rgb(self) -> np.ndarray:
        data = _image_bytes(self.parameter_values.get("image"))
        pil = Image.open(io.BytesIO(data))
        pil.load()
        has_alpha = pil.mode in ("RGBA", "LA", "PA") or "transparency" in pil.info
        self._alpha = pil.convert("RGBA").getchannel("A") if has_alpha else None
        return np.ascontiguousarray(np.asarray(pil.convert("RGB")))

    # ------------------------------------------------------------------ value changes

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "live":
            if self._applying_live:
                return
            if not (isinstance(value, dict) and value.get("_fromWidget")):
                return
            action = str(value.get("_action") or "")
            # Persist only real settings: a button press must never replay on reload,
            # and node-side status pushes must not look like widget edits.
            self.parameter_values["live"] = {k: v for k, v in value.items() if k not in _WIDGET_ONLY_KEYS}
            self._sync_canvas_width(str(value.get("view") or "wipe"))
            if action == "restart":
                self._on_restart_clicked()
                return
            if action == "stop":
                self._on_stop_clicked()
                return
            if action == "bake":
                self._on_bake_clicked()
                return
            session = self._session()
            if session is not None and session.running:
                session.update_settings(self._settings())  # no-op when /cmd already applied it
            return
        if parameter.name == "image":
            image_input = value
            if isinstance(image_input, ImageUrlArtifact):
                self.parameter_output_values["image"] = image_input
            try:
                self._start_live()
            except Exception as exc:  # noqa: BLE001
                self._set_live_status("error", message=str(exc))
                self._log(f"Live image could not start: {exc}\n")

    # ------------------------------------------------------------------ live

    def _on_restart_clicked(self, *_args: Any) -> None:
        self.log_params.clear_logs()
        try:
            self._start_live()
        except Exception as exc:  # noqa: BLE001
            self._set_live_status("error", message=str(exc))
            self._log(f"Live image could not start: {exc}\n")

    def _on_stop_clicked(self, *_args: Any) -> None:
        _stop_session(self._session_key)
        self._set_live_status("stopped", message="Live preview stopped.")

    def _start_live(self) -> None:
        _stop_session(self._session_key)
        runtime = self._runtime()
        rgb = self._load_rgb()
        preview_width = int(self.get_parameter_value("preview_width") or DEFAULT_PREVIEW_WIDTH)
        node_ref = weakref.ref(self)
        self._log(runtime.describe() + "\n")
        self._log(f"Live still {rgb.shape[1]}x{rgb.shape[0]}{' + alpha' if self._alpha is not None else ''}\n")

        session = LiveImageSession(
            rgb,
            runtime,
            self._settings(),
            preview_width=preview_width,
            log=self._log,
            on_state=lambda state, ref=node_ref: _on_session_state(ref, state),
            node_alive=_NodeAlive(self),
        )
        with _sessions_lock:
            _sessions[self._session_key] = session
        url = session.start()
        self._sync_canvas_width(str(self._live_value().get("view") or "wipe"))
        self._set_live_status("running", url=url, message="starting worker...")
        self._log(f"Live image preview at {url}\n")

    # ------------------------------------------------------------------ bake

    def _on_bake_clicked(self, *_args: Any) -> None:
        if self._bake_thread is not None and self._bake_thread.is_alive():
            self._log("A bake is already running.\n")
            return

        def run() -> None:
            try:
                self._bake()
            except Exception as exc:  # noqa: BLE001
                self._log(f"Bake failed: {exc}\n")
                report = f"bake failed: {exc}"
                self.set_parameter_value("report", report)
                with contextlib.suppress(Exception):
                    self.publish_update_to_parameter("report", report)

        self._bake_thread = threading.Thread(target=run, name="dlss5-live-image-bake", daemon=True)
        self._bake_thread.start()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        """Running the node in a flow starts live (if needed) then bakes."""
        self.log_params.clear_logs()
        image_input = self.parameter_values.get("image")
        if isinstance(image_input, ImageUrlArtifact):
            self.parameter_output_values["image"] = image_input
        session = self._session()
        if session is None or not session.running:
            try:
                self._start_live()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(str(exc)) from None
        self._bake()

    def _bake(self) -> None:
        import time

        from dlss5_live_bridge import NativeLiveWorker

        runtime = self._runtime()
        settings = self._settings()
        rgb = self._load_rgb()
        height, width = rgb.shape[:2]
        fmt = str(self.get_parameter_value("output_format") or FORMAT_PNG)

        with _bake_lock:
            self._log(
                f"Bake: {settings.upscale_mode}, style {settings.nr_style}, "
                f"tone {settings.local_tone:.2f}, structure {settings.local_structure:.2f}, "
                f"skin {settings.skin_structure:.2f}, mask {'on' if settings.auto_mask else 'off'}\n"
            )
            t0 = time.perf_counter()
            worker = NativeLiveWorker(settings, width, height, runtime)
            try:
                setup = worker.start()
                self._log(
                    f"Worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms; "
                    f"{width}x{height} -> {setup.output_width}x{setup.output_height}\n"
                )
                t1 = time.perf_counter()
                out = worker.process(rgb, settings=settings, reset=True)
                ms = (time.perf_counter() - t1) * 1000
            finally:
                worker.close()

        result = Image.fromarray(out)
        if self._alpha is not None and fmt != FORMAT_JPEG:
            alpha = self._alpha
            if alpha.size != result.size:
                alpha = alpha.resize(result.size, Image.LANCZOS)
            result = result.convert("RGBA")
            result.putalpha(alpha)

        buf = io.BytesIO()
        if fmt == FORMAT_JPEG:
            result.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
        elif fmt == FORMAT_WEBP:
            result.save(buf, format="WEBP", quality=95, method=4)
        else:
            result.save(buf, format="PNG", compress_level=3)

        dest = self._output_file.build_file(file_extension=_FORMAT_EXT[fmt].lstrip("."))
        saved = dest.write_bytes(buf.getvalue())
        self._log(f"Saved image to {saved.resolve()}\n")
        artifact = ImageUrlArtifact(saved.location)
        report = (
            f"{width}x{height} -> {out.shape[1]}x{out.shape[0]} | {settings.upscale_mode} | {ms:.1f} ms"
        )
        self.parameter_output_values["output_image"] = artifact
        self.parameter_output_values["report"] = report
        self.set_parameter_value("report", report)
        with contextlib.suppress(Exception):
            self.publish_update_to_parameter("output_image", artifact)
            self.publish_update_to_parameter("report", report)
        self._log("Done. " + report + "\n")


def _on_session_state(node_ref: weakref.ReferenceType, state: dict[str, Any]) -> None:
    node = node_ref()
    if node is None:
        return
    message = str(state.get("error") or state.get("message") or "")
    status = "error" if state.get("error") else ("running" if state.get("running") else "stopped")
    url = ""
    session = None
    with _sessions_lock:
        session = _sessions.get(node._session_key)
    if session is not None:
        url = session.url
    # The session reports after every evaluate (each slider tick). Only push to the GUI
    # when something the widget shows actually changed; otherwise every frame would
    # re-render the node and fight the slider being dragged.
    if node._last_pushed_state == (status, url, message):
        return
    try:
        node._set_live_status(status, url=url, message=message)
    except Exception:  # noqa: BLE001
        pass
