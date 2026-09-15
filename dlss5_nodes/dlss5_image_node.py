"""DLSS 5 Neural Rendering for a single image.

Same neural pass, same look controls and quality presets as the video node, applied to one
still. The image is evaluated as a single frame (no temporal history), which is exactly how
the video node's "Test one frame" mode works, so results match a Single Frame render of the
same picture. Alpha is preserved: the RGB goes through DLSS, the alpha channel is resized to
the output size and re-attached.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from PIL import Image

try:
    import dlss5_worker_bridge  # noqa: F401
except ImportError:  # ensure sibling modules resolve regardless of loader
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

from dlss5_video_node import (  # noqa: E402  (shared presets / session; the video node itself is untouched)
    QUALITY_CHOICES,
    QUALITY_CONTROLS,
    QUALITY_CUSTOM,
    QUALITY_PRESETS,
    QUALITY_ULTRA,
    TRANSFER_CHOICES,
    TRANSFER_ENCODED,
    TRANSFER_LINEAR,
    DLSS5NeuralRenderNode,
    _Session,
    _worker_lock,
)
from dlss5_worker_bridge import (  # noqa: E402
    BACKEND_MERSERK,
    MERSERK_ENV,
    MODEL_PRESETS,
    MV_MODE_NONE,
    NR_PRESETS,
    NR_STYLES,
    UPSCALE_MODES,
    DLSS5Settings,
    DLSS5WorkerError,
    resolve_backend,
)

DEFAULT_OUTPUT_FILENAME = "dlss5.png"
FORMAT_PNG = "PNG (lossless)"
FORMAT_JPEG = "JPEG (quality 95)"
FORMAT_WEBP = "WebP (quality 95)"
FORMAT_CHOICES = [FORMAT_PNG, FORMAT_JPEG, FORMAT_WEBP]
_FORMAT_EXT = {FORMAT_PNG: ".png", FORMAT_JPEG: ".jpg", FORMAT_WEBP: ".webp"}


def _image_bytes(image_input: Any) -> bytes:
    """Bytes of the connected image, whatever artifact form it arrived in."""
    if image_input is None:
        raise ValueError("An image input is required.")
    if isinstance(image_input, ImageUrlArtifact):
        return File(image_input.value).read_bytes()
    if isinstance(image_input, ImageArtifact):
        value = image_input.value
        return value if isinstance(value, bytes) else base64.b64decode(value)
    if isinstance(image_input, dict):
        value = image_input.get("value") or image_input.get("url")
        if isinstance(value, str):
            if str(image_input.get("type", "")).endswith("ImageArtifact") and not value.startswith(("http", "{", "/", "file:")):
                return base64.b64decode(value)
            return File(value).read_bytes()
        if isinstance(value, bytes):
            return value
    if isinstance(image_input, str) and image_input:
        return File(image_input).read_bytes()
    raise ValueError(f"Unsupported image input: {type(image_input).__name__}")


class DLSS5NeuralRenderImageNode(ControlNode):
    """Apply NVIDIA DLSS 5 Neural Rendering (+ optional DLSS Super Resolution) to one image."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)

        # -- image in / out at the top -----------------------------------------------
        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact", "dict"],
                type="ImageArtifact",
                output_type="ImageUrlArtifact",
                tooltip=(
                    "Input image to neural-render. The right-hand port passes the same original image through "
                    "untouched (e.g. into a Compare Images node next to output_image)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output_image",
                output_type="ImageUrlArtifact",
                tooltip="Neural-rendered image. Connect to Display Image / Compare Images / Save Image.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )

        # -- main controls (identical to the video node) -----------------------------
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
                    "and auto_mask. Ultra = the strongest settings measured; Low = a subtle touch. "
                    "Touching any of those controls switches this to Custom."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
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
        slider = DLSS5NeuralRenderNode._slider  # noqa: SLF001 - shared parameter factory
        self.add_parameter(
            slider(
                "intensity", 1.0, 0.0, 2.0,
                "Strength of the neural pass. Current runtimes clamp at 1.0 (values above do nothing); below 1.0 blends back toward the source.",
            )
        )
        self.add_parameter(slider("local_tone", 1.0, 0.0, 2.0, "Low-frequency tone / lighting response (Tone Intensity)."))
        self.add_parameter(
            slider(
                "local_structure", 2.0, 0.0, 2.0,
                "High-frequency detail: AO, reflections, materials (Structure Intensity). 2.0 = most realistic faces.",
            )
        )
        self.add_parameter(
            slider(
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

        # -- advanced --------------------------------------------------------------------
        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced:
            Parameter(
                name="output_format",
                input_types=["str"],
                type="str",
                default_value=FORMAT_PNG,
                traits={Options(choices=FORMAT_CHOICES)},
                tooltip="File format for output_file. PNG keeps alpha; JPEG drops it.",
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
                    "execution and fail if it cannot be proven."
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
            mv_mode=MV_MODE_NONE,  # a still has no motion: single-frame evaluation
            dis_preset="Balanced (640p)",
            warmup_frames=0,
        )

    def _process(self) -> None:
        self.log_params.clear_logs()

        runtime_dir = str(self.get_parameter_value("runtime_dir") or "").strip()
        if not runtime_dir:
            try:
                runtime_dir = str(GriptapeNodes.ConfigManager().get_config_value("dlss5.runtime_dir", default="") or "").strip()
            except Exception:  # noqa: BLE001 - settings are optional
                runtime_dir = ""
        try:
            backend, root = resolve_backend(runtime_dir)
        except DLSS5WorkerError as exc:
            raise RuntimeError(str(exc)) from None

        image_input = self.parameter_values.get("image")
        data = _image_bytes(image_input)
        # Pass the untouched source out of the image row's right-hand port (for compare nodes).
        if isinstance(image_input, ImageUrlArtifact):
            self.parameter_output_values["image"] = image_input
        elif isinstance(image_input, dict) and isinstance(image_input.get("value"), str) and not str(
            image_input.get("type", "")
        ).endswith("ImageArtifact"):
            self.parameter_output_values["image"] = ImageUrlArtifact(image_input["value"])

        pil = Image.open(io.BytesIO(data))
        pil.load()
        has_alpha = pil.mode in ("RGBA", "LA", "PA") or "transparency" in pil.info
        alpha = pil.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = np.ascontiguousarray(np.asarray(pil.convert("RGB")))
        height, width = rgb.shape[:2]

        settings = self._settings()
        linear = str(self.get_parameter_value("color_transfer") or TRANSFER_ENCODED) == TRANSFER_LINEAR
        verify = bool(self.get_parameter_value("verify_neural_rendering"))
        out_w, out_h = settings.output_size(width, height)

        backend_label = (
            f"DLSS 5 Visual Enhancer worker at {root}" if backend == BACKEND_MERSERK else f"bundled native worker at {root}"
        )
        self.log_params.append_to_logs(f"Backend: {backend_label}\n")
        self.log_params.append_to_logs(
            f"Input {width}x{height}{' + alpha' if alpha is not None else ''} -> output {out_w}x{out_h} [{settings.upscale_mode}]\n"
        )

        try:
            with _worker_lock:
                t0 = time.perf_counter()
                session = _Session(
                    backend, root, settings, width, height,
                    linear=linear, sequence=False, scene_threshold=0.24, frame_count=1,
                )
                self.log_params.append_to_logs(f"Worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms.\n")
                try:
                    t1 = time.perf_counter()
                    out = session.run(rgb, first=True)
                    ms = (time.perf_counter() - t1) * 1000
                    report_info = session.finish(verify)
                except BaseException:
                    session.abort()
                    raise
        except DLSS5WorkerError as exc:
            raise RuntimeError(str(exc)) from None
        verified = DLSS5NeuralRenderNode._report_verification(self, report_info)  # noqa: SLF001 - same log_params shape
        self.log_params.append_to_logs(f"Rendered in {ms:.1f} ms -> {out.shape[1]}x{out.shape[0]}\n")

        result = Image.fromarray(out)
        fmt = str(self.get_parameter_value("output_format") or FORMAT_PNG)
        if alpha is not None and fmt != FORMAT_JPEG:
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
        self.log_params.append_to_logs(f"Saved image to {saved.resolve()}\n")
        self.parameter_output_values["output_image"] = ImageUrlArtifact(saved.location)
        report = f"{width}x{height} -> {out.shape[1]}x{out.shape[0]} | {settings.upscale_mode} | {ms:.1f} ms{verified}"
        self.parameter_output_values["report"] = report
        self.log_params.append_to_logs("Done. " + report + "\n")
