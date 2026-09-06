"""Pure-Python clients for the two DLSS 5 neural-rendering worker processes.

Both workers evaluate NGX Feature 18 (DLSS Neural Rendering) and, when upscaling,
DLSS Super Resolution, and both speak a small packed binary protocol over
stdin/stdout with the same magic numbers, frame headers and setup response.

Backend "native" - ``runtime/DLSS5Worker.exe`` built from the ``worker/`` directory
of https://github.com/KJzzzKJ/DLSS5-for-Nuke . Calls the NGX snippet directly via a
caller shim, takes linear RGBA16F, estimates optical flow itself, and needs a
user-supplied ``nvngx_dlssnr.dll`` next to it.

    VideoHeader  (96 bytes)  ->  worker
    SetupResponse(48 bytes)  <-  worker
    per frame:
        FrameHeader(24 bytes) + RGBA16F pixels [+ RG16F motion] [+ f32 depth] [+ f32 mask]  -> worker
        FrameResponse(28 bytes) + RGBA16F pixels                                          <- worker

Backend "merserk" - the worker shipped inside Merserk's "DLSS 5 Visual Enhancer"
(``bin/runtime/nvngx.dll``, an executable despite the name; hosts ReShade + the
RenoDX DLSS 5 add-on). Protocol version 4, as documented by
https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer (MIT):

    VideoHeader  (72 bytes)  ->  worker
    SetupResponse(48 bytes)  <-  worker
    per frame:
        FrameHeader(24 bytes) + RGBA8 pixels (render size) + FP16 motion (render size)  -> worker
        FrameResponse(28 bytes) + RGBA8 pixels (output size)                            <- worker

Motion vectors for the merserk backend are estimated client-side (``TemporalGuide``,
OpenCV DIS) and successful Feature 18 execution is proven from ``ReShade.log``
after the worker exits (``verify_feature_18``).

This module has no Griptape dependency so it can be exercised from a plain script.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import struct
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # optional: only needed for client-side optical flow (merserk backend, Sequence mode)
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

# ----------------------------------------------------------------------- consts

MAGIC_VIDEO_RGBA16F = 0x35563544  # 'D5V5'
MAGIC_VIDEO_RGBA8 = 0x34563544  # 'D5V4' (legacy; not used here)
MAGIC_SETUP = 0x34505553  # 'SUP4'
MAGIC_FRAME = 0x314D5246  # 'FRM1'
MAGIC_OUT = 0x3154554F  # 'OUT1'
MAGIC_END = 0x31444E45  # 'END1' (merserk >= v6: completion record for unknown-length streams)

GUIDE_DEPTH = 1 << 0
GUIDE_CONTROL_MASK = 1 << 1

# struct VideoHeader { 14 x uint32, 4 x float, 5 x uint32, 1 x float }  (#pragma pack(1))  - native worker
_VIDEO_HEADER = struct.Struct("<14I4f5If")
# Protocol-4 VideoHeader { 14 x uint32, 4 x float }  - merserk worker
_V4_VIDEO_HEADER = struct.Struct("<14I4f")
# struct SetupResponse { 12 x uint32 }
_SETUP_RESPONSE = struct.Struct("<12I")
# struct FrameHeader { 4 x uint32, int64 }
_FRAME_HEADER = struct.Struct("<4Iq")
# struct FrameResponse { 5 x uint32, int64 }
_FRAME_RESPONSE = struct.Struct("<5Iq")

assert _VIDEO_HEADER.size == 96
assert _V4_VIDEO_HEADER.size == 72
assert _SETUP_RESPONSE.size == 48
assert _FRAME_HEADER.size == 24
assert _FRAME_RESPONSE.size == 28

RUNTIME_DIR = Path(__file__).parent / "runtime"
DEFAULT_WORKER_EXE = RUNTIME_DIR / "DLSS5Worker.exe"
NR_SNIPPET_NAME = "nvngx_dlssnr.dll"

# Merserk "DLSS 5 Visual Enhancer" runtime folder (bin/runtime). Two layouts exist:
#   flat  (<= v5): bin/runtime/{nvngx.dll, dxgi.dll, renodx-dlss5.addon64, nvngx_dlss.dll, nvngx_dlssnr.dll}
#   split (>= v6): bin/runtime/host/{nvngx.dll, dxgi.dll, ReShade.log}, dlss/nvngx_dlss.dll,
#                  dlssnr/{renodx-dlss5.addon64, nvngx_dlssnr.dll}
MERSERK_ENV = "DLSS5_RUNTIME_DIR"
MERSERK_WORKER_NAME = "nvngx.dll"  # executable; the snippet checks the caller's image name
MERSERK_REQUIRED_FILES = (  # kept for compatibility; see MerserkLayout.missing()
    MERSERK_WORKER_NAME,
    "dxgi.dll",
    "renodx-dlss5.addon64",
    "nvngx_dlss.dll",
    "nvngx_dlssnr.dll",
)
MERSERK_RESHADE_LOG = "ReShade.log"

BACKEND_NATIVE = "native"
BACKEND_MERSERK = "merserk"

# Output size limits enforced by the merserk worker (8K).
MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320

# Upscaling modes: (label, scale factor, NGX PerfQualityValue).
# PerfQuality follows the NGX enum: MaxPerf=0, Balanced=1, MaxQuality=2, UltraPerf=3, DLAA=5.
UPSCALE_MODES: dict[str, tuple[float, int]] = {
    "1.0x (DLAA / native)": (1.0, 5),
    "1.5x (Quality)": (1.5, 2),
    "1.72x (Balanced)": (1.7241379, 1),
    "2.0x (Performance)": (2.0, 0),
    "3.0x (Ultra Performance)": (3.0, 3),
}

# DLSS SR model presets -> NVSDK_NGX_DLSS_Hint_Render_Preset values.
MODEL_PRESETS: dict[str, int] = {"Default": 0, "J": 10, "K": 11, "L": 12, "M": 13}

NR_STYLES: dict[str, int] = {"Default": 0, "Natural": 1, "Cinematic": 2}
NR_PRESETS: dict[str, int] = {"Default": 0, "Preset #1": 1, "Preset #2": 2, "Preset #3": 3}

MV_MODE_NONE = 0
MV_MODE_EXTERNAL = 1
MV_MODE_AUTO_DIS = 2

# DIS optical-flow presets -> (preset id, flow width, iterations). Mirrors the Nuke node.
DIS_PRESETS: dict[str, tuple[int, int, int]] = {
    "Fast Preview (480p)": (0, 480, 12),
    "Balanced (640p)": (1, 640, 25),
    "High Quality (960p)": (2, 960, 32),
    "Extreme (1280p)": (3, 1280, 48),
}


# ---------------------------------------------------------------------- errors


class DLSS5WorkerError(RuntimeError):
    """Raised when the worker fails to start, initialise or process a frame."""


# -------------------------------------------------------------------- settings


def _even(value: float) -> int:
    """Round to the nearest even pixel count (half-up), matching the merserk worker."""
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


@dataclass
class DLSS5Settings:
    """Everything that goes into the worker's VideoHeader (except the image size).

    Defaults follow the measurements published by the ComfyUI-DLSS5-Enhancer
    project on generated footage: model preset M, automatic mask on, skin 2.0,
    local structure 1.5. Note that ``intensity`` is clamped to 1.0 by current
    runtimes (only values below 1.0 change the output) and ``nr_preset`` is inert.
    """

    upscale_mode: str = "1.0x (DLAA / native)"
    model_preset: str = "M"
    nr_style: str = "Default"
    nr_preset: str = "Default"
    auto_mask: bool = True  # also gates skin_structure
    intensity: float = 1.0
    local_tone: float = 1.0
    local_structure: float = 1.5
    skin_structure: float = 2.0  # -1 inherits local_structure; only active with auto_mask
    mv_mode: int = MV_MODE_NONE  # native worker only; merserk motion is client-side
    dis_preset: str = "Balanced (640p)"
    warmup_frames: int = 0  # extra evaluations of the first frame (merserk); ignored by the native worker

    @property
    def scale_factor(self) -> float:
        return UPSCALE_MODES[self.upscale_mode][0]

    @property
    def perf_quality(self) -> int:
        return UPSCALE_MODES[self.upscale_mode][1]

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """Output resolution for the given input, rounded to even numbers."""
        s = self.scale_factor
        out_w, out_h = _even(int(width) * s), _even(int(height) * s)
        if max(out_w, out_h) > MAX_LONG_EDGE or min(out_w, out_h) > MAX_SHORT_EDGE:
            usable = [
                label for label, (f, _) in UPSCALE_MODES.items()
                if max(_even(width * f), _even(height * f)) <= MAX_LONG_EDGE
                and min(_even(width * f), _even(height * f)) <= MAX_SHORT_EDGE
            ]
            hint = f" Use {usable[-1]} or lower for this source." if usable else " The source alone exceeds the limit."
            raise DLSS5WorkerError(
                f"A {out_w}x{out_h} output exceeds the supported {MAX_LONG_EDGE}x{MAX_SHORT_EDGE} boundary.{hint}"
            )
        return out_w, out_h

    def pack_header_v4(self, width: int, height: int, frame_count: int | None = None) -> bytes:
        """Protocol-4 header for the merserk worker.

        ``frame_count`` is a contract: the worker expects exactly that many frames.
        ``None`` (or 0) selects streaming mode for unknown-length input; the client
        then sends an END1 completion record (worker >= v6).
        """
        out_w, out_h = self.output_size(width, height)
        return _V4_VIDEO_HEADER.pack(
            MAGIC_VIDEO_RGBA8,
            int(width),
            int(height),
            out_w,
            out_h,
            max(0, int(self.warmup_frames)),
            0 if not frame_count else int(frame_count),
            self.perf_quality,
            MODEL_PRESETS[self.model_preset],
            0,  # profile
            NR_PRESETS[self.nr_preset],
            NR_STYLES[self.nr_style],
            1 if self.auto_mask else 0,
            0,  # ui_correction
            float(self.intensity),
            float(self.local_tone),
            float(self.local_structure),
            float(self.skin_structure),
        )

    def pack_header(self, width: int, height: int) -> bytes:
        """96-byte header for the native worker."""
        out_w, out_h = self.output_size(width, height)
        dis_id, dis_w, dis_iters = DIS_PRESETS[self.dis_preset]
        return _VIDEO_HEADER.pack(
            MAGIC_VIDEO_RGBA16F,
            int(width),
            int(height),
            out_w,
            out_h,
            max(0, int(self.warmup_frames)),
            100000,  # frame_count (informational)
            self.perf_quality,
            MODEL_PRESETS[self.model_preset],
            0,  # profile
            NR_PRESETS[self.nr_preset],
            NR_STYLES[self.nr_style],
            1 if self.auto_mask else 0,
            0,  # ui_correction
            float(self.intensity),
            float(self.local_tone),
            float(self.local_structure),
            float(self.skin_structure),
            int(self.mv_mode),
            dis_id,
            dis_w,
            dis_iters,
            0,  # _reserved_scene_cut
            0.0,  # _reserved_thresh
        )


@dataclass
class SetupInfo:
    setup_ok: bool
    setup_result: int
    render_width: int
    render_height: int
    output_width: int
    output_height: int
    applied_model_preset: int
    stderr: str = field(default="")

    @classmethod
    def from_bytes(cls, raw: bytes, stderr: str = "") -> "SetupInfo":
        vals = _SETUP_RESPONSE.unpack(raw)
        if vals[0] != MAGIC_SETUP:
            raise DLSS5WorkerError(f"Bad SetupResponse magic 0x{vals[0]:08X}. {stderr}".strip())
        return cls(
            setup_ok=vals[1] == 1,
            setup_result=vals[2],
            render_width=vals[3],
            render_height=vals[4],
            output_width=vals[5],
            output_height=vals[6],
            applied_model_preset=vals[11],
            stderr=stderr,
        )


# ------------------------------------------------------------------ runtime chk


def runtime_status(runtime_dir: Path = RUNTIME_DIR) -> dict[str, bool]:
    """Report which runtime pieces are present next to the worker."""
    return {
        "worker_exe": (runtime_dir / "DLSS5Worker.exe").is_file(),
        "caller_shim": (runtime_dir / "nvngx.dll").is_file(),
        "nr_snippet": (runtime_dir / NR_SNIPPET_NAME).is_file(),
    }


def runtime_problem(runtime_dir: Path = RUNTIME_DIR) -> str | None:
    """Human-readable description of what is missing for the native backend, or None."""
    status = runtime_status(runtime_dir)
    if not status["worker_exe"]:
        return (
            f"DLSS5Worker.exe not found in {runtime_dir}. Run build_worker.ps1 in the library root "
            "(needs VS2022 Build Tools, CMake and Ninja)."
        )
    if not status["caller_shim"]:
        return f"Caller shim nvngx.dll not found in {runtime_dir}. Re-run build_worker.ps1."
    if not status["nr_snippet"]:
        return (
            f"{NR_SNIPPET_NAME} not found in {runtime_dir}. This is NVIDIA's DLSS 5 Neural Rendering "
            "runtime and is NOT redistributed with this library. Obtain it legitimately (it ships with "
            "DLSS 5 game installs / the DLSS 5 SDK) and copy it next to DLSS5Worker.exe."
        )
    return None


@dataclass(frozen=True)
class MerserkLayout:
    """Resolved file locations inside a DLSS 5 Visual Enhancer runtime folder."""

    root: Path  # bin/runtime
    host_dir: Path  # worker + dxgi + ReShade.log live here (worker cwd)
    layout: str  # "flat" or "split"

    @property
    def worker(self) -> Path:
        return self.host_dir / MERSERK_WORKER_NAME

    @property
    def dxgi(self) -> Path:
        return self.host_dir / "dxgi.dll"

    @property
    def reshade_log(self) -> Path:
        return self.host_dir / MERSERK_RESHADE_LOG

    @property
    def addon(self) -> Path:
        return (self.root / "dlssnr" if self.layout == "split" else self.root) / "renodx-dlss5.addon64"

    @property
    def neural_runtime(self) -> Path:
        return (self.root / "dlssnr" if self.layout == "split" else self.root) / "nvngx_dlssnr.dll"

    @property
    def superres(self) -> Path:
        return (self.root / "dlss" if self.layout == "split" else self.root) / "nvngx_dlss.dll"

    def missing(self) -> list[str]:
        return [
            str(p.relative_to(self.root)) for p in (self.worker, self.dxgi, self.addon, self.superres, self.neural_runtime)
            if not p.is_file()
        ]


def _layout_at(root: Path) -> MerserkLayout | None:
    """Recognise a runtime root by its worker location; None if there is no worker."""
    if (root / "host" / MERSERK_WORKER_NAME).is_file():
        return MerserkLayout(root=root, host_dir=root / "host", layout="split")
    if (root / MERSERK_WORKER_NAME).is_file():
        return MerserkLayout(root=root, host_dir=root, layout="flat")
    return None


def resolve_merserk_layout(path: str | os.PathLike[str]) -> MerserkLayout | None:
    """Accept the app root, ``bin/runtime``, or the ``host`` folder and return the layout."""
    cand = Path(str(path).strip()).expanduser()
    roots = [cand, cand / "bin" / "runtime", cand / "runtime"]
    if cand.name.lower() == "host":
        roots.insert(0, cand.parent)
    for root in roots:
        layout = _layout_at(root)
        if layout is not None:
            return MerserkLayout(root=layout.root.resolve(), host_dir=layout.host_dir.resolve(), layout=layout.layout)
    return None


def merserk_runtime_problem(root: Path | str) -> str | None:
    """Describe what is missing from a Merserk runtime folder, or None if complete."""
    layout = resolve_merserk_layout(root)
    if layout is None:
        p = Path(root)
        if not p.is_dir():
            return f"{p} is not a directory."
        return f"No DLSS 5 Visual Enhancer worker ({MERSERK_WORKER_NAME}) found under {p}."
    missing = layout.missing()
    if missing:
        return (
            f"The DLSS 5 Visual Enhancer runtime in {layout.root} ({layout.layout} layout) is incomplete. "
            f"Missing: {', '.join(missing)}."
        )
    return None


def find_merserk_runtime(override: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate a Merserk runtime root: explicit override first, then $DLSS5_RUNTIME_DIR.

    Accepts the app install root, its ``bin/runtime`` folder, or the ``host`` folder.
    Returns the ``bin/runtime`` root, or None when nothing usable is found.
    """
    candidates: list[str] = []
    if override and str(override).strip():
        candidates.append(str(override))
    env_value = os.environ.get(MERSERK_ENV)
    if env_value:
        candidates.append(env_value)
    for cand in candidates:
        layout = resolve_merserk_layout(cand)
        if layout is not None:
            return layout.root
    return None


def resolve_backend(runtime_dir: str | os.PathLike[str] | None = None) -> tuple[str, Path]:
    """Decide which worker to use.

    Order: an explicit/env Merserk runtime folder wins; otherwise the bundled native
    worker if its NR snippet is present. Raises DLSS5WorkerError with instructions
    when neither is usable.
    """
    merserk = find_merserk_runtime(runtime_dir)
    if merserk is not None:
        problem = merserk_runtime_problem(merserk)
        if problem:
            raise DLSS5WorkerError(problem)
        return BACKEND_MERSERK, merserk
    if runtime_dir and str(runtime_dir).strip():
        hint = ""
        p = Path(str(runtime_dir).strip())
        if (p / "bin" / "README.md").is_file() and not (p / "bin" / "runtime").is_dir():
            hint = (
                " This looks like a git clone of the source repository; the worker binaries only ship in the "
                "release zip (https://github.com/Merserk/dlss5-visual-enhancer/releases). Unzip a release and "
                "point runtime_dir at it."
            )
        raise DLSS5WorkerError(
            f"No DLSS 5 Visual Enhancer worker ({MERSERK_WORKER_NAME}) found under {runtime_dir}. "
            f"Point runtime_dir at the unzipped app (or its bin\\runtime folder).{hint}"
        )
    native_problem = runtime_problem()
    if native_problem is None:
        return BACKEND_NATIVE, RUNTIME_DIR
    raise DLSS5WorkerError(
        "No usable DLSS 5 runtime. Either\n"
        "  (a) set runtime_dir (or the DLSS5_RUNTIME_DIR environment variable) to the bin\\runtime folder of a "
        "'DLSS 5 Visual Enhancer' install (https://github.com/Merserk/dlss5-visual-enhancer/releases), or\n"
        f"  (b) complete the bundled native worker: {native_problem}"
    )


# -------------------------------------------------------------------- pixels


def rgb_u8_to_rgba16f(frame: np.ndarray, linearize: bool = False) -> bytes:
    """uint8 HxWx3 (or HxWx4) -> tightly packed RGBA half-float rows, alpha = 1."""
    if frame.dtype != np.uint8:
        raise ValueError("expected uint8 frame")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 or HxWx4 frame, got {frame.shape}")
    rgb = frame[:, :, :3].astype(np.float32) * (1.0 / 255.0)
    if linearize:
        rgb = _srgb_to_linear(rgb)
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.float16)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 1.0
    return rgba.tobytes()


def rgba16f_to_rgb_u8(raw: bytes, width: int, height: int, delinearize: bool = False) -> np.ndarray:
    rgba = np.frombuffer(raw, dtype=np.float16).reshape(height, width, 4)
    rgb = rgba[:, :, :3].astype(np.float32)
    if delinearize:
        rgb = _linear_to_srgb(rgb)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


_CAST_THREADS = 8
_cast_pool: ThreadPoolExecutor | None = None
_cast_pool_lock = threading.Lock()


def to_float16(array: np.ndarray) -> np.ndarray:
    """float32 -> contiguous float16, casting row chunks on a thread pool.

    numpy's half-precision cast is scalar software code (~10 ms for a 1080p RG field); it releases
    the GIL, so splitting the rows over threads gets it down to ~2 ms.
    """
    global _cast_pool
    if array.dtype == np.float16:
        return np.ascontiguousarray(array)
    src = np.ascontiguousarray(array, dtype=np.float32)
    rows = src.shape[0]
    if src.ndim < 2 or rows < _CAST_THREADS * 16 or src.size < 262_144:
        return src.astype(np.float16)
    with _cast_pool_lock:
        if _cast_pool is None:
            _cast_pool = ThreadPoolExecutor(max_workers=_CAST_THREADS, thread_name_prefix="dlss5-f16")
    out = np.empty(src.shape, dtype=np.float16)
    bounds = [(i * rows // _CAST_THREADS, (i + 1) * rows // _CAST_THREADS) for i in range(_CAST_THREADS)]

    def cast(b: tuple[int, int]) -> None:
        np.copyto(out[b[0]:b[1]], src[b[0]:b[1]], casting="same_kind")

    list(_cast_pool.map(cast, bounds))
    return out


def motion_to_rg16f(motion: np.ndarray) -> bytes:
    """HxWx2 float motion vectors (pixels, pointing current -> previous) -> RG16F bytes."""
    if motion.ndim != 3 or motion.shape[2] != 2:
        raise ValueError(f"expected HxWx2 motion array, got {motion.shape}")
    return np.ascontiguousarray(motion, dtype=np.float16).tobytes()


def rgb_u8_to_rgba8(frame: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 (or HxWx4) -> contiguous HxWx4 uint8 with opaque alpha."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected uint8 HxWx3/4 frame, got {frame.dtype} {frame.shape}")
    if frame.shape[2] == 4:
        return np.ascontiguousarray(frame)
    if cv2 is not None:  # SIMD path, ~5x faster than the numpy copy below
        return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2RGBA)
    h, w = frame.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame
    rgba[:, :, 3] = 255
    return rgba


def rgba8_to_rgb_u8(rgba: np.ndarray) -> np.ndarray:
    """Contiguous uint8 HxWx4 -> contiguous HxWx3 (drops alpha)."""
    if cv2 is not None:
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(rgba[:, :, :3])


def fit_frame_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a uint8 HxWx3 frame into width x height without distortion."""
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return np.ascontiguousarray(frame[:, :, :3])
    scale = min(width / src_w, height / src_h)
    fit_w = max(1, min(width, int(round(src_w * scale))))
    fit_h = max(1, min(height, int(round(src_h * scale))))
    if cv2 is not None:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(frame[:, :, :3], (fit_w, fit_h), interpolation=interp)
    else:
        from PIL import Image  # PIL is a library dependency; cv2 is optional

        resized = np.asarray(Image.fromarray(frame[:, :, :3]).resize((fit_w, fit_h), Image.LANCZOS))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - fit_h) // 2, (width - fit_w) // 2
    canvas[top:top + fit_h, left:left + fit_w] = resized
    return canvas


# ------------------------------------------------------------- temporal guide


@dataclass(frozen=True)
class Guide:
    motion: np.ndarray  # HxWx2 float16, current -> previous, in render pixels
    reset: bool
    scene_score: float


class TemporalGuide:
    """Client-side motion vectors for the merserk backend (OpenCV DIS optical flow).

    DLSS wants backward motion (where each pixel came from) at the render size.
    Flow is computed on downscaled grayscale (<= ``flow_width`` px wide) and scaled
    up; a mean-luminance jump above ``scene_change_threshold`` resets history.
    With ``enabled=False`` zero motion is sent and history is reset only on the
    first frame.

    Two-phase use for pipelines: :meth:`prepare` is the cheap sequential part
    (grayscale + scene-cut score, keeps the frame history) and :meth:`flow` is the
    expensive, stateless DIS solve that may run on any thread, several frames at a
    time. :meth:`process` does both in one call.
    """

    # OpenCV's default pool (one thread per hardware thread) oversubscribes DIS badly on big CPUs:
    # measured 27 ms at 64 threads vs 10 ms at 4. Each solve is capped to this many threads.
    DIS_THREADS = 4

    def __init__(
        self,
        width: int,
        height: int,
        *,
        flow_width: int = 640,
        scene_change_threshold: float = 0.24,
        enabled: bool = True,
    ) -> None:
        self.width, self.height = int(width), int(height)
        self.enabled = enabled
        self.scene_change_threshold = float(scene_change_threshold)
        self.zero_motion = np.zeros((self.height, self.width, 2), dtype=np.float16)
        self._previous_gray: np.ndarray | None = None
        self._seen_first = False
        scale = min(1.0, flow_width / max(1, self.width))
        self.flow_width = max(64, int(round(self.width * scale / 2) * 2))
        self.flow_height = max(64, int(round(self.height * scale / 2) * 2))
        self._flow_scale = np.array([self.width / self.flow_width, self.height / self.flow_height], dtype=np.float32)
        self._local = threading.local()  # one DIS solver per calling thread (the solver is not thread-safe)
        if enabled and cv2 is None:
            raise DLSS5WorkerError(
                "Sequence mode with the DLSS 5 Visual Enhancer worker needs OpenCV for optical flow "
                "(pip install opencv-python-headless), or use Single Frame mode."
            )

    def _solver(self):  # noqa: ANN202 - cv2.DISOpticalFlow
        solver = getattr(self._local, "solver", None)
        if solver is None:
            solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            solver.setUseSpatialPropagation(True)
            solver.setFinestScale(1)
            self._local.solver = solver
        return solver

    def _gray(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)

    def prepare(self, rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, bool, float]:
        """Sequential phase. Returns (current_gray, previous_gray, reset, scene_score).

        ``current_gray`` is None when motion is disabled; ``previous_gray`` is None when there is
        no usable history (first frame or scene cut) - send zero motion in both cases.
        """
        if not self.enabled:
            first = not self._seen_first
            self._seen_first = True
            return None, None, first, 0.0
        current = self._gray(rgb)
        previous = self._previous_gray
        self._previous_gray = current
        if previous is None:
            return current, None, True, 1.0
        score = float(np.mean(cv2.absdiff(current, previous))) / 255.0
        if score > self.scene_change_threshold:
            return current, None, True, score
        return current, previous, False, score

    def flow(self, current_gray: np.ndarray, previous_gray: np.ndarray) -> np.ndarray:
        """Stateless phase: DIS flow current -> previous, upsampled to render size as float16 HxWx2."""
        with cv_thread_budget(self.DIS_THREADS):
            field = self._solver().calc(current_gray, previous_gray, None)
        field *= self._flow_scale  # scale while the field is still small
        field = cv2.resize(field, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return to_float16(field)

    def process(self, rgb: np.ndarray) -> Guide:
        current, previous, reset, score = self.prepare(rgb)
        if current is None or previous is None:
            return Guide(self.zero_motion, reset, score)
        return Guide(self.flow(current, previous), reset, score)


_cv_threads_lock = threading.Lock()
_cv_threads_users = 0
_cv_threads_saved: int | None = None


@contextlib.contextmanager
def cv_thread_budget(threads: int):  # noqa: ANN201
    """Cap OpenCV's global thread pool; restored when the last (outermost) user leaves.

    Resizing the pool while another thread is inside a cv2 call is unsafe, so pipelines should hold
    this for the whole render (nested uses are no-ops) rather than toggling it around each solve.
    """
    global _cv_threads_users, _cv_threads_saved
    if cv2 is None:
        yield
        return
    with _cv_threads_lock:
        if _cv_threads_users == 0:
            _cv_threads_saved = cv2.getNumThreads()
            if _cv_threads_saved > threads:
                cv2.setNumThreads(threads)
        _cv_threads_users += 1
    try:
        yield
    finally:
        with _cv_threads_lock:
            _cv_threads_users -= 1
            if _cv_threads_users == 0 and _cv_threads_saved is not None:
                cv2.setNumThreads(_cv_threads_saved)
                _cv_threads_saved = None


# -------------------------------------------------------- feature 18 evidence

# Version tolerant: the runtime version in the first marker changes between builds.
FEATURE_18_MARKERS = (
    ("signed DLSSNR runtime initialized", re.compile(r"signed DLSSNR [\d.]+ D3D12 runtime initialized")),
    ("feature 18 created", re.compile(r"feature 18 created via the signed snippet")),
    ("feature 18 evaluated", re.compile(r"inline feature 18 evaluation succeeded")),
)


def relevant_log_lines(reshade_log: str, limit: int = 60) -> list[str]:
    lines = reshade_log.splitlines()
    picked = [
        ln for ln in lines
        if "DLSS 5 Neural Rendering" in ln or "DLSSNR" in ln or "feature 18" in ln
        or "exception" in ln.lower() or "failed" in ln.lower()
    ]
    return (picked or lines)[-limit:]


def verify_feature_18(reshade_log: str) -> dict:
    """Return evidence that signed Feature 18 ran; raise DLSS5WorkerError if it cannot be shown.

    Without this a render can complete with plain DLSS upscaling and look like success.
    """
    missing = [name for name, pat in FEATURE_18_MARKERS if not pat.search(reshade_log)]
    if missing:
        evidence = "\n".join(relevant_log_lines(reshade_log, 40))
        raise DLSS5WorkerError(
            "Frames were rendered, but signed DLSSNR feature-18 execution was not verified. Missing evidence: "
            + "; ".join(missing)
            + (f"\n{evidence}" if evidence else "\n(ReShade.log was empty or not written)")
        )
    return {
        "verified": True,
        "native_fallback": "NR upscaling fell back to native" in reshade_log,
        "evidence": [
            ln for ln in reshade_log.splitlines()
            if "signed DLSSNR" in ln or "feature 18 created" in ln
            or "feature 18 evaluation succeeded" in ln or "NR upscaling fell back" in ln
        ][-20:],
    }


# -------------------------------------------------------------------- worker


class DLSS5Worker:
    """One worker process bound to a fixed input size and settings.

    Usage::

        worker = DLSS5Worker(settings, width, height)
        setup = worker.start()
        out = worker.process(rgb_u8_frame, reset=True)
        ...
        worker.close()

    Settings and size are fixed for the process lifetime (the worker allocates its
    D3D12 textures and NGX features once). Change anything -> new worker.
    """

    def __init__(
        self,
        settings: DLSS5Settings,
        width: int,
        height: int,
        exe_path: Path | str = DEFAULT_WORKER_EXE,
        linear_transfer: bool = False,
    ) -> None:
        self.settings = settings
        self.width = int(width)
        self.height = int(height)
        self.exe_path = Path(exe_path)
        self.linear_transfer = linear_transfer
        self.out_width, self.out_height = settings.output_size(self.width, self.height)
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._frame_index = 0
        self.setup: SetupInfo | None = None
        self._pending: deque[int] = deque()  # frame indices sent but not yet received
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> SetupInfo:
        problem = runtime_problem(self.exe_path.parent)
        if problem:
            raise DLSS5WorkerError(problem)

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [str(self.exe_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.exe_path.parent),
            creationflags=creationflags,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        try:
            self._write(self.settings.pack_header(self.width, self.height))
            raw = self._read(_SETUP_RESPONSE.size)
        except DLSS5WorkerError:
            self.close()
            raise DLSS5WorkerError(f"Worker exited during setup. {self.stderr_text()}".strip()) from None

        setup = SetupInfo.from_bytes(raw, self.stderr_text())
        if not setup.setup_ok:
            self.close()
            raise DLSS5WorkerError(
                f"Worker setup failed (ngx result 0x{setup.setup_result:08X}). {self.stderr_text()}".strip()
            )
        self.setup = setup
        self._frame_index = 0
        self._pending.clear()
        return setup

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> "DLSS5Worker":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- frames ------------------------------------------------------------

    def process(
        self,
        frame_rgb_u8: np.ndarray,
        reset: bool = False,
        motion: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        control_mask: np.ndarray | None = None,
        pts: int | None = None,
    ) -> np.ndarray:
        """Run one frame through the worker and return uint8 HxWx3 output."""
        self.send(frame_rgb_u8, reset=reset, motion=motion, depth=depth, control_mask=control_mask, pts=pts)
        return self.receive()

    def send(
        self,
        frame_rgb_u8: np.ndarray,
        reset: bool = False,
        motion: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        control_mask: np.ndarray | None = None,
        pts: int | None = None,
    ) -> int:
        """Queue one frame without waiting for the result (see MerserkWorker.send)."""
        if not self.running:
            raise DLSS5WorkerError(f"Worker is not running. {self.stderr_text()}".strip())
        h, w = frame_rgb_u8.shape[:2]
        if (w, h) != (self.width, self.height):
            raise DLSS5WorkerError(f"Frame is {w}x{h}, worker was started for {self.width}x{self.height}")

        guide_flags = 0
        payload = [rgb_u8_to_rgba16f(frame_rgb_u8, linearize=self.linear_transfer)]

        if self.settings.mv_mode == MV_MODE_EXTERNAL:
            if motion is None:
                motion = np.zeros((h, w, 2), dtype=np.float16)
            payload.append(motion_to_rg16f(motion))
        if depth is not None:
            guide_flags |= GUIDE_DEPTH
            payload.append(np.ascontiguousarray(depth, dtype=np.float32).reshape(h, w).tobytes())
        if control_mask is not None:
            guide_flags |= GUIDE_CONTROL_MASK
            payload.append(np.ascontiguousarray(control_mask, dtype=np.float32).reshape(h, w).tobytes())

        with self._send_lock:
            index = self._frame_index
            self._frame_index += 1
            header = _FRAME_HEADER.pack(MAGIC_FRAME, index, 1 if reset else 0, guide_flags, int(pts if pts is not None else index))
            try:
                self._write(header + b"".join(payload))
            except DLSS5WorkerError:
                raise DLSS5WorkerError(f"Worker died while processing frame {index}. {self.stderr_text()}".strip()) from None
            self._pending.append(index)
        return index

    def receive(self) -> np.ndarray:
        """Wait for the oldest outstanding frame and return it as uint8 HxWx3."""
        with self._recv_lock:
            if not self._pending:
                raise DLSS5WorkerError("receive() called with no frame outstanding")
            index = self._pending.popleft()
            try:
                resp = _FRAME_RESPONSE.unpack(self._read(_FRAME_RESPONSE.size))
            except DLSS5WorkerError:
                raise DLSS5WorkerError(f"Worker died while processing frame {index}. {self.stderr_text()}".strip()) from None
            return self._finish_receive(index, resp)

    @property
    def outstanding(self) -> int:
        return len(self._pending)

    def _finish_receive(self, index: int, resp: tuple[int, ...]) -> np.ndarray:
        magic, out_index, ok, byte_count, ngx_result, _out_pts = resp
        if magic != MAGIC_OUT:
            raise DLSS5WorkerError(f"Bad FrameResponse magic 0x{magic:08X} on frame {index}")
        if ok != 1:
            raise DLSS5WorkerError(
                f"Worker failed on frame {index} (ngx result 0x{ngx_result:08X}). {self.stderr_text()}".strip()
            )
        expected = self.out_width * self.out_height * 8
        if byte_count != expected:
            raise DLSS5WorkerError(f"Unexpected output size {byte_count} (expected {expected}) on frame {index}")
        raw = self._read(byte_count)
        return rgba16f_to_rgb_u8(raw, self.out_width, self.out_height, delinearize=self.linear_transfer)

    # -- plumbing ----------------------------------------------------------

    def stderr_text(self) -> str:
        return "".join(self._stderr_lines).strip()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            self._stderr_lines.append(line.decode("utf-8", errors="replace"))
            if len(self._stderr_lines) > 200:
                del self._stderr_lines[:-200]

    def _write(self, data: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise DLSS5WorkerError("worker stdin closed")
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5WorkerError(f"write failed: {exc}") from exc

    def _read(self, n: int) -> bytes:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise DLSS5WorkerError("worker stdout closed")
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                raise DLSS5WorkerError("worker closed its stdout")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


# --------------------------------------------------------- merserk worker


class MerserkWorker:
    """One session against the DLSS 5 Visual Enhancer worker (protocol version 4).

    Started once per render and closed at the end; ReShade only flushes its log on
    exit, so :meth:`feature_report` is available after :meth:`close`. Frames handed
    to :meth:`process` are letterboxed to the negotiated render size automatically.
    Motion vectors must be supplied by the caller (see :class:`TemporalGuide`);
    ``None`` sends zero motion.

    ``frame_count`` is a contract with the worker: pass the exact number of frames
    you will send, or ``None`` when unknown (streaming mode: the worker is told the
    count in an END1 record at close; needs Visual Enhancer >= v6).
    """

    backend = BACKEND_MERSERK

    def __init__(
        self,
        settings: DLSS5Settings,
        width: int,
        height: int,
        runtime_root: Path | str,
        frame_count: int | None = None,
        command: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self.width, self.height = int(width), int(height)
        self.runtime_root = Path(runtime_root)
        layout = resolve_merserk_layout(self.runtime_root)
        # Without a recognisable worker fall back to a flat layout rooted here (tests with a command override).
        self.layout = layout or MerserkLayout(root=self.runtime_root, host_dir=self.runtime_root, layout="flat")
        self.frame_count: int | None = int(frame_count) if frame_count else None
        self.streaming = self.frame_count is None
        # Test hook: replace the worker command line (default: <host>/nvngx.dll --video).
        self.command = command
        self.out_width, self.out_height = settings.output_size(self.width, self.height)
        self.render_width, self.render_height = self.width, self.height
        self.setup: SetupInfo | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._frame_index = 0
        self._closed = False
        self._started_at = 0.0
        self._pending: deque[int] = deque()  # frame indices sent but not yet received
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._zero_motion_cache = b""

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> SetupInfo:
        if self.command is None:
            problem = merserk_runtime_problem(self.runtime_root)
            if problem:
                raise DLSS5WorkerError(problem)
        self._started_at = time.time() - 1.0  # filesystem timestamps can lag the clock
        self._log_before = self._log_stat()  # an unchanged file afterwards is an old worker's log
        try:
            self._proc = subprocess.Popen(
                self.command or [str(self.layout.worker), "--video"],
                cwd=str(self.layout.host_dir),  # required: the ReShade carrier loads relative to cwd
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                bufsize=0,
            )
        except OSError as exc:
            raise DLSS5WorkerError(
                f"The DLSS 5 Visual Enhancer worker at {self.layout.worker} could not be "
                f"started: {exc}. It is an executable despite the .dll name, so antivirus software often blocks "
                "it; add the runtime folder to the exclusion list."
            ) from exc
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        try:
            self._write(self.settings.pack_header_v4(self.width, self.height, self.frame_count))
            raw = self._read(_SETUP_RESPONSE.size)
        except DLSS5WorkerError:
            code = self._wait(10)
            self.abort()
            raise DLSS5WorkerError(
                f"The worker failed during setup or does not speak the version-4 protocol (exit {code}).\n"
                + self.stderr_text()
                + (
                    "\nUnknown-length (streaming) input needs DLSS 5 Visual Enhancer v6.0 or newer; older workers "
                    "require an exact frame count."
                    if self.streaming
                    else ""
                )
            ) from None

        vals = _SETUP_RESPONSE.unpack(raw)
        if vals[0] != MAGIC_SETUP:
            self.abort()
            raise DLSS5WorkerError(
                f"Bad setup magic 0x{vals[0]:08X}: this runtime does not speak protocol version 4 "
                "(use DLSS 5 Visual Enhancer v3.0)."
            )
        setup = SetupInfo.from_bytes(raw, self.stderr_text())
        if not setup.setup_ok:
            self.abort()
            raise DLSS5WorkerError(
                f"DLSS {self.settings.upscale_mode} is unavailable for {self.out_width}x{self.out_height} "
                f"(NGX 0x{setup.setup_result:08X}). Pick a lower upscaling mode or update the NVIDIA driver.\n"
                + self.stderr_text()
            )
        if (setup.output_width, setup.output_height) != (self.out_width, self.out_height):
            self.abort()
            raise DLSS5WorkerError(
                f"The worker negotiated {setup.output_width}x{setup.output_height} instead of the requested "
                f"{self.out_width}x{self.out_height}."
            )
        requested_preset = MODEL_PRESETS[self.settings.model_preset]
        if setup.applied_model_preset != requested_preset:
            self.abort()
            raise DLSS5WorkerError(
                f"The worker applied DLSS model preset {setup.applied_model_preset} instead of the requested "
                f"{self.settings.model_preset} ({requested_preset}). This runtime does not support that model; "
                "set model_preset to Default."
            )
        if setup.render_width < 64 or setup.render_height < 64:
            self.abort()
            raise DLSS5WorkerError(
                f"DLSS returned an unusable render size {setup.render_width}x{setup.render_height}."
            )
        self.render_width, self.render_height = setup.render_width, setup.render_height
        self.setup = setup
        self._frame_index = 0
        self._pending.clear()
        return setup

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._closed

    def process(
        self,
        frame_rgb_u8: np.ndarray,
        reset: bool = False,
        motion: np.ndarray | None = None,
        pts: int | None = None,
        **_ignored: object,
    ) -> np.ndarray:
        """Send one frame (any size; letterboxed to render size) and return uint8 HxWx3 output."""
        self.send(frame_rgb_u8, reset=reset, motion=motion, pts=pts)
        return self.receive()

    def send(
        self,
        frame_rgb_u8: np.ndarray,
        reset: bool = False,
        motion: np.ndarray | None = None,
        pts: int | None = None,
    ) -> int:
        """Queue one frame for the worker without waiting for its result; returns the frame index.

        Pair every ``send`` with a later :meth:`receive`. Sending frame N+1 while frame N is
        still being rendered overlaps this process's decode / convert / optical-flow work with
        the worker's GPU time. Call ``send`` from one thread and ``receive`` from another (or
        the same one, sequentially); the two never overlap on the same pipe end.
        """
        if not self.running:
            raise DLSS5WorkerError(f"Worker is not running. {self.stderr_text()}".strip())
        rgb = fit_frame_rgb(frame_rgb_u8, self.render_width, self.render_height)
        rgba = rgb_u8_to_rgba8(rgb)
        if motion is None:
            motion_bytes: bytes | memoryview = self._zero_motion()
        else:
            if motion.shape[:2] != (self.render_height, self.render_width):
                raise DLSS5WorkerError(
                    f"Motion is {motion.shape[1]}x{motion.shape[0]}, render size is {self.render_width}x{self.render_height}"
                )
            motion_bytes = memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B")
        with self._send_lock:
            index = self._frame_index
            self._frame_index += 1
            header = _FRAME_HEADER.pack(MAGIC_FRAME, index, 1 if reset else 0, 0, int(pts if pts is not None else index))
            try:
                # Three writes instead of one concatenated copy: the frame goes straight from numpy to the pipe.
                self._write_parts((header, memoryview(rgba).cast("B"), motion_bytes))
            except DLSS5WorkerError:
                self._raise_failure(index)
            self._pending.append(index)
        return index

    def receive(self) -> np.ndarray:
        """Wait for the oldest outstanding frame and return it as uint8 HxWx3."""
        with self._recv_lock:
            if not self._pending:
                raise DLSS5WorkerError("receive() called with no frame outstanding")
            index = self._pending.popleft()
            try:
                resp = _FRAME_RESPONSE.unpack(self._read(_FRAME_RESPONSE.size))
            except DLSS5WorkerError:
                self._raise_failure(index)
            magic, out_index, ok, byte_count, ngx_result, _pts = resp
            if magic != MAGIC_OUT:
                raise DLSS5WorkerError(f"Bad FrameResponse magic 0x{magic:08X} on frame {index}")
            expected = self.out_width * self.out_height * 4
            if ok != 1 or out_index != index or byte_count != expected:
                raise DLSS5WorkerError(
                    f"Invalid response for frame {index} (index {out_index}, ok={ok}, {byte_count} of {expected} bytes, "
                    f"ngx 0x{ngx_result:08X}).\n" + self.stderr_text()
                )
            if ngx_result != 1:
                raise DLSS5WorkerError(f"Feature-18 evaluation failed on frame {index}: 0x{ngx_result:08X}")
            try:
                raw = self._read(byte_count)
            except DLSS5WorkerError:
                self._raise_failure(index)
        out = np.frombuffer(raw, dtype=np.uint8).reshape(self.out_height, self.out_width, 4)
        return rgba8_to_rgb_u8(out)

    @property
    def outstanding(self) -> int:
        """Frames sent but not yet received."""
        return len(self._pending)

    def _zero_motion(self) -> bytes:
        size = self.render_width * self.render_height * 4  # RG16F
        if len(self._zero_motion_cache) != size:
            self._zero_motion_cache = bytes(size)
        return self._zero_motion_cache

    def close(self) -> None:
        """Finish the stream and wait for a clean exit (flushes ReShade.log)."""
        if self._closed or self._proc is None:
            return
        while self._pending:  # drain anything still in flight so the frame count matches
            self.receive()
        sent = self._frame_index
        try:
            if self.streaming and sent > 0:
                # Completion record: tells the worker how many frames the stream had.
                self._write(_FRAME_HEADER.pack(MAGIC_END, sent, 0, 0, 0))
        except DLSS5WorkerError:
            pass  # a dead worker is reported via the exit code below
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        code = self._wait(60)
        if code is None:
            self.abort()
            raise DLSS5WorkerError("The DLSS 5 worker did not exit within 60 seconds and was killed.")
        ack_error = ""
        if self.streaming and sent > 0 and code == 0:
            # The 28-byte acknowledgement sits in the pipe after the worker exits.
            try:
                ack = _FRAME_RESPONSE.unpack(self._read(_FRAME_RESPONSE.size))
                if ack != (MAGIC_END, sent, 1, 0, 1, 0):
                    ack_error = f"invalid completion acknowledgement {ack}"
            except DLSS5WorkerError as exc:
                ack_error = f"missing completion acknowledgement ({exc})"
        self._closed = True
        self._finish()
        if code:
            raise DLSS5WorkerError(f"The DLSS 5 worker exited with code {code}:\n" + self.stderr_text())
        if ack_error:
            raise DLSS5WorkerError(f"The DLSS 5 worker returned an {ack_error} for {sent} frames.")

    def abort(self) -> None:
        """Kill the worker without raising."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass
        self._finish()

    def __enter__(self) -> "MerserkWorker":
        self.start()
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    # -- evidence ----------------------------------------------------------

    def _log_stat(self) -> tuple[float, int] | None:
        try:
            st = self.layout.reshade_log.stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    def reshade_log(self) -> str:
        """This worker's ReShade output ('' if the file is missing or predates the run).

        ReShade truncates the log on init and flushes it on exit, so the file holds
        exactly one worker's output and is complete only after close().
        """
        path = self.layout.reshade_log
        try:
            st = path.stat()
            if st.st_mtime < self._started_at or (st.st_mtime, st.st_size) == self._log_before:
                return ""  # belongs to an earlier worker
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # ReShade prefixes lines with [pid]; keep only this worker's lines when present.
        if self._proc is not None:
            marker = f"[{self._proc.pid}]"
            mine = [ln for ln in text.splitlines() if marker in ln]
            if mine:
                return "\n".join(mine)
        return text

    def feature_report(self) -> dict:
        """Prove signed Feature 18 ran. Only meaningful after close()."""
        if not self._closed:
            raise DLSS5WorkerError("feature_report() is only available after close().")
        return verify_feature_18(self.reshade_log())

    # -- plumbing ----------------------------------------------------------

    def stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-60:]).strip()

    def _raise_failure(self, index: int) -> None:
        code = self._wait(10)
        evidence = "\n".join([*self._stderr_lines[-40:], *relevant_log_lines(self.reshade_log(), 40)])
        access_violation = "0xC0000005" in evidence or (code is not None and (code & 0xFFFFFFFF) == 0xC0000005)
        self.abort()
        headline = (
            f"The DLSS 5 worker crashed inside feature-18 evaluation (access violation) on frame {index}. "
            "Update the NVIDIA driver and make sure the runtime files come from one matching release."
            if access_violation
            else f"The DLSS 5 worker stopped on frame {index} (exit {code})."
        )
        raise DLSS5WorkerError(f"{headline}\n{evidence}".strip()) from None

    def _wait(self, timeout: float) -> int | None:
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def _finish(self) -> None:
        proc = self._proc
        if proc is not None:
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
                    if len(self._stderr_lines) > 600:
                        # keep the first 100 (setup evidence) and the most recent 400
                        self._stderr_lines = self._stderr_lines[:100] + self._stderr_lines[-400:]
        except (OSError, ValueError):
            pass

    def _write(self, data: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise DLSS5WorkerError("worker stdin closed")
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5WorkerError(f"write failed: {exc}") from exc

    def _write_parts(self, parts: tuple[bytes | memoryview, ...]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise DLSS5WorkerError("worker stdin closed")
        try:
            for part in parts:  # unbuffered pipe: each part goes straight to the worker, no concatenation copy
                view = memoryview(part)
                while len(view):
                    written = proc.stdin.write(view)
                    view = view[written or len(view):]
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5WorkerError(f"write failed: {exc}") from exc

    def _read(self, n: int) -> bytearray:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise DLSS5WorkerError("worker stdout closed")
        buf = bytearray(n)
        view = memoryview(buf)
        offset = 0
        while offset < n:
            count = proc.stdout.readinto(view[offset:])
            if not count:
                raise DLSS5WorkerError(f"worker stopped after {offset} of {n} bytes")
            offset += count
        return buf  # no bytes() copy; callers unpack / frombuffer directly


DLSS5Worker.backend = BACKEND_NATIVE  # type: ignore[attr-defined]


def create_worker(
    settings: DLSS5Settings,
    width: int,
    height: int,
    *,
    runtime_dir: str | os.PathLike[str] | None = None,
    linear_transfer: bool = False,
    frame_count: int | None = None,
) -> "DLSS5Worker | MerserkWorker":
    """Pick a backend (see :func:`resolve_backend`) and return an un-started worker."""
    backend, root = resolve_backend(runtime_dir)
    if backend == BACKEND_MERSERK:
        return MerserkWorker(settings, width, height, root, frame_count=frame_count)
    return DLSS5Worker(settings, width, height, exe_path=root / "DLSS5Worker.exe", linear_transfer=linear_transfer)


__all__ = [
    "BACKEND_MERSERK",
    "BACKEND_NATIVE",
    "DIS_PRESETS",
    "DEFAULT_WORKER_EXE",
    "DLSS5Settings",
    "DLSS5Worker",
    "DLSS5WorkerError",
    "Guide",
    "MERSERK_ENV",
    "MODEL_PRESETS",
    "MV_MODE_AUTO_DIS",
    "MV_MODE_EXTERNAL",
    "MV_MODE_NONE",
    "MerserkLayout",
    "MerserkWorker",
    "NR_PRESETS",
    "NR_SNIPPET_NAME",
    "NR_STYLES",
    "RUNTIME_DIR",
    "SetupInfo",
    "TemporalGuide",
    "UPSCALE_MODES",
    "create_worker",
    "find_merserk_runtime",
    "fit_frame_rgb",
    "merserk_runtime_problem",
    "resolve_backend",
    "resolve_merserk_layout",
    "runtime_problem",
    "runtime_status",
    "verify_feature_18",
]
