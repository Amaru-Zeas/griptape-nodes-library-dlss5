"""Client for the native worker's *live* mode (per-frame settings, resident process).

The native ``DLSS5Worker.exe`` (built by ``build_worker.ps1`` from ``worker_patches/``)
accepts a third header magic, ``'D5L1'``. In that mode

* pixels travel as RGBA8 both ways (the worker converts with F16C on the CPU),
* every frame is preceded by a 32-byte ``LiveSettings`` block that is applied before
  the evaluate - tone, structure, skin, mask and style change on the next frame,
* the process stays alive until stdin is closed, so a preview loop can run for hours.

Fixed at start: input size, upscale mode, motion-vector mode, DIS preset and the DLSS
model preset (the latter only steers the super-resolution pass; it has no measurable
effect on the neural-rendering pass). ``intensity`` below 1.0 merely darkens the image
on current runtimes, so callers should leave it at 1.0.

The worker is told where NVIDIA's DLLs live with ``--nr-dir``/``--sr-dir``; nothing is
copied. By default they are taken from the same "DLSS 5 Visual Enhancer" install that
the render node already uses (``dlss5.runtime_dir``), so there is no extra setup.

This module is additive: it imports from :mod:`dlss5_worker_bridge` and changes nothing
in it.
"""

from __future__ import annotations

import contextlib
import os
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from dlss5_worker_bridge import (
    DIS_PRESETS,
    MODEL_PRESETS,
    MV_MODE_EXTERNAL,
    NR_PRESETS,
    NR_STYLES,
    RUNTIME_DIR,
    DLSS5Settings,
    DLSS5Worker,
    DLSS5WorkerError,
    SetupInfo,
    _FRAME_HEADER,
    _FRAME_RESPONSE,
    _SETUP_RESPONSE,
    _VIDEO_HEADER,
    find_merserk_runtime,
    motion_to_rg16f,
    resolve_merserk_layout,
    rgb_u8_to_rgba8,
    rgba8_to_rgb_u8,
)

MAGIC_VIDEO_LIVE = 0x314C3544  # 'D5L1'
MAGIC_FRAME_LIVE = 0x324D5246  # 'FRM2'
MAGIC_OUT = 0x3154554F  # 'OUT1'

# struct LiveSettings { 4 x uint32, 4 x float }
_LIVE_SETTINGS = struct.Struct("<4I4f")
assert _LIVE_SETTINGS.size == 32

NR_SNIPPET = "nvngx_dlssnr.dll"
SR_SNIPPET = "nvngx_dlss.dll"
WORKER_EXE = "DLSS5Worker.exe"
SHIM_DLL = "nvngx.dll"

# Fields of DLSS5Settings that change on the next frame without restarting the worker
# (measured: all of them are evaluate-time NGX parameters).
LIVE_FIELDS = ("nr_style", "nr_preset", "auto_mask", "intensity", "local_tone", "local_structure", "skin_structure")
# Fields that need a new worker process: they size GPU resources / the wire format, or
# are read when the super-resolution feature is created (model_preset).
RESTART_FIELDS = ("upscale_mode", "mv_mode", "dis_preset", "model_preset")


# ------------------------------------------------------------------- runtime


@dataclass(frozen=True)
class NativeRuntime:
    """Resolved locations for the native worker and the NVIDIA runtimes it loads."""

    exe: Path
    nr_dir: Path  # folder holding nvngx_dlssnr.dll
    sr_dir: Path | None  # folder holding nvngx_dlss.dll (needed for upscaling only)

    @property
    def exe_dir(self) -> Path:
        return self.exe.parent

    def command(self) -> list[str]:
        cmd = [str(self.exe), "--nr-dir", str(self.nr_dir)]
        if self.sr_dir is not None:
            cmd += ["--sr-dir", str(self.sr_dir)]
        return cmd

    def describe(self) -> str:
        sr = str(self.sr_dir) if self.sr_dir else "(none - upscaling unavailable)"
        return f"worker {self.exe}\n  neural rendering: {self.nr_dir}\\{NR_SNIPPET}\n  super resolution: {sr}"


def find_native_runtime(
    runtime_dir: str | os.PathLike[str] | None = None,
    exe_dir: Path = RUNTIME_DIR,
) -> NativeRuntime:
    """Locate ``DLSS5Worker.exe`` + shim and the NVIDIA DLLs, or raise with instructions.

    DLL search order: next to the worker, then inside the DLSS 5 Visual Enhancer install
    given by ``runtime_dir`` / ``$DLSS5_RUNTIME_DIR`` (both flat and split layouts),
    then ``runtime_dir`` itself if the DLL sits directly in it.
    """
    exe = exe_dir / WORKER_EXE
    if not exe.is_file() or not (exe_dir / SHIM_DLL).is_file():
        raise DLSS5WorkerError(
            f"The native worker is not built ({WORKER_EXE} / {SHIM_DLL} missing in {exe_dir}). "
            "Run build_worker.ps1 in the library root (needs VS 2022 Build Tools with C++, CMake, Ninja)."
        )

    nr_dir: Path | None = None
    sr_dir: Path | None = None
    if (exe_dir / NR_SNIPPET).is_file():
        nr_dir = exe_dir
    if (exe_dir / SR_SNIPPET).is_file():
        sr_dir = exe_dir

    merserk_root = find_merserk_runtime(runtime_dir)
    if merserk_root is not None:
        layout = resolve_merserk_layout(merserk_root)
        if layout is not None:
            if nr_dir is None and layout.neural_runtime.is_file():
                nr_dir = layout.neural_runtime.parent
            if sr_dir is None and layout.superres.is_file():
                sr_dir = layout.superres.parent

    if runtime_dir and str(runtime_dir).strip():
        direct = Path(str(runtime_dir).strip()).expanduser()
        if nr_dir is None and (direct / NR_SNIPPET).is_file():
            nr_dir = direct
        if sr_dir is None and (direct / SR_SNIPPET).is_file():
            sr_dir = direct

    if nr_dir is None:
        raise DLSS5WorkerError(
            f"{NR_SNIPPET} (NVIDIA's DLSS 5 Neural Rendering runtime) was not found. It is not redistributed "
            "with this library. Point runtime_dir (or the DLSS5_RUNTIME_DIR environment variable) at an unzipped "
            "'DLSS 5 Visual Enhancer' release - the same folder the render node uses - or copy a legitimately "
            f"obtained {NR_SNIPPET} next to {exe}."
        )
    return NativeRuntime(exe=exe.resolve(), nr_dir=nr_dir.resolve(), sr_dir=sr_dir.resolve() if sr_dir else None)


# ------------------------------------------------------------------- packing


def pack_live_header(settings: DLSS5Settings, width: int, height: int) -> bytes:
    """96-byte VideoHeader with the live magic. Same layout as the Nuke header."""
    out_w, out_h = settings.output_size(width, height)
    dis_id, dis_w, dis_iters = DIS_PRESETS[settings.dis_preset]
    return _VIDEO_HEADER.pack(
        MAGIC_VIDEO_LIVE,
        int(width),
        int(height),
        out_w,
        out_h,
        0,  # warmup_frames
        0,  # frame_count: unbounded
        settings.perf_quality,
        MODEL_PRESETS[settings.model_preset],
        0,  # profile
        NR_PRESETS[settings.nr_preset],
        NR_STYLES[settings.nr_style],
        1 if settings.auto_mask else 0,
        0,  # ui_correction
        float(settings.intensity),
        float(settings.local_tone),
        float(settings.local_structure),
        float(settings.skin_structure),
        int(settings.mv_mode),
        dis_id,
        dis_w,
        dis_iters,
        0,
        0.0,
    )


def pack_live_settings(settings: DLSS5Settings) -> bytes:
    """32-byte LiveSettings block sent in front of every live frame."""
    return _LIVE_SETTINGS.pack(
        MODEL_PRESETS[settings.model_preset],
        NR_STYLES[settings.nr_style],
        1 if settings.auto_mask else 0,
        NR_PRESETS[settings.nr_preset],
        float(settings.intensity),
        float(settings.local_tone),
        float(settings.local_structure),
        float(settings.skin_structure),
    )


def needs_restart(old: DLSS5Settings, new: DLSS5Settings) -> bool:
    """True when the change cannot be applied per frame (see RESTART_FIELDS)."""
    return any(getattr(old, f) != getattr(new, f) for f in RESTART_FIELDS)


# -------------------------------------------------------------------- worker


class NativeLiveWorker(DLSS5Worker):
    """Resident native worker; every ``send`` may carry new settings.

    Usage::

        rt = find_native_runtime(runtime_dir)
        w = NativeLiveWorker(settings, width, height, rt)
        w.start()
        out = w.process(frame, settings=settings, reset=True)   # uint8 HxWx3
        out = w.process(frame2, settings=replace(settings, intensity=0.5))
        w.close()

    ``send``/``receive`` can be split across threads for pipelining exactly like the
    other workers; results come back in send order.
    """

    backend = "native-live"

    def __init__(self, settings: DLSS5Settings, width: int, height: int, runtime: NativeRuntime) -> None:
        super().__init__(settings, width, height, exe_path=runtime.exe)
        self.runtime = runtime
        self.last_worker_ms = 0.0  # time the worker spent on the last received frame
        self._motion_zero: bytes | None = None
        if settings.scale_factor > 1.0 and runtime.sr_dir is None:
            raise DLSS5WorkerError(
                f"{settings.upscale_mode} needs {SR_SNIPPET} (DLSS Super Resolution), which was not found in the "
                "runtime folder. Use 1.0x or add the DLSS 5 Visual Enhancer install to runtime_dir."
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> SetupInfo:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            self.runtime.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.runtime.exe_dir),
            creationflags=creationflags,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        try:
            self._write(pack_live_header(self.settings, self.width, self.height))
            raw = self._read(_SETUP_RESPONSE.size)
        except DLSS5WorkerError:
            time.sleep(0.05)  # let the stderr drain thread catch the reason
            self.close()
            raise DLSS5WorkerError(f"Native worker exited during setup. {self.stderr_text()}".strip()) from None

        setup = SetupInfo.from_bytes(bytes(raw), self.stderr_text())
        if not setup.setup_ok:
            time.sleep(0.05)
            self.close()
            raise DLSS5WorkerError(
                f"Native worker setup failed (ngx result 0x{setup.setup_result:08X}). {self.stderr_text()}".strip()
            )
        self.setup = setup
        self._frame_index = 0
        self._pending.clear()
        return setup

    def close(self) -> None:
        """Close stdin (the worker exits on EOF) and kill it if it has not left within a second."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        with contextlib.suppress(OSError):
            if proc.stdin:
                proc.stdin.close()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.5)

    # -- frames ------------------------------------------------------------

    def process(  # type: ignore[override]
        self,
        frame_rgb_u8: np.ndarray,
        settings: DLSS5Settings | None = None,
        reset: bool = False,
        motion: np.ndarray | None = None,
    ) -> np.ndarray:
        self.send(frame_rgb_u8, settings=settings, reset=reset, motion=motion)
        return self.receive()

    def send(  # type: ignore[override]
        self,
        frame_rgb_u8: np.ndarray,
        settings: DLSS5Settings | None = None,
        reset: bool = False,
        motion: np.ndarray | None = None,
        pts: int | None = None,
    ) -> int:
        """Queue one frame. ``settings`` (per-frame fields only) apply to this frame onwards."""
        if not self.running:
            raise DLSS5WorkerError(f"Native worker is not running. {self.stderr_text()}".strip())
        h, w = frame_rgb_u8.shape[:2]
        if (w, h) != (self.width, self.height):
            raise DLSS5WorkerError(f"Frame is {w}x{h}, worker was started for {self.width}x{self.height}")
        live = settings if settings is not None else self.settings
        if needs_restart(self.settings, live):
            raise DLSS5WorkerError(
                "upscale_mode / mv_mode / dis_preset cannot change on a running worker; start a new one."
            )
        self.settings = replace(self.settings, **{f: getattr(live, f) for f in LIVE_FIELDS})

        parts: list[bytes | memoryview] = [pack_live_settings(live), memoryview(rgb_u8_to_rgba8(frame_rgb_u8)).cast("B")]
        if self.settings.mv_mode == MV_MODE_EXTERNAL:
            if motion is None:
                if self._motion_zero is None:
                    self._motion_zero = bytes(h * w * 4)
                parts.append(self._motion_zero)
            else:
                parts.append(motion_to_rg16f(motion))

        with self._send_lock:
            index = self._frame_index
            self._frame_index += 1
            header = _FRAME_HEADER.pack(MAGIC_FRAME_LIVE, index, 1 if reset else 0, 0, int(pts if pts is not None else index))
            try:
                self._write_parts((header, *parts))
            except DLSS5WorkerError:
                raise DLSS5WorkerError(f"Native worker died while processing frame {index}. {self.stderr_text()}".strip()) from None
            self._pending.append(index)
        return index

    def _finish_receive(self, index: int, resp: tuple[int, ...]) -> np.ndarray:
        magic, out_index, ok, byte_count, ngx_result, _out_pts = resp
        if magic != MAGIC_OUT:
            raise DLSS5WorkerError(f"Bad FrameResponse magic 0x{magic:08X} on frame {index}")
        if ok != 1:
            raise DLSS5WorkerError(
                f"Native worker failed on frame {index} (ngx result 0x{ngx_result:08X}). {self.stderr_text()}".strip()
            )
        expected = self.out_width * self.out_height * 4
        if byte_count != expected:
            raise DLSS5WorkerError(f"Unexpected output size {byte_count} (expected {expected}) on frame {index}")
        self.last_worker_ms = float(ngx_result)  # on success the worker reports its own ms here
        raw = self._read(byte_count)
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(self.out_height, self.out_width, 4)
        return rgba8_to_rgb_u8(rgba)

    # -- plumbing ----------------------------------------------------------

    def _write_parts(self, parts: tuple[bytes | memoryview, ...]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise DLSS5WorkerError("worker stdin closed")
        try:
            for part in parts:
                view = memoryview(part)
                while len(view):
                    written = proc.stdin.write(view)
                    view = view[written or len(view):]
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5WorkerError(f"write failed: {exc}") from exc

    def _read(self, n: int) -> bytearray:  # type: ignore[override]
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
        return buf


__all__ = [
    "LIVE_FIELDS",
    "RESTART_FIELDS",
    "NativeLiveWorker",
    "NativeRuntime",
    "find_native_runtime",
    "needs_restart",
    "pack_live_header",
    "pack_live_settings",
]
