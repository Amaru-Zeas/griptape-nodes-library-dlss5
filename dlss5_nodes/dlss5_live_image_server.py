"""Live still preview: one image through the resident native worker, served as JPEG.

Reuses ``LiveServer`` from :mod:`dlss5_live_server` (same /frame.jpg, /state, /cmd, CORS).
A single background thread re-evaluates the still whenever look settings change; upscale /
model_preset hot-swap the worker. There is no playback loop - the composed frame is
republished whenever the result changes.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from dlss5_live_bridge import LIVE_FIELDS, NativeLiveWorker, NativeRuntime, needs_restart
from dlss5_live_server import (
    VIEW_AFTER,
    VIEW_BEFORE,
    VIEW_SPLIT,
    VIEW_WIPE,
    VIEWS,
    LiveServer,
)
from dlss5_worker_bridge import MV_MODE_NONE, DLSS5Settings, DLSS5WorkerError

DEFAULT_PREVIEW_WIDTH = 1280
IDLE_PAUSE_SECONDS = 30.0
DEBOUNCE_MS = 40.0  # coalesce rapid slider drags into one evaluate


class LiveImageSession:
    """Resident native worker evaluating one still; look controls apply on the next evaluate."""

    def __init__(
        self,
        source_rgb: np.ndarray,
        runtime: NativeRuntime,
        settings: DLSS5Settings,
        *,
        preview_width: int = DEFAULT_PREVIEW_WIDTH,
        log: Callable[[str], None] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        node_alive: Callable[[], bool] | None = None,
    ) -> None:
        if source_rgb.ndim != 3 or source_rgb.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 RGB, got shape {source_rgb.shape}")
        self.source = np.ascontiguousarray(source_rgb[:, :, :3], dtype=np.uint8)
        self.height, self.width = self.source.shape[:2]
        self.runtime = runtime
        self.preview_max_width = max(320, int(preview_width))
        self._log = log or (lambda _m: None)
        self._on_state = on_state
        self._alive = node_alive or (lambda: True)

        self._lock = threading.Lock()
        self._settings = replace(settings, mv_mode=MV_MODE_NONE, intensity=1.0)
        self._view = VIEW_WIPE
        self._wipe = 0.5
        self._dirty = True
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: NativeLiveWorker | None = None
        self._out: np.ndarray | None = None
        self.out_width = self.width
        self.out_height = self.height
        self.worker_ms = 0.0
        self.encode_ms = 0.0
        self.message = "starting"
        self.error: str | None = None
        self.running = False
        self.seq = 0

        self.server = LiveServer(on_command=self._command, get_state=self.state)
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ public

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.port}" if self.server.port else ""

    def start(self) -> str:
        if self.running:
            return self.url
        if cv2 is None:
            raise RuntimeError("opencv-python-headless is required for the live image preview.")
        base = self.server.start()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="dlss5-live-image", daemon=True)
        self._thread.start()
        return base

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None
        self.server.stop()
        self.running = False

    def update_settings(self, settings: DLSS5Settings) -> None:
        with self._lock:
            self._settings = replace(settings, mv_mode=MV_MODE_NONE, intensity=1.0)
            self._dirty = True
        self._wake.set()

    def command(self, **kwargs: Any) -> None:
        self._command({k: str(v) for k, v in kwargs.items()})

    def state(self) -> dict[str, Any]:
        with self._lock:
            s = self._settings
        return {
            "running": self.running,
            "message": self.message,
            "error": self.error,
            "width": self.width,
            "height": self.height,
            "out_width": self.out_width,
            "out_height": self.out_height,
            "worker_ms": round(self.worker_ms, 1),
            "encode_ms": round(self.encode_ms, 1),
            "view": self._view,
            "wipe": round(self._wipe, 3),
            "seq": self.seq,
            "look": {
                "nr_style": s.nr_style,
                "nr_preset": s.nr_preset,
                "auto_mask": s.auto_mask,
                "local_tone": s.local_tone,
                "local_structure": s.local_structure,
                "skin_structure": s.skin_structure,
                "upscale_mode": s.upscale_mode,
                "model_preset": s.model_preset,
            },
        }

    # ------------------------------------------------------------------ cmds

    def _command(self, args: dict[str, str]) -> None:
        changed = False
        if "wipe" in args:
            try:
                self._wipe = min(1.0, max(0.0, float(args["wipe"])))
                changed = True
            except ValueError:
                pass
        if "view" in args and args["view"] in VIEWS:
            self._view = args["view"]
            changed = True
        # Per-frame look fields from the widget (bypass the node round-trip).
        live_updates: dict[str, Any] = {}
        if "nr_style" in args:
            live_updates["nr_style"] = args["nr_style"]
        if "auto_mask" in args:
            live_updates["auto_mask"] = args["auto_mask"] in ("1", "true", "True", "yes")
        for key in ("local_tone", "local_structure", "skin_structure"):
            if key in args:
                try:
                    live_updates[key] = float(args[key])
                except ValueError:
                    pass
        restart_updates: dict[str, Any] = {}
        if "upscale_mode" in args:
            restart_updates["upscale_mode"] = args["upscale_mode"]
        if "model_preset" in args:
            restart_updates["model_preset"] = args["model_preset"]
        if live_updates or restart_updates:
            with self._lock:
                self._settings = replace(self._settings, **live_updates, **restart_updates)
                self._dirty = True
            self._wake.set()
            return
        if changed:
            # Recompose without re-running the worker (wipe/view only).
            self._publish_current()
            self._emit_state()

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        try:
            with self._lock:
                settings = self._settings
            self._worker = self._start_worker(settings)
            self.message = "live"
            self._emit_state()
            idle_since: float | None = None

            while not self._stop.is_set():
                if not self._alive():
                    self._log("Node gone - stopping live image preview.\n")
                    break

                watching = (time.monotonic() - self.server.broadcast.last_pull) < IDLE_PAUSE_SECONDS
                if not watching and self.server.broadcast.last_pull > 0:
                    if idle_since is None:
                        idle_since = time.monotonic()
                        self.message = "idle (no viewer)"
                        self._emit_state()
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue
                if idle_since is not None:
                    idle_since = None
                    self.message = "live"
                    self._emit_state()

                with self._lock:
                    wanted = self._settings
                    dirty = self._dirty
                    self._dirty = False

                if not dirty and self._out is not None:
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue

                # Debounce rapid slider motion.
                self._wake.wait(DEBOUNCE_MS / 1000.0)
                self._wake.clear()
                with self._lock:
                    if self._dirty:
                        wanted = self._settings
                        self._dirty = False

                worker = self._worker
                assert worker is not None
                if needs_restart(worker.settings, wanted):
                    self._log(f"Hot-swapping worker for {wanted.upscale_mode} / preset {wanted.model_preset}...\n")
                    self._close_worker(worker)
                    worker = self._start_worker(wanted)
                    self._worker = worker

                try:
                    t0 = time.perf_counter()
                    out = worker.process(self.source, settings=wanted, reset=True)
                    self.worker_ms = (time.perf_counter() - t0) * 1000
                    self._out = out
                    self.out_width, self.out_height = out.shape[1], out.shape[0]
                    self.error = None
                    self.message = "live"
                except DLSS5WorkerError as exc:
                    self.error = str(exc)
                    self.message = "error"
                    self._log(f"Worker error: {exc}\n")
                    self._emit_state()
                    # Try to recover with a fresh worker on the next dirty tick.
                    self._close_worker(worker)
                    self._worker = None
                    with contextlib.suppress(Exception):
                        self._worker = self._start_worker(wanted)
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue

                self._publish_current()
                self._emit_state()
        except Exception as exc:  # noqa: BLE001 - surfaces in the widget
            self.error = str(exc)
            self.message = "error"
            self._log(f"Live image session crashed: {exc}\n")
            self._emit_state()
        finally:
            self._close_worker(self._worker)
            self._worker = None
            self.message = "stopped"
            self._emit_state()

    def _start_worker(self, settings: DLSS5Settings) -> NativeLiveWorker:
        t0 = time.perf_counter()
        worker = NativeLiveWorker(settings, self.width, self.height, self.runtime)
        setup = worker.start()
        self.out_width, self.out_height = setup.output_width, setup.output_height
        self._log(
            f"Native worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms "
            f"[{settings.upscale_mode}, preset {settings.model_preset}] "
            f"-> {setup.output_width}x{setup.output_height}\n"
        )
        return worker

    @staticmethod
    def _close_worker(worker: NativeLiveWorker | None) -> None:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.close()

    def _publish_current(self) -> None:
        if self._out is None or cv2 is None:
            return
        t0 = time.perf_counter()
        jpeg = self._compose(self.source, self._out)
        self.encode_ms = (time.perf_counter() - t0) * 1000
        self.server.broadcast.publish(jpeg)
        self.seq += 1

    def _emit_state(self) -> None:
        if self._on_state is not None:
            with contextlib.suppress(Exception):
                self._on_state(self.state())

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        assert cv2 is not None
        h, w = img.shape[:2]
        if w <= self.preview_max_width:
            return img
        scale = self.preview_max_width / w
        return cv2.resize(img, (self.preview_max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    def _compose(self, src: np.ndarray, out: np.ndarray) -> bytes:
        assert cv2 is not None
        a = self._shrink(src)
        b = self._shrink(out)
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        view = self._view
        if view == VIEW_BEFORE:
            frame = a
        elif view == VIEW_AFTER:
            frame = b
        elif view == VIEW_SPLIT:
            mid = a.shape[1] // 2
            frame = np.concatenate([a[:, :mid], b[:, mid:]], axis=1)
        else:  # wipe
            mid = int(round(self._wipe * a.shape[1]))
            frame = a.copy()
            frame[:, mid:] = b[:, mid:]
            cv2.line(frame, (mid, 0), (mid, frame.shape[0] - 1), (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()


__all__ = ["DEFAULT_PREVIEW_WIDTH", "LiveImageSession"]
