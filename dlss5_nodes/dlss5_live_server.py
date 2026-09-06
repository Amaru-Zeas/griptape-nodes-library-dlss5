"""Live preview engine: loops a clip through the resident native worker and serves it as MJPEG.

Two pieces, both free of Griptape imports so they can be exercised from a script:

``LiveServer``  - tiny HTTP server on 127.0.0.1 (random free port). Endpoints:
    /stream.mjpg       multipart/x-mixed-replace stream of the composed preview
    /frame.jpg         the latest composed frame
    /state             JSON: playing, index, count, fps, timings, settings, message
    /cmd?...           play=1|0, toggle=1, seek=<index>, step=<+-n>, wipe=<0..1>, view=wipe|after|before|split,
                       speed=<multiplier, 0 = unpaced>
    All responses carry CORS headers so the node widget (served from the editor's origin) can talk to it.

``LiveSession`` - three cooperating threads: a decoder fills the RAM frame cache ahead of
    playback, the main loop keeps ``PIPELINE_DEPTH`` frames inside a ``NativeLiveWorker``
    (sent with the *current* settings) and receives them in order, and an encoder composes
    the before/after view at preview size, JPEG-encodes it and hands it to the server.
    Look-control changes are applied on the next frame; upscale/temporal/preset changes
    hot-swap the worker (the old one keeps rendering until the new one is ready). Playback
    is paced to the clip's fps times ``speed`` (0 = as fast as the pipeline goes).
"""

from __future__ import annotations

import contextlib
import json
import queue
import socket
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import imageio.v2 as imageio
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from dlss5_live_bridge import LIVE_FIELDS, NativeLiveWorker, NativeRuntime, needs_restart
from dlss5_worker_bridge import MV_MODE_AUTO_DIS, MV_MODE_NONE, DLSS5Settings, DLSS5WorkerError

VIEW_WIPE = "wipe"
VIEW_AFTER = "after"
VIEW_BEFORE = "before"
VIEW_SPLIT = "split"
VIEWS = (VIEW_WIPE, VIEW_AFTER, VIEW_BEFORE, VIEW_SPLIT)

DEFAULT_FPS = 24.0
DEFAULT_CACHE_BYTES = 8 * 1024**3
DEFAULT_PREVIEW_WIDTH = 1280  # the stream is encoded at most this wide; the bake is untouched
IDLE_PAUSE_SECONDS = 20.0  # nobody watching the stream for this long -> stop burning GPU
PIPELINE_DEPTH = 2  # frames in flight inside the worker (send N+1 while N renders)
SPEEDS = (0.25, 0.5, 1.0, 2.0, 0.0)  # 0 = unpaced (as fast as the pipeline goes)
BOUNDARY = b"dlss5frame"


# -------------------------------------------------------------------- server


class _Broadcast:
    """Latest JPEG + a condition so stream clients can wait for the next one."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes = b""
        self._seq = 0
        self.last_pull = 0.0  # monotonic time a client last consumed a frame

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> bytes:
        with self._cond:
            return self._jpeg

    def wait_next(self, seq: int, timeout: float) -> tuple[bytes, int] | None:
        self.last_pull = time.monotonic()  # a waiting client counts as a viewer (wakes an idle loop)
        with self._cond:
            if self._seq == seq and not self._cond.wait_for(lambda: self._seq != seq, timeout=timeout):
                return None
            self.last_pull = time.monotonic()
            return self._jpeg, self._seq


class LiveServer:
    """HTTP server exposing the preview stream and a command endpoint."""

    def __init__(self, on_command: Callable[[dict[str, str]], None], get_state: Callable[[], dict[str, Any]]) -> None:
        self.broadcast = _Broadcast()
        self._on_command = on_command
        self._get_state = get_state
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def start(self) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # silence
                return

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _send_bytes(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802, C901
                url = urlparse(self.path)
                path = url.path
                if path == "/stream.mjpg":
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    seq = -1
                    try:
                        while True:
                            got = server.broadcast.wait_next(seq, timeout=1.0)
                            if got is None:
                                continue  # keep-alive by waiting; nothing new yet
                            jpeg, seq = got
                            if not jpeg:
                                continue
                            self.wfile.write(
                                b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                            )
                            self.wfile.flush()
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                        return
                elif path == "/frame.jpg":
                    server.broadcast.last_pull = time.monotonic()
                    self._send_bytes(200, server.broadcast.latest() or b"", "image/jpeg")
                elif path == "/state":
                    self._send_bytes(200, json.dumps(server._get_state()).encode(), "application/json")
                elif path == "/cmd":
                    args = {k: v[-1] for k, v in parse_qs(url.query).items()}
                    try:
                        server._on_command(args)
                        self._send_bytes(200, json.dumps({"ok": True}).encode(), "application/json")
                    except Exception as exc:  # noqa: BLE001
                        self._send_bytes(400, json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json")
                else:
                    self._send_bytes(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="dlss5-live-http", daemon=True)
        self._thread.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def _free_port_available() -> bool:  # pragma: no cover - sanity helper
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return True


# ------------------------------------------------------------------- session


@dataclass
class LiveStats:
    frame_ms: float = 0.0  # wall time per delivered frame (pipeline throughput)
    worker_ms: float = 0.0  # time inside the worker process
    encode_ms: float = 0.0  # compose + JPEG on the encoder thread
    fps: float = 0.0  # achieved playback fps
    swaps: int = 0  # worker hot-swaps performed


@dataclass
class _InFlight:
    index: int
    frame: np.ndarray
    live_key: tuple
    worker: NativeLiveWorker


class LiveSession:
    """Owns the worker, the frame cache, the play loop and the HTTP server."""

    def __init__(
        self,
        video_path: Path,
        runtime: NativeRuntime,
        settings: DLSS5Settings,
        *,
        sequence: bool,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        jpeg_quality: int = 85,
        preview_max_width: int = DEFAULT_PREVIEW_WIDTH,
        log: Callable[[str], None] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        alive: Callable[[], bool] | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.runtime = runtime
        self._settings = self._for_mode(settings, sequence)
        self.cache_bytes = int(cache_bytes)
        self.jpeg_quality = int(jpeg_quality)
        self.preview_max_width = int(preview_max_width)
        self._log = log or (lambda _m: None)
        self._on_state = on_state
        self._alive = alive or (lambda: True)

        self.server = LiveServer(self._command, self.state)
        self.stats = LiveStats()
        self.error: str | None = None
        self.message = "starting"
        self.playing = True
        self.speed = 1.0  # playback speed relative to the clip's fps; 0 = unpaced
        self.wipe = 0.5
        self.view = VIEW_WIPE
        self.index = 0
        self.fps = DEFAULT_FPS
        self.width = 0
        self.height = 0
        self.out_width = 0
        self.out_height = 0
        self.count = 0  # frames decoded so far (grows while the decoder thread runs)
        self.total_frames: int | None = None  # None until the clip has been read once
        self.truncated = False

        self._frames: list[np.ndarray] = []
        self._reader: Any = None
        self._worker: NativeLiveWorker | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._decoder: threading.Thread | None = None
        self._decoded = threading.Event()  # pulsed by the decoder whenever a frame lands
        self._encoder: threading.Thread | None = None
        self._encode_q: queue.Queue[tuple[int, np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=2)
        self._receiver: threading.Thread | None = None
        self._recv_q: queue.Queue[_InFlight | None] = queue.Queue()
        self._slots = threading.Semaphore(PIPELINE_DEPTH)  # frames allowed inside the worker at once
        self._recv_error: DLSS5WorkerError | None = None
        self._t_prev_delivery = time.perf_counter()
        self._dirty = True  # something changed while paused -> re-render current frame
        self._seeked = False  # index was moved by a seek/step -> continue from there
        self._fps_window: deque[float] = deque(maxlen=48)
        self._last_out: tuple[int, np.ndarray, np.ndarray, tuple] | None = None  # idx, frame, out, live_key

    # -- public --------------------------------------------------------------

    @staticmethod
    def _for_mode(settings: DLSS5Settings, sequence: bool) -> DLSS5Settings:
        return replace(settings, mv_mode=MV_MODE_AUTO_DIS if sequence else MV_MODE_NONE, intensity=1.0)

    @property
    def settings(self) -> DLSS5Settings:
        return self._settings

    @property
    def sequence(self) -> bool:
        return self._settings.mv_mode == MV_MODE_AUTO_DIS

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def url(self) -> str:
        return self.server.url

    def start(self) -> str:
        url = self.server.start()
        self._thread = threading.Thread(target=self._run, name="dlss5-live", daemon=True)
        self._thread.start()
        return url

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.server.stop()

    def update_settings(self, settings: DLSS5Settings, *, sequence: bool | None = None) -> None:
        """Apply new look settings; restart-class changes are hot-swapped by the loop."""
        with self._lock:
            seq = self.sequence if sequence is None else sequence
            self._settings = self._for_mode(settings, seq)
            self._dirty = True
        self._wake.set()

    def state(self) -> dict[str, Any]:
        s = self._settings
        return {
            "running": self.running,
            "playing": self.playing,
            "speed": self.speed,
            "index": self.index,
            "count": self.count,
            "total": self.total_frames,
            "decoding": self.total_frames is None,
            "truncated": self.truncated,
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "out_width": self.out_width,
            "out_height": self.out_height,
            "wipe": self.wipe,
            "view": self.view,
            "frame_ms": round(self.stats.frame_ms, 1),
            "worker_ms": round(self.stats.worker_ms, 1),
            "encode_ms": round(self.stats.encode_ms, 1),
            "play_fps": round(self.stats.fps, 1),
            "swaps": self.stats.swaps,
            "sequence": self.sequence,
            "upscale": s.upscale_mode,
            "message": self.error or self.message,
            "error": self.error,
        }

    # -- commands (from the widget via HTTP or from the node) -----------------

    def _command(self, args: dict[str, str]) -> None:
        if "play" in args:
            self.playing = args["play"] in ("1", "true", "True")
        if "toggle" in args:
            self.playing = not self.playing
        if "seek" in args and self.count:
            self.index = max(0, min(self.count - 1, int(float(args["seek"]))))
            self._dirty = self._seeked = True
        if "step" in args and self.count:
            self.playing = False
            self.index = (self.index + int(args["step"])) % self.count
            self._dirty = self._seeked = True
        if "wipe" in args:
            self.wipe = max(0.0, min(1.0, float(args["wipe"])))
            self._dirty = True
        if "view" in args and args["view"] in VIEWS:
            self.view = args["view"]
            self._dirty = True
        if "speed" in args:
            with contextlib.suppress(ValueError):
                self.speed = max(0.0, min(8.0, float(args["speed"])))
        self._wake.set()

    def command(self, **kwargs: Any) -> None:
        self._command({k: str(v) for k, v in kwargs.items()})

    # -- loop ----------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._open_clip()
            self._decoder = threading.Thread(target=self._decode_all, name="dlss5-live-decode", daemon=True)
            self._decoder.start()
            self._encoder = threading.Thread(target=self._encode_all, name="dlss5-live-encode", daemon=True)
            self._encoder.start()
            self._receiver = threading.Thread(target=self._receive_all, name="dlss5-live-receive", daemon=True)
            self._receiver.start()
            self._loop()
        except DLSS5WorkerError as exc:
            self.error = str(exc)
            self._log(f"Live preview stopped: {exc}\n")
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._log(f"Live preview crashed: {self.error}\n")
        finally:
            self._stop.set()
            self._close_worker(self._worker)  # unblocks a receiver stuck in receive()
            self._worker = None
            self._recv_q.put(None)
            if self._receiver is not None:
                self._receiver.join(timeout=5.0)
            with contextlib.suppress(queue.Full):
                self._encode_q.put_nowait(None)
            for th in (self._decoder, self._encoder):
                if th is not None:
                    th.join(timeout=5.0)
            if self._reader is not None:
                with contextlib.suppress(Exception):
                    self._reader.close()
                self._reader = None
            self.message = "stopped"
            self._emit_state()

    def _emit_state(self) -> None:
        if self._on_state is not None:
            with contextlib.suppress(Exception):
                self._on_state(self.state())

    def _open_clip(self) -> None:
        self._reader = imageio.get_reader(str(self.video_path), format="FFMPEG")
        meta = self._reader.get_meta_data()
        self.fps = float(meta.get("fps") or DEFAULT_FPS) or DEFAULT_FPS
        size = meta.get("size") or (0, 0)
        w, h = int(size[0]), int(size[1])
        first = np.ascontiguousarray(np.asarray(self._reader.get_data(0))[:, :, :3])
        if w <= 0 or h <= 0:
            h, w = first.shape[:2]
        if (w, h) != (first.shape[1], first.shape[0]):
            h, w = first.shape[:2]
        self.width, self.height = w, h
        self._frames = [first]
        self.count = 1
        # imageio's frame index iterator; we re-open on wrap when the cache is truncated.
        self._decode_iter = iter(self._reader.iter_data())
        next(self._decode_iter, None)  # frame 0 already taken
        self.message = "decoding"
        self._log(f"Live source {w}x{h} @ {self.fps:.3f} fps ({self.video_path.name})\n")

    def _decode_next(self) -> bool:
        """Append one more frame to the cache. Returns False when the clip is exhausted or the budget is hit."""
        if self.total_frames is not None:
            return False
        budget_frames = max(1, self.cache_bytes // max(1, self.width * self.height * 3))
        if self.count >= budget_frames:
            self.total_frames = self.count
            self.truncated = True
            self._log(
                f"Frame cache budget reached: looping the first {self.count} frames "
                f"({self.count / self.fps:.1f} s). Raise max_cache_gb in Advanced to loop more.\n"
            )
            return False
        frame = next(self._decode_iter, None)
        if frame is None:
            self.total_frames = self.count
            return False
        self._frames.append(np.ascontiguousarray(np.asarray(frame)[:, :, :3]))
        self.count = len(self._frames)
        return True

    def _decode_all(self) -> None:
        """Decoder thread: fill the frame cache ahead of playback (ffmpeg is the slowest stage at 4K)."""
        try:
            while not self._stop.is_set() and self._decode_next():
                self._decoded.set()
        except Exception as exc:  # noqa: BLE001 - a truncated file just ends the clip early
            self._log(f"Decoder stopped early: {exc}\n")
            self.total_frames = self.count
        finally:
            self._decoded.set()
            if self.total_frames is not None and not self.truncated:
                self._log(f"Decoded {self.count} frames ({self.count / self.fps:.1f} s) into RAM.\n")

    def _receive_all(self) -> None:
        """Receiver thread: takes results back from the worker in order and hands them to the encoder.

        Sending and receiving must live on different threads: the worker blocks on its stdout once
        the pipe is full, so a single thread that sends ahead deadlocks against it.
        """
        while True:
            item = self._recv_q.get()
            if item is None:
                return
            try:
                out = item.worker.receive()
            except DLSS5WorkerError as exc:
                if self._recv_error is None:
                    self._recv_error = exc
                self._wake.set()
                continue
            finally:
                self._slots.release()
                self._recv_q.task_done()
            self.stats.worker_ms = item.worker.last_worker_ms
            self.index = item.index
            self._last_out = (item.index, item.frame, out, item.live_key)
            now = time.perf_counter()
            self.stats.frame_ms = (now - self._t_prev_delivery) * 1000.0
            self._t_prev_delivery = now
            if self.playing:
                self._fps_window.append(now)
                if len(self._fps_window) >= 2:
                    span = self._fps_window[-1] - self._fps_window[0]
                    self.stats.fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0
            self._encode_q.put((item.index, item.frame, out))

    def _encode_all(self) -> None:
        """Encoder thread: compose + JPEG + publish, overlapping the worker round trip on the main loop."""
        while True:
            item = self._encode_q.get()
            if item is None:
                return
            idx, frame, out = item
            t0 = time.perf_counter()
            try:
                jpeg = self._compose(frame, out)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Preview encode failed on frame {idx}: {exc}\n")
                continue
            self.server.broadcast.publish(jpeg)
            self.stats.encode_ms = (time.perf_counter() - t0) * 1000.0

    def _start_worker(self, settings: DLSS5Settings) -> NativeLiveWorker:
        t0 = time.perf_counter()
        worker = NativeLiveWorker(settings, self.width, self.height, self.runtime)
        setup = worker.start()
        self.out_width, self.out_height = setup.output_width, setup.output_height
        self._log(
            f"Native worker ready in {(time.perf_counter() - t0) * 1000:.0f} ms "
            f"[{settings.upscale_mode}, {'sequence' if settings.mv_mode == MV_MODE_AUTO_DIS else 'single-frame'}, "
            f"preset {settings.model_preset}] -> {self.out_width}x{self.out_height}\n"
        )
        return worker

    @staticmethod
    def _close_worker(worker: NativeLiveWorker | None) -> None:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.close()

    @staticmethod
    def _live_key(settings: DLSS5Settings) -> tuple:
        return tuple(getattr(settings, f) for f in LIVE_FIELDS)

    def _restart_worker(self, settings: DLSS5Settings, why: str) -> NativeLiveWorker:
        self._log(f"{why}\nRestarting worker...\n")
        self._close_worker(self._worker)  # pending receives now fail fast instead of hanging
        self._drain()
        self._recv_error = None
        self._worker = self._start_worker(settings)
        return self._worker

    def _drain(self) -> None:
        """Wait until every frame handed to the worker has been received (used before swaps / idling)."""
        while self._recv_q.unfinished_tasks and not self._stop.is_set():
            time.sleep(0.005)

    def _pace(self, deadline: float) -> bool:
        """Sleep until ``deadline`` (perf_counter). False if a command/stop interrupted the wait.

        Event.wait() on Windows has ~15.6 ms granularity, which turned a 41.7 ms frame budget into
        ~47 ms (21 fps instead of 24). Coarse-wait most of it, then finish with short sleeps.
        """
        remaining = deadline - time.perf_counter()
        if remaining > 0.02:
            self._wake.wait(remaining - 0.018)
            self._wake.clear()
            if self._dirty or self._stop.is_set():
                return False
        while (remaining := deadline - time.perf_counter()) > 0:
            time.sleep(0.0005 if remaining > 0.002 else 0)
        return True

    def _acquire_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slots.acquire(timeout=0.1):
                return True
        return False

    def _loop(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Sender loop. Four threads cooperate:

        decoder   -> fills ``_frames`` ahead of playback
        this loop -> picks frames, paces playback, sends them with the current settings
        receiver  -> takes results back in order (up to ``PIPELINE_DEPTH`` in flight)
        encoder   -> composes the before/after view at preview size, JPEG-encodes, publishes

        so the per-frame wall time is the slowest stage instead of the sum of all of them.
        """
        with self._lock:
            settings = self._settings
        self._worker = self._start_worker(settings)
        self.message = "live"
        self._emit_state()
        first = True
        last_emit = 0.0
        idle_since: float | None = None
        last_scheduled = -1  # index of the last frame handed to the worker
        t_prev_send = time.perf_counter()

        while not self._stop.is_set():
            if not self._alive():
                self._log("Node gone - stopping live preview.\n")
                break
            worker = self._worker
            assert worker is not None

            # Receiver hit a worker error: bring the worker back and start clean.
            if self._recv_error is not None:
                worker = self._restart_worker(settings, f"Worker error: {self._recv_error}")
                first = True
                last_scheduled = -1
                continue

            # Nobody watching? Pause the GPU loop (worker stays resident) until a client returns.
            watching = (time.monotonic() - self.server.broadcast.last_pull) < IDLE_PAUSE_SECONDS
            if not watching and self.server.broadcast.last_pull > 0:
                if idle_since is None:
                    self._drain()
                    idle_since = time.monotonic()
                    self.message = "idle (no viewer)"
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if idle_since is not None:
                idle_since = None
                self.message = "live"

            # Settings: pick up changes; hot-swap when needed.
            with self._lock:
                wanted = self._settings
                dirty, seeked = self._dirty, self._seeked
                self._dirty = self._seeked = False
            if needs_restart(settings, wanted):
                self.message = "switching worker"
                self._emit_state()
                try:
                    new_worker = self._start_worker(wanted)
                except DLSS5WorkerError as exc:
                    self._log(f"Could not switch worker ({exc}); keeping the previous settings.\n")
                    with self._lock:
                        self._settings = replace(
                            wanted, **{f: getattr(settings, f) for f in ("upscale_mode", "mv_mode", "dis_preset", "model_preset")}
                        )
                    wanted = self._settings
                else:
                    self._drain()  # the old worker finishes what it has, then goes
                    old, self._worker = self._worker, new_worker
                    self._close_worker(old)
                    worker = new_worker
                    self.stats.swaps += 1
                    first = True
                    self.message = "live"
            settings = wanted
            key = self._live_key(settings)

            # Decide which frame (if any) to hand to the worker this iteration.
            idx: int | None = None
            if self.playing:
                nxt = self.index if (seeked or last_scheduled < 0) else last_scheduled + 1
                if nxt < self.count:
                    idx = nxt
                elif self.total_frames is None:
                    self._decoded.wait(0.05)  # decoder has not produced this frame yet
                    self._decoded.clear()
                    continue
                else:
                    idx = 0  # wrap
            elif dirty or first:
                idx = max(0, min(self.index, self.count - 1))
                last = self._last_out
                if not first and last is not None and last[0] == idx and last[3] == key and not self._recv_q.unfinished_tasks:
                    # Only the wipe / view changed: re-compose the cached result, no worker call.
                    self._encode_q.put((idx, last[1], last[2]))
                    continue
            else:
                self._fps_window.clear()
                self.stats.fps = 0.0
                self._wake.wait(0.25)
                self._wake.clear()
                continue

            # Pace sends to the clip's fps x speed (0 = unpaced).
            if self.playing and self.speed > 0:
                deadline = t_prev_send + 1.0 / (max(1.0, self.fps) * self.speed)
                if not self._pace(deadline):
                    continue  # a command arrived (seek/settings): re-decide instead of sending a stale frame

            if not self._acquire_slot():
                break
            frame = self._frames[idx]
            reset = first or not self.sequence or idx == 0 or idx != last_scheduled + 1
            t_prev_send = time.perf_counter()  # pace from send *start*: the pipe write itself can take 10-20 ms at 4K
            try:
                worker.send(frame, settings=settings, reset=reset)
            except DLSS5WorkerError as exc:
                self._slots.release()
                worker = self._restart_worker(settings, f"Worker error on frame {idx}: {exc}")
                first = True
                last_scheduled = -1
                continue
            self._recv_q.put(_InFlight(idx, frame, key, worker))
            last_scheduled = idx
            first = False

            if time.monotonic() - last_emit > 1.0:
                last_emit = time.monotonic()
                self._emit_state()

    # -- compose ------------------------------------------------------------

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        """Downscale to the preview width (the widget is never wider than that; the bake is full-res)."""
        if self.preview_max_width and img.shape[1] > self.preview_max_width:
            scale = self.preview_max_width / img.shape[1]
            return cv2.resize(img, (self.preview_max_width, max(1, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        return img

    def _compose(self, src: np.ndarray, out: np.ndarray) -> bytes:
        """Build the preview image (before/after) at preview size and JPEG-encode it."""
        if cv2 is None:  # pragma: no cover - cv2 is a library dependency
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(out).save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()

        after = self._shrink(out)
        oh, ow = after.shape[:2]
        if self.view == VIEW_AFTER:
            img = after
        else:
            before = self._shrink(src)
            if (before.shape[0], before.shape[1]) != (oh, ow):
                before = cv2.resize(before, (ow, oh), interpolation=cv2.INTER_CUBIC)
            if self.view == VIEW_BEFORE:
                img = before
            elif self.view == VIEW_SPLIT:
                img = np.concatenate([before, after], axis=1)
            else:  # wipe: left = before, right = after
                x = int(round(self.wipe * ow))
                img = after.copy()
                if x > 0:
                    img[:, :x] = before[:, :x]
                if 0 < x < ow:
                    img[:, max(0, x - 1) : min(ow, x + 1)] = (255, 255, 255)
        ok, enc = cv2.imencode(
            ".jpg", cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return enc.tobytes()


def stage_video(source_bytes: bytes, suffix: str = ".mp4") -> Path:
    """Write clip bytes to a private temp file the session can decode from repeatedly."""
    tmp = tempfile.NamedTemporaryFile(prefix="dlss5_live_", suffix=suffix, delete=False)
    with tmp:
        tmp.write(source_bytes)
    return Path(tmp.name)


__all__ = [
    "DEFAULT_CACHE_BYTES",
    "DEFAULT_PREVIEW_WIDTH",
    "SPEEDS",
    "VIEWS",
    "LiveServer",
    "LiveSession",
    "LiveStats",
    "stage_video",
]
