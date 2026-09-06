"""One-click installer for the DLSS 5 runtime (Merserk's "DLSS 5 Visual Enhancer" release).

The NVIDIA DLSS 5 runtimes cannot be redistributed with this library, so the user fetches
them from Merserk's GitHub release exactly as they would by hand - this module just removes
the manual steps: query the latest release, download the zip with progress, verify the
SHA-256 GitHub publishes for the asset, extract, locate ``nvngx_dlssnr.dll``, check that it
carries NVIDIA's Authenticode signature, and return the folder to use as ``runtime_dir``.

No Griptape imports: usable from a script. Nothing here touches the nodes' code paths; the
Instructions node calls :func:`install` from a button and stores the result in the library
setting ``dlss5.runtime_dir``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = "Merserk/dlss5-visual-enhancer"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
USER_AGENT = "griptape-nodes-library-dlss5"
NR_SNIPPET = "nvngx_dlssnr.dll"
SR_SNIPPET = "nvngx_dlss.dll"
CHUNK = 1 << 20

Log = Callable[[str], None]


class RuntimeInstallError(RuntimeError):
    """Download / verification failed; message tells the user what to do by hand."""


@dataclass(frozen=True)
class ReleaseAsset:
    tag: str
    name: str
    url: str
    size: int
    sha256: str | None  # from the GitHub "digest" field when present

    @property
    def size_mb(self) -> float:
        return self.size / 1024**2


def install_root() -> Path:
    """Where releases are unpacked: %LOCALAPPDATA%\\griptape_nodes\\dlss5 (home dir fallback)."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "griptape_nodes" / "dlss5"


def find_installed(root: Path | None = None) -> Path | None:
    """Return the newest already-installed release folder that still has the NR snippet, else None."""
    root = root or install_root()
    if not root.is_dir():
        return None
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in candidates:
        app = _locate_app_root(folder)
        if app is not None:
            return app
    return None


def latest_release(timeout: float = 30.0) -> ReleaseAsset:
    """Ask the GitHub API for the latest release and pick its zip asset."""
    req = urllib.request.Request(API_LATEST, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeInstallError(
            f"Could not reach the GitHub API ({exc}). Check your internet connection, or download the release "
            f"by hand from {RELEASES_PAGE} and set runtime_dir to the unzipped folder."
        ) from None
    zips = [a for a in data.get("assets", []) if str(a.get("name", "")).lower().endswith(".zip")]
    if not zips:
        raise RuntimeInstallError(
            f"The latest release ({data.get('tag_name', '?')}) has no zip asset. Download it by hand from {RELEASES_PAGE}."
        )
    asset = max(zips, key=lambda a: int(a.get("size", 0)))  # the app zip is by far the largest
    digest = str(asset.get("digest") or "")
    sha256 = digest.split(":", 1)[1].lower() if digest.startswith("sha256:") else None
    return ReleaseAsset(
        tag=str(data.get("tag_name") or "latest"),
        name=str(asset["name"]),
        url=str(asset["browser_download_url"]),
        size=int(asset.get("size", 0)),
        sha256=sha256,
    )


def _download(asset: ReleaseAsset, dest: Path, log: Log, timeout: float = 60.0) -> str:
    """Stream the asset to ``dest``; returns its SHA-256 hex digest. Logs progress every ~5 %."""
    req = urllib.request.Request(asset.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    done = 0
    next_report = 0.0
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
            total = int(resp.headers.get("Content-Length") or asset.size or 0)
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                digest.update(block)
                done += len(block)
                frac = done / total if total else 0.0
                if frac >= next_report or not total:
                    speed = done / max(1e-6, time.perf_counter() - t0) / 1024**2
                    log(f"  {done / 1024**2:6.0f} / {total / 1024**2:.0f} MB  ({frac * 100:3.0f} %, {speed:.0f} MB/s)\n")
                    next_report = frac + 0.05
    except Exception as exc:  # noqa: BLE001
        raise RuntimeInstallError(f"Download failed after {done / 1024**2:.0f} MB: {exc}. Try again, or download by hand from {RELEASES_PAGE}.") from None
    if asset.size and done != asset.size:
        raise RuntimeInstallError(f"Download incomplete: got {done} bytes, expected {asset.size}. Try again.")
    return digest.hexdigest()


def _extract(zip_path: Path, target: Path, log: Log) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total = sum(m.file_size for m in members) or 1
        done = 0
        next_report = 0.0
        target_resolved = target.resolve()
        for m in members:
            # Zip-slip guard: every entry must land inside ``target``.
            out = (target / m.filename).resolve()
            if not str(out).startswith(str(target_resolved)):
                raise RuntimeInstallError(f"Refusing to extract '{m.filename}' outside the install folder.")
            zf.extract(m, target)
            done += m.file_size
            if done / total >= next_report:
                log(f"  extracting... {done / total * 100:3.0f} %\n")
                next_report = done / total + 0.25


def _locate_app_root(folder: Path) -> Path | None:
    """The folder to use as runtime_dir: the one that contains bin\\runtime (Merserk's app root)."""
    for hit in folder.rglob(NR_SNIPPET):
        # .../<app>/bin/runtime/dlssnr/nvngx_dlssnr.dll -> <app>
        for parent in hit.parents:
            if (parent / "bin" / "runtime").is_dir():
                return parent
        return hit.parent  # snippet found but not Merserk's layout: point straight at it
    return None


def _authenticode_subject(dll: Path) -> tuple[str, str] | None:
    """(status, signer subject) via PowerShell; None when PowerShell is unavailable (non-Windows)."""
    if os.name != "nt":
        return None
    script = (
        f"$s = Get-AuthenticodeSignature -LiteralPath '{dll}'; "
        "Write-Output $s.Status; if ($s.SignerCertificate) { Write-Output $s.SignerCertificate.Subject }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[0], (lines[1] if len(lines) > 1 else "")


def verify_snippet(app_root: Path, log: Log) -> Path:
    """Locate nvngx_dlssnr.dll under ``app_root`` and report the Authenticode status of the NVIDIA DLLs.

    Integrity is guaranteed by the SHA-256 check against GitHub's published digest. The signature
    is reported, not enforced: the NR snippet in the Visual Enhancer releases is NOT signed
    (v7.0 ships a modified "universal" build; the SR snippet next to it is NVIDIA-signed).
    """
    hits = list(app_root.rglob(NR_SNIPPET))
    if not hits:
        raise RuntimeInstallError(f"{NR_SNIPPET} not found under {app_root}. The release layout may have changed.")
    for dll in (hits[0], *app_root.rglob(SR_SNIPPET)):
        sig = _authenticode_subject(dll)
        if sig is None:
            log(f"  {dll.name}: {dll.stat().st_size / 1024**2:.0f} MB (signature check needs PowerShell; skipped)\n")
            continue
        status, subject = sig
        signer = subject.split(",")[0] if subject else "-"
        note = "" if (status == "Valid" and "NVIDIA" in subject.upper()) else "  <- not NVIDIA-signed (modified build)"
        log(f"  {dll.name}: {dll.stat().st_size / 1024**2:.0f} MB, Authenticode {status}, signer {signer}{note}\n")
    return hits[0]


def install(log: Log | None = None, *, root: Path | None = None, reuse_existing: bool = True) -> Path:
    """Download + verify + extract the latest release. Returns the folder to use as runtime_dir."""
    log = log or (lambda _m: None)
    root = root or install_root()
    root.mkdir(parents=True, exist_ok=True)

    asset = latest_release()
    log(f"Latest release: {asset.tag} - {asset.name} ({asset.size_mb:.0f} MB)\n")
    target = root / asset.tag
    if reuse_existing:
        existing = _locate_app_root(target) if target.is_dir() else None
        if existing is not None:
            log(f"Already installed at {existing}\n")
            verify_snippet(existing, log)
            return existing

    free = shutil.disk_usage(root).free
    if free < asset.size * 2.5:
        raise RuntimeInstallError(
            f"Not enough free disk space on {root.anchor}: need ~{asset.size_mb * 2.5 / 1024:.1f} GB "
            f"(zip + extracted files), have {free / 1024**3:.1f} GB."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="dlss5_runtime_", dir=root))
    zip_path = tmp_dir / asset.name
    try:
        log(f"Downloading {asset.url}\n")
        digest = _download(asset, zip_path, log)
        if asset.sha256:
            if digest != asset.sha256:
                raise RuntimeInstallError(f"SHA-256 mismatch (got {digest[:16]}..., GitHub says {asset.sha256[:16]}...). Download corrupted; try again.")
            log(f"SHA-256 verified ({digest[:16]}...)\n")
        else:
            log(f"SHA-256 {digest[:16]}... (GitHub published no digest to compare against)\n")

        extract_dir = tmp_dir / "unpacked"
        extract_dir.mkdir()
        log("Extracting...\n")
        _extract(zip_path, extract_dir, log)
        zip_path.unlink(missing_ok=True)

        app_root = _locate_app_root(extract_dir)
        if app_root is None:
            raise RuntimeInstallError(f"{NR_SNIPPET} not found in the release zip. The release layout may have changed; download by hand from {RELEASES_PAGE}.")
        verify_snippet(app_root, log)

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(app_root), str(target))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    sr = "found" if list(target.rglob(SR_SNIPPET)) else "not found (upscaling unavailable)"
    log(f"Installed to {target}\n  neural rendering: {NR_SNIPPET} ok\n  super resolution: {SR_SNIPPET} {sr}\n")
    return target


__all__ = ["ReleaseAsset", "RuntimeInstallError", "find_installed", "install", "install_root", "latest_release", "verify_snippet"]
