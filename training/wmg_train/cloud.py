"""Mac side of training on Modal (`retrain run --trainer modal`; app defined in training/modal_app.py).

    python -m wmg_train.cloud sync                 # upload cache-v3 / cache-phone if they changed
    python -m wmg_train.cloud train 10s|5s         # train one model remotely, download runs/<ts>/

Only the feature caches go up (derived log-mels, no audio); the finished run folder comes back
into training/runs/ so compare/install work exactly as with a local run. `modal` is imported
lazily: the local trainer never needs it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TRAINING = Path(__file__).resolve().parents[1]
RUNS = TRAINING / "runs"
APP_NAME = "wmg-train"
VOLUME_NAME = "wmg-train"
CACHES = ("cache-v3", "cache-phone")
FINGERPRINT = ".fingerprint"
POLL_S = 5
SETUP_HINT = ("Modal is not set up: in training/, run `.venv-arm64/bin/modal setup` once, then "
              "`.venv-arm64/bin/modal deploy modal_app.py`")

Line = Callable[[str], None]


class CloudError(RuntimeError):
    pass


def _modal():
    try:
        import modal
    except ImportError as e:  # pragma: no cover
        raise CloudError("the `modal` package is missing: uv pip install --python .venv-arm64/bin/python modal") from e
    return modal


def _volume():
    return _modal().Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _function():
    modal = _modal()
    try:
        return modal.Function.from_name(APP_NAME, "train")
    except Exception as e:  # NotFoundError / AuthError: the message tells the user what to run
        raise CloudError(f"{SETUP_HINT} ({type(e).__name__}: {e})") from e


def fingerprint(cache: Path) -> str:
    """Identity of a cache: meta + index content and the windows file's size (reading 2.6 GB
    to hash it would take longer than checking it)."""
    h = hashlib.sha1()
    for name in ("meta.json", "index.parquet"):
        h.update((cache / name).read_bytes())
    h.update(str((cache / "windows.f16").stat().st_size).encode())
    return h.hexdigest()


def _read(vol, path: str) -> bytes | None:
    try:
        return b"".join(vol.read_file(path))
    except Exception:  # missing file (FileNotFoundError or a modal NotFoundError)
        return None


def sync_caches(line: Line = print, caches=CACHES, root: Path = TRAINING, vol=None) -> list[str]:
    """Upload each cache whose fingerprint differs from the Volume's copy. Returns what was uploaded."""
    vol = vol or _volume()
    uploaded = []
    for name in caches:
        local = root / name
        if not (local / "windows.f16").exists():
            raise CloudError(f"{local} is not a feature cache (no windows.f16)")
        fp = fingerprint(local)
        remote = _read(vol, f"/{name}/{FINGERPRINT}")
        if remote is not None and remote.decode().strip() == fp:
            line(f"{name}: unchanged on Modal")
            continue
        size_gb = sum(p.stat().st_size for p in local.iterdir() if p.is_file()) / 1e9
        line(f"{name}: uploading {size_gb:.1f} GB to Modal volume {VOLUME_NAME}...")
        t0 = time.monotonic()
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(local), f"/{name}")
            batch.put_file(io.BytesIO(fp.encode()), f"/{name}/{FINGERPRINT}")
        line(f"{name}: uploaded in {time.monotonic() - t0:.0f} s")
        uploaded.append(name)
    return uploaded


def _tail(vol, path: str, seen: int, line: Line) -> int:
    """Emit log lines past byte offset `seen`; returns the new offset."""
    data = _read(vol, path)
    if not data or len(data) <= seen:
        return seen
    chunk = data[seen:]
    end = chunk.rfind(b"\n")
    if end < 0:
        return seen
    for text in chunk[:end].decode(errors="replace").splitlines():
        line(text)
    return seen + end + 1


def download_run(vol, run: str, runs: Path = RUNS, line: Line = print) -> Path:
    dest = runs / run
    dest.mkdir(parents=True, exist_ok=True)
    for entry in vol.listdir(f"/runs/{run}"):
        name = Path(entry.path).name
        data = _read(vol, f"/runs/{run}/{name}")
        if data is None:
            continue
        (dest / name).write_bytes(data)
        line(f"downloaded runs/{run}/{name} ({len(data) / 1e6:.1f} MB)")
    if not (dest / "model.pt").exists():
        raise CloudError(f"runs/{run} came back without model.pt")
    return dest


def train_remote(name: str, extra_args: list[str], line: Line = print, runs: Path = RUNS,
                 vol=None, fn=None, poll_s: float = POLL_S) -> Path:
    """Spawn `train` on Modal, follow its log into `line`, download the finished run.
    Any exception while waiting (incl. the retrain's Cancelled) cancels the remote call."""
    vol = vol or _volume()
    fn = fn or _function()
    tag = f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    log_path = f"/logs/{tag}.log"
    line(f"Modal: spawning train({name}) as {tag}")
    call = fn.spawn(name, list(extra_args), tag)
    seen = 0
    try:
        while True:
            try:
                result = call.get(timeout=0)
                break
            except TimeoutError:
                seen = _tail(vol, log_path, seen, line)
                time.sleep(poll_s)
    except BaseException:
        try:
            call.cancel()
            line(f"Modal: cancelled {tag}")
        except Exception:
            pass
        raise
    _tail(vol, log_path, seen, line)
    if result.get("rc") != 0 or not result.get("run"):
        raise CloudError(f"remote training failed (exit {result.get('rc')}); log: modal volume get "
                         f"{VOLUME_NAME} logs/{tag}.log -")
    line(f"Modal: {name} trained on {result.get('device')} -> runs/{result['run']}")
    return download_run(vol, result["run"], runs, line)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="upload cache-v3 and cache-phone if they changed")
    t = sub.add_parser("train", help="train one model on Modal and download the run")
    t.add_argument("name", choices=["10s", "5s"])
    args = ap.parse_args()
    try:
        if args.cmd == "sync":
            sync_caches()
        else:
            from .retrain import MODELS  # the same per-model arguments the local trainer uses

            sync_caches()
            print(train_remote(args.name, MODELS[args.name]))
    except CloudError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
