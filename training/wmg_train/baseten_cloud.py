"""Training on Baseten (`retrain run --trainer baseten`; the job is defined in training/baseten/).

    python -m wmg_train.baseten_cloud train 10s|5s [--reseed]   # train one model, download runs/<ts>/
    python -m wmg_train.baseten_cloud smoke [--reseed]          # one short epoch, no CV folds
    python -m wmg_train.baseten_cloud job                       # inside the container (baseten/run.sh)

Everything goes through Baseten; there is no bucket of our own:

- Feature caches (derived log-mels, no audio) ride along in the job upload, but only when this
  Mac hasn't shipped that fingerprint to the project yet (.baseten-seeded.json). The job copies
  them into the project cache ($BT_PROJECT_CACHE_DIR/caches/<cache>/<fingerprint>/), which
  persists between jobs, so later jobs upload only code. If Baseten ever drops the project cache,
  the job says so; rerun with --reseed.
- The job trains into $BT_CHECKPOINT_DIR; the finished run folder comes back through the
  checkpoint download links into training/runs/, so compare/install work exactly as with a
  local run.

The upload limit is 5 GB, so cache-v3 + cache-phone (4.4 GB) only just fit in one job. `truss`
is imported lazily: the local trainer never needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cloud import CACHES, CloudError, fingerprint

TRAINING = Path(__file__).resolve().parents[1]
RUNS = TRAINING / "runs"
JOB_DIR = TRAINING / "baseten"
PROJECT = "wmg-detector"  # the training project in baseten/config.py
SEEDED = TRAINING / ".baseten-seeded.json"  # {remote/project: {cache: fingerprint}} shipped to the project cache
COMPLETE = ".complete"  # written last, so a half-copied cache never counts
POLL_S = 10
CHECKPOINT_WAIT_S = 120  # checkpoint files can show up a little after the job completes
DONE = {"TRAINING_JOB_COMPLETED", "TRAINING_JOB_FAILED", "TRAINING_JOB_DEPLOY_FAILED", "TRAINING_JOB_STOPPED"}
SETUP_HINT = ("Baseten is not set up: in training/, run `.venv-arm64/bin/truss login` once "
              "(or set BASETEN_TRUSS_AUTH_REMOTE_URL and BASETEN_TRUSS_AUTH_API_KEY)")

Line = Callable[[str], None]


# ---------------------------------------------------------------- caches

def local_fingerprints(root: Path = TRAINING, caches=CACHES) -> dict[str, str]:
    out = {}
    for name in caches:
        if not (root / name / "windows.f16").exists():
            raise CloudError(f"{root / name} is not a feature cache (no windows.f16)")
        out[name] = fingerprint(root / name)
    return out


class Seeded:
    """Which cache fingerprints this Mac has already shipped to a Baseten project."""

    def __init__(self, key: str, path: Path = SEEDED):
        self.key, self.path = key, path

    def _all(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def to_ship(self, fingerprints: dict[str, str], reseed: bool = False) -> list[str]:
        have = {} if reseed else self._all().get(self.key, {})
        return [name for name, fp in fingerprints.items() if have.get(name) != fp]

    def record(self, shipped: dict[str, str]) -> None:
        data = self._all()
        data[self.key] = {**data.get(self.key, {}), **shipped}
        self.path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------- Baseten API

class Baseten:
    """Push / status / logs / stop / checkpoints for training jobs, through the truss SDK."""

    def __init__(self, remote: str | None = None):
        try:
            from truss.remote.remote_factory import RemoteFactory
        except ImportError as e:  # pragma: no cover
            raise CloudError("the `truss` package is missing: pip install -e '.[baseten]'") from e
        try:
            names = RemoteFactory.get_available_config_names()
            self.remote = remote or os.environ.get("WMG_BASETEN_REMOTE") or (names[0] if names else "baseten")
            self.api = RemoteFactory.create(remote=self.remote).api
        except Exception as e:
            raise CloudError(f"{SETUP_HINT} ({type(e).__name__}: {e})") from e

    def push(self, project, source_dir: Path) -> tuple[str, str]:
        from truss_train import push

        try:
            job = push(project, remote=self.remote, source_dir=source_dir)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 403:
                raise CloudError("Baseten refused the training job (403): this account is not enabled for "
                                 "Baseten Training yet. Ask Baseten to turn on Training for the workspace.") from e
            raise CloudError(f"pushing the Baseten job failed ({type(e).__name__}: {e})") from e
        return job["training_project"]["id"], job["id"]

    def status(self, project_id: str, job_id: str) -> tuple[str, str | None]:
        job = self.api.get_training_job(project_id, job_id)["training_job"]
        return job["current_status"], job.get("error_message")

    def logs(self, project_id: str, job_id: str, since_ms: int | None) -> list[dict]:
        """[{timestamp (epoch ns, str), message}], oldest first."""
        return self.api.get_training_job_logs(project_id, job_id, start_epoch_millis=since_ms)

    def stop(self, project_id: str, job_id: str) -> None:
        self.api.stop_training_job(project_id, job_id)

    def checkpoint_files(self, project_id: str, job_id: str) -> list[dict]:
        """[{relative_file_name, url}] of everything the job wrote to $BT_CHECKPOINT_DIR."""
        return self.api.get_training_job_checkpoint_presigned_url(project_id, job_id, page_size=1000)

    def fetch(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)


class LogTail:
    """Emits each job log line once, in order, across overlapping polls."""

    def __init__(self, line: Line):
        self.line, self.last_ns, self.seen = line, 0, set()

    @property
    def since_ms(self) -> int | None:
        return self.last_ns // 1_000_000 if self.last_ns else None

    def feed(self, logs: list[dict]) -> None:
        for e in logs:
            ts, msg = int(e["timestamp"]), e["message"].rstrip("\n")
            if ts < self.last_ns or (ts, msg) in self.seen:
                continue
            if ts > self.last_ns:
                self.seen = set()  # only lines sharing the newest timestamp can come back
            self.seen.add((ts, msg))
            self.last_ns = ts
            self.line(msg)


def new_tag(name: str) -> str:
    return f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _make_project(name, extra_args, tag, fingerprints, ship=()):
    import importlib.util

    spec = importlib.util.spec_from_file_location("wmg_baseten_config", JOB_DIR / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.make_project(name, list(extra_args), tag, fingerprints, ship=list(ship))


def download_run(api, project_id: str, job_id: str, runs: Path = RUNS, line: Line = print,
                 wait_s: float = CHECKPOINT_WAIT_S, poll_s: float = POLL_S) -> Path:
    """The checkpoint folder that holds model.pt, into runs/<that folder's name>/."""
    deadline = time.monotonic() + wait_s
    while True:
        files = api.checkpoint_files(project_id, job_id)
        models = [f["relative_file_name"] for f in files if f["relative_file_name"].endswith("/model.pt")]
        if models or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    if not models:
        raise CloudError(f"job {job_id} finished without a <run>/model.pt in its checkpoints")
    run_dir = min(models, key=len).rsplit("/", 1)[0]
    run = run_dir.rsplit("/", 1)[-1]
    for f in files:
        if f["relative_file_name"].startswith(run_dir + "/"):
            rel = f["relative_file_name"][len(run_dir) + 1:]
            api.fetch(f["url"], runs / run / rel)
            line(f"downloaded runs/{run}/{rel} ({(runs / run / rel).stat().st_size / 1e6:.1f} MB)")
    return runs / run


def train_remote(name: str, extra_args: list[str], line: Line = print, runs: Path = RUNS,
                 api: Baseten | None = None, make_project=_make_project, fingerprints: dict[str, str] | None = None,
                 seeded: Seeded | None = None, reseed: bool = False, poll_s: float = POLL_S,
                 wait_s: float = CHECKPOINT_WAIT_S) -> Path:
    """Push one training job, follow its log into `line`, download the finished run. Caches the
    project doesn't have yet go up with the job. Any exception while waiting (incl. the retrain's
    Cancelled) stops the job."""
    api = api or Baseten()
    fingerprints = local_fingerprints() if fingerprints is None else fingerprints
    seeded = seeded or Seeded(f"{getattr(api, 'remote', 'baseten')}/{PROJECT}")
    ship = seeded.to_ship(fingerprints, reseed)
    tag = new_tag(name)
    line(f"Baseten: pushing train({name}) as {tag}"
         + (f", with {' + '.join(ship)} for the project cache" if ship else ""))
    project_id, job_id = api.push(make_project(name, extra_args, tag, fingerprints, ship), JOB_DIR)
    line(f"Baseten: job {job_id} (project {project_id})")
    tail, state, error = LogTail(line), None, None
    try:
        while True:
            state, error = api.status(project_id, job_id)
            tail.feed(api.logs(project_id, job_id, tail.since_ms))
            if state in DONE:
                break
            time.sleep(poll_s)
    except BaseException:
        try:
            api.stop(project_id, job_id)
            line(f"Baseten: stopped job {job_id}")
        except Exception:
            pass
        raise
    if state != "TRAINING_JOB_COMPLETED":
        raise CloudError(f"remote training failed ({state}{', ' + error if error else ''}); "
                         f"log: truss train logs --job-id {job_id}")
    seeded.record({n: fingerprints[n] for n in ship})
    return download_run(api, project_id, job_id, runs, line, wait_s, poll_s)


# ---------------------------------------------------------------- inside the job

def place_cache(name: str, fp: str, store: Path, workdir: Path = Path(".")) -> Path:
    """The cache with this fingerprint in the project cache (store/<name>/<fp>), copied in from the
    job upload when it came along. Older fingerprints of the same cache are removed."""
    dest = store / name / fp
    shipped = workdir / name
    if shipped.is_dir():
        if fingerprint(shipped) != fp:
            raise CloudError(f"{name}: the uploaded cache is not fingerprint {fp[:10]}")
        print(f"{name}: copying {fp[:10]} into the project cache", flush=True)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for p in shipped.iterdir():
            if p.is_file():
                shutil.copyfile(p, dest / p.name)
        (dest / COMPLETE).write_text(fp)
    elif (dest / COMPLETE).exists():
        print(f"{name}: {fp[:10]} already in the project cache", flush=True)
    else:
        raise CloudError(f"{name} {fp[:10]} is not in the project cache: rerun with --reseed")
    for old in (store / name).iterdir():
        if old.name != fp:
            shutil.rmtree(old, ignore_errors=True)
    return dest


def _device() -> str:
    try:
        import torch

        return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def run_job(env=os.environ, train=None, workdir: Path = Path(".")) -> int:
    """Place the caches, run `train_baseline` into the checkpoint folder. Returns the exit code."""
    store = Path(env.get("BT_PROJECT_CACHE_DIR", "/root/.cache/user_artifacts")) / "caches"
    out = Path(env.get("BT_CHECKPOINT_DIR", "/tmp/training_checkpoints"))
    caches = {name: place_cache(name, env[f"WMG_FP_{name.replace('-', '_').upper()}"], store, workdir)
              for name in CACHES}
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.iterdir()}
    cmd = [sys.executable, "-m", "wmg_train.train_baseline", "--cache", str(caches["cache-v3"]),
           "--phone-cache", str(caches["cache-phone"]), "--out", str(out), *env.get("WMG_TRAIN_ARGS", "").split()]
    device = _device()
    print(f"[{env.get('WMG_MODEL')}] {device}\n$ {' '.join(cmd)}", flush=True)
    rc = (train or subprocess.call)(cmd)
    new = sorted(p.name for p in out.iterdir() if p.name not in before and (p / "model.pt").exists())
    if not new:
        print("no run folder with model.pt was written", flush=True)
        return rc or 1
    (out / new[-1] / "baseten.json").write_text(json.dumps({"tag": env.get("WMG_TAG"), "device": device}) + "\n")
    print(f"run {new[-1]} written to the checkpoint folder", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train one model on Baseten and download the run")
    t.add_argument("name", choices=["10s", "5s"])
    s = sub.add_parser("smoke", help="one short epoch without CV folds: proves image, GPU and caches")
    for p in (t, s):
        p.add_argument("--reseed", action="store_true", help="upload the caches again even if already shipped")
    sub.add_parser("job", help="run inside the Baseten container")
    args = ap.parse_args()
    try:
        if args.cmd == "job":
            try:
                sys.exit(run_job())
            except CloudError as e:
                print(e, flush=True)
                sys.exit(1)
        if args.cmd == "train":
            from .retrain import MODELS  # the same per-model arguments the local trainer uses

            print(train_remote(args.name, MODELS[args.name], reseed=args.reseed))
        else:
            print(train_remote("smoke", ["--epochs", "1", "--steps", "5", "--folds", "none"], reseed=args.reseed))
    except CloudError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
