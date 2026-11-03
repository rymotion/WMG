"""Training on Baseten (`retrain run --trainer baseten`; the job is defined in training/baseten/).

    python -m wmg_train.baseten_cloud sync                 # upload cache-v3 / cache-phone if they changed
    python -m wmg_train.baseten_cloud train 10s|5s         # train one model on Baseten, download runs/<ts>/
    python -m wmg_train.baseten_cloud smoke                # one short epoch, no CV folds
    python -m wmg_train.baseten_cloud job                  # inside the Baseten container (baseten/run.sh)

An S3-compatible bucket (R2 or S3) sits between the Mac and the job, because the caches are far
over Baseten's upload limit and its checkpoint links expire after 7 days:

    caches/<cache>/<fingerprint>/...   immutable; written by `sync`, read by the job
    runs/<ts>/...                      the finished run folder, written by the job
    jobs/<tag>.json                    {"run", "rc", "device"} of one job

Only the feature caches go up (derived log-mels, no audio); the run folder comes back into
training/runs/ so compare/install work exactly as with a local run. Bucket settings: WMG_BUCKET,
WMG_S3_ENDPOINT (R2: https://<account>.r2.cloudflarestorage.com), WMG_S3_REGION, and the usual
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or AWS_PROFILE). `boto3` and `truss` are imported
lazily: the local trainer never needs them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cloud import CACHES, CloudError, fingerprint

TRAINING = Path(__file__).resolve().parents[1]
RUNS = TRAINING / "runs"
JOB_DIR = TRAINING / "baseten"
COMPLETE = ".complete"  # written last, so a half-uploaded or half-downloaded cache never counts
POLL_S = 10
DONE = {"TRAINING_JOB_COMPLETED", "TRAINING_JOB_FAILED", "TRAINING_JOB_DEPLOY_FAILED", "TRAINING_JOB_STOPPED"}
SETUP_HINT = ("Baseten is not set up: in training/, run `.venv-arm64/bin/truss login` once "
              "(or set BASETEN_TRUSS_AUTH_REMOTE_URL and BASETEN_TRUSS_AUTH_API_KEY)")

Line = Callable[[str], None]


# ---------------------------------------------------------------- bucket

class Bucket:
    """The few S3 calls this module needs, on one bucket."""

    def __init__(self, name: str | None = None, client=None):
        self.name = name or os.environ.get("WMG_BUCKET")
        if not self.name:
            raise CloudError("WMG_BUCKET is not set (the R2/S3 bucket between the Mac and Baseten)")
        if client is None:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover
                raise CloudError("the `boto3` package is missing: pip install -e '.[baseten]'") from e
            client = boto3.client("s3", endpoint_url=os.environ.get("WMG_S3_ENDPOINT") or None,
                                  region_name=os.environ.get("WMG_S3_REGION") or None)
        self.s3 = client

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.name, Key=key)
            return True
        except Exception:  # botocore ClientError 404 (or 403 without ListBucket)
            return False

    def keys(self, prefix: str) -> list[str]:
        out, token = [], None
        while True:
            kw = {"Bucket": self.name, "Prefix": prefix, **({"ContinuationToken": token} if token else {})}
            page = self.s3.list_objects_v2(**kw)
            out += [o["Key"] for o in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return out
            token = page["NextContinuationToken"]

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.name, Key=key, Body=data)

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.name, Key=key)["Body"].read()
        except Exception:
            return None

    def upload(self, path: Path, key: str) -> None:
        self.s3.upload_file(str(path), self.name, key)  # multipart for the 2.6 GB windows file

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.name, key, str(path))


def cache_prefix(name: str, fp: str) -> str:
    return f"caches/{name}/{fp}/"


def local_fingerprints(root: Path = TRAINING, caches=CACHES) -> dict[str, str]:
    out = {}
    for name in caches:
        if not (root / name / "windows.f16").exists():
            raise CloudError(f"{root / name} is not a feature cache (no windows.f16)")
        out[name] = fingerprint(root / name)
    return out


def sync_caches(line: Line = print, caches=CACHES, root: Path = TRAINING, bucket: Bucket | None = None) -> list[str]:
    """Upload each cache whose fingerprint is not in the bucket yet. Returns what was uploaded."""
    bucket = bucket or Bucket()
    uploaded = []
    for name, fp in local_fingerprints(root, caches).items():
        prefix = cache_prefix(name, fp)
        if bucket.exists(prefix + COMPLETE):
            line(f"{name}: already in bucket {bucket.name} ({fp[:10]})")
            continue
        files = [p for p in (root / name).iterdir() if p.is_file()]
        line(f"{name}: uploading {sum(p.stat().st_size for p in files) / 1e9:.1f} GB to bucket {bucket.name}...")
        t0 = time.monotonic()
        for p in files:
            bucket.upload(p, prefix + p.name)
        bucket.put(prefix + COMPLETE, fp.encode())
        line(f"{name}: uploaded in {time.monotonic() - t0:.0f} s")
        uploaded.append(name)
    return uploaded


def download_run(bucket: Bucket, run: str, runs: Path = RUNS, line: Line = print) -> Path:
    dest = runs / run
    for key in bucket.keys(f"runs/{run}/"):
        rel = key[len(f"runs/{run}/"):]
        bucket.download(key, dest / rel)
        line(f"downloaded runs/{run}/{rel} ({(dest / rel).stat().st_size / 1e6:.1f} MB)")
    if not (dest / "model.pt").exists():
        raise CloudError(f"runs/{run} came back without model.pt")
    return dest


# ---------------------------------------------------------------- Baseten API

class Baseten:
    """Push / status / logs / stop for training jobs, through the truss SDK."""

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

        job = push(project, remote=self.remote, source_dir=source_dir)
        return job["training_project"]["id"], job["id"]

    def status(self, project_id: str, job_id: str) -> tuple[str, str | None]:
        job = self.api.get_training_job(project_id, job_id)["training_job"]
        return job["current_status"], job.get("error_message")

    def logs(self, project_id: str, job_id: str, since_ms: int | None) -> list[dict]:
        """[{timestamp (epoch ns, str), message}], oldest first."""
        return self.api.get_training_job_logs(project_id, job_id, start_epoch_millis=since_ms)

    def stop(self, project_id: str, job_id: str) -> None:
        self.api.stop_training_job(project_id, job_id)


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


def _make_project(name, extra_args, tag, fingerprints):
    import importlib.util

    spec = importlib.util.spec_from_file_location("wmg_baseten_config", JOB_DIR / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.make_project(name, list(extra_args), tag, fingerprints)


def train_remote(name: str, extra_args: list[str], line: Line = print, runs: Path = RUNS,
                 bucket: Bucket | None = None, api: Baseten | None = None, make_project=_make_project,
                 fingerprints: dict[str, str] | None = None, poll_s: float = POLL_S) -> Path:
    """Push one training job, follow its log into `line`, download the finished run. Any exception
    while waiting (incl. the retrain's Cancelled) stops the job. Run `sync_caches` first."""
    bucket = bucket or Bucket()
    api = api or Baseten()
    fingerprints = fingerprints or local_fingerprints()
    tag = new_tag(name)
    line(f"Baseten: pushing train({name}) as {tag}")
    project_id, job_id = api.push(make_project(name, extra_args, tag, fingerprints), JOB_DIR)
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
    raw = bucket.get(f"jobs/{tag}.json")
    result = json.loads(raw) if raw else {}
    if state != "TRAINING_JOB_COMPLETED" or result.get("rc") != 0 or not result.get("run"):
        raise CloudError(f"remote training failed ({state}, exit {result.get('rc')}"
                         f"{', ' + error if error else ''}); log: truss train logs --job-id {job_id}")
    line(f"Baseten: {name} trained on {result.get('device')} -> runs/{result['run']}")
    return download_run(bucket, result["run"], runs, line)


# ---------------------------------------------------------------- inside the job

def fetch_cache(bucket: Bucket, name: str, fp: str, root: Path) -> Path:
    """The cache with this fingerprint under root/<name>/<fp>, downloaded only if it isn't there
    yet; older fingerprints of the same cache are removed to keep the project cache small."""
    dest = root / name / fp
    if not (dest / COMPLETE).exists():
        if not bucket.exists(cache_prefix(name, fp) + COMPLETE):
            raise CloudError(f"{name} {fp[:10]} is not in bucket {bucket.name}: run `baseten_cloud sync` first")
        print(f"{name}: downloading {fp[:10]} from bucket {bucket.name}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        for key in bucket.keys(cache_prefix(name, fp)):
            if not key.endswith(COMPLETE):
                bucket.download(key, dest / key.rsplit("/", 1)[1])
        (dest / COMPLETE).write_text(fp)
    else:
        print(f"{name}: {fp[:10]} already in the project cache", flush=True)
    for old in (root / name).iterdir():
        if old.name != fp:
            shutil.rmtree(old, ignore_errors=True)
    return dest


def _device() -> str:
    try:
        import torch

        return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def run_job(env=os.environ, bucket: Bucket | None = None, train=None) -> int:
    """Fetch the caches, run `train_baseline`, upload the run and jobs/<tag>.json. Returns the exit code."""
    bucket = bucket or Bucket()
    tag = env["WMG_TAG"]
    store = Path(env.get("BT_PROJECT_CACHE_DIR", "/root/.cache/user_artifacts")) / "caches"
    out = Path(env.get("BT_CHECKPOINT_DIR", "/tmp/training_checkpoints"))
    caches = {name: fetch_cache(bucket, name, env[f"WMG_FP_{name.replace('-', '_').upper()}"], store)
              for name in CACHES}
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.iterdir()}
    cmd = [sys.executable, "-m", "wmg_train.train_baseline", "--cache", str(caches["cache-v3"]),
           "--phone-cache", str(caches["cache-phone"]), "--out", str(out), *env.get("WMG_TRAIN_ARGS", "").split()]
    device = _device()
    print(f"[{env.get('WMG_MODEL')}] {device}\n$ {' '.join(cmd)}", flush=True)
    rc = (train or subprocess.call)(cmd)
    new = sorted(p.name for p in out.iterdir() if p.name not in before and (p / "model.pt").exists())
    run = new[-1] if new else None
    if run:
        for p in sorted((out / run).rglob("*")):
            if p.is_file():
                bucket.upload(p, f"runs/{run}/{p.relative_to(out / run)}")
        print(f"uploaded runs/{run}", flush=True)
    bucket.put(f"jobs/{tag}.json", json.dumps({"run": run, "rc": rc, "device": device, "tag": tag}).encode())
    return rc if run else (rc or 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="upload cache-v3 and cache-phone if they changed")
    t = sub.add_parser("train", help="train one model on Baseten and download the run")
    t.add_argument("name", choices=["10s", "5s"])
    sub.add_parser("smoke", help="one short epoch without CV folds: proves image, GPU and bucket")
    sub.add_parser("job", help="run inside the Baseten container")
    args = ap.parse_args()
    try:
        if args.cmd == "job":
            sys.exit(run_job())
        sync_caches()
        if args.cmd == "train":
            from .retrain import MODELS  # the same per-model arguments the local trainer uses

            print(train_remote(args.name, MODELS[args.name]))
        elif args.cmd == "smoke":
            print(train_remote("smoke", ["--epochs", "1", "--steps", "5", "--folds", "none"]))
    except CloudError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
