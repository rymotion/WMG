"""Modal app that trains the detector models in the cloud (`wmg_train.retrain --trainer modal`).

One-time setup on the Mac, from training/:

    .venv-arm64/bin/modal setup                      # log in (opens the browser)
    .venv-arm64/bin/modal deploy modal_app.py        # build the image, deploy app "wmg-train"

Then `python -m wmg_train.cloud sync` uploads the feature caches and
`python -m wmg_train.cloud train 10s` (or `retrain run --trainer modal`) trains on a GPU.

What leaves the Mac: `cache-v3/` and `cache-phone/` (float16 log-mel windows, their index and
meta: derived features, no audio) and `configs/student.yaml`. Results come back as the usual
`runs/<ts>/` folder (model.pt, model.onnx, metrics.json, oof_*.csv) through the same Volume.

Smoke test of image + GPU + Volume layout (one short epoch, no CV folds):

    .venv-arm64/bin/modal run modal_app.py::smoke
"""

from __future__ import annotations

import modal

APP_NAME = "wmg-train"
VOLUME_NAME = "wmg-train"
DATA = "/data"  # Volume mount: /data/cache-v3, /data/cache-phone, /data/runs, /data/logs
GPU = ["L4", "A10", "T4"]  # BandNet is ~0.4 M params; the cheapest card that is free wins
TIMEOUT_S = 3 * 3600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Same versions as training/.venv-arm64 so cloud and Mac metrics stay comparable.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "numpy==2.5.4",
        "pandas==3.0.6",
        "pyarrow==26.0.0",
        "scikit-learn==1.9.1",
        "pyyaml==6.0.3",
        "onnx==1.23.2",
    )
    .env({"PYTHONUNBUFFERED": "1", "WMG_STUDENT_CFG": "/root/configs/student.yaml"})
    .add_local_file("configs/student.yaml", "/root/configs/student.yaml", copy=True)
    .add_local_python_source("wmg_train")
)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={DATA: volume})
def train(name: str, extra_args: list[str], tag: str) -> dict:
    """Run `wmg_train.train_baseline` on the Volume's caches. The log is written to
    /data/logs/<tag>.log and committed every few seconds so the Mac can follow it; returns
    {"run": "<ts>" | None, "rc": exit code, "device": "cuda" | ...}."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    import torch

    data = Path(DATA)
    runs, logs = data / "runs", data / "logs"
    runs.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    before = {p.name for p in runs.iterdir()}
    cmd = [sys.executable, "-m", "wmg_train.train_baseline", "--cache", str(data / "cache-v3"),
           "--phone-cache", str(data / "cache-phone"), "--out", str(runs), *extra_args]
    device = f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    with (logs / f"{tag}.log").open("w") as log:
        log.write(f"[{name}] {device}\n$ {' '.join(cmd)}\n")
        log.flush()
        volume.commit()
        p = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = time.monotonic()
        for line in p.stdout:
            log.write(line)
            log.flush()
            if time.monotonic() - last > 10:
                volume.commit()
                last = time.monotonic()
        rc = p.wait()
        log.write(f"[{name}] exit {rc}\n")
    volume.commit()
    new = sorted(p.name for p in runs.iterdir() if p.name not in before and (p / "model.pt").exists())
    return {"run": new[-1] if new else None, "rc": rc, "device": device, "tag": tag}


@app.local_entrypoint()
def smoke():
    """One short training without CV folds: proves the image, the GPU and the Volume layout."""
    import time

    tag = f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    res = train.remote("smoke", ["--epochs", "1", "--steps", "5", "--folds", "none"], tag)
    print(res)
    if res["rc"] != 0 or not res["run"]:
        raise SystemExit(f"smoke training failed; see `modal volume get {VOLUME_NAME} logs/{tag}.log -`")
