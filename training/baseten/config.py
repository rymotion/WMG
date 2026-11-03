"""Baseten Training job that trains one detector model (`wmg_train.retrain --trainer baseten`).

`wmg_train.baseten_cloud` builds the job with `make_project()` and pushes it with this folder
as the source directory. From training/:

    .venv-arm64/bin/truss login                                  # once: paste a Baseten API key
    .venv-arm64/bin/python -m wmg_train.baseten_cloud sync       # caches -> bucket (when changed)
    .venv-arm64/bin/python -m wmg_train.baseten_cloud smoke      # one short epoch, no CV folds

The uploaded archive is this folder plus ../wmg_train and ../configs (see `Workspace`); the
feature caches are far over Baseten's upload limit, so the job pulls them from the bucket
(caches/<name>/<fingerprint>/) into the project cache and only downloads a cache once.
Bucket settings come from the environment of whoever pushes (WMG_BUCKET, WMG_S3_ENDPOINT,
WMG_S3_REGION); the keys come from the Baseten secrets `wmg_s3_access_key_id` and
`wmg_s3_secret_access_key`.
"""

from __future__ import annotations

import os

from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    AvailabilityModel,
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
    Workspace,
)

PROJECT = "wmg-detector"
BASE_IMAGE = "python:3.12-slim"  # torch wheels from PyPI bundle CUDA; run.sh pins the versions


def make_project(name: str, extra_args: list[str], tag: str, fingerprints: dict[str, str],
                 gpu: str | None = None, spot: bool | None = None) -> TrainingProject:
    """One training job: `train_baseline <extra_args>` on the bucket's caches with these fingerprints.
    `gpu="none"` (or WMG_BASETEN_GPU=none) asks for a CPU-only job."""
    gpu = gpu or os.environ.get("WMG_BASETEN_GPU", "H100")
    spot = os.environ.get("WMG_BASETEN_SPOT", "1") == "1" if spot is None else spot
    env: dict[str, str | SecretReference] = {
        "PYTHONUNBUFFERED": "1",
        "WMG_MODEL": name,
        "WMG_TRAIN_ARGS": " ".join(extra_args),
        "WMG_TAG": tag,
        "WMG_BUCKET": os.environ.get("WMG_BUCKET", ""),
        "WMG_S3_ENDPOINT": os.environ.get("WMG_S3_ENDPOINT", ""),
        "WMG_S3_REGION": os.environ.get("WMG_S3_REGION", "auto"),
        "AWS_ACCESS_KEY_ID": SecretReference(name="wmg_s3_access_key_id"),
        "AWS_SECRET_ACCESS_KEY": SecretReference(name="wmg_s3_secret_access_key"),
        **{f"WMG_FP_{k.replace('-', '_').upper()}": v for k, v in fingerprints.items()},
    }
    compute = Compute(
        cpu_count=8,
        memory="32Gi",
        accelerator=None if gpu.lower() == "none" else AcceleratorSpec(accelerator=gpu, count=1),
        availability_model=AvailabilityModel.SPOT if spot else AvailabilityModel.DEDICATED,
    )
    job = TrainingJob(
        name=tag,
        image=Image(base_image=BASE_IMAGE),
        compute=compute,
        runtime=Runtime(
            start_commands=["sh ./run.sh"],
            environment_variables=env,
            cache_config=CacheConfig(enabled=True),
            checkpointing_config=CheckpointingConfig(enabled=True),
        ),
        workspace=Workspace(external_dirs=["../wmg_train", "../configs"]),
    )
    return TrainingProject(name=PROJECT, job=job)

