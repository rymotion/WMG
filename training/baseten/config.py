"""Baseten Training job that trains one detector model (`wmg_train.retrain --trainer baseten`).

`wmg_train.baseten_cloud` builds the job with `make_project()` and pushes it with this folder
as the source directory. From training/:

    .venv-arm64/bin/truss login                                  # once: paste a Baseten API key
    .venv-arm64/bin/python -m wmg_train.baseten_cloud smoke      # one short epoch, no CV folds

The uploaded archive is this folder plus ../wmg_train and ../configs (see `Workspace`), plus
../cache-v3 and/or ../cache-phone when the project cache doesn't have that fingerprint yet
(`ship`). No secrets: the job needs no credentials of its own.
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
    TrainingJob,
    TrainingProject,
    Workspace,
)

PROJECT = "wmg-detector"
BASE_IMAGE = "python:3.12-slim"  # torch wheels from PyPI bundle CUDA; run.sh pins the versions


def make_project(name: str, extra_args: list[str], tag: str, fingerprints: dict[str, str],
                 ship: list[str] = (), gpu: str | None = None, spot: bool | None = None) -> TrainingProject:
    """One training job: `train_baseline <extra_args>` on the caches with these fingerprints;
    the caches in `ship` go up with the job. `gpu="none"` (or WMG_BASETEN_GPU=none) asks for a
    CPU-only job."""
    gpu = gpu or os.environ.get("WMG_BASETEN_GPU", "H100")
    spot = os.environ.get("WMG_BASETEN_SPOT", "1") == "1" if spot is None else spot
    env: dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "WMG_MODEL": name,
        "WMG_TRAIN_ARGS": " ".join(extra_args),
        "WMG_TAG": tag,
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
        workspace=Workspace(external_dirs=["../wmg_train", "../configs", *(f"../{c}" for c in ship)]),
    )
    return TrainingProject(name=PROJECT, job=job)

