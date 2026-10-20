"""Pick the training/inference device: CUDA, then Apple MPS, then CPU."""

from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """`prefer` (or $WMG_DEVICE) forces "cuda" / "mps" / "cpu"; otherwise cuda -> mps -> cpu.

    MPS needs an arm64 Python + torch build; under an x86_64 (Rosetta) interpreter it is never
    available and this falls back to CPU.
    """
    prefer = prefer or os.environ.get("WMG_DEVICE")
    if prefer:
        dev = torch.device(prefer)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
        if dev.type == "mps" and not mps_available():
            raise RuntimeError("mps requested but unavailable (needs arm64 Python + macOS 12.3+)")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available() and backend.is_built())
