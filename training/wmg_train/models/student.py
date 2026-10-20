"""BandNet: the small student detector distilled from the PaSST/MERT teacher.

Input is the wmg_features log-mel [B, 1, 128, 1000] (absolute dB). A shared depthwise-separable
stem feeds three branches that each see only their mel-bin range (bass 20-250 Hz, mid
250-4000 Hz, treble 4000-16000 Hz). Each branch emits per-frame logits; one attention head
pools all bands over time, giving a clip score plus explainable per-band and per-stem scores.

Only ops that convert cleanly to ONNX, CoreML and LiteRT are used: Conv2d (incl. depthwise),
BatchNorm, ReLU/Hardswish, AvgPool2d, Linear, Softmax, Sigmoid, elementwise mul/add and sum.
No custom ops and no data-dependent shapes: the band slices are fixed at construction.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

BAND_NAMES = ("bass", "mid", "treble")
DEFAULT_BANDS = {"bass": (20.0, 250.0), "mid": (250.0, 4000.0), "treble": (4000.0, 16000.0)}
DEFAULT_STEMS = ("vocals", "bass", "drums", "other")
OUTPUT_NAMES = ("p_ai", "band_scores", "stem_scores", "frame_logits", "attention")

# Slaney mel scale (librosa htk=False), matching wmg_features' mel_scale = "slaney".
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = math.log(6.4) / 27.0


def hz_to_mel(f: float) -> float:
    return f / _F_SP if f < _MIN_LOG_HZ else _MIN_LOG_MEL + math.log(f / _MIN_LOG_HZ) / _LOGSTEP


def mel_to_hz(m: float) -> float:
    return m * _F_SP if m < _MIN_LOG_MEL else _MIN_LOG_HZ * math.exp(_LOGSTEP * (m - _MIN_LOG_MEL))


def mel_band_centers_hz(n_mels: int = 128, fmin: float = 20.0, fmax: float = 16000.0) -> list[float]:
    """Centre frequency of each mel filter (n_mels + 2 equally spaced mel points, inner ones)."""
    lo, hi = hz_to_mel(fmin), hz_to_mel(fmax)
    step = (hi - lo) / (n_mels + 1)
    return [mel_to_hz(lo + step * (i + 1)) for i in range(n_mels)]


def band_bin_ranges(
    bands: dict[str, tuple[float, float]] | None = None,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
) -> dict[str, tuple[int, int]]:
    """Mel-bin [start, end) per band: bins whose centre lies in [lo, hi) (last band closed)."""
    bands = bands or DEFAULT_BANDS
    centers = mel_band_centers_hz(n_mels, fmin, fmax)
    names = list(bands)
    out: dict[str, tuple[int, int]] = {}
    for k, name in enumerate(names):
        lo, hi = bands[name]
        last = k == len(names) - 1
        idx = [i for i, c in enumerate(centers) if lo <= c and (c <= hi if last else c < hi)]
        if not idx:
            raise ValueError(f"band {name} {lo}-{hi} Hz contains no mel bins")
        out[name] = (idx[0], idx[-1] + 1)
    return out


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "hardswish": nn.Hardswish}[name]()


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by BN + activation."""

    def __init__(self, c_in: int, c_out: int, stride: tuple[int, int], act: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            _act(act),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            _act(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BandBranch(nn.Module):
    """DS-conv stack over one band's frequency slice -> [B, C, 1, T'] (freq averaged out)."""

    def __init__(self, c_in: int, channels: list[int], freq_bins: int, act: str):
        super().__init__()
        strides = [(2, 2)] + [(2, 1)] + [(1, 1)] * (len(channels) - 2)
        layers, c = [], c_in
        for c_out, s in zip(channels, strides):
            layers.append(DSConv(c, c_out, s, act))
            c = c_out
        self.convs = nn.Sequential(*layers)
        f = freq_bins
        for s, _ in strides[: len(channels)]:
            f = (f + 2 - 3) // s + 1  # conv output size with k=3, p=1
        self.pool = nn.AvgPool2d((f, 1))  # static kernel: exports as plain AveragePool
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.convs(x))


class BandNet(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        frames: int = 1000,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        stem_channels: tuple[int, ...] = (32, 64, 96),
        branch_channels: tuple[int, ...] = (128, 192, 256),
        attention_hidden: int = 128,
        activation: str = "hardswish",
        dropout: float = 0.1,
        bands: dict[str, tuple[float, float]] | None = None,
        stems: tuple[str, ...] = DEFAULT_STEMS,
    ):
        super().__init__()
        if len(stem_channels) != 3 or len(branch_channels) < 2:
            raise ValueError("stem_channels needs 3 entries, branch_channels at least 2")
        self.n_mels, self.frames = n_mels, frames
        self.band_names = tuple(bands or DEFAULT_BANDS)
        self.stems = tuple(stems)
        self.mel_ranges = band_bin_ranges(bands, n_mels, fmin, fmax)

        c0, c1, c2 = stem_channels
        # Input BN learns the dB offset/scale; it folds into the first conv at export.
        self.input_norm = nn.BatchNorm2d(1)
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(c0),
            _act(activation),
            DSConv(c0, c1, (2, 2), activation),  # freq /2, time /4 overall
            DSConv(c1, c2, (1, 1), activation),
        )
        # Band slices in stem coordinates (freq downsampled by 2), fixed Python ints.
        self.stem_ranges = {
            name: (lo // 2, max(lo // 2 + 1, -(-hi // 2))) for name, (lo, hi) in self.mel_ranges.items()
        }
        self.branches = nn.ModuleDict({
            name: BandBranch(c2, list(branch_channels), hi - lo, activation)
            for name, (lo, hi) in self.stem_ranges.items()
        })
        c = branch_channels[-1]
        n_bands = len(self.band_names)
        self.frame_heads = nn.ModuleDict({name: nn.Conv2d(c, 1, 1) for name in self.band_names})
        self.attention = nn.Sequential(
            nn.Conv2d(c * n_bands, attention_hidden, 1),
            _act(activation),
            nn.Conv2d(attention_hidden, 1, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.clip_head = nn.Linear(c * n_bands + n_bands, 1)
        self.stem_head = nn.Linear(c * n_bands, len(self.stems))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 1, n_mels, frames] log-mel dB."""
        h = self.stem(self.input_norm(x))  # [B, C, F/2, T/4]

        feats, frame_logits = [], []
        for name in self.band_names:
            lo, hi = self.stem_ranges[name]
            f = self.branches[name](h[:, :, lo:hi, :])  # [B, C, 1, T']
            feats.append(f)
            frame_logits.append(self.frame_heads[name](f))  # [B, 1, 1, T']
        feat = torch.cat(feats, dim=1)  # [B, 3C, 1, T']
        frames = torch.cat(frame_logits, dim=1).flatten(2)  # [B, 3, T']

        att = torch.softmax(self.attention(feat).flatten(1), dim=-1)  # [B, T']
        w = att.unsqueeze(1)  # [B, 1, T']
        pooled = (feat.flatten(2) * w).sum(dim=-1)  # [B, 3C]
        band_logits = (frames * w).sum(dim=-1)  # [B, 3]

        z = self.dropout(pooled)
        ai_logit = self.clip_head(torch.cat([z, band_logits], dim=1)).squeeze(1)  # [B]
        stem_logits = self.stem_head(z)  # [B, 4]
        return {
            "p_ai": torch.sigmoid(ai_logit),
            "band_scores": torch.sigmoid(band_logits),
            "stem_scores": torch.sigmoid(stem_logits),
            "frame_logits": frames,
            "attention": att,
            # Logits for numerically stable BCE during training; not exported.
            "ai_logit": ai_logit,
            "stem_logits": stem_logits,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BandNet":
        i, m = cfg["input"], cfg["model"]
        return cls(
            n_mels=i["n_mels"], frames=i["frames"], fmin=i["fmin"], fmax=i["fmax"],
            stem_channels=tuple(m["stem_channels"]), branch_channels=tuple(m["branch_channels"]),
            attention_hidden=m["attention_hidden"], activation=m["activation"],
            dropout=m["dropout"], bands={k: tuple(v) for k, v in m["bands"].items()},
            stems=tuple(m["stems"]),
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class _ExportWrapper(nn.Module):
    """Tuple outputs in OUTPUT_NAMES order (exporters prefer tuples over dicts)."""

    def __init__(self, model: BandNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[k] for k in OUTPUT_NAMES)


def export_onnx(model: BandNet, path: str | Path, opset: int = 17) -> Path:
    path = Path(path)
    # The wrapper must be in eval mode too: export restores the wrapper's original mode
    # afterwards, which would otherwise flip `model` back into training mode.
    wrapper = _ExportWrapper(model).eval()
    dummy = torch.zeros(1, 1, model.n_mels, model.frames)
    kwargs: dict[str, Any] = dict(
        input_names=["logmel"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={"logmel": {0: "batch"}, **{k: {0: "batch"} for k in OUTPUT_NAMES}},
        opset_version=opset,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # TorchScript exporter: stable, no onnxscript needed
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy,), str(path), **kwargs)
    return path
