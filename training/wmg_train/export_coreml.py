"""Convert a trained BandNet to Core ML for on-device inference (iOS / macOS).

    python -m wmg_train.export_coreml runs/<ts>/model.pt [--plugin ../packages/wmg_detector]

Writes `runs/<ts>/coreml/WmgDetector.mlpackage`, a compiled `WmgDetector.mlmodelc` (loaded at
runtime with MLModel(contentsOf:)), and `parity.json` comparing Core ML with PyTorch on real
feature windows. With --plugin, copies the compiled model + metadata into the Flutter plugin.

Input `logmel` is [batch 1-16, 1, 128, 1000] absolute dB from wmg_features (level-normalised).
Outputs: `ai_logit` [B], `p_ai` [B], `band_scores` [B, 3] (bass, mid, treble),
`frame_logits` [B, 3, 125], `attention` [B, 125], `embedding` [B, 771] (input to the final
layer: ai_logit = head_weights . embedding + head_bias). A clip's score is sigmoid(mean ai_logit).
model_meta.json carries head_weights/head_bias so the app can fine-tune that layer on device.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

from . import features
from .score import load_model

OUTPUTS = ("ai_logit", "p_ai", "band_scores", "frame_logits", "attention", "embedding")
MAX_BATCH = 16


class _Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[k] for k in OUTPUTS)


def sample_windows(n: int = 8) -> np.ndarray:
    """Real feature windows for tracing/parity: a few songs from training_data, level-normalised."""
    from .groups import inventory
    from .score import normalize_level
    from .augment import decode
    import tempfile

    root = Path(__file__).resolve().parents[2] / "training_data"
    songs = inventory(root)
    pick = [s for s in songs if s.label == 1][:2] + [s for s in songs if s.label == 0][:2]
    mels = []
    with tempfile.TemporaryDirectory() as d:
        for s in pick:
            x, sr = decode(s.path, Path(d))
            mono = normalize_level(x.astype(np.float64).mean(axis=1))
            mels += [w[2] for w in features.extract_windows(mono, sr)[: n // len(pick)]]
    return np.stack(mels).astype(np.float32)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--plugin", type=Path, help="Flutter plugin dir to install the compiled model into")
    args = ap.parse_args()

    run_dir = args.model.resolve().parent
    out = run_dir / "coreml"
    out.mkdir(exist_ok=True)
    model = load_model(args.model).eval()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    wrap = _Wrap(model).eval()

    frames = model.frames  # 1000 for 10 s windows, 500 for 5 s
    x = np.ascontiguousarray(sample_windows()[..., :frames])
    with torch.no_grad():
        traced = torch.jit.trace(wrap, torch.from_numpy(x[:1]))
        ref = [t.numpy() for t in wrap(torch.from_numpy(x))]

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="logmel", shape=(ct.RangeDim(1, MAX_BATCH, default=1), 1, 128, frames),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name=n) for n in OUTPUTS],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS15,
    )
    meta = {
        "model_run": run_dir.name,
        "feature_spec_hash": ckpt["feature_spec_hash"],
        "level_target_dbfs": -14.0,
        "precision": args.precision,
        "outputs": list(OUTPUTS),
        "bands": ["bass", "mid", "treble"],
        "embedding_dim": int(model.clip_head.in_features),
        "frames": int(frames),
        "window_seconds": frames / 100,
    }
    head = {"head_weights": model.clip_head.weight.detach().reshape(-1).tolist(),
            "head_bias": float(model.clip_head.bias.detach().item())}
    mlmodel.short_description = "AI-generated music detector (BandNet). Score = sigmoid(mean ai_logit over 10 s windows)."
    for k, v in meta.items():
        mlmodel.user_defined_metadata[k] = json.dumps(v) if not isinstance(v, str) else v
    pkg = out / "WmgDetector.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(str(pkg))

    # Parity: Core ML vs PyTorch on the same windows (prediction runs on macOS only).
    pred = mlmodel.predict({"logmel": x})
    diff = {k: float(np.abs(np.asarray(pred[k]).reshape(r.shape) - r).max()) for k, r in zip(OUTPUTS, ref)}
    p_ref = 1 / (1 + np.exp(-ref[0]))
    p_ml = 1 / (1 + np.exp(-np.asarray(pred["ai_logit"]).reshape(-1)))
    # The app's personalised logit is computed from embedding + head weights; check it matches.
    emb = np.asarray(pred["embedding"]).reshape(len(x), -1)
    w = model.clip_head.weight.detach().numpy().reshape(-1)
    head_logit = emb @ w + float(model.clip_head.bias.detach().item())
    parity = {"windows": int(len(x)), "max_abs_diff": diff, "max_abs_diff_p_ai": float(np.abs(p_ref - p_ml).max()),
              "max_abs_diff_head_logit_vs_ai_logit": float(np.abs(head_logit - np.asarray(pred["ai_logit"]).reshape(-1)).max())}
    (out / "parity.json").write_text(json.dumps(parity, indent=1))
    print(json.dumps(parity, indent=1))

    compiled = out / "WmgDetector.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out)], check=True, capture_output=True)
    # Head weights go only into the app-side metadata (too long for Core ML metadata strings).
    (out / "model_meta.json").write_text(json.dumps(meta | head, indent=1))
    print(f"saved {pkg} and {compiled}")

    if args.plugin:
        dest = args.plugin / "darwin" / "wmg_detector" / "Resources"
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "WmgDetector.mlmodelc").exists():
            shutil.rmtree(dest / "WmgDetector.mlmodelc")
        shutil.copytree(compiled, dest / "WmgDetector.mlmodelc")
        shutil.copy(out / "model_meta.json", dest / "model_meta.json")
        print(f"installed into {dest}")


if __name__ == "__main__":
    main()
