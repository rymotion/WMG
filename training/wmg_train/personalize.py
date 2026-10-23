"""Bring an on-device fine-tuned layer back into a full model.

The app re-fits BandNet's final layer (clip_head) on recordings labelled on the device and
offloads a `.wmgmodel` bundle (format `wmg-personal-head/1`): the layer's weights, the training
report, and the labelled embeddings it was fitted on (no audio).

    python -m wmg_train.personalize list   [--api http://localhost:9000]
    python -m wmg_train.personalize merge  <file.wmgmodel | server id> [--api ...] [--coreml]

`merge` writes runs/<base_run>-personal-<id>/:
    model.pt       the base model with clip_head replaced (loads anywhere model.pt does)
    personal.json  device, training report, and accuracy of base vs personalised layer
    samples.npz    embeddings / labels / recording ids, for analysis or further training
With --coreml it also exports Core ML, so the merged model can be bundled into the app
(packages/wmg_detector/tool/install.sh runs/<new run>).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

RUNS = Path(__file__).resolve().parents[1] / "runs"
FORMAT = "wmg-personal-head/1"


def fetch(api: str, path: str):
    with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=60) as r:
        return json.load(r)


def load_bundle(src: str, api: str) -> dict:
    p = Path(src)
    bundle = json.loads(p.read_text()) if p.exists() else fetch(api, f"/models/personalized/{src}")
    if bundle.get("format") != FORMAT:
        raise SystemExit(f"not a {FORMAT} bundle")
    return bundle


def accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(((x @ w + b > 0).astype(int) == y).mean()) if len(y) else float("nan")


def cmd_list(args) -> None:
    rows = fetch(args.api, "/models/personalized")
    if not rows:
        print("no personalised models on the server")
    for r in rows:
        t = r.get("train") or {}
        print(f"{r['id']}  base={r['base_model_run']}  device={r.get('device')}  samples={r['samples']}  "
              f"acc {t.get('accuracy_before')}->{t.get('accuracy_after')}  "
              f"held-out {t.get('held_out_accuracy_before')}->{t.get('held_out_accuracy_after')}")


def cmd_merge(args) -> None:
    bundle = load_bundle(args.source, args.api)
    base_run = bundle["base_model_run"]
    base_pt = RUNS / base_run / "model.pt"
    if not base_pt.exists():
        raise SystemExit(f"base model {base_pt} not found; the head only fits the model it was trained from")
    ckpt = torch.load(base_pt, map_location="cpu", weights_only=False)
    if ckpt["feature_spec_hash"] != bundle["feature_spec_hash"]:
        raise SystemExit(f"feature spec mismatch: base {ckpt['feature_spec_hash']} vs head {bundle['feature_spec_hash']}")

    sd = ckpt["state_dict"]
    w_base = sd["clip_head.weight"].reshape(-1).numpy().astype(np.float64)
    b_base = float(sd["clip_head.bias"].item())
    w = np.asarray(bundle["head"]["weights"], dtype=np.float64)
    b = float(bundle["head"]["bias"])
    if w.shape != w_base.shape:
        raise SystemExit(f"head has {w.size} weights, base clip_head has {w_base.size}")

    samples = bundle.get("samples", [])
    x = np.asarray([s["embedding"] for s in samples], dtype=np.float64).reshape(len(samples), -1) if samples else np.zeros((0, w.size))
    y = np.asarray([s["label"] for s in samples], dtype=int)
    sessions = np.asarray([s["session_id"] for s in samples])

    out = RUNS / f"{base_run}-personal-{bundle['id']}"
    out.mkdir(parents=True, exist_ok=args.force)
    sd = {k: v.clone() for k, v in sd.items()}
    sd["clip_head.weight"] = torch.tensor(w, dtype=sd["clip_head.weight"].dtype).reshape(sd["clip_head.weight"].shape)
    sd["clip_head.bias"] = torch.tensor([b], dtype=sd["clip_head.bias"].dtype)
    torch.save({**ckpt, "state_dict": sd, "personalized_from": bundle["id"], "base_model_run": base_run}, out / "model.pt")
    np.savez_compressed(out / "samples.npz", embedding=x.astype(np.float32), label=y, session_id=sessions)
    report = {
        "head_id": bundle["id"], "base_model_run": base_run, "device": bundle.get("device"),
        "exported_at": bundle.get("exported_at"), "device_train_report": bundle.get("train"),
        "samples": int(len(y)), "ai_samples": int(y.sum()), "recordings": int(len(set(sessions))),
        "accuracy_base_head": accuracy(x, y, w_base, b_base), "accuracy_personal_head": accuracy(x, y, w, b),
        "weight_change_l2": float(np.linalg.norm(w - w_base)), "feature_spec_hash": bundle["feature_spec_hash"],
    }
    (out / "personal.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"saved {out}/model.pt")
    if args.coreml:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "wmg_train.export_coreml", str(out / "model.pt")], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="personalised models uploaded to the backend")
    ls.add_argument("--api", default="http://localhost:9000")
    mg = sub.add_parser("merge", help="merge a .wmgmodel (file or server id) into a full model.pt")
    mg.add_argument("source")
    mg.add_argument("--api", default="http://localhost:9000")
    mg.add_argument("--coreml", action="store_true", help="also export Core ML for the app")
    mg.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"list": cmd_list, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
