"""`wmg-check`: build the manifest and run every dataset acceptance check.

    wmg-check --data-root ../training_data --out manifests/

Exit code 0 only if: no nulls, splits leak nothing, buckets are balanced and the
codec/loudness baseline is at chance (AUC <= 0.55).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features
from .balance import BalanceError, check_balance
from .config import load_config
from .confound import ConfoundError, check_confound
from .manifest import ManifestError, build_manifest, write_manifest
from .splits import SplitError, held_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wmg-check", description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--config", default="data", help="config name or YAML path (default: data)")
    ap.add_argument("--out", type=Path, default=Path("manifests"))
    ap.add_argument("--feature-crate-commit",
                    help="override when wmg_features is not installed (not recommended)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if features.available():
        features.check_spec(cfg)
    elif not args.feature_crate_commit:
        print("warning: wmg_features not installed; feature_crate_commit will be null "
              "and the manifest check will fail. Install it or pass --feature-crate-commit.",
              file=sys.stderr)

    failed = False
    try:
        df = build_manifest(args.data_root, cfg, args.feature_crate_commit)
    except (ManifestError, SplitError) as e:
        print(f"FAIL manifest: {e}", file=sys.stderr)
        return 1
    pq, csv = write_manifest(df, args.out)
    print(f"ok   manifest: {len(df)} windows, {df['track_id'].nunique()} tracks -> {pq}, {csv}")
    print("     split sizes: " + ", ".join(f"{k}={v}" for k, v in df["split"].value_counts().items()))
    print(f"     held-out test generators: {sorted(held_out(df, 'generator', 1))}")
    print(f"     held-out test artists: {len(held_out(df, 'artist_id', 0))}")

    try:
        report = check_balance(df, cfg["balance"])
        print(f"ok   balance: {len(report)} buckets")
    except BalanceError as e:
        e.report.to_csv(args.out / "balance_report.csv", index=False)
        print(f"FAIL balance: {e}", file=sys.stderr)
        failed = True

    try:
        r = check_confound(df, cfg["confound"])
        print(f"ok   confound: test AUC {r.auc:.3f} <= {r.max_auc}")
    except ConfoundError as e:
        print(f"FAIL confound: {e}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
