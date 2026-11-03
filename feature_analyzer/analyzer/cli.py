"""Command line for the analyzer (run from feature_analyzer/ with training/.venv-arm64's python):

    python -m analyzer analyze <paths…> [--out reports/analyzer/<ts>] [--category human|ai_assisted|ai]
                                        [--generator G] [--artist A] [--notes N] [--no-models] [--force]
    python -m analyzer sources add <path…> --category C [--generator G] [--artist A] [--notes N] [--sidecar]
    python -m analyzer sources list
    python -m analyzer retrain --trainer local|modal
    python -m analyzer serve [--port 9200]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import config


def _targets(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p).expanduser()
        if pp.is_dir():
            out += [f for f in sorted(pp.rglob("*")) if f.is_file() and f.suffix.lower() in config.AUDIO_EXTS
                    and not f.name.startswith(".")]
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"skip {pp}: not found", file=sys.stderr)
    return out


def cmd_analyze(args) -> int:
    from . import analysis as A
    out = Path(args.out) if args.out else config.REPORTS_OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"category": args.category, "generator": args.generator, "artist": args.artist, "notes": args.notes}
    results = []
    for p in _targets(args.paths):
        print(f"== {p}")
        a = A.analyze(p, meta=meta, models=not args.no_models, force=args.force, log=lambda s: print(f"   {s}"))
        results.append(a)
        dest = out / a.name
        dest.mkdir(exist_ok=True)
        shutil.copy2(a.dir / "analysis.json", dest / "analysis.json")
        for img in a.images.values():
            shutil.copy2(a.dir / img, dest / img)
        print(f"   P(AI) 10s={A._fmt(a.p_ai('10s'))} 5s={A._fmt(a.p_ai('5s'))}  fc={A._fmt((a.features.get('raw') or {}).get('fc_hz'))} Hz")
    (out / "summary.md").write_text(A.summary_markdown(results))
    print(f"wrote {out / 'summary.md'}")
    return 0 if results else 1


def cmd_sources(args) -> int:
    from .sources import Registry, sidecar_path, write_sidecar
    reg = Registry()
    if args.sub == "list":
        for s in reg.items:
            sc = "sidecar" if sidecar_path(s.path).is_file() else "no sidecar"
            print(f"{s.category:12s} {sc:11s} {s.path}  {s.generator} {s.artist} {s.notes}".rstrip())
        print(f"{len(reg.items)} source(s) in {reg.path}")
        return 0
    n = 0
    for p in _targets(args.paths):
        s = reg.add(p, args.category, args.generator, args.artist, args.notes)
        if args.sidecar:
            sc = write_sidecar(s, reg)
            print(f"wrote {sc}")
        n += 1
    print(f"added {n} source(s) to {reg.path}")
    return 0 if n else 1


def cmd_retrain(args) -> int:
    from .app import retrain_start, retrain_status
    st = retrain_status()
    if not st["reachable"]:
        print(f"retrain service unreachable ({st['error']}): {st['hint']}")
        return 2
    if args.status_only:
        print(json.dumps({k: v for k, v in st["status"].items() if k != "log_tail"}, indent=1))
        return 0
    r = retrain_start(args.trainer)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("analyzer.app:app", host=config.HOST, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analyzer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse files/folders, write a report folder with summary.md")
    a.add_argument("paths", nargs="+")
    a.add_argument("--out", help="report folder (default reports/analyzer/<timestamp>)")
    a.add_argument("--category", choices=config.CATEGORIES)
    a.add_argument("--generator", default="")
    a.add_argument("--artist", default="")
    a.add_argument("--notes", default="")
    a.add_argument("--no-models", action="store_true", help="skip the detector models")
    a.add_argument("--force", action="store_true", help="ignore the cached analysis")
    a.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("sources", help="labelled-audio registry")
    ss = s.add_subparsers(dest="sub", required=True)
    add = ss.add_parser("add")
    add.add_argument("paths", nargs="+")
    add.add_argument("--category", required=True, choices=config.CATEGORIES)
    add.add_argument("--generator", default="")
    add.add_argument("--artist", default="")
    add.add_argument("--notes", default="")
    add.add_argument("--sidecar", action="store_true", help="also write the wmg_train.manifest sidecar JSON")
    ss.add_parser("list")
    s.set_defaults(fn=cmd_sources)

    r = sub.add_parser("retrain", help="start a retrain through the retrain service on :9100")
    r.add_argument("--trainer", choices=["local", "modal"], default="local")
    r.add_argument("--status-only", action="store_true")
    r.set_defaults(fn=cmd_retrain)

    v = sub.add_parser("serve", help="run the web app")
    v.add_argument("--port", type=int, default=config.PORT)
    v.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)
