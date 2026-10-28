"""One-click retrain: app labels -> phone captures -> both models -> compare -> install.

    python -m wmg_train.retrain run [--build-app] [--yes]   # what Retrain.command runs
    python -m wmg_train.retrain approve | reject            # decide on a run held for review
    python -m wmg_train.retrain serve [--port 9100]         # backs the app's "Retrain models" button
    python -m wmg_train.retrain status

`run` does, in order:
  1. labels    collect labels -- for whole recordings and for single clips -- from the app's
               saved sessions, the exported `.wmgmodel` bundles (exported-model/) and any
               labels the app posted, then `playlist label-session` every labelled backend
               session not captured yet, or captured from the app with different labels
               (audio comes from the backend). Conflicting labels are reported, never guessed.
  2. cache     rebuild cache-phone (AI-assisted recordings are kept but skipped).
  3. train     10 s model, then 5 s model, same settings as the bundled ones.
  4. compare   each new model against the one in the app, on its own cross-validation:
               confound gate, clean and phone AUC, AI recall and human false positives
               (allowed movement in TOLERANCE). Always reported.
  5. install   if neither is worse: install.sh both, regenerate the parity fixture and
               (with --build-app) rebuild the macOS app. If either is worse the pair is held
               for review: `approve` installs it, `reject` keeps the current models.

Progress goes to training/retrain/status.json and retrain.log; every finished run is appended
to retrain/history.jsonl. A lock keeps the script and the app button from running together.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import labels
from .playlist import CAPTURES

TRAINING = Path(__file__).resolve().parents[1]
REPO = TRAINING.parent
RUNS = TRAINING / "runs"
STATE = TRAINING / "retrain"
STATUS, LOG, HISTORY, PENDING, LOCK = (STATE / n for n in
                                       ("status.json", "retrain.log", "history.jsonl", "pending.json", "lock"))
POSTED = STATE / "posted_labels.json"
EXPORTS = REPO / "exported-model"
APP_SOURCE = "app label"  # source_path of captures made from the app's labels
APP_ID = "org.activeie.wmgFrontend"
APP_HISTORY = Path.home() / f"Library/Containers/{APP_ID}/Data/Library/Application Support/{APP_ID}/history"
RESOURCES = REPO / "packages/wmg_detector/darwin/wmg_detector/Resources"
BUILT_APP = REPO / "frontend/build/macos/Build/Products/Release/wmg_frontend.app"
PY = sys.executable

TRAIN_ARGS = ["--cache", "cache-v3", "--phone-cache", "cache-phone"]
MODELS = {"10s": [], "5s": ["--window-seconds", "5", "--save-oof-windows"]}
META = {"10s": "model_meta.json", "5s": "model_meta_5s.json"}
# How far a check may move the wrong way before a new model counts as worse (roughly the
# cross-validation noise between two runs). +1: higher is better; -1: lower is better.
TOLERANCE = {"song_auc": (0.01, 1), "phone_auc": (0.02, 1), "ai_recall": (0.05, 1), "human_fpr": (0.03, -1),
             "phone_ai_recall": (0.05, 1), "phone_human_fpr": (0.05, -1)}
GATE_RANGE = (0.40, 0.55)
MIN_FREE_GB = 4
STEPS = ["labels", "cache", "train_10s", "train_5s", "compare", "install", "build_app"]


# ---------------------------------------------------------------- status / log

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def write_json(p: Path, data) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


class Run:
    """Status of the current run, rewritten to status.json on every change."""

    def __init__(self, build_app: bool):
        self.s = {"state": "running", "pid": os.getpid(), "started_at": now(), "finished_at": None,
                  "step": None, "steps": {k: {"state": "pending"} for k in STEPS if k != "build_app" or build_app},
                  "labels": None, "comparison": None, "installed": None, "message": None, "last_line": ""}
        self._t = 0.0
        self.save()

    def save(self) -> None:
        write_json(STATUS, self.s)

    def step(self, name: str, state: str = "running", detail: str | None = None) -> None:
        st = self.s["steps"].setdefault(name, {})
        st["state"] = state
        if detail is not None:
            st["detail"] = detail
        if state == "running":
            self.s["step"], st["started_at"] = name, now()
        else:
            st["finished_at"] = now()
        log(f"== {name}: {state}{f' ({detail})' if detail else ''}")
        self.save()

    def line(self, text: str) -> None:
        self.s["last_line"] = text[-300:]
        if time.monotonic() - self._t > 1:
            self._t = time.monotonic()
            self.save()

    def finish(self, state: str, message: str) -> None:
        self.s.update(state=state, message=message, finished_at=now(), step=None)
        self.save()
        log(f"== {state}: {message}")


def log(text: str) -> None:
    print(text, flush=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(text + "\n")


def sh(cmd: list[str], run: Run | None, cwd: Path = TRAINING, env: dict | None = None) -> int:
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "LANG": "en_US.UTF-8", "PYTHONUNBUFFERED": "1", **(env or {})})
    for line in p.stdout:
        line = line.rstrip("\n")
        log(line)
        if run:
            run.line(line)
    return p.wait()


def take_lock():
    STATE.mkdir(parents=True, exist_ok=True)
    fh = LOCK.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("a retrain is already running (training/retrain/lock)")
    return fh


def locked() -> bool:
    if not LOCK.exists():
        return False
    with LOCK.open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


# ---------------------------------------------------------------- 1. labels

def _category(sample: dict) -> str | None:
    c = sample.get("category")
    if c in labels.CATEGORIES:
        return c
    return {1: "ai", 0: "human", -1: "ai_assisted"}.get(sample.get("label"))


WHOLE = "*"  # key for a label on the whole recording; clip labels are keyed by clip index


def collect_labels(exports: Path, history: Path, posted: Path | None) -> dict[str, dict]:
    """session id -> {WHOLE or clip index: {category: {where it came from}}}."""
    out: dict[str, dict] = {}

    def add(sid, key, cat, src):
        if sid and cat in labels.CATEGORIES:
            out.setdefault(sid, {}).setdefault(key, {}).setdefault(cat, set()).add(src)

    for f in sorted(exports.glob("*.wmgmodel")) if exports.is_dir() else []:
        b = read_json(f, {})
        for x in b.get("samples", []):
            add(x.get("session_id"), WHOLE, _category(x), f"bundle {f.name}")
    for f in sorted(history.glob("*/session.json")) if history.is_dir() else []:
        s = read_json(f, {})
        src = f"app session {s.get('id')}"
        add(s.get("server_session_id"), WHOLE, (s.get("label") or {}).get("category"), src)
        for c in s.get("clips") or []:
            add(s.get("server_session_id"), c.get("index"), (c.get("label") or {}).get("category"), src)
    for x in (read_json(posted, []) if posted else []):
        idx = x.get("clip_index")
        add(x.get("session_id"), WHOLE if idx is None else int(idx), x.get("category"),
            f"app session {x.get('id', '')}".strip())
    return out


def captured(sid: str) -> tuple[dict[int, str], bool]:
    """Labels already in the training data, by clip index, and whether the app made them."""
    p = CAPTURES / sid / "segments.csv"
    if not p.exists():
        return {}, False
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    have = {int(r["clip_index"]): labels.category_of(r) for r in rows if str(r.get("clip_index", "")).isdigit()}
    return have, bool(rows) and all(r.get("source_path") == APP_SOURCE for r in rows)


def step_labels(run: Run, args) -> dict:
    """Bring the app's labels into the training data. Captures the app made itself are redone
    when a label changes; ones labelled by hand (playlist label-session) are never overwritten."""
    found = collect_labels(args.exports, args.app_history, args.posted)
    report = {"new": [], "updated": [], "already": [], "conflicts": [], "missing_audio": []}
    for sid, keys in sorted(found.items(), key=lambda kv: kv[0]):
        clashes = {str(k): {c: sorted(s) for c, s in cats.items()} for k, cats in keys.items() if len(cats) > 1}
        if clashes:
            report["conflicts"].append({"session_id": sid, "labels": clashes})
            continue
        want = {k: next(iter(cats)) for k, cats in keys.items()}
        default = want.pop(WHOLE, None)
        entry = {"session_id": sid, "category": default, "clips": {str(i): c for i, c in sorted(want.items())}}
        have, by_app = captured(sid)
        if have:
            changed = sorted(i for i in have if want.get(i, default) != have[i])
            if not changed:
                report["already"].append(entry)
                continue
            if not by_app:
                report["conflicts"].append({"session_id": sid, "labels": {
                    str(i): {str(want.get(i, default)): ["app"], have[i]: ["training data"]} for i in changed}})
                continue
        cmd = [PY, "-m", "wmg_train.playlist", "label-session", "--session", sid, "--source", APP_SOURCE,
               "--api", args.api]
        if default:
            cmd += ["--label", default]
        for i, c in sorted(want.items()):
            cmd += ["--clip-label", f"{i}={c}"]
        if have:
            cmd.append("--force")
        rc = sh(cmd, run)
        ok = rc == 0 and captured(sid)[0]
        report["updated" if have and ok else "new" if ok else "missing_audio"].append(entry)
    for c in report["conflicts"]:
        log(f"  conflict {c['session_id']}: {c['labels']} -- not used; fix the label in the app, or relabel "
            f"with `playlist label-session --session {c['session_id']} --label <x> [--clip-label i=<x>] --force`")
    run.s["labels"] = report
    return report


# ---------------------------------------------------------------- 3/4. train + compare

def train(run: Run, name: str) -> Path:
    before = set(RUNS.iterdir()) if RUNS.exists() else set()
    rc = sh([PY, "-m", "wmg_train.train_baseline", *TRAIN_ARGS, *MODELS[name]], run)
    new = sorted(p for p in set(RUNS.iterdir()) - before if (p / "model.pt").exists())
    if rc != 0 or not new:
        raise StepFailed(f"{name} training failed (exit {rc}); see training/retrain/retrain.log")
    return new[-1]


def installed_run(name: str) -> str | None:
    return (read_json(RESOURCES / META[name], {}) or {}).get("model_run")


def summary(run_name: str | None) -> dict | None:
    m = read_json(RUNS / run_name / "metrics.json") if run_name else None
    if not m:
        return None
    oof, ph = m.get("oof", {}), m.get("oof_phone") or {}
    return {"run": run_name, "gate_auc": m.get("confound_gate_auc"),
            "song_auc": oof.get("song_auc"), "ai_recall": oof.get("ai_recall@0.5"),
            "human_fpr": oof.get("human_fpr@0.5"), "phone_auc": ph.get("segment_auc"),
            "phone_ai_recall": ph.get("ai_recall@0.5"), "phone_human_fpr": ph.get("human_fpr@0.5"),
            "phone_sessions": (m.get("phone") or {}).get("sessions"),
            "phone_windows": (m.get("phone") or {}).get("windows")}


def regressions(new: dict, old: dict | None) -> list[str]:
    out = []
    g = new.get("gate_auc")
    if g is None or not GATE_RANGE[0] <= g <= GATE_RANGE[1]:
        out.append(f"confound gate AUC {g} outside {GATE_RANGE[0]}-{GATE_RANGE[1]}")
    if old:
        for k, (tol, sign) in TOLERANCE.items():
            a, b = new.get(k), old.get(k)
            if a is not None and b is not None and sign * (b - a) > tol:
                out.append(f"{k.replace('_', ' ')} {b:.3f} -> {a:.3f} (more than {tol} worse)")
    return out


def compare(new_runs: dict[str, str]) -> dict:
    out = {}
    for name, run_name in new_runs.items():
        new, old = summary(run_name), summary(installed_run(name))
        out[name] = {"new": new, "current": old, "regressions": regressions(new, old)}
        log(f"  {name}: current {old and old['run']} -> new {run_name}")
        for k in ("song_auc", "ai_recall", "human_fpr", "phone_auc", "phone_ai_recall", "phone_human_fpr", "gate_auc"):
            a, b = (old or {}).get(k), new.get(k)
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            log(f"    {k:16s} {fmt(a):>8} -> {fmt(b)}")
        for r in out[name]["regressions"]:
            log(f"    WORSE: {r}")
    return out


# ---------------------------------------------------------------- 5. install

def install(run: Run, runs: dict[str, str], build_app: bool) -> None:
    run.step("install")
    r10, r5 = (str(RUNS / runs[k]) for k in ("10s", "5s"))
    if sh([str(REPO / "packages/wmg_detector/tool/install.sh"), r10, r5], run, cwd=REPO) != 0:
        raise StepFailed("install.sh failed")
    if sh([PY, "-m", "wmg_train.parity_fixture", f"{r10}/model.pt", "--model5", f"{r5}/model.pt"], run) != 0:
        raise StepFailed("parity fixture failed")
    run.s["installed"] = runs
    run.step("install", "done", f"10 s {runs['10s']}, 5 s {runs['5s']}")
    if build_app:
        run.step("build_app")
        flutter = shutil.which("flutter") or str(Path.home() / "SDKs/flutter/bin/flutter")
        if sh([flutter, "build", "macos", "--release"], run, cwd=REPO / "frontend") != 0:
            raise StepFailed("flutter build macos failed; models are installed, rebuild the app by hand")
        run.step("build_app", "done", str(BUILT_APP.relative_to(REPO)))


def record(run: Run) -> None:
    with HISTORY.open("a") as f:
        f.write(json.dumps({k: run.s.get(k) for k in
                            ("state", "started_at", "finished_at", "labels", "comparison", "installed", "message")}) + "\n")


class StepFailed(Exception):
    pass


class Cancelled(Exception):
    pass


def _cancel(*_):
    raise Cancelled


def finish_message(installed: dict, build_app: bool) -> str:
    tail = (" The macOS app is rebuilt; quit and reopen it to load them." if build_app
            else " Rebuild the app (flutter run / flutter build macos) to load them.")
    return f"Installed 10 s {installed['10s']} and 5 s {installed['5s']}.{tail}"


def cmd_run(args) -> None:
    lock = take_lock()  # noqa: F841  held until exit
    LOG.write_text("")
    run = Run(args.build_app)
    signal.signal(signal.SIGTERM, _cancel)
    try:
        free = shutil.disk_usage(TRAINING).free / 1e9
        if free < MIN_FREE_GB:
            raise StepFailed(f"only {free:.1f} GB free; training needs about {MIN_FREE_GB} GB")

        run.step("labels")
        rep = step_labels(run, args)
        run.step("labels", "done", f"{len(rep['new'])} new, {len(rep['updated'])} relabelled, "
                                   f"{len(rep['already'])} already in training, "
                                   f"{len(rep['conflicts'])} conflicts, {len(rep['missing_audio'])} without audio")

        run.step("cache")
        if sh([PY, "-m", "wmg_train.phone_cache", "--out", "cache-phone"], run) != 0:
            raise StepFailed("phone cache build failed")
        meta = read_json(TRAINING / "cache-phone/meta.json", {})
        run.step("cache", "done", f"{meta.get('n')} phone windows from {len(meta.get('sessions', []))} sessions")

        new_runs = {}
        for name in MODELS:
            run.step(f"train_{name}")
            new_runs[name] = train(run, name).name
            run.step(f"train_{name}", "done", new_runs[name])

        run.step("compare")
        cmp = compare(new_runs)
        run.s["comparison"] = cmp
        worse = [f"{n}: {r}" for n, c in cmp.items() for r in c["regressions"]]
        run.step("compare", "done", "worse: " + "; ".join(worse) if worse else "no regressions")

        approve = not worse
        if worse and args.yes:
            approve = True
        elif worse and sys.stdin.isatty():
            approve = input("New models are worse on the checks above. Install anyway? [y/N] ").strip().lower() == "y"
            if not approve:
                run.step("install", "skipped", "kept the current models")
                run.finish("rejected", "New models were worse; kept the current ones.")
                record(run)
                return
        if not approve:
            write_json(PENDING, {"runs": new_runs, "comparison": cmp, "build_app": args.build_app, "at": now()})
            run.step("install", "waiting", "needs your approval")
            run.finish("awaiting_approval", "New models are worse on: " + "; ".join(worse) +
                       ". Approve to install them anyway, or keep the current ones.")
            record(run)
            return
        install(run, new_runs, args.build_app)
        run.finish("done", finish_message(new_runs, args.build_app))
    except StepFailed as e:
        if run.s["step"]:
            run.step(run.s["step"], "failed", str(e))
        run.finish("failed", str(e))
    except (Cancelled, KeyboardInterrupt):
        if run.s["step"]:
            run.step(run.s["step"], "cancelled")
        run.finish("cancelled", "Stopped before finishing; the app's models are unchanged.")
    except Exception as e:  # keep status.json truthful for the app
        run.finish("failed", f"{type(e).__name__}: {e}")
        raise
    finally:
        if run.s["state"] in ("done", "failed", "cancelled"):
            record(run)


def cmd_decide(args) -> None:
    lock = take_lock()  # noqa: F841
    pending = read_json(PENDING)
    if not pending:
        raise SystemExit("nothing is waiting for approval")
    status = read_json(STATUS, {})
    run = Run.__new__(Run)
    run.s, run._t = status, 0.0
    run.s.update(state="running", pid=os.getpid())
    run.save()
    if args.cmd == "reject":
        PENDING.unlink()
        run.step("install", "skipped", "kept the current models")
        run.finish("rejected", "Kept the current models.")
        record(run)
        return
    try:
        install(run, pending["runs"], pending["build_app"])
        PENDING.unlink()
        run.finish("done", "Approved. " + finish_message(pending["runs"], pending["build_app"]))
    except StepFailed as e:
        run.finish("failed", str(e))
    record(run)


# ---------------------------------------------------------------- serve

class Server:
    proc: subprocess.Popen | None = None
    lock = threading.Lock()

    @classmethod
    def spawn(cls, argv: list[str]) -> None:
        cls.proc = subprocess.Popen([PY, "-m", "wmg_train.retrain", *argv], cwd=TRAINING,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)

    @staticmethod
    def status() -> dict:
        s = read_json(STATUS, {"state": "idle"})
        if s.get("state") == "running" and not locked():
            s.update(state="failed", message=s.get("message") or "the retrain process stopped unexpectedly")
        try:
            tail = LOG.read_text().splitlines()[-40:]
        except OSError:
            tail = []
        return {**s, "log_tail": tail, "pending": read_json(PENDING),
                "current": {k: installed_run(k) for k in META},
                "built_app": BUILT_APP.exists()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a) -> None:
        pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/status", "/retrain/status", ""):
            return self._send(200, Server.status())
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path, body = self.path.rstrip("/"), self._body()
        with Server.lock:
            busy = locked()
            if path == "/retrain":
                if busy:
                    return self._send(409, {"error": "a retrain is already running"})
                if PENDING.exists():
                    return self._send(409, {"error": "a previous run is waiting for approval"})
                STATE.mkdir(parents=True, exist_ok=True)
                write_json(POSTED, [x for x in body.get("labels", []) if isinstance(x, dict)])
                Server.spawn(["run", "--posted", str(POSTED), *(["--build-app"] if body.get("build_app", True) else [])])
                return self._send(202, {"ok": True})
            if path in ("/approve", "/reject"):
                if busy:
                    return self._send(409, {"error": "a retrain is already running"})
                if not PENDING.exists():
                    return self._send(409, {"error": "nothing is waiting for approval"})
                Server.spawn([path[1:]])
                return self._send(202, {"ok": True})
            if path == "/cancel":
                p = Server.proc
                if p and p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    return self._send(202, {"ok": True})
                return self._send(409, {"error": "nothing to cancel"})
            if path == "/relaunch":
                if not BUILT_APP.exists():
                    return self._send(409, {"error": "no built app"})
                # The app quits itself after this returns; open the rebuilt copy once it has.
                subprocess.Popen(["sh", "-c", f"sleep 2; open -n '{BUILT_APP}'"], start_new_session=True)
                return self._send(202, {"ok": True})
        self._send(404, {"error": "not found"})


def cmd_serve(args) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"retrain service on http://127.0.0.1:{args.port} (app: Training page -> Retrain models). Ctrl-C to stop.")
    srv.serve_forever()


def cmd_status(_args) -> None:
    s = Server.status()
    print(json.dumps({k: v for k, v in s.items() if k != "log_tail"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="label, cache, train both models, compare, install")
    r.add_argument("--exports", type=Path, default=EXPORTS, help="folder of exported .wmgmodel bundles")
    r.add_argument("--app-history", type=Path, default=APP_HISTORY, help="the macOS app's saved sessions")
    r.add_argument("--posted", type=Path, help="labels sent by the app ([{session_id, category}])")
    r.add_argument("--api", default="http://localhost:9000")
    r.add_argument("--build-app", action="store_true", help="rebuild the macOS app after installing")
    r.add_argument("--yes", action="store_true", help="install even if the new models are worse")
    sub.add_parser("approve", help="install the models held for review")
    sub.add_parser("reject", help="discard the held models and keep the current ones")
    s = sub.add_parser("serve", help="HTTP service for the app's Retrain button")
    s.add_argument("--port", type=int, default=9100)
    sub.add_parser("status")
    args = ap.parse_args()
    {"run": cmd_run, "approve": cmd_decide, "reject": cmd_decide, "serve": cmd_serve, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
