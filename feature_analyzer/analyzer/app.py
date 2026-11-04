"""FastAPI routes for the analyzer (uvicorn on 127.0.0.1:9200; see Analyzer.command)."""

from __future__ import annotations

import json
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import analysis as A
from . import config
from .jobs import manager
from .sources import Registry, sidecar_path, write_sidecar

app = FastAPI(title="WMG feature analyzer")
app.mount("/static", StaticFiles(directory=str(config.HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(config.HERE / "templates"))


# ---------------------------------------------------------------- template helpers

def fmt(v, digits: int = 3) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "–"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{digits}g}"
    return str(v)


def band(b) -> str:
    return "–" if not b else f"{fmt(b[0])} · {fmt(b[1])} · {fmt(b[2])}"


def pclass(p) -> str:
    """CSS class for a probability cell."""
    if p is None:
        return ""
    return "p-ai" if p >= A.P_AI else "p-mid" if p >= A.P_HUMAN else "p-human"


def median(values) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


templates.env.filters.update({"fmt": fmt, "band": band, "pclass": pclass, "median": median})
templates.env.globals.update({"CATEGORIES": config.CATEGORIES, "MODEL_NAMES": A.MODEL_NAMES,
                              "CONDITIONS": A.CONDITIONS, "SIDE_KEYS": ("flux_p2m", "mod_peak_db", "sfm_mean")})


def render(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"request": request, "active_jobs": len(manager.active()), **ctx})


# ---------------------------------------------------------------- retrain service client

def retrain_status() -> dict:
    try:
        with urllib.request.urlopen(f"{config.RETRAIN_URL}/status", timeout=3) as r:
            body = r.read()
    except (urllib.error.URLError, OSError) as e:
        return {"reachable": False, "error": f"nothing answered at {config.RETRAIN_URL} ({getattr(e, 'reason', e)})",
                "hint": config.RETRAIN_HINT}
    try:
        status = json.loads(body)
        if not isinstance(status, dict) or "state" not in status:
            raise ValueError("no state field")
    except ValueError:
        return {"reachable": False, "hint": config.RETRAIN_HINT,
                "error": f"{config.RETRAIN_URL} answered, but not with the retrain service's JSON: another program "
                         "(Flutter DevTools often takes port 9100) may hold the port; stop it first, then"}
    return {"reachable": True, "status": status}


def retrain_start(trainer: str) -> dict:
    body = json.dumps({"labels": [], "build_app": False, "trainer": trainer}).encode()
    req = urllib.request.Request(f"{config.RETRAIN_URL}/retrain", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"ok": True, "code": r.status, "body": json.load(r)}
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e)
        except ValueError:
            detail = {"error": e.reason}
        return {"ok": False, "code": e.code, "body": detail}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "code": 0, "body": {"error": f"service unreachable ({getattr(e, 'reason', e)}): {config.RETRAIN_HINT}"}}


# ---------------------------------------------------------------- jobs

def submit_analysis(path: Path, meta: dict, force: bool = False):
    def run(log):
        a = A.analyze(path, meta=meta, force=force, log=log)
        return {"analysis_id": a.id, "name": a.name, "p_ai": a.headline()}
    return manager.submit(f"analyze {path.name}", run)


def scan_folder(folder: Path, limit: int = 200) -> list[Path]:
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix.lower() in config.AUDIO_EXTS and not p.name.startswith(".")]
    return files[:limit]


# ---------------------------------------------------------------- routes

@app.get("/")
def home():
    return RedirectResponse("/analyses", status_code=303)


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return render(request, "upload.html", data_dir=str(config.DATA))


@app.post("/upload")
async def upload(request: Request, files: list[UploadFile] | None = None, folder: str = Form(""),
                 category: str = Form(""), generator: str = Form(""), artist: str = Form(""),
                 notes: str = Form(""), force: str = Form("")):
    meta = {"category": category or None, "generator": generator, "artist": artist, "notes": notes}
    queued, problems = [], []
    config.UPLOADS.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in config.AUDIO_EXTS:
            problems.append(f"{name}: not an audio extension")
            continue
        tmp = config.UPLOADS / f"incoming-{name}"
        with open(tmp, "wb") as out:
            shutil.copyfileobj(f.file, out)
        sha = A.sha1_of(tmp)
        dest = config.UPLOADS / sha[:12] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(dest)
        queued.append(submit_analysis(dest, meta, force=bool(force)))
    folder = folder.strip()
    if folder:
        p = Path(folder).expanduser()
        if p.is_file():
            queued.append(submit_analysis(p, meta, force=bool(force)))
        elif p.is_dir():
            found = scan_folder(p)
            if not found:
                problems.append(f"{p}: no audio files found")
            for fp in found:
                queued.append(submit_analysis(fp, meta, force=bool(force)))
        else:
            problems.append(f"{p}: no such file or folder")
    if problems and not queued:
        return render(request, "upload.html", data_dir=str(config.DATA), problems=problems)
    return RedirectResponse(f"/analyses?queued={len(queued)}" + (f"&problems={len(problems)}" if problems else ""),
                            status_code=303)


@app.get("/analyses", response_class=HTMLResponse)
def analyses(request: Request, queued: int = 0, problems: int = 0):
    return render(request, "analyses.html", analyses=A.list_analyses(), queued=queued, problems=problems,
                  jobs=manager.recent(), reference=A.reference())


@app.get("/analyses/{aid}", response_class=HTMLResponse)
def analysis_detail(request: Request, aid: str):
    a = A.load(aid)
    if a is None:
        return HTMLResponse("<p>no such analysis</p>", status_code=404)
    ref = A.reference()
    return render(request, "analysis.html", a=a, rows=A.feature_rows(a, ref), reference=ref,
                  sidecar_exists=sidecar_path(a.source_path).is_file(), source_exists=Path(a.source_path).is_file())


@app.get("/analyses/{aid}/json")
def analysis_json(aid: str):
    a = A.load(aid)
    return JSONResponse(a.to_dict() if a else {"error": "no such analysis"}, status_code=200 if a else 404)


@app.get("/analyses/{aid}/files/{name}")
def analysis_file(aid: str, name: str):
    p = config.ANALYSES / aid / Path(name).name
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p))


@app.post("/analyses/{aid}/meta")
def analysis_meta(aid: str, category: str = Form(""), generator: str = Form(""), artist: str = Form(""),
                  notes: str = Form("")):
    a = A.load(aid)
    if a is not None:
        a.category, a.generator, a.artist, a.notes = (category or None), generator, artist, notes
        a.save()
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


@app.post("/analyses/{aid}/reanalyze")
def analysis_reanalyze(aid: str):
    a = A.load(aid)
    if a is not None and Path(a.source_path).is_file():
        submit_analysis(Path(a.source_path), {"category": a.category, "generator": a.generator,
                                              "artist": a.artist, "notes": a.notes}, force=True)
    return RedirectResponse("/analyses", status_code=303)


@app.post("/analyses/{aid}/sidecar")
def analysis_sidecar(aid: str):
    """Register the analysed file as a source and write its sidecar."""
    a = A.load(aid)
    if a is not None and a.category and Path(a.source_path).is_file():
        reg = Registry()
        s = reg.add(a.source_path, a.category, a.generator, a.artist, a.notes)
        write_sidecar(s, reg, profile=a.profile)
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


@app.post("/analyses/{aid}/delete")
def analysis_delete(aid: str):
    A.delete(aid)
    return RedirectResponse("/analyses", status_code=303)


@app.get("/compare", response_class=HTMLResponse)
def compare(request: Request):
    ids = request.query_params.getlist("ids")
    chosen = [a for a in (A.load(i) for i in ids) if a is not None]
    problem = None
    if len(chosen) < 2:
        problem = "pick at least two analyses"
    elif len(chosen) > 6:
        problem = "pick at most six analyses"
        chosen = chosen[:6]
    ref = A.reference()
    rows = {a.id: A.feature_rows(a, ref) for a in chosen}
    return render(request, "compare.html", analyses=chosen, rows=rows, reference=ref, problem=problem,
                  all_analyses=A.list_analyses())


@app.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request, msg: str = ""):
    reg = Registry()
    items = [{"s": s, "exists": Path(s.path).is_file(), "sidecar": sidecar_path(s.path).is_file()} for s in reg.items]
    return render(request, "sources.html", items=items, msg=msg, registry_path=str(reg.path))


@app.post("/sources/add")
def sources_add(path: str = Form(...), category: str = Form(...), generator: str = Form(""),
                artist: str = Form(""), notes: str = Form(""), sidecar: str = Form("")):
    reg = Registry()
    p = Path(path.strip()).expanduser()
    try:
        targets = scan_folder(p) if p.is_dir() else [p]
        n = 0
        for t in targets:
            s = reg.add(t, category, generator, artist, notes)
            if sidecar:
                write_sidecar(s, reg)
            n += 1
        msg = f"added {n} file(s)" + (" and wrote sidecars" if sidecar else "")
    except (ValueError, FileNotFoundError) as e:
        msg = f"error: {e}"
    return RedirectResponse(f"/sources?msg={urllib.request.quote(msg)}", status_code=303)


@app.post("/sources/action")
def sources_action(path: str = Form(...), action: str = Form(...)):
    reg = Registry()
    s = reg.find(path)
    msg = "unknown source"
    if s is not None:
        if action == "sidecar":
            try:
                sc = write_sidecar(s, reg)
                msg = f"wrote {sc.name}"
            except Exception as e:  # noqa: BLE001
                msg = f"sidecar failed: {type(e).__name__}: {e}"
        elif action == "analyze":
            submit_analysis(Path(s.path), {"category": s.category, "generator": s.generator,
                                           "artist": s.artist, "notes": s.notes})
            msg = f"queued {Path(s.path).name}"
        elif action == "remove":
            reg.remove(s.path)
            msg = f"removed {Path(s.path).name} from the registry (file and sidecar untouched)"
    return RedirectResponse(f"/sources?msg={urllib.request.quote(msg)}", status_code=303)


@app.get("/retrain", response_class=HTMLResponse)
def retrain_page(request: Request, msg: str = ""):
    return render(request, "retrain.html", rs=retrain_status(), msg=msg, retrain_url=config.RETRAIN_URL)


@app.get("/retrain/status", response_class=HTMLResponse)
def retrain_status_partial(request: Request):
    return templates.TemplateResponse(request, "partials/retrain_status.html", {"request": request, "rs": retrain_status()})


@app.get("/retrain/status.json")
def retrain_status_json():
    return JSONResponse(retrain_status())


@app.post("/retrain/start")
def retrain_start_route(trainer: str = Form("local")):
    trainer = trainer if trainer in ("local", "modal") else "local"
    r = retrain_start(trainer)
    msg = f"retrain started ({trainer})" if r["ok"] else f"not started: {r['body'].get('error', r['body'])}"
    return RedirectResponse(f"/retrain?msg={urllib.request.quote(msg)}", status_code=303)


@app.get("/jobs")
def jobs_json():
    return JSONResponse([j.to_dict() for j in manager.recent()])


@app.get("/jobs/partial", response_class=HTMLResponse)
def jobs_partial(request: Request):
    return templates.TemplateResponse(request, "partials/jobs.html", {"request": request, "jobs": manager.recent()})


@app.post("/jobs/clear")
def jobs_clear():
    manager.clear_finished()
    return RedirectResponse("/analyses", status_code=303)
