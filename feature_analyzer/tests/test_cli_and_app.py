from pathlib import Path

from fastapi.testclient import TestClient

from analyzer import analysis as A
from analyzer import cli


def test_cli_analyze_smoke(data_dir, tmp_path, wav_wall, models_ok):
    out = tmp_path / "report"
    argv = ["analyze", str(wav_wall), "--out", str(out), "--category", "ai", "--generator", "synthetic"]
    if not models_ok:
        argv.append("--no-models")
    assert cli.main(argv) == 0
    md = (out / "summary.md").read_text()
    assert md.startswith("# Analyzer report") and "noise_wall16k.wav" in md and "| fc_hz |" in md
    assert (out / "noise_wall16k.wav" / "analysis.json").is_file()
    assert (out / "noise_wall16k.wav" / "spectrum.png").is_file()


def test_cli_sources(data_dir, wav_full, capsys):
    assert cli.main(["sources", "add", str(wav_full), "--category", "human", "--artist", "noise band"]) == 0
    assert cli.main(["sources", "list"]) == 0
    assert "noise_fullband.wav" in capsys.readouterr().out


def test_app_pages(data_dir, wav_wall, wav_full, models_ok):
    from analyzer.app import app
    a = A.analyze(wav_wall, meta={"category": "ai"}, models=models_ok)
    b = A.analyze(wav_full, meta={"category": "human"}, models=models_ok)
    c = TestClient(app)
    assert c.get("/analyses").status_code == 200
    r = c.get(f"/analyses/{a.id}")
    assert r.status_code == 200 and "fc_hz" in r.text and "spectrum.png" in r.text
    r = c.get(f"/compare?ids={a.id}&ids={b.id}")
    assert r.status_code == 200 and "noise_wall16k.wav" in r.text and "noise_fullband.wav" in r.text
    assert "pick at least two" in c.get(f"/compare?ids={a.id}").text
    r = c.post(f"/analyses/{a.id}/meta", data={"category": "ai_assisted", "generator": "g", "artist": "", "notes": ""}, follow_redirects=False)
    assert r.status_code == 303 and A.load(a.id).category == "ai_assisted"
    assert c.get("/sources").status_code == 200
    # Retrain page renders whether or not the service is up.
    r = c.get("/retrain")
    assert r.status_code == 200 and ("Retrain Service.command" in r.text or "installed runs" in r.text)
    js = c.get("/retrain/status.json").json()
    assert "reachable" in js
    # Upload a file through the form; the job runs in the background worker.
    with open(wav_full, "rb") as f:
        r = c.post("/upload", files={"files": ("noise_fullband.wav", f, "audio/wav")},
                   data={"category": "human"}, follow_redirects=False)
    assert r.status_code == 303 and "queued=1" in r.headers["location"]
    assert c.get("/jobs").json()[0]["label"] == "analyze noise_fullband.wav"
