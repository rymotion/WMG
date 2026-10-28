import json

from wmg_train import retrain as r


def _m(**kw):
    base = {"gate_auc": 0.45, "song_auc": 0.94, "ai_recall": 0.77, "human_fpr": 0.05,
            "phone_auc": 0.89, "phone_ai_recall": 0.73, "phone_human_fpr": 0.12}
    return base | kw


def test_regressions_flag_only_moves_past_tolerance():
    old = _m()
    assert r.regressions(_m(song_auc=0.935, human_fpr=0.07), old) == []
    worse = r.regressions(_m(song_auc=0.92, human_fpr=0.09, phone_ai_recall=0.6), old)
    assert [w.split(" ")[0:2] for w in worse] == [["song", "auc"], ["human", "fpr"], ["phone", "ai"]]
    assert r.regressions(_m(gate_auc=0.7), old)[0].startswith("confound gate")
    assert r.regressions(_m(), None) == []  # nothing installed yet: only the gate applies


def test_collect_labels_merges_sources_clips_and_keeps_conflicts(tmp_path):
    exports, hist = tmp_path / "exp", tmp_path / "hist"
    exports.mkdir()
    (exports / "a.wmgmodel").write_text(json.dumps({"samples": [
        {"session_id": "s1", "label": 1, "category": "ai"},
        {"session_id": "s2", "label": -1},  # old bundle without category
    ]}))
    (hist / "x").mkdir(parents=True)
    (hist / "x" / "session.json").write_text(json.dumps(
        {"id": "x", "server_session_id": "s1", "label": {"category": "human"},
         "clips": [{"index": 0}, {"index": 3, "label": {"category": "ai"}}]}))
    posted = tmp_path / "p.json"
    posted.write_text(json.dumps([{"session_id": "s3", "category": "ai_assisted"}, {"session_id": "s4"},
                                  {"session_id": "s3", "clip_index": 2, "category": "human"}]))
    got = r.collect_labels(exports, hist, posted)
    assert set(got) == {"s1", "s2", "s3"}
    assert set(got["s1"][r.WHOLE]) == {"ai", "human"}  # conflict is reported, not resolved
    assert set(got["s1"][3]) == {"ai"}
    assert set(got["s2"][r.WHOLE]) == {"ai_assisted"}
    assert set(got["s3"][2]) == {"human"}


def _capture(tmp_path, sid, cats, source="app label"):
    d = tmp_path / sid
    d.mkdir(parents=True)
    with (d / "segments.csv").open("w") as fh:
        fh.write("clip_index,label,category,source_path\n")
        for i, c in enumerate(cats):
            fh.write(f"{i},{ {'ai': 1, 'human': 0, 'ai_assisted': -1}[c] },{c},{source}\n")


def test_step_labels_new_relabel_and_hand_labelled(monkeypatch, tmp_path):
    caps = tmp_path / "caps"
    monkeypatch.setattr(r, "CAPTURES", caps)
    monkeypatch.setattr(r, "STATE", tmp_path / "st")
    monkeypatch.setattr(r, "LOG", tmp_path / "st" / "log")
    _capture(caps, "same", ["human", "ai"])
    _capture(caps, "app", ["human", "human"])
    _capture(caps, "hand", ["human", "human"], source="Spotify")
    found = {
        "same": {r.WHOLE: {"human": {"a"}}, 1: {"ai": {"a"}}},
        "app": {r.WHOLE: {"human": {"a"}}, 1: {"ai": {"a"}}},   # clip 1 changed -> redo
        "hand": {r.WHOLE: {"human": {"a"}}, 1: {"ai": {"a"}}},  # never overwrite hand labels
        "fresh": {1: {"ai": {"a"}}},                            # only one clip labelled
    }
    monkeypatch.setattr(r, "collect_labels", lambda *a: found)
    calls = []

    def fake_sh(cmd, run, **k):
        calls.append(cmd)
        sid = cmd[cmd.index("--session") + 1]
        if sid == "fresh":
            _capture(caps, "fresh", ["ai"])
        return 0

    monkeypatch.setattr(r, "sh", fake_sh)
    run = type("R", (), {"s": {}})()
    args = type("A", (), {"exports": None, "app_history": None, "posted": None, "api": "x"})()
    rep = r.step_labels(run, args)
    assert [e["session_id"] for e in rep["already"]] == ["same"]
    assert [e["session_id"] for e in rep["updated"]] == ["app"]
    assert [e["session_id"] for e in rep["new"]] == ["fresh"]
    assert [c["session_id"] for c in rep["conflicts"]] == ["hand"]
    app_cmd = next(c for c in calls if "app" in c)
    assert app_cmd[-5:] == ["--label", "human", "--clip-label", "1=ai", "--force"]
    fresh_cmd = next(c for c in calls if "fresh" in c)
    assert "--label" not in fresh_cmd and "1=ai" in fresh_cmd


def _fake_pipeline(monkeypatch, tmp_path, regress: bool):
    st = tmp_path / "state"
    for name, attr in [("status.json", "STATUS"), ("retrain.log", "LOG"), ("history.jsonl", "HISTORY"),
                       ("pending.json", "PENDING"), ("lock", "LOCK")]:
        monkeypatch.setattr(r, attr, st / name)
    monkeypatch.setattr(r, "STATE", st)
    monkeypatch.setattr(r, "step_labels", lambda run, args: {"new": [], "updated": [], "already": [], "conflicts": [], "missing_audio": []})
    monkeypatch.setattr(r, "sh", lambda *a, **k: 0)
    monkeypatch.setattr(r, "train", lambda run, name: tmp_path / f"new{name}")
    cmp = {n: {"new": {"run": f"new{n}"}, "current": None, "regressions": ["song auc worse"] if regress else []}
           for n in r.MODELS}
    monkeypatch.setattr(r, "compare", lambda runs: cmp)
    installed = []
    monkeypatch.setattr(r, "install", lambda run, runs, build: installed.append(runs))
    args = type("A", (), {"exports": tmp_path, "app_history": tmp_path, "posted": None, "api": "",
                          "build_app": False, "yes": False})()
    return args, installed


def test_run_installs_when_not_worse(monkeypatch, tmp_path):
    args, installed = _fake_pipeline(monkeypatch, tmp_path, regress=False)
    r.cmd_run(args)
    s = json.loads(r.STATUS.read_text())
    assert s["state"] == "done" and installed == [{"10s": "new10s", "5s": "new5s"}]
    assert json.loads(r.HISTORY.read_text().splitlines()[-1])["state"] == "done"


def test_worse_models_wait_for_approval(monkeypatch, tmp_path):
    args, installed = _fake_pipeline(monkeypatch, tmp_path, regress=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)  # as when the app starts it
    r.cmd_run(args)
    s = json.loads(r.STATUS.read_text())
    assert s["state"] == "awaiting_approval" and s["steps"]["install"]["state"] == "waiting"
    assert installed == [] and r.PENDING.exists()
    r.cmd_decide(type("A", (), {"cmd": "approve"})())
    assert installed == [{"10s": "new10s", "5s": "new5s"}] and not r.PENDING.exists()
    assert json.loads(r.STATUS.read_text())["state"] == "done"


def test_reject_keeps_current(monkeypatch, tmp_path):
    args, installed = _fake_pipeline(monkeypatch, tmp_path, regress=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    r.cmd_run(args)
    r.cmd_decide(type("A", (), {"cmd": "reject"})())
    assert installed == [] and json.loads(r.STATUS.read_text())["state"] == "rejected"
