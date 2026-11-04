"""wmg_train.baseten_cloud against a fake Baseten API: nothing here talks to Baseten."""

import json
from pathlib import Path

import pytest

from wmg_train import baseten_cloud as bc
from wmg_train import retrain
from wmg_train.cloud import CloudError


class FakeApi:
    """A job that logs one line per poll, then ends in `final` with `checkpoints` ({rel path: bytes})."""

    def __init__(self, log_lines, final="TRAINING_JOB_COMPLETED", checkpoints=None):
        self.log_lines, self.final, self.checkpoints = log_lines, final, checkpoints or {}
        self.polls, self.stopped, self.pushed = 0, False, None
        self.remote = "fake"

    def push(self, project, source_dir):
        self.pushed = project
        return "proj", "job"

    def status(self, project_id, job_id):
        self.polls += 1
        if self.polls <= len(self.log_lines):
            return "TRAINING_JOB_RUNNING", None
        return self.final, None

    def logs(self, project_id, job_id, since_ms):
        # The whole log so far on every poll: LogTail must not repeat lines.
        return [{"timestamp": str((i + 1) * 1_000_000_000), "message": m}
                for i, m in enumerate(self.log_lines[: self.polls])]

    def stop(self, project_id, job_id):
        self.stopped = True

    def checkpoint_files(self, project_id, job_id):
        return [{"relative_file_name": k, "url": k} for k in self.checkpoints]

    def fetch(self, url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.checkpoints[url])


def _cache(root: Path, name: str, size: int = 10) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text('{"n": 1}')
    (d / "index.parquet").write_bytes(b"idx")
    (d / "windows.f16").write_bytes(b"\0" * size)
    return d


def _project(name, extra_args, tag, fingerprints, ship):
    return {"name": name, "args": extra_args, "tag": tag, "fps": fingerprints, "ship": ship}


RUN = {"rank-0/20260101-000000/model.pt": b"pt", "rank-0/20260101-000000/metrics.json": b"{}",
       "rank-0/20260101-000000/coreml/x.bin": b"x", "rank-0/other.txt": b"?"}


def _train(tmp_path, api, fps, **kw):
    return bc.train_remote("10s", ["--x"], kw.pop("line", lambda _: None), runs=tmp_path / "runs", api=api,
                           make_project=_project, fingerprints=fps,
                           seeded=kw.pop("seeded", bc.Seeded("fake/p", tmp_path / "seeded.json")),
                           poll_s=0, wait_s=0, **kw)


def test_train_remote_follows_the_log_and_downloads_the_run(tmp_path):
    api = FakeApi(["epoch 1", "epoch 2"], checkpoints=RUN)
    lines = []
    out = _train(tmp_path, api, {"cache-v3": "f1"}, line=lines.append)
    assert api.pushed["args"] == ["--x"] and api.pushed["tag"].startswith("10s-")
    assert api.pushed["fps"] == {"cache-v3": "f1"}
    assert out == tmp_path / "runs" / "20260101-000000" and (out / "model.pt").read_bytes() == b"pt"
    assert (out / "coreml" / "x.bin").exists() and not (tmp_path / "runs" / "other.txt").exists()
    assert lines.count("epoch 1") == 1 and lines.count("epoch 2") == 1 and not api.stopped


def test_caches_ship_once_per_fingerprint(tmp_path):
    seeded = bc.Seeded("fake/p", tmp_path / "seeded.json")
    api = FakeApi([], checkpoints=RUN)
    _train(tmp_path, api, {"cache-v3": "a", "cache-phone": "b"}, seeded=seeded)
    assert api.pushed["ship"] == ["cache-v3", "cache-phone"]
    _train(tmp_path, api, {"cache-v3": "a", "cache-phone": "b"}, seeded=seeded)
    assert api.pushed["ship"] == []
    _train(tmp_path, api, {"cache-v3": "a", "cache-phone": "b2"}, seeded=seeded)  # phone cache rebuilt
    assert api.pushed["ship"] == ["cache-phone"]
    _train(tmp_path, api, {"cache-v3": "a", "cache-phone": "b2"}, seeded=seeded, reseed=True)
    assert api.pushed["ship"] == ["cache-v3", "cache-phone"]
    # Another project (or account) starts from nothing.
    assert bc.Seeded("other/p", tmp_path / "seeded.json").to_ship({"cache-v3": "a"}) == ["cache-v3"]


def test_a_failed_job_does_not_count_as_shipped(tmp_path):
    seeded = bc.Seeded("fake/p", tmp_path / "seeded.json")
    with pytest.raises(CloudError, match="remote training failed"):
        _train(tmp_path, FakeApi([], final="TRAINING_JOB_FAILED"), {"cache-v3": "a"}, seeded=seeded)
    assert seeded.to_ship({"cache-v3": "a"}) == ["cache-v3"]


def test_train_remote_stops_the_job_when_interrupted(tmp_path):
    api = FakeApi(["a"] * 100)

    def interrupt(text):
        if api.polls == 3:
            raise retrain.Cancelled

    with pytest.raises(retrain.Cancelled):
        _train(tmp_path, api, {}, line=interrupt)
    assert api.stopped


def test_a_completed_job_without_a_model_fails(tmp_path):
    with pytest.raises(CloudError, match="model.pt"):
        _train(tmp_path, FakeApi([], checkpoints={"rank-0/log.txt": b""}), {})


def test_cli_requires_a_cache(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(CloudError):
        bc.local_fingerprints(tmp_path, ("x",))


def test_log_tail_handles_overlapping_polls_and_shared_timestamps():
    lines = []
    tail = bc.LogTail(lines.append)
    tail.feed([{"timestamp": "5", "message": "a"}, {"timestamp": "5", "message": "b"}])
    tail.feed([{"timestamp": "5", "message": "a"}, {"timestamp": "5", "message": "b"},
               {"timestamp": "7", "message": "c\n"}])
    tail.feed([{"timestamp": "7", "message": "c"}])
    assert lines == ["a", "b", "c"]


def _job_env(tmp_path, fps):
    return {"WMG_TAG": "10s-x", "WMG_MODEL": "10s", "WMG_TRAIN_ARGS": "--window-seconds 5",
            "WMG_FP_CACHE_V3": fps["cache-v3"], "WMG_FP_CACHE_PHONE": fps["cache-phone"],
            "BT_PROJECT_CACHE_DIR": str(tmp_path / "pc"), "BT_CHECKPOINT_DIR": str(tmp_path / "ck")}


def _trainer(tmp_path, seen):
    def train(cmd):
        seen.append(cmd)
        run = tmp_path / "ck" / f"run{len(seen)}"
        run.mkdir()
        (run / "model.pt").write_bytes(b"pt")
        return 0
    return train


def test_job_places_shipped_caches_then_reuses_them(tmp_path):
    work = tmp_path / "work"
    fps = {n: bc.fingerprint(_cache(work, n)) for n in ("cache-v3", "cache-phone")}
    (tmp_path / "pc" / "caches" / "cache-v3" / "stale").mkdir(parents=True)
    seen = []
    assert bc.run_job(_job_env(tmp_path, fps), _trainer(tmp_path, seen), work) == 0
    cmd = seen[0]
    assert cmd[cmd.index("--cache") + 1] == str(tmp_path / "pc/caches/cache-v3" / fps["cache-v3"])
    assert cmd[cmd.index("--phone-cache") + 1] == str(tmp_path / "pc/caches/cache-phone" / fps["cache-phone"])
    assert cmd[-2:] == ["--window-seconds", "5"]
    assert not (tmp_path / "pc/caches/cache-v3/stale").exists()
    assert json.loads((tmp_path / "ck/run1/baseten.json").read_text())["tag"] == "10s-x"
    # Next job ships nothing: the project cache is used as is.
    assert bc.run_job(_job_env(tmp_path, fps), _trainer(tmp_path, seen), tmp_path / "empty") == 0
    assert (tmp_path / "pc/caches/cache-v3" / fps["cache-v3"] / "windows.f16").exists()


def test_job_without_the_cache_asks_for_a_reseed(tmp_path):
    with pytest.raises(CloudError, match="--reseed"):
        bc.run_job(_job_env(tmp_path, {"cache-v3": "a", "cache-phone": "b"}), lambda cmd: 0, tmp_path)


def test_job_refuses_a_shipped_cache_with_the_wrong_fingerprint(tmp_path):
    _cache(tmp_path / "work", "cache-v3")
    with pytest.raises(CloudError, match="fingerprint"):
        bc.run_job(_job_env(tmp_path, {"cache-v3": "a", "cache-phone": "b"}), lambda cmd: 0, tmp_path / "work")


def test_job_without_a_run_reports_failure(tmp_path):
    work = tmp_path / "work"
    fps = {n: bc.fingerprint(_cache(work, n)) for n in ("cache-v3", "cache-phone")}
    assert bc.run_job(_job_env(tmp_path, fps), lambda cmd: 0, work) == 1


def test_retrain_dispatches_to_baseten(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(bc, "train_remote", lambda name, args, line: seen.setdefault("train", (name, args)) and tmp_path)
    monkeypatch.setattr(retrain, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(retrain, "LOG", tmp_path / "retrain.log")
    monkeypatch.setattr(retrain, "STATE", tmp_path)
    run = retrain.Run(build_app=False, trainer="baseten")
    assert retrain.train(run, "10s") == tmp_path
    assert seen == {"train": ("10s", retrain.MODELS["10s"])}
    assert run.s["steps"]["train_10s"]["detail"] == "training on Baseten"


def test_job_definition_packages_the_training_code(tmp_path):
    pytest.importorskip("truss_train")
    from truss_train import deployment

    project = bc._make_project("5s", ["--window-seconds", "5"], "5s-x", {"cache-v3": "a", "cache-phone": "b"})
    env = project.job.runtime.environment_variables
    assert env["WMG_TRAIN_ARGS"] == "--window-seconds 5"
    assert env["WMG_FP_CACHE_V3"] == "a" and env["WMG_FP_CACHE_PHONE"] == "b"
    assert all(isinstance(v, str) for v in env.values())  # no secrets needed
    assert project.job.compute.accelerator.accelerator == "H100"
    gathered = deployment._gather_training_dir(bc.JOB_DIR, project.job.workspace)
    names = {p.name for p in gathered.iterdir()}
    assert {"run.sh", "config.py", "wmg_train", "configs"} <= names and "cache-v3" not in names
    assert (gathered / "wmg_train" / "train_baseline.py").exists()
    assert (gathered / "configs" / "student.yaml").exists()


def test_job_definition_ships_requested_caches():
    pytest.importorskip("truss_train")
    project = bc._make_project("10s", [], "t", {}, ship=["cache-phone"])
    assert "../cache-phone" in project.job.workspace.external_dirs
    assert "../cache-v3" not in project.job.workspace.external_dirs


def test_cpu_only_job(monkeypatch):
    pytest.importorskip("truss_train")
    monkeypatch.setenv("WMG_BASETEN_GPU", "none")
    project = bc._make_project("10s", [], "t", {})
    assert project.job.compute.accelerator is None
