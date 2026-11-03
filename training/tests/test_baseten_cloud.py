"""wmg_train.baseten_cloud against a fake S3 client and a fake Baseten API: nothing here talks to
Baseten or a bucket."""

import io
import json
from pathlib import Path

import pytest

from wmg_train import baseten_cloud as bc
from wmg_train import retrain
from wmg_train.cloud import CloudError


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads = 0

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 2]  # tiny pages, so pagination is exercised
        more = start + 2 < len(keys)
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + 2)} if more else {})}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, path, bucket, key):
        self.uploads += 1
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


class FakeApi:
    """A job that logs one line per poll, then ends in `final`; `on_done` plays the job's uploads."""

    def __init__(self, log_lines, final="TRAINING_JOB_COMPLETED", on_done=None):
        self.log_lines, self.final, self.on_done = log_lines, final, on_done
        self.polls, self.stopped, self.pushed = 0, False, None

    def push(self, project, source_dir):
        self.pushed = project
        return "proj", "job"

    def status(self, project_id, job_id):
        self.polls += 1
        if self.polls <= len(self.log_lines):
            return "TRAINING_JOB_RUNNING", None
        if self.on_done:
            self.on_done(self.pushed)
            self.on_done = None
        return self.final, None

    def logs(self, project_id, job_id, since_ms):
        # The whole log so far on every poll: LogTail must not repeat lines.
        return [{"timestamp": str((i + 1) * 1_000_000_000), "message": m}
                for i, m in enumerate(self.log_lines[: self.polls])]

    def stop(self, project_id, job_id):
        self.stopped = True


def _cache(root: Path, name: str, size: int = 10) -> Path:
    d = root / name
    d.mkdir()
    (d / "meta.json").write_text('{"n": 1}')
    (d / "index.parquet").write_bytes(b"idx")
    (d / "windows.f16").write_bytes(b"\0" * size)
    return d


def _project(name, extra_args, tag, fingerprints):
    return {"name": name, "args": extra_args, "tag": tag, "fps": fingerprints}


def test_sync_uploads_only_changed_caches(tmp_path):
    _cache(tmp_path, "a")
    _cache(tmp_path, "b")
    s3 = FakeS3()
    bucket = bc.Bucket("wmg", client=s3)
    assert bc.sync_caches(lambda _: None, caches=("a", "b"), root=tmp_path, bucket=bucket) == ["a", "b"]
    fp_b = bc.fingerprint(tmp_path / "b")
    assert f"caches/b/{fp_b}/windows.f16" in s3.objects and f"caches/b/{fp_b}/.complete" in s3.objects
    n = s3.uploads
    assert bc.sync_caches(lambda _: None, caches=("a", "b"), root=tmp_path, bucket=bucket) == []
    assert s3.uploads == n
    # A rebuilt cache goes up under its new fingerprint; the old copy stays (it's immutable).
    (tmp_path / "b" / "index.parquet").write_bytes(b"idx2")
    assert bc.sync_caches(lambda _: None, caches=("a", "b"), root=tmp_path, bucket=bucket) == ["b"]
    assert f"caches/b/{bc.fingerprint(tmp_path / 'b')}/.complete" in s3.objects


def test_a_half_uploaded_cache_is_uploaded_again(tmp_path):
    _cache(tmp_path, "a")
    s3 = FakeS3()
    s3.objects[f"caches/a/{bc.fingerprint(tmp_path / 'a')}/windows.f16"] = b"partial"  # no .complete
    assert bc.sync_caches(lambda _: None, caches=("a",), root=tmp_path, bucket=bc.Bucket("wmg", client=s3)) == ["a"]


def test_sync_refuses_a_folder_that_is_not_a_cache(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(CloudError):
        bc.sync_caches(lambda _: None, caches=("x",), root=tmp_path, bucket=bc.Bucket("wmg", client=FakeS3()))


def test_bucket_name_is_required(monkeypatch):
    monkeypatch.delenv("WMG_BUCKET", raising=False)
    with pytest.raises(CloudError, match="WMG_BUCKET"):
        bc.Bucket(client=FakeS3())


def test_train_remote_follows_the_log_and_downloads_the_run(tmp_path):
    s3 = FakeS3()

    def job_uploads(project):
        s3.objects["runs/20260101-000000/model.pt"] = b"pt"
        s3.objects["runs/20260101-000000/metrics.json"] = b"{}"
        s3.objects["runs/20260101-000000/coreml/x.bin"] = b"x"
        s3.objects[f"jobs/{project['tag']}.json"] = json.dumps(
            {"run": "20260101-000000", "rc": 0, "device": "cuda:H100"}).encode()

    api = FakeApi(["epoch 1", "epoch 2"], on_done=job_uploads)
    lines = []
    out = bc.train_remote("10s", ["--x"], lines.append, runs=tmp_path, bucket=bc.Bucket("wmg", client=s3),
                          api=api, make_project=_project, fingerprints={"cache-v3": "f1"}, poll_s=0)
    assert api.pushed["args"] == ["--x"] and api.pushed["tag"].startswith("10s-")
    assert api.pushed["fps"] == {"cache-v3": "f1"}
    assert out == tmp_path / "20260101-000000" and (out / "model.pt").read_bytes() == b"pt"
    assert (out / "coreml" / "x.bin").exists()
    assert lines.count("epoch 1") == 1 and lines.count("epoch 2") == 1 and not api.stopped


def test_train_remote_stops_the_job_when_interrupted(tmp_path):
    api = FakeApi(["a"] * 100)

    def interrupt(text):
        if api.polls == 3:
            raise retrain.Cancelled

    with pytest.raises(retrain.Cancelled):
        bc.train_remote("5s", [], interrupt, runs=tmp_path, bucket=bc.Bucket("wmg", client=FakeS3()), api=api,
                        make_project=_project, fingerprints={}, poll_s=0)
    assert api.stopped


@pytest.mark.parametrize("final, result", [
    ("TRAINING_JOB_FAILED", None),
    ("TRAINING_JOB_COMPLETED", {"run": None, "rc": 1}),
])
def test_train_remote_reports_a_failed_run(tmp_path, final, result):
    s3 = FakeS3()

    def job_uploads(project):
        if result:
            s3.objects[f"jobs/{project['tag']}.json"] = json.dumps(result).encode()

    api = FakeApi([], final=final, on_done=job_uploads)
    with pytest.raises(CloudError, match="remote training failed"):
        bc.train_remote("10s", [], lambda _: None, runs=tmp_path, bucket=bc.Bucket("wmg", client=s3), api=api,
                        make_project=_project, fingerprints={}, poll_s=0)


def test_log_tail_handles_overlapping_polls_and_shared_timestamps():
    lines = []
    tail = bc.LogTail(lines.append)
    tail.feed([{"timestamp": "5", "message": "a"}, {"timestamp": "5", "message": "b"}])
    tail.feed([{"timestamp": "5", "message": "a"}, {"timestamp": "5", "message": "b"},
               {"timestamp": "7", "message": "c\n"}])
    tail.feed([{"timestamp": "7", "message": "c"}])
    assert lines == ["a", "b", "c"]


def test_job_fetches_caches_once_trains_and_uploads(tmp_path):
    s3 = FakeS3()
    bucket = bc.Bucket("wmg", client=s3)
    for name, fp in (("cache-v3", "v3fp"), ("cache-phone", "phfp")):
        s3.objects[f"caches/{name}/{fp}/windows.f16"] = b"w"
        s3.objects[f"caches/{name}/{fp}/meta.json"] = b"{}"
        s3.objects[f"caches/{name}/{fp}/.complete"] = fp.encode()
    env = {"WMG_TAG": "10s-x", "WMG_MODEL": "10s", "WMG_TRAIN_ARGS": "--window-seconds 5",
           "WMG_FP_CACHE_V3": "v3fp", "WMG_FP_CACHE_PHONE": "phfp",
           "BT_PROJECT_CACHE_DIR": str(tmp_path / "pc"), "BT_CHECKPOINT_DIR": str(tmp_path / "ck")}
    (tmp_path / "pc" / "caches" / "cache-v3" / "stale").mkdir(parents=True)
    seen = []

    def train(cmd):
        seen.append(cmd)
        run = tmp_path / "ck" / "20260101-000000"
        run.mkdir()
        (run / "model.pt").write_bytes(b"pt")
        return 0

    assert bc.run_job(env, bucket, train) == 0
    cmd = seen[0]
    assert cmd[cmd.index("--cache") + 1] == str(tmp_path / "pc/caches/cache-v3/v3fp")
    assert cmd[cmd.index("--phone-cache") + 1] == str(tmp_path / "pc/caches/cache-phone/phfp")
    assert cmd[-2:] == ["--window-seconds", "5"]
    assert not (tmp_path / "pc/caches/cache-v3/stale").exists()
    assert s3.objects["runs/20260101-000000/model.pt"] == b"pt"
    assert json.loads(s3.objects["jobs/10s-x.json"])["run"] == "20260101-000000"
    # Second job on the same fingerprints: nothing is downloaded again.
    s3.objects["caches/cache-v3/v3fp/windows.f16"] = b"changed"
    (tmp_path / "ck" / "20260101-000000").rename(tmp_path / "ck" / "old")
    bc.run_job(env, bucket, train)
    assert (tmp_path / "pc/caches/cache-v3/v3fp/windows.f16").read_bytes() == b"w"


def test_job_without_a_run_reports_failure(tmp_path):
    s3 = FakeS3()
    for name in ("cache-v3", "cache-phone"):
        s3.objects[f"caches/{name}/fp/.complete"] = b"fp"
    env = {"WMG_TAG": "t", "WMG_FP_CACHE_V3": "fp", "WMG_FP_CACHE_PHONE": "fp",
           "BT_PROJECT_CACHE_DIR": str(tmp_path / "pc"), "BT_CHECKPOINT_DIR": str(tmp_path / "ck")}
    assert bc.run_job(env, bc.Bucket("wmg", client=s3), lambda cmd: 0) == 1
    assert json.loads(s3.objects["jobs/t.json"]) ["run"] is None


def test_retrain_dispatches_to_baseten(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(bc, "sync_caches", lambda line: seen.setdefault("synced", True))
    monkeypatch.setattr(bc, "train_remote", lambda name, args, line: seen.setdefault("train", (name, args)) and tmp_path)
    monkeypatch.setattr(retrain, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(retrain, "LOG", tmp_path / "retrain.log")
    monkeypatch.setattr(retrain, "STATE", tmp_path)
    run = retrain.Run(build_app=False, trainer="baseten")
    assert retrain.train(run, "10s") == tmp_path
    assert seen == {"synced": True, "train": ("10s", retrain.MODELS["10s"])}
    assert run.s["steps"]["train_10s"]["detail"] == "training on Baseten"


def test_job_definition_packages_the_training_code(monkeypatch):
    pytest.importorskip("truss_train")
    from truss_train import deployment

    monkeypatch.setenv("WMG_BUCKET", "wmg")
    project = bc._make_project("5s", ["--window-seconds", "5"], "5s-x", {"cache-v3": "a", "cache-phone": "b"})
    env = project.job.runtime.environment_variables
    assert env["WMG_TRAIN_ARGS"] == "--window-seconds 5" and env["WMG_BUCKET"] == "wmg"
    assert env["WMG_FP_CACHE_V3"] == "a" and env["WMG_FP_CACHE_PHONE"] == "b"
    assert env["AWS_ACCESS_KEY_ID"].name == "wmg_s3_access_key_id"
    assert project.job.compute.accelerator.accelerator == "H100"
    gathered = deployment._gather_training_dir(bc.JOB_DIR, project.job.workspace)
    names = {p.name for p in gathered.iterdir()}
    assert {"run.sh", "config.py", "wmg_train", "configs"} <= names
    assert (gathered / "wmg_train" / "train_baseline.py").exists()
    assert (gathered / "configs" / "student.yaml").exists()


def test_cpu_only_job(monkeypatch):
    pytest.importorskip("truss_train")
    monkeypatch.setenv("WMG_BASETEN_GPU", "none")
    project = bc._make_project("10s", [], "t", {})
    assert project.job.compute.accelerator is None
