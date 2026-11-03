"""wmg_train.cloud against a fake Modal Volume / Function: nothing here talks to Modal."""

import io
from pathlib import Path

import pytest

from wmg_train import cloud, retrain


class FakeBatch:
    def __init__(self, vol):
        self.vol = vol

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def put_directory(self, local, remote):
        for p in Path(local).iterdir():
            self.vol.files[f"{remote}/{p.name}"] = p.read_bytes()

    def put_file(self, src, remote):
        self.vol.files[remote] = src.read() if isinstance(src, io.IOBase) else Path(src).read_bytes()


class FakeVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.uploads = 0

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def batch_upload(self, force=False):
        self.uploads += 1
        return FakeBatch(self)

    def listdir(self, path):
        class E:
            def __init__(self, p):
                self.path = p
        return [E(p) for p in self.files if p.startswith(path + "/")]


class FakeCall:
    def __init__(self, vol, result, log_lines):
        self.vol, self.result, self.polls, self.cancelled = vol, result, 0, False
        self.log_lines, self.tag = log_lines, None

    def get(self, timeout=0):
        self.polls += 1
        # The remote log grows by one line per poll; the call finishes after the log is complete.
        self.vol.files[f"/logs/{self.tag}.log"] = "".join(self.log_lines[: self.polls]).encode()
        if self.polls <= len(self.log_lines):
            raise TimeoutError
        return self.result

    def cancel(self):
        self.cancelled = True


class FakeFunction:
    def __init__(self, call):
        self.call, self.args = call, None

    def spawn(self, *args):
        self.args = args
        self.call.tag = args[2]
        return self.call


def _cache(root: Path, name: str, size: int = 10) -> Path:
    d = root / name
    d.mkdir()
    (d / "meta.json").write_text('{"n": 1}')
    (d / "index.parquet").write_bytes(b"idx")
    (d / "windows.f16").write_bytes(b"\0" * size)
    return d


def test_sync_uploads_only_changed_caches(tmp_path):
    _cache(tmp_path, "a")
    _cache(tmp_path, "b")
    vol = FakeVolume()
    lines = []
    assert cloud.sync_caches(lines.append, caches=("a", "b"), root=tmp_path, vol=vol) == ["a", "b"]
    assert vol.uploads == 2 and "/a/windows.f16" in vol.files and "/b/.fingerprint" in vol.files
    # Unchanged: nothing goes up.
    assert cloud.sync_caches(lines.append, caches=("a", "b"), root=tmp_path, vol=vol) == []
    assert vol.uploads == 2
    # A rebuilt cache (new index) goes up again; the other does not.
    (tmp_path / "b" / "index.parquet").write_bytes(b"idx2")
    assert cloud.sync_caches(lines.append, caches=("a", "b"), root=tmp_path, vol=vol) == ["b"]
    assert vol.uploads == 3


def test_sync_refuses_a_folder_that_is_not_a_cache(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(cloud.CloudError):
        cloud.sync_caches(lambda _: None, caches=("x",), root=tmp_path, vol=FakeVolume())


def test_train_remote_follows_the_log_and_downloads_the_run(tmp_path):
    vol = FakeVolume()
    vol.files["/runs/20260101-000000/model.pt"] = b"pt"
    vol.files["/runs/20260101-000000/metrics.json"] = b"{}"
    call = FakeCall(vol, {"run": "20260101-000000", "rc": 0, "device": "cuda:L4"}, ["epoch 1\n", "epoch 2\n"])
    fn = FakeFunction(call)
    lines = []
    out = cloud.train_remote("10s", ["--x"], lines.append, runs=tmp_path, vol=vol, fn=fn, poll_s=0)
    assert fn.args[:2] == ("10s", ["--x"]) and fn.args[2].startswith("10s-")
    assert out == tmp_path / "20260101-000000" and (out / "model.pt").read_bytes() == b"pt"
    assert "epoch 1" in lines and "epoch 2" in lines and not call.cancelled


def test_train_remote_cancels_the_call_when_interrupted(tmp_path):
    vol = FakeVolume()
    call = FakeCall(vol, {"run": None, "rc": 0}, ["a\n"] * 100)
    fn = FakeFunction(call)

    def interrupt(_):
        if call.polls == 3:
            raise retrain.Cancelled

    with pytest.raises(retrain.Cancelled):
        cloud.train_remote("5s", [], interrupt, runs=tmp_path, vol=vol, fn=fn, poll_s=0)
    assert call.cancelled


def test_train_remote_reports_a_failed_run(tmp_path):
    vol = FakeVolume()
    call = FakeCall(vol, {"run": None, "rc": 1}, [])
    with pytest.raises(cloud.CloudError, match="exit 1"):
        cloud.train_remote("10s", [], lambda _: None, runs=tmp_path, vol=vol, fn=FakeFunction(call), poll_s=0)


def test_retrain_dispatches_to_modal(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cloud, "sync_caches", lambda line, **kw: seen.setdefault("synced", True))
    monkeypatch.setattr(cloud, "train_remote", lambda name, args, line: seen.setdefault("train", (name, args)) and tmp_path)
    monkeypatch.setattr(retrain, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(retrain, "LOG", tmp_path / "retrain.log")
    monkeypatch.setattr(retrain, "STATE", tmp_path)
    run = retrain.Run(build_app=False, trainer="modal")
    assert retrain.train(run, "5s") == tmp_path
    assert seen == {"synced": True, "train": ("5s", retrain.MODELS["5s"])}
    assert run.s["trainer"] == "modal" and run.s["steps"]["train_5s"]["detail"] == "training on Modal"
