import pytest
import torch

from wmg_train.config import load_config
from wmg_train.device import pick_device
from wmg_train.models.student import (
    OUTPUT_NAMES, BandNet, band_bin_ranges, count_params, export_onnx, mel_band_centers_hz,
)


@pytest.fixture(scope="module")
def model() -> BandNet:
    torch.manual_seed(0)
    return BandNet.from_config(load_config("student")).eval()


def test_band_bins_follow_mel_scale():
    ranges = band_bin_ranges()
    centers = mel_band_centers_hz()
    assert ranges["bass"][0] == 0 and ranges["treble"][1] == 128
    assert ranges["bass"][1] == ranges["mid"][0] and ranges["mid"][1] == ranges["treble"][0]
    assert centers[ranges["bass"][1] - 1] < 250 <= centers[ranges["mid"][0]]
    assert centers[ranges["mid"][1] - 1] < 4000 <= centers[ranges["treble"][0]]


def test_pick_device(monkeypatch):
    monkeypatch.delenv("WMG_DEVICE", raising=False)
    assert pick_device().type in {"cuda", "mps", "cpu"}
    assert pick_device("cpu").type == "cpu"
    monkeypatch.setenv("WMG_DEVICE", "cpu")
    assert pick_device().type == "cpu"


def test_output_shapes(model):
    dev = pick_device()
    m = BandNet.from_config(load_config("student")).eval().to(dev)
    m.load_state_dict(model.state_dict())
    x = (torch.randn(2, 1, 128, 1000) * 20 - 50).to(dev)
    with torch.no_grad():
        out = {k: v.cpu() for k, v in m(x).items()}
    t = out["frame_logits"].shape[-1]
    assert t == 125
    assert out["p_ai"].shape == (2,)
    assert out["band_scores"].shape == (2, 3)
    assert out["stem_scores"].shape == (2, 4)
    assert out["frame_logits"].shape == (2, 3, t)
    assert out["attention"].shape == (2, t)
    assert torch.allclose(out["attention"].sum(-1), torch.ones(2), atol=1e-5)
    for k in ("p_ai", "band_scores", "stem_scores"):
        assert ((out[k] >= 0) & (out[k] <= 1)).all()


def test_param_budget(model):
    assert count_params(model) <= load_config("student")["model"]["max_params"]


def test_only_mobile_friendly_layers(model):
    allowed = (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Hardswish,
               torch.nn.AvgPool2d, torch.nn.Linear, torch.nn.Dropout)
    leaves = [m for m in model.modules() if not list(m.children())]
    assert all(isinstance(m, allowed) for m in leaves), {type(m) for m in leaves}


def test_onnx_export(model, tmp_path):
    onnx = pytest.importorskip("onnx")
    path = export_onnx(model, tmp_path / "bandnet.onnx")
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    assert [o.name for o in proto.graph.output] == list(OUTPUT_NAMES)
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"Loop", "If", "Scan"}, ops

    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(path))
    x = torch.randn(3, 1, 128, 1000) * 20 - 50
    got = sess.run(None, {"logmel": x.numpy()})
    with torch.no_grad():
        want = model(x)
    for name, arr in zip(OUTPUT_NAMES, got):
        assert torch.allclose(torch.from_numpy(arr), want[name], atol=1e-4), name
