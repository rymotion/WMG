"""Smoke test for the wmg_features Python module. Run after installing the wheel."""

import json

import numpy as np
import wmg_features as wf


def sine(f, amp, sr, secs):
    return amp * np.sin(2 * np.pi * f * np.arange(int(sr * secs)) / sr)


def main():
    spec = json.loads(wf.SPEC_JSON)
    assert spec["sample_rate"] == wf.SAMPLE_RATE == 32000
    assert wf.N_MELS == 128 and wf.FRAMES_PER_WINDOW == 1000
    assert len(wf.SPEC_HASH) == 16

    mel = wf.DetectorMel()
    m = mel.logmel(sine(1000, 1.0, 32000, 2))
    assert m.shape == (128, 200) and m.dtype == np.float32
    assert abs(m[:, :190].max()) < 0.1, m.max()

    x = np.concatenate([sine(440, 0.5, 44100, 10), sine(440, 0.001, 44100, 10)])
    windows = mel.extract_windows(x, 44100)
    assert [w[0] for w in windows] == [0.0, 5.0], [w[0] for w in windows]
    assert all(w[2].shape == (128, 1000) for w in windows)

    assert len(wf.resample(sine(440, 0.5, 16000, 1), 16000, 32000)) == 32000
    assert abs(wf.rms_dbfs(sine(440, 1.0, 32000, 1)) + 3.01) < 0.01

    print(f"ok spec={wf.SPEC_HASH} commit={wf.COMMIT[:12]}")


if __name__ == "__main__":
    main()
