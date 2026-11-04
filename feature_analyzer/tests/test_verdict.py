from analyzer import analysis as A


def _a(p10, p5):
    return A.Analysis(id="x", name="x", source_path="x", created_at="t", scores={
        "10s": {"p_ai": 0.5, "windows": [{"p_ai": p} for p in p10]},
        "5s": {"p_ai": 0.5, "windows": [{"p_ai": p} for p in p5]},
    })


def test_mostly_red_is_ai():
    v = A.verdict([0.9, 0.95, 0.8, 0.5, 0.1])
    assert v["key"] == "ai" and (v["red"], v["yellow"], v["green"]) == (3, 1, 1)


def test_mostly_yellow_is_likely_ai_at_75_80():
    v = A.verdict([0.5, 0.6, 0.45, 0.65, 0.2, 0.9])
    assert v["key"] == "likely_ai" and v["estimate"] == "75–80%" and v["confidence"] == "medium"
    # A red + yellow majority with neither alone past half still reads as likely AI.
    assert A.verdict([0.9, 0.9, 0.5, 0.5, 0.1, 0.1, 0.1])["key"] == "likely_ai"


def test_mostly_green_is_human_and_the_rest_undetermined():
    assert A.verdict([0.05, 0.1, 0.2, 0.3, 0.5])["key"] == "human"
    assert A.verdict([0.1, 0.1, 0.5, 0.9])["key"] == "undetermined"  # half green, half red/yellow
    assert A.verdict([])["key"] == "none"


def test_analysis_pools_both_models_windows():
    v = _a([0.5, 0.55], [0.5, 0.6, 0.2]).verdict()
    assert v["n"] == 5 and v["key"] == "likely_ai" and "both models" in v["reason"]
    assert A.Analysis(id="y", name="y", source_path="y", created_at="t").verdict()["key"] == "none"
    assert "likely AI-generated (75–80%)" in A.summary_markdown([_a([0.5], [0.5])], ref={})
