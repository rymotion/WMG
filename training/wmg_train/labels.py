"""Recording categories shared by the capture tools, matching the app's `AiCategory`.

    human        human-made                  AI share  0-10%   binary label 0
    ai_assisted  human-made with AI parts    AI share 10-50%   not used for binary training yet
    ai           AI-generated                AI share 50-100%  binary label 1

AI-assisted recordings are kept (segments, audio, labels) for a future AI-share model but
left out of the current Human-vs-AI training. Files written before this category existed have
only `label` (0/1); `category_of` derives it from that.
"""

from __future__ import annotations

CATEGORIES = {
    "human": {"label": 0, "ai_share": (0.0, 0.1)},
    "ai_assisted": {"label": -1, "ai_share": (0.1, 0.6)},
    "ai": {"label": 1, "ai_share": (0.4, 1.0)},
}
BINARY = {"human", "ai"}


def label_of(category: str) -> int:
    """Binary target: 0 human, 1 AI, -1 = not used for binary training."""
    return CATEGORIES[category]["label"]


def category_of(row) -> str:
    """Category of a segments.csv row / dict, falling back to the binary label for older files."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    c = get("category", None)
    if isinstance(c, str) and c in CATEGORIES:
        return c
    return "ai" if int(get("label")) == 1 else "human"


def columns(category: str) -> dict:
    lo, hi = CATEGORIES[category]["ai_share"]
    return {"label": label_of(category), "category": category, "ai_share_min": lo, "ai_share_max": hi}
