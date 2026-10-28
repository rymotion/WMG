"""Codec/loudness confound baseline.

If bitrate, original sample rate and loudness alone predict AI vs human, a detector can learn
the delivery chain instead of the music. Fit logistic regression on those features on the
train split and require test AUC <= max_auc (0.55).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConfoundError(ValueError):
    pass


@dataclass(frozen=True)
class ConfoundResult:
    auc: float
    max_auc: float
    n_train: int
    n_test: int
    coefficients: dict[str, float]

    @property
    def passed(self) -> bool:
        return self.auc <= self.max_auc


def confound_auc(df: pd.DataFrame, cfg: dict) -> ConfoundResult:
    feats = list(cfg["features"])
    data = df[~df["hybrid"].astype(bool)]
    train, test = data[data["split"] == "train"], data[data["split"] == "test"]
    for name, part in (("train", train), ("test", test)):
        if part["label"].nunique() < 2:
            raise ConfoundError(f"{name} split needs both classes for the confound baseline")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[feats].to_numpy(np.float64), train["label"].to_numpy())
    scores = model.predict_proba(test[feats].to_numpy(np.float64))[:, 1]
    auc = float(roc_auc_score(test["label"].to_numpy(), scores))
    coef = model[-1].coef_[0]
    return ConfoundResult(auc, float(cfg["max_auc"]), len(train), len(test),
                          {f: float(c) for f, c in zip(feats, coef)})


def check_confound(df: pd.DataFrame, cfg: dict) -> ConfoundResult:
    result = confound_auc(df, cfg)
    if not result.passed:
        raise ConfoundError(
            f"codec/loudness baseline test AUC {result.auc:.3f} > {result.max_auc:.2f}: the "
            f"delivery chain predicts the label (standardised coefs {result.coefficients}). "
            "Re-encode/loudness-match sources or rebalance codecs before training.")
    return result
