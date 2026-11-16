"""models.py -- Probability models. 

Each has fit(windows, y) and predict(windows), where windows is the (N, 9, 9) array from the dataset and predict returns P(mine).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inf_minesweeper.features import FEATURE_NAMES, handcrafted, local_ratios



class Constant:
    """Baseline: the same probability for every cell. With p=None it uses the mien rate of the training rows"""
    
    def __init__(self, p: float | None = None) -> None:
        self.p = p
    
    def fit(self, windows, y):
        if self.p is None:
            self.p = float(np.mean(y))
        return self
    
    def predict(self, windows) -> np.ndarray:
        return np.full(len(windows), self.p)


class LocalRatio:
    """Baseline with nothing to learn: read each touching number on its own as (mines it still needs) / 
    (unknown cells around it), and average them. This is the usual back-of-the-envelope estimate a player makes."""
    
    def fit(self, windows, y):
        return self
    
    def predict(self, windows) -> np.ndarray:
        return np.clip(np.nanmean(local_ratios(windows), axis=1), 0.01, 0.99)


class LogisticFeatures:
    """Baseline: logistic regression on the 8 handcrafted features."""

    def __init__(self, C: float = 1.0) -> None:
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))

    def fit(self, windows, y):
        self.model.fit(handcrafted(windows), y)
        return self

    def predict(self, windows) -> np.ndarray:
        return self.model.predict_proba(handcrafted(windows))[:, 1]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.model[-1].coef_[0]))
