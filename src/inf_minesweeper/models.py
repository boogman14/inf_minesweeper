"""models.py -- Probability models. 

Each has fit(windows, y) and predict(windows), where windows is the (N, 9, 9) array from the dataset and predict returns P(mine).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inf_minesweeper.features import handcrafted, local_ratios


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
        from inf_minesweeper.features import FEATURE_NAMES
        return dict(zip(FEATURE_NAMES, self.model[-1].coef_[0]))
    

def tree_features(windows: np.ndarray, k: int = 5) -> np.ndarray:
    """One-hot k x k window plus the 8 handcrafted features, as float32."""
    from inf_minesweeper.features import one_hot
    return np.hstack([one_hot(windows, k), handcrafted(windows)]).astype(np.float32)


class GradientBoosted:
    """LightGBM on the one-hot k x k window plus the handcrafted features.

    fit(windows, y, val=(windows_val, y_val)) stops adding trees once the
    validation loss has not improved for `patience` rounds. The fitted object
    can be pickled and used as a simulate.py policy via its predict method.
    """

    def __init__(self, k: int = 5, patience: int = 100, **params: Any) -> None:
        self.k = k
        self.patience = patience
        self.params: dict[str, Any] = dict(
            n_estimators=3000, learning_rate=0.05, num_leaves=63,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, verbose=-1)
        self.params.update(params)
        self.model: Any = None      # an lgb.LGBMClassifier once fitted

    def fit(self, windows, y, val=None) -> "GradientBoosted":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**self.params)
        kw: dict[str, Any] = {}
        if val is not None:
            kw["eval_set"] = [(tree_features(val[0], self.k), val[1])]
            kw["callbacks"] = [lgb.early_stopping(self.patience, verbose=False)]
        model.fit(tree_features(windows, self.k), y, **kw)
        self.model = model
        return self

    @property
    def n_trees(self) -> int:
        """Trees actually used: the early-stopping point, else all of them."""
        return int(self._fitted().best_iteration_ or self.params["n_estimators"])

    def predict(self, windows) -> np.ndarray:
        proba = self._fitted().predict_proba(tree_features(windows, self.k))
        return np.asarray(proba)[:, 1]

    def _fitted(self) -> Any:
        if self.model is None:
            raise RuntimeError("call fit() before using the model")
        return self.model