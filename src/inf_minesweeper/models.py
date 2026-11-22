"""models.py -- Probability models. 

Each has fit(windows, y) and predict(windows), where windows is the (N, 9, 9) array from the dataset and predict returns P(mine).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inf_minesweeper.features import features_, local_ratios
from inf_minesweeper.metrics import log_loss


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
        self.model.fit(features_(windows), y)
        return self

    def predict(self, windows) -> np.ndarray:
        return self.model.predict_proba(features_(windows))[:, 1]

    def coefficients(self) -> dict[str, float]:
        from inf_minesweeper.features import FEATURE_NAMES
        return dict(zip(FEATURE_NAMES, self.model[-1].coef_[0]))
    

def tree_features(windows: np.ndarray, k: int = 5) -> np.ndarray:
    """One-hot k x k window plus the 8 handcrafted features, as float32."""
    from inf_minesweeper.features import one_hot
    oh = one_hot(windows, k)
    X = np.empty((len(windows), oh.shape[1] + 8), dtype=np.float32)
    X[:, :oh.shape[1]] = oh
    del oh
    X[:, oh_cols(k):] = features_(windows)
    return X
    
    
def oh_cols(k: int) -> int:
    """Number of one-hot columns for a k x k window (centre excluded)."""
    from inf_minesweeper.board import N_STATES
    return (k * k - 1) * N_STATES


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
            kw["eval_X"] = tree_features(val[0], self.k)
            kw["eval_y"] = val[1]
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
    

class ConvNeuralNet:
    """Convolutional network on the k x k window, k = 2 * layers + 1.

    Each 3x3 convolution has no padding, so every layer trims one cell from
    each side: a k x k window shrinks to a single output, the candidate cell.
    The network therefore sees exactly the k x k window, which makes `layers`
    the window-size knob (1 = 3x3, 2 = 5x5, 3 = 7x7, 4 = 9x9), the same
    question the LightGBM window sizes ask.

    Training: Adam on log-loss, a random rotation or reflection of every batch
    (the rules are the same in all 8 orientations), and early stopping on the
    validation loss, keeping the best epoch. Windows are one-hot encoded on
    the device batch by batch, so memory stays at one byte per cell.

    The fitted object pickles to the CPU and can be used as a simulate.py
    policy; call .to("cuda") after loading to predict on the GPU.
    """

    def __init__(self, layers: int = 4, width: int = 64, epochs: int = 30,
                 batch: int = 2048, lr: float = 2e-3, weight_decay: float = 1e-4,
                 patience: int = 3, augment: bool = True, device: str | None = None,
                 seed: int = 0) -> None:
        self.layers, self.width = layers, width
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.patience, self.augment = weight_decay, patience, augment
        self.seed = seed
        self.symmetrize = False   # default for predict(); True averages 8 orientations
        self.device = device or _default_device()
        self.net: Any = None
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0

    @property
    def k(self) -> int:
        return 2 * self.layers + 1

    # ---------------------------------------------------------------- fitting
    def fit(self, windows, y, val=None) -> "ConvNeuralNet":
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        self.net = _build_net(self.layers, self.width).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=0)

        Xt = self._windows_tensor(windows)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        best, best_state, waited = np.inf, None, 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            order = torch.randperm(len(Xt), generator=gen).to(self.device)
            total = 0.0
            for i in range(0, len(Xt), self.batch):
                idx = order[i:i + self.batch]
                xb = Xt[idx]
                if self.augment:
                    xb = _random_symmetry(xb, gen)
                logits = self.net(_one_hot(xb)).flatten()
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            row = {"epoch": epoch, "train_loss": total / len(Xt)}
            if val is not None:
                row["val_loss"] = log_loss(val[1], self.predict(val[0]))
                sched.step(row["val_loss"])
                if row["val_loss"] < best - 1e-5:
                    best, waited, self.best_epoch = row["val_loss"], 0, epoch
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    waited += 1
            self.history.append(row)
            if val is not None and waited >= self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        else:
            self.best_epoch = len(self.history)
        return self

    # ------------------------------------------------------------- prediction
    def predict(self, windows, symmetrize: bool | None = None) -> np.ndarray:
        """P(mine) for each window. With symmetrize=True, average over all 8
        rotations and reflections (8 times slower, usually better). If not
        given, the model's `symmetrize` attribute decides, so setting it
        before pickling makes a simulate.py policy use it too."""
        if symmetrize is None:
            symmetrize = self.symmetrize
        import torch

        net = self._fitted()
        net.eval()
        Xt = self._windows_tensor(windows)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 8 * self.batch):
                xb = Xt[i:i + 8 * self.batch]
                if symmetrize:
                    p = torch.stack([torch.sigmoid(net(_one_hot(s)).flatten())
                                     for s in _all_symmetries(xb)]).mean(0)
                else:
                    p = torch.sigmoid(net(_one_hot(xb)).flatten())
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def to(self, device: str) -> "ConvNeuralNet":
        self.device = device
        if self.net is not None:
            self.net.to(device)
        return self

    # ---------------------------------------------------------------- helpers
    def _windows_tensor(self, windows):
        import torch
        from inf_minesweeper.loader import crop
        w = np.ascontiguousarray(crop(np.asarray(windows, dtype=np.uint8), self.k))
        return torch.as_tensor(w, device=self.device)

    def _fitted(self) -> Any:
        if self.net is None:
            raise RuntimeError("call fit() before using the model")
        return self.net

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.net is not None:
            state["net"] = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        state["device"] = "cpu"
        return state

    def __setstate__(self, state: dict) -> None:
        import multiprocessing
        import torch
        weights = state.pop("net")
        state.setdefault("symmetrize", False)   # models saved before this option
        self.__dict__.update(state)
        self.net = None
        if weights is not None:
            self.net = _build_net(self.layers, self.width)
            self.net.load_state_dict(weights)
        if multiprocessing.current_process().name != "MainProcess":
            torch.set_num_threads(1)   # many simulate.py workers share the CPU


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_net(layers: int, width: int):
    import torch.nn as nn
    from inf_minesweeper.board import N_STATES
    mods: list = []
    c = N_STATES
    for _ in range(layers):
        mods += [nn.Conv2d(c, width, 3), nn.BatchNorm2d(width), nn.ReLU()]
        c = width
    mods += [nn.Conv2d(c, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1)]
    return nn.Sequential(*mods)


def _one_hot(xb):
    import torch.nn.functional as F
    from inf_minesweeper.board import N_STATES
    return F.one_hot(xb.long(), N_STATES).permute(0, 3, 1, 2).float()


def _random_symmetry(xb, gen):
    import torch
    r = int(torch.randint(4, (1,), generator=gen))
    xb = torch.rot90(xb, r, dims=(1, 2))
    if int(torch.randint(2, (1,), generator=gen)):
        xb = xb.flip(2)
    return xb


def _all_symmetries(xb):
    import torch
    for flip in (False, True):
        x = xb.flip(2) if flip else xb
        for r in range(4):
            yield torch.rot90(x, r, dims=(1, 2))
