"""metrics.py -- Scores for probability models. 

`y` is the true label (1 = mine), `p` the model's probability, `q` the oracle's exact probability.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=np.float64), _clip(p)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def excess_log_loss(q, p) -> float:
    """Expected log-loss above the oracle's: the mean KL divergence from the oracle's probability to the model's. 
    Zero only for a perfect model. It uses q directly, so it is far less noisy than log_loss(y, p) minus log_loss(y, q)."""
    q, p = _clip(q), _clip(p)
    return float((q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))).mean())


def gap_closed(model: float, baseline: float, oracle: float = 0.0) -> float:
    """Share of the baseline-to-oracle gap a model closes: 0 = no better than the baseline, 1 = as good as the oracle. 
    Works with any loss; with excess_log_loss the oracle's value is 0."""
    return (baseline - model) / (baseline - oracle)


def reliability(y, p, bins: int = 10):
    """Per bin (mean predicted, observed mine rate, count), skipping empty bins."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]
    pred, obs, n = map(np.array, zip(*rows))
    return pred, obs, n


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: the average gap between predicted and observed mine rate, weighted by bin size."""
    pred, obs, n = reliability(y, p, bins)
    return float((np.abs(pred - obs) * n).sum() / n.sum())


def guess_quality(p, q, game, position) -> dict[str, float]:
    """How good the model's choices would be. At each forced guess the model picks the candidate with its lowest p.

    regret      mean of (oracle q of the model's pick) - (best q available):
                the extra chance of hitting a mine per guess
    agreement   share of positions where the model's pick is as safe as the
                oracle's best
    hit_rate    mean oracle q of the model's pick: its chance of a mine
    best_rate   the same for the oracle's own pick
    """
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    _, group = np.unique(np.stack([game, position]), axis=1, return_inverse=True)
    group = group.ravel()
    order = np.lexsort((p, group))             # by position, then by p
    first = np.r_[True, group[order][1:] != group[order][:-1]]
    picked = q[order][first]                   # oracle q of the model's pick
    best = np.full(group.max() + 1, np.inf)
    np.minimum.at(best, group, q)
    return dict(
        regret=float((picked - best).mean()),
        agreement=float((picked <= best + 1e-9).mean()),
        hit_rate=float(picked.mean()),
        best_rate=float(best.mean()),
    )


def evaluate(y, p, q, game, position, baseline_p=None) -> dict[str, float]:
    """All the headline numbers for one model."""
    out = dict(
        log_loss=log_loss(y, p),
        brier=brier(y, p),
        excess=excess_log_loss(q, p),
        ece=ece(y, p),
    )
    if baseline_p is not None:
        out["gap_closed"] = gap_closed(out["excess"], excess_log_loss(q, baseline_p))
    out.update(guess_quality(p, q, game, position))
    return out
