import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inf_minesweeper.board import HIDDEN
from inf_minesweeper.models import ConvNeuralNet


def random_windows(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 9, size=(n, 9, 9), dtype=np.uint8)
    w[rng.random(w.shape) < 0.5] = HIDDEN
    w[:, 4, 4] = HIDDEN
    return w


def test_sees_exactly_its_window():
    """With 2 layers the model sees the 5x5 window: changing a cell outside it
    must not change the prediction, changing one inside should."""
    w = random_windows(64)
    y = (w[:, 3, 4] >= 2).astype(np.float32)        # depends on a neighbor
    model = ConvNeuralNet(layers=2, width=8, epochs=1, device="cpu").fit(w, y)
    base = model.predict(w)
    outside = w.copy()
    outside[:, 0, :] = 8                             # row 0 is outside the 5x5
    assert np.allclose(model.predict(outside), base)
    inside = w.copy()
    inside[:, 3, 4] = 0
    assert not np.allclose(model.predict(inside), base)


def test_learns_a_simple_rule_and_stops_early():
    w = random_windows(4000, seed=1)
    # a rule that, like Minesweeper's, is the same in every orientation
    ring = (w[:, 3:6, 3:6] == HIDDEN).sum(axis=(1, 2)) - 1
    y = (ring >= 5).astype(np.float32)
    vw, vy = w[3000:], y[3000:]
    model = ConvNeuralNet(layers=1, width=16, epochs=8, batch=256, device="cpu").fit(
        w[:3000], y[:3000], val=(vw, vy))
    p = model.predict(vw)
    assert ((p > 0.5) == (vy > 0.5)).mean() > 0.95
    assert 1 <= model.best_epoch <= len(model.history)


def test_pickle_round_trip_and_symmetrize():
    w = random_windows(50)
    model = ConvNeuralNet(layers=1, width=8, epochs=1, device="cpu").fit(w, np.zeros(50))
    again = pickle.loads(pickle.dumps(model))
    assert np.allclose(again.predict(w), model.predict(w), atol=1e-6)
    p = again.predict(w, symmetrize=True)
    assert p.shape == (50,) and ((p > 0) & (p < 1)).all()
    again.symmetrize = True                      # becomes the default
    assert np.allclose(pickle.loads(pickle.dumps(again)).predict(w), p, atol=1e-6)