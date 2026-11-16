import numpy as np

from inf_minesweeper.board import FLAGGED, HIDDEN
from inf_minesweeper.features import FEATURE_NAMES, channels, handcrafted, local_ratios, one_hot, symmetries
from inf_minesweeper.metrics import ece, excess_log_loss, gap_closed, guess_quality, log_loss
from inf_minesweeper.models import Constant, LocalRatio, LogisticFeatures


def blank():
    return np.full((1, 9, 9), HIDDEN, dtype=np.uint8)


def test_local_ratio_by_hand():
    w = blank()
    # A revealed 1 directly above the centre, everything else hidden:
    # it has 8 unknown neighbors, so each is a mine with ratio 1/8.
    w[0, 3, 4] = 1
    r = local_ratios(w)[0]
    assert np.nansum(r) == 1 / 8 and np.isnan(r).sum() == 7
    # Flag one of its other neighbors: the 1 is used up, ratio drops to 0.
    w[0, 2, 4] = FLAGGED
    assert np.nansum(local_ratios(w)[0]) == 0.0
    f = dict(zip(FEATURE_NAMES, handcrafted(w)[0]))
    assert f["n_numbers"] == 1 and f["n_unknown"] == 7 and f["sum_numbers"] == 1


def test_encodings_have_the_right_shape():
    rng = np.random.default_rng(0)
    w = rng.integers(0, 13, size=(20, 9, 9), dtype=np.uint8)
    w[:, 4, 4] = HIDDEN
    assert one_hot(w, 5).shape == (20, 24 * 13)
    assert (one_hot(w, 3).sum(1) == 8).all()
    c = channels(w, 7)
    assert c.shape == (20, 13, 7, 7) and (c.sum(1) == 1).all()


def test_symmetries_keep_the_features():
    rng = np.random.default_rng(1)
    w = rng.integers(0, 9, size=(30, 9, 9), dtype=np.uint8)
    w[rng.random(w.shape) < 0.5] = HIDDEN
    w[:, 4, 4] = HIDDEN
    s = symmetries(w)
    assert s.shape == (8, 30, 9, 9)
    assert len({a.tobytes() for a in s}) == 8
    base = handcrafted(w)
    for a in s:
        assert np.allclose(handcrafted(np.ascontiguousarray(a)), base)


def test_metrics_by_hand():
    y = np.array([1, 0, 1, 0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(log_loss(y, q), np.log(2))
    assert excess_log_loss(q, q) == 0.0
    assert excess_log_loss(q, np.full(4, 0.9)) > 0
    assert np.isclose(gap_closed(0.02, 0.10), 0.8)
    assert ece(y, q) == 0.0
    # two positions; the model picks the worse cell in the second
    g = guess_quality(p=[0.1, 0.9, 0.8, 0.2], q=[0.2, 0.6, 0.3, 0.5],
                      game=[0, 0, 0, 0], position=[0, 0, 1, 1])
    assert np.isclose(g["regret"], (0 + 0.2) / 2) and g["agreement"] == 0.5


def test_models_run():
    rng = np.random.default_rng(2)
    w = rng.integers(0, 9, size=(200, 9, 9), dtype=np.uint8)
    w[rng.random(w.shape) < 0.5] = HIDDEN
    w[:, 4, 4] = HIDDEN
    w[:, 3, 4] = 1                      # every cell touches a number
    y = rng.integers(0, 2, size=200)
    for model in (Constant(), Constant(0.21), LocalRatio(), LogisticFeatures()):
        p = model.fit(w, y).predict(w)
        assert p.shape == (200,) and ((p > 0) & (p < 1)).all()