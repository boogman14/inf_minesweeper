import numpy as np
import pickle
from dataclasses import replace

from inf_minesweeper.board import HIDDEN, Board
from inf_minesweeper.loader import crop, load, split_by_game
from inf_minesweeper.simulate import Config, play_game, run
from inf_minesweeper.models import GradientBoosted, LocalRatio

SMALL = Config(radius=2)


def test_rows_are_consistent_with_the_board():
    for seed in range(8):
        rows, summary = play_game(seed, SMALL)
        truth = Board(seed, density=SMALL.density)
        truth.reveal(*SMALL.start)          # applies the first-click rule
        for w, q, mine, x, y in zip(rows["window"], rows["q"], rows["mine"],
                                    rows["x"], rows["y"]):
            assert w.shape == (9, 9)
            assert w[4, 4] == HIDDEN        # the cell itself is unknown
            assert q != q or 0.0 < q < 1.0  # certain cells are never logged; NaN = no q
            assert mine == truth.is_mine(x, y)
        # exactly one cell is guessed at each position
        for p in set(rows["position"]):
            picked = [c for c, pos in zip(rows["chosen"], rows["position"]) if pos == p]
            assert sum(picked) <= 1
        assert summary["cleared"] + summary["locked"] <= summary["sectors"] == 9


def test_games_replay_exactly():
    a, sa = play_game(3, SMALL)
    b, sb = play_game(3, SMALL)
    assert sa == sb
    assert a["q"] == b["q"] and a["x"] == b["x"] and a["chosen"] == b["chosen"]


def test_run_save_load_and_split(tmp_path):
    data = run(games=6, cfg=SMALL, workers=2, progress=False)
    path = tmp_path / "d.npz"
    np.savez_compressed(path, allow_pickle=False, **data)
    d = load(str(path))
    assert d["config"]["radius"] == 2
    n = len(d["q"])
    assert d["window"].shape == (n, 9, 9) and len(d["game"]) == n
    assert len(d["game_seed"]) == 6

    train, test = split_by_game(d["game"], test_frac=0.34)
    assert train.sum() + test.sum() == n
    assert not set(d["game"][train]) & set(d["game"][test])

    assert crop(d["window"], 5).shape == (n, 5, 5)
    assert (crop(d["window"], 5)[:, 2, 2] == HIDDEN).all()


def test_budget_rows_are_kept_with_nan_q(tmp_path):
    cfg = replace(SMALL, max_nodes=50)          # tiny budget: many NaN rows
    data = run(games=4, cfg=cfg, workers=2, progress=False)
    nan = np.isnan(data["q"])
    assert nan.any() and (~nan).any()
    path = tmp_path / "d.npz"
    np.savez_compressed(path, allow_pickle=False, **data)
    assert not np.isnan(load(str(path))["q"]).any()          # default: dropped
    full = load(str(path), unscored=True)
    assert len(full["q"]) == len(data["q"]) and len(full["window"]) == len(data["q"])


class FirstCandidate:
    """Policy that always prefers the first candidate (lowest score)."""
    def predict(self, windows):
        return np.arange(len(windows), dtype=float)


def test_policy_picks_its_lowest_cell():
    cfg = replace(SMALL, epsilon=0.0)
    rows, summary = play_game(5, cfg, FirstCandidate().predict)
    assert summary["random_guesses"] == 0
    first = {}
    for pos, chosen in zip(rows["position"], rows["chosen"]):
        first.setdefault(pos, chosen)
    assert all(first.values())                 # the chosen row is each position's first


def test_trained_model_plays_in_parallel():
    data = run(games=6, cfg=SMALL, workers=2, progress=False)
    model = GradientBoosted(k=3, n_estimators=20).fit(data["window"], data["mine"])
    model = pickle.loads(pickle.dumps(model))  # what --policy loads
    played = run(games=4, first_seed=1000, cfg=replace(SMALL, epsilon=0.0),
                 workers=2, progress=False, policy=model.predict)
    assert played["game_random_guesses"].sum() == 0
    assert played["game_pick_q_n"].sum() > 0
    local = run(games=2, first_seed=1000, cfg=replace(SMALL, epsilon=0.0),
                workers=2, progress=False, policy=LocalRatio().predict)
    assert len(local["q"]) > 0
