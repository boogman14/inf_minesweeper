import numpy as np

from inf_minesweeper.board import HIDDEN, Board
from inf_minesweeper.loader import crop, load, split_by_game
from inf_minesweeper.simulate import Config, play_game, run

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
            assert 0.0 < q < 1.0            # certain cells are never logged
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
    np.savez_compressed(path, **data)       #type:ignore
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
