"""loader.py -- Load the simulated dataset and prepare it for modeling."""

from __future__ import annotations

import json

import numpy as np


def load(path: str, unscored: bool = False) -> dict:
    """Read a file written by simulate.py.

    Per-row arrays (one row per candidate cell at a forced guess):
      window    (N, 9, 9) uint8  visible cell states around the cell
      q         (N,)             exact mine probability from the oracle
      mine      (N,)             1 if the cell really is a mine
      game      (N,)             seed of the game the row came from
      position  (N,)             index of the forced guess within its game
      chosen    (N,)             True for the cell that was actually guessed
      x, y      (N,)             board coordinates of the cell
    Per-game arrays start with `game_` (score, cleared, locked, ...).
    `config` is the simulation settings as a dict.
    """
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["config"] = json.loads(str(data["config"]))
    if not unscored:
        keep = ~np.isnan(data["q"])
        if not keep.all():
            n = len(keep)
            for k, v in data.items():
                if isinstance(v, np.ndarray) and not k.startswith("game_") and len(v) == n:
                    data[k] = v[keep]
    return data


def split_by_game(game: np.ndarray, test_frac: float = 0.2, seed: int = 0):
    """Boolean (train, test) row masks with every game entirely on one side.

    Rows from the same game are correlated, so splitting by row would leak
    information from train into test.
    """
    games = np.unique(game)
    rng = np.random.default_rng(seed)
    test_games = rng.choice(games, size=max(1, round(test_frac * len(games))), replace=False)
    test = np.isin(game, test_games)
    return ~test, test


def crop(windows: np.ndarray, k: int) -> np.ndarray:
    """Centre k x k crop of the stored windows (k odd, at most 9)."""
    full = windows.shape[-1]
    if k % 2 == 0 or not 1 <= k <= full:
        raise ValueError(f"k must be odd and between 1 and {full}")
    a = (full - k) // 2
    return windows[..., a:a + k, a:a + k]
