"""features.py -- Turn stored 9x9 windows into model inputs.

A window holds player-visible cell states (see board.py): 0-8 revealed number, 9 hidden, 10 flagged, 
11 mine hit, 12 wrong flag. The centre cell is the candidate being scored and is always hidden.
"""


from __future__ import annotations

import numpy as np

from inf_minesweeper.board import FLAGGED, HIDDEN, MINE_HIT, N_STATES
from inf_minesweeper.loader import crop

FEATURE_NAMES = [
    "n_numbers",            # revealed numbers touching the cell
    "n_unknown",            # hidden neighbors
    "n_known_mines",        # flagged or exploded neighbors
    "sum_numbers",          # sum of adjacent numbers
    "ratio_mean",           # local mine ratio, averaged over adjacent numbers
    "ratio_max",
    "ratio_min",
    "ratio_union",          # 1 - prod(1 - ratio): "any number says mine"
]

_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _neighbor_sum(mask: np.ndarray) -> np.ndarray:
    """For every cell, how many of its 8 neighbors are set in `mask` (cells outside the window count as not set)."""
    padded = np.pad(mask.astype(np.int16), ((0, 0), (1, 1), (1, 1)))
    h, w = mask.shape[1:]
    out = np.zeros(mask.shape, dtype=np.int16)
    for dy, dx in _OFFSETS:
        out += padded[:, (1 + dy):(1 + dy + h), (1 + dx):(1 + dx + w)]
    return out


def local_ratios(windows: np.ndarray) -> np.ndarray:
    """(N, 8) array: for each of the centre's 8 neighbors that is a revealed number, the share of its unknown 
    neighbors that must be mines, (number - known mines around it) / (unknown cells around it). NaN where the 
    neighbor is not a number. This is the estimate a player makes from one number at a time. It needs the 5x5 
    around the centre, so any window of side 5 or more works."""
    w = crop(windows, 5)
    number = w <= 8
    unknown = w == HIDDEN
    known_mine = (w == FLAGGED) | (w == MINE_HIT)
    need = np.where(number, w.astype(np.int16) - _neighbor_sum(known_mine), 0)
    room = _neighbor_sum(unknown)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(number & (room > 0), need / room, np.nan)
    return np.stack([ratio[:, 2 + dy, 2 + dx] for dy, dx in _OFFSETS], axis=1)


def handcrafted(windows: np.ndarray) -> np.ndarray:
    """(N, 8) float array of simple features; columns are FEATURE_NAMES"""
    w3 = crop(windows, 3).reshape(len(windows), 9)
    w3 = np.delete(w3, 4, axis=1)
    number = w3 <= 8
    ratios = np.clip(local_ratios(windows), 0.0, 1.0)
    has = ~np.isnan(ratios)
    filled = np.where(has, ratios, 0.0)
    count = np.maximum(has.sum(1), 1)
    return np.column_stack([
        number.sum(1),
        (w3 == HIDDEN).sum(1),
        ((w3 == FLAGGED) | (w3 == MINE_HIT)).sum(1),
        np.where(number, w3, 0).sum(1),
        filled.sum(1) / count,
        np.where(has, ratios, -np.inf).max(1).clip(0.0),
        np.where(has, ratios, np.inf).min(1).clip(None, 1.0),
        1.0 - np.prod(1.0 - filled, axis=1),
    ]).astype(np.float64)
    
    
def one_hot(windows: np.ndarray, k: int = 5) -> np.ndarray:
    """(N, (k*k - 1) * 13) flat one-hot encoding of the k x k window, without the centre cell (it is always hidden). 
    For linear models and trees."""
    w = crop(windows, k).reshape(len(windows), k * k)
    w = np.delete(w, (k *k) // 2, axis=1)
    return np.eye(N_STATES, dtype=np.uint8)[w].reshape(len(windows), -1)


def channels(windows: np.ndarray, k: int = 9) -> np.ndarray:
    """(N, 13, k, k) float32 one-hot encoding, one channel per state. For convolutional networks."""
    w = crop(windows, k)
    return np.moveaxis(np.eye(N_STATES, dtype=np.float32)[w], -1, 1)


def symmetries(windows: np.ndarray) -> np.ndarray:
    """(8, N, k, k): the windows under the 8 rotations and reflections of the square. The rules of the game are unchanged 
    by these, so they multiply the training data by 8 (repeat the labels 8 times to match)."""
    out = []
    for flip in (False, True):
        w = windows[..., ::-1] if flip else windows
        for r in range(4):
            out.append(np.rot90(w, r, axes=(-2, -1)))
    return np.stack(out)
