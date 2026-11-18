"""screenshots.py -- Read cleared sectors from screenshots of the Infinite Minesweeper app.

Sector boundaries are the long pink lines. Each sector between them is cut
into 8x8 cells, and each cell is classified from the pixels near its centre:
a flag is about half white with pink in it, a revealed cell is plain light
gray. Sectors with any other kind of cell (unfinished, dark, covered by an
icon) are skipped, and a sector seen in more than one screenshot is counted
once. In a cleared sector every flag is a mine, so the flags give its exact
mine layout.

Needs Pillow.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SECTOR = 8


def _peaks(profile: np.ndarray, threshold: float = 0.45) -> list[float]:
    """Centres of runs where `profile` exceeds `threshold` (the pink lines)."""
    runs: list[list[int]] = []
    for i in np.where(profile > threshold)[0].tolist():
        if runs and i - runs[-1][-1] <= 2:
            runs[-1].append(i)
        else:
            runs.append([i])
    return [sum(r) / len(r) for r in runs]


def _cell_stats(patch: np.ndarray) -> tuple[float, float, float]:
    """(share of white pixels, share of pink pixels, mean brightness)."""
    px = patch.reshape(-1, 3)
    white = (px.min(1) > 235).mean()
    pink = ((px[:, 0] > 200) & (px[:, 0] - px[:, 1] > 50)).mean()
    return white, pink, px.mean()


def sector_grids(path: str) -> list[list[list[tuple[float, float, float]]]]:
    """Per-cell pixel statistics for every whole sector in one screenshot."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    pink = (r > 200) & (g < 170) & (b < 200) & (r - g > 60)
    xs, ys = _peaks(pink.mean(0)), _peaks(pink.mean(1))
    grids = []
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            cw, ch = (x1 - x0) / SECTOR, (y1 - y0) / SECTOR
            grid = []
            for cy in range(SECTOR):
                row = []
                for cx in range(SECTOR):
                    a, b_ = int(x0 + (cx + 0.2) * cw), int(x0 + (cx + 0.8) * cw)
                    c, d = int(y0 + (cy + 0.2) * ch), int(y0 + (cy + 0.8) * ch)
                    row.append(_cell_stats(im[c:d, a:b_]))
                grid.append(row)
            grids.append(grid)
    return grids


def cleared_patterns(paths: list[str]) -> list[tuple[int, ...]]:
    """Mine layouts (64 zeros and ones, row by row) of the distinct cleared
    sectors found in the screenshots."""
    seen: dict[tuple[int, ...], None] = {}
    for path in paths:
        for grid in sector_grids(path):
            pattern, ok = [], True
            for row in grid:
                for white, pink, lum in row:
                    if 0.25 <= white <= 0.75 and pink > 0.1:
                        pattern.append(1)        # flag = mine
                    elif white < 0.05 and pink < 0.03 and 165 <= lum <= 220:
                        pattern.append(0)        # revealed
                    else:
                        ok = False
            if ok:
                seen.setdefault(tuple(pattern), None)
    return list(seen)


def density_summary(paths: list[str]) -> dict[str, float]:
    """Density estimate and mines-per-sector statistics."""
    counts = np.array([sum(p) for p in cleared_patterns(paths)])
    n = len(counts)
    p = counts.sum() / (SECTOR * SECTOR * n)
    se = (p * (1 - p) / (SECTOR * SECTOR * n)) ** 0.5
    return dict(
        sectors=n,
        mines=int(counts.sum()),
        density=p,
        low=p - 1.96 * se,
        high=p + 1.96 * se,
        min=int(counts.min()),
        max=int(counts.max()),
        mean=counts.mean(),
        variance=counts.var(ddof=1),
        binomial_variance=SECTOR * SECTOR * p * (1 - p),
    )
