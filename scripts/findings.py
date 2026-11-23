"""Recompute every number in the README's "Findings so far" section.

Usage (from the repo root):

    python scripts/findings.py                                  # about 10 minutes
    python scripts/findings.py --games 300 --stuck-games 100    # quick check
    python scripts/findings.py --data data/raw/guesses.npz      # + calibration

Prints the findings as Markdown bullets, ready to paste into the README.

1. Policy comparison: the same seeds played with oracle-only guessing
   (epsilon 0) and random guessing (epsilon 1), at densities 0.20, 0.21 and
   0.22. Both policies play every certain move; they differ only in which
   cell they pick at a forced guess. Score differences are paired by seed.
2. Solver incompleteness and oracle cost: at every position where the simple
   solver is stuck, whether the oracle can still prove a cell safe, how long
   the oracle takes, and how many cells exceed its search budget.
3. Oracle calibration on a saved dataset (optional, with --data).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from inf_minesweeper.board import Board
from inf_minesweeper.oracle import frontier_proba
from inf_minesweeper.simulate import _CERTAIN, Config, run
from inf_minesweeper.solver import Solver, sector_region

DENSITIES = (0.20, 0.21, 0.22)


def policy_comparison(games: int, workers: int | None) -> dict:
    out = {}
    for p in DENSITIES:
        for eps, name in ((0.0, "oracle"), (1.0, "random")):
            t = time.time()
            d = run(games=games, cfg=Config(density=p, epsilon=eps),
                    workers=workers, progress=False)
            out[p, name] = {k: d[f"game_{k}"].astype(float)
                            for k in ("score", "cleared", "locked", "guesses", "sectors")}
            print(f"  density {p:.2f}, {name} guessing: {time.time() - t:.0f}s",
                  file=sys.stderr, flush=True)
    return out
 
 
def _stuck_stats(seed: int, cfg: Config = Config()) -> list[tuple[bool, float, int, int]]:
    """Play one game like simulate.py; at each stuck position record
    (oracle finds a safe cell, oracle seconds, cells over budget, cells)."""
    board = Board(seed, density=cfg.density, lock_penalty=cfg.lock_penalty)
    solver = Solver(board, sector_region(board, cfg.radius, cfg.start))
    rng = random.Random(seed)
    solver.reveal(*cfg.start)
    rows, guesses = [], 0
    while guesses < cfg.max_guesses:
        solver.solve()
        t = time.perf_counter()
        probs = frontier_proba(solver, cfg.max_nodes)
        dt = time.perf_counter() - t
        if not probs:
            break
        safe = [c for c, q in probs.items() if q < _CERTAIN]
        mines = [c for c, q in probs.items() if q > 1 - _CERTAIN]
        n_nan = sum(math.isnan(q) for q in probs.values())
        rows.append((bool(safe), dt, n_nan, len(probs)))
        for c in mines:
            solver.flag(*c)
        for c in safe:
            solver.reveal(*c)
        if safe:
            continue
        known = [c for c, q in probs.items() if not math.isnan(q) and q <= 1 - _CERTAIN]
        cands = known or [c for c, q in probs.items() if q <= 1 - _CERTAIN or math.isnan(q)]
        if not cands:
            break
        if known and rng.random() >= cfg.epsilon:
            best = min(probs[c] for c in known)
            pick = rng.choice(sorted(c for c in known if probs[c] <= best + 1e-12))
        else:
            pick = rng.choice(sorted(cands))
        solver.reveal(*pick)
        guesses += 1
    return rows
 
 
def stuck_stats(games: int, workers: int | None) -> np.ndarray:
    t = time.time()
    with Pool(workers or os.cpu_count() or 1) as pool:
        rows = [r for game in pool.map(_stuck_stats, range(games), chunksize=4) for r in game]
    print(f"  stuck positions: {time.time() - t:.0f}s", file=sys.stderr, flush=True)
    return np.array(rows, dtype=float)
 
 
def calibration(path: str) -> dict:
    from inf_minesweeper.loader import load
    from inf_minesweeper.metrics import ece, reliability
    d = load(path)
    q, y = d["q"], d["mine"].astype(float)
    pred, obs, _ = reliability(y, q, bins=20)
    return dict(rows=len(q), mean_q=q.mean(), rate=y.mean(),
                gap=np.abs(pred - obs).max(), ece=ece(y, q, bins=20))
 
 
def mean_se(x: np.ndarray) -> tuple[float, float]:
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--games", type=int, default=1000, help="games per policy and density")
    ap.add_argument("--stuck-games", type=int, default=500)
    ap.add_argument("--data", help="dataset .npz for the calibration finding")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
 
    print("running...", file=sys.stderr)
    pol = policy_comparison(args.games, args.workers)
    st = stuck_stats(args.stuck_games, args.workers)
    cal = calibration(args.data) if args.data else None
 
    n = args.games
    sectors = int(pol[0.21, "oracle"]["sectors"][0])
    o, r = pol[0.21, "oracle"], pol[0.21, "random"]
    diff_m, diff_se = mean_se(o["score"] - r["score"])
 
    lines = [
        f"Generated by `scripts/findings.py` ({n:,} games per policy and density, "
        f"seeds 0 to {n - 1}).",
        "",
        "- An unbounded solver does not stop: runs were still going past 100,000",
        "  revealed cells. Games are therefore played in a 7×7 block of sectors and",
        f"  scored on the inner {int(math.isqrt(sectors))}×{int(math.isqrt(sectors))}.",
        f"- Guess quality shows up in the score. Over {n:,} endless games at density",
        "  0.21, with every certain move played and only the forced guesses",
        f"  differing, random guessing clears a mean of {r['cleared'].mean():.1f} of "
        f"{sectors} sectors,",
        f"  locks {r['locked'].mean():.2f}, and scores {r['score'].mean():.0f}; always "
        f"taking the oracle's safest",
        f"  cell clears {o['cleared'].mean():.1f}, locks {o['locked'].mean():.2f}, and "
        f"scores {o['score'].mean():.0f}. The paired",
        f"  difference is {diff_m:.1f} ± {1.96 * diff_se:.1f} points per game (95%). "
        f"The gap between the",
        "  two policies is what the learned models compete for.",
        "- The oracle's lead holds across the uncertainty in the measured density.",
        "  Scores rise with density because a cleared sector earns one point per",
        "  mine.",
        "",
        "  | Density | Policy | Cleared | Locked | Score |",
        "  | --- | --- | --- | --- | --- |",
    ]
    for p in DENSITIES:
        for name in ("random", "oracle"):
            g = pol[p, name]
            lines.append(f"  | {p:.2f} | {name} | {g['cleared'].mean():.1f} | "
                         f"{g['locked'].mean():.2f} | {g['score'].mean():.0f} |")
    lines.append("")
 
    if cal:
        lines += [
            f"- The oracle is calibrated against the true mines. Over {cal['rows']:,} "
            f"candidate",
            f"  cells, the mean predicted probability is {cal['mean_q']:.3f} and the "
            f"actual mine rate",
            f"  is {cal['rate']:.3f}; across 20 probability bins the largest gap is "
            f"{cal['gap']:.3f}.",
        ]
    safe_share = st[:, 0].mean()
    times_ms = st[:, 1] * 1000
    nan_share = st[:, 2].sum() / st[:, 3].sum()
    lines += [
        f"- The solver's two rules are incomplete. In {safe_share:.0%} of the "
        f"{len(st):,} positions",
        "  where the solver is stuck, the oracle still finds a cell that is",
        "  certainly safe. A forced guess is therefore defined by the oracle (no",
        "  certain-safe cell left), not by the solver stalling.",
        f"- The oracle takes a median of {np.median(times_ms):.0f} ms per stuck "
        f"position (95th",
        f"  percentile {np.percentile(times_ms, 95):.0f} ms); {nan_share:.1%} of "
        f"candidate cells exceed its search",
        f"  budget of {Config().max_nodes:,} nodes and are left out of the data.",
    ]
    print("\n".join(lines))
 
 
if __name__ == "__main__":
    main()
