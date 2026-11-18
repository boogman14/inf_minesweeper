#!usr/bin/env python3
"""simulate.py -- generates dataset, plays games, logs frontier cells → data/raw

One game, inside a box of sectors around the first click:

1. The solver plays every simple certain move.
2. The oracle computes the exact mine probability q of every candidate cell.
   Cells with q = 1 are flagged and cells with q = 0 are revealed, then back
   to step 1. These moves are certain, so they are not guesses.
3. If no certain cell is left, the position is a forced guess. Every
   candidate cell becomes one dataset row: its 9x9 window of visible cell
   states, its oracle probability q, and whether it really is a mine.
4. One candidate is revealed: the oracle's safest cell, or with probability
   `epsilon` a random one. A mine locks the sector and the game goes on.
   
Candidates whose oracle search exceeds its budget are logged too, with q =
NaN: their mine label is still known. `loader.load` leaves them out unless
asked, so the oracle-based metrics never see a missing q.

The game ends when the box has no candidates left or after `max_guesses`.

Sectors on the edge of the play box can be impossible to finish, because
nothing outside the box may be played. Games are therefore played on a box of
radius `radius` and scored on the inner box of radius `radius - 1`.

Command line (writes one compressed .npz file):

    python -m inf_minesweeper.simulate --games 2000 --out data/raw/guesses.npz
    python -m inf_minesweeper.simulate --minutes 5 --out data/raw/guesses.npz

Test sets for distribution shift use other seeds and densities, e.g.

    python -m inf_minesweeper.simulate --games 1000 --first-seed 100000 \
        --density 0.20 --out data/raw/shift_p020.npz

and games played by a trained model (saved with pickle) use --policy:

    python -m inf_minesweeper.simulate --games 1000 --first-seed 200000 \
        --epsilon 0 --policy results/lgb5.pkl --out data/raw/played_lgb5.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from itertools import count
import multiprocessing as mp
from typing import Callable

import numpy as np

from inf_minesweeper.board import Board
from inf_minesweeper.solver import Solver, sector_region
from inf_minesweeper.oracle import frontier_proba

WINDOW = 9
_HALF = WINDOW // 2
_CERTAIN = 1e-9


@dataclass
class Config:
    density: float = 0.21
    radius: int = 3
    epsilon: float = 0.1            # share of guesses made at random
    max_guesses: int = 300
    max_nodes: int = 200_000        # oracle search budget per component
    lock_penalty: float = 5.0
    start: tuple[int, int] = (4, 4)
    
    
def window(board: Board, x: int, y: int) -> np.ndarray:
    """WINDOW x WINDOW array of player-visible states centered on (x, y)"""
    out = np.empty((WINDOW, WINDOW), dtype=np.uint8)
    for j in range(WINDOW):
        for i in range(WINDOW):
            out[j, i] = board.state(x + i - _HALF, y + j - _HALF)
    return out


Policy = Callable[[np.ndarray], np.ndarray]   # (n, 9, 9) windows -> P(mine)


def play_game(seed: int, cfg: Config = Config(), policy: Policy | None = None) -> tuple[dict, dict]:
    """Play one game. Returns (rows, summary); rows hold one entry per 
    candidate cell at every forced guess.
    
    `policy` picks the guess in place of the oracle: it gets the candidates'
    windows and the candidate with the lowest returned value is revealed.
    Random guesses (`epsilon`) still apply on top."""
    board = Board(seed, density=cfg.density, lock_penalty=cfg.lock_penalty)
    solver = Solver(board, sector_region(board, cfg.radius, cfg.start))
    rng = random.Random(seed)
    solver.reveal(*cfg.start)
    
    rows = {k: [] for k in ("window", "q", "mine", "x", "y", "position", "chosen")}
    guesses = random_guesses = certain_moves = dropped = 0
    pick_q = pick_q_n = regret = 0.0
    
    while guesses < cfg.max_guesses:
        solver.solve()
        probs = frontier_proba(solver, cfg.max_nodes)
        if not probs:
            break
        
        # Certain cells first
        sure_mines = [c for c, q in probs.items() if q > 1 - _CERTAIN]
        sure_safe = [c for c, q in probs.items() if q < _CERTAIN]
        for c in sure_mines:
            solver.flag(*c)
        for c in sure_safe:
            solver.reveal(*c)
        certain_moves += len(sure_mines) + len(sure_safe)
        if sure_safe:
            continue
        
        # Forced guess
        cands = sorted(c for c, q in probs.items() if q <= 1 - _CERTAIN or math.isnan(q))
        known = [c for c in cands if not math.isnan(probs[c])]
        dropped += len(cands) - len(known)
        if not cands:
            break
        windows = [window(board, *c) for c in cands]
        if policy is not None:
            if rng.random() >= cfg.epsilon:
                p = np.asarray(policy(np.stack(windows)), dtype=float)
                pick = cands[int(np.argmin(p))]
            else:
                pick = rng.choice(cands)
                random_guesses += 1
        elif known and rng.random() >= cfg.epsilon:
            best = min(probs[c] for c in known)
            pick = rng.choice([c for c in known if probs[c] <= best + 1e-12])
        else:
            pick = rng.choice(cands)
            random_guesses += 1
        if known and not math.isnan(probs[pick]):
            pick_q += probs[pick]
            pick_q_n += 1
            regret += probs[pick] - min(probs[c] for c in known)
            
        for c, w in zip(cands, windows):
            rows["window"].append(w)
            rows["q"].append(probs[c])
            rows["mine"].append(board.is_mine(*c))
            rows["x"].append(c[0])
            rows["y"].append(c[1])
            rows["position"].append(guesses)
            rows["chosen"].append(c == pick)
        
        solver.reveal(*pick)
        guesses += 1
    
    # score the inner box only w/ ground-truth mine counts per sector
    sx, sy = board.sector_of(*cfg.start)
    r = cfg.radius - 1
    inner = [(i, j) for i in range(sx - r, sx + r + 1) for j in range(sy - r, sy + r + 1)]
    cleared = [s for s in inner if board.sector_cleared(*s)]
    locked = [s for s in inner if s in board.locked]
    cleared_mines = sum(len(board.sector_mines(*s)) for s in cleared)
    summary = dict(
        seed=seed,
        guesses=guesses,
        random_guesses=random_guesses,
        mines_hit=len(board.hits),
        certain_moves=certain_moves,
        dropped_cells=dropped,
        pick_q_sum=pick_q,
        pick_q_n=pick_q_n,
        regret_sum=regret,
        sectors=len(inner),
        cleared=len(cleared),
        locked=len(locked),
        cleared_mines=cleared_mines,
        locked_mines=sum(len(board.sector_mines(*s)) for s in locked),
        all_mines=sum(len(board.sector_mines(*s)) for s in inner),
        score=cleared_mines - cfg.lock_penalty * len(locked),
    )
    return rows, summary


def _job(args: tuple[int, Config, Policy | None]) -> tuple[dict, dict]:
    return play_game(*args)


def run(
    games: int | None = None,
    minutes: float | None = None,
    first_seed: int = 0,
    cfg: Config = Config(),
    workers: int | None = None,
    progress: bool = True,
    policy: Policy | None = None
) -> dict[str, np.ndarray]:
    """Play `games` games (seeds first_seed, first_seed + 1, ...) or as many as
    fit in `minutes`, in parallel. Returns the dataset as a dict of arrays.
    
    `policy` (see play_game) is sent to every worker process, so it must be
    picklable: a bound method like `model.predict` is fine, a lambda is not.
    Workers are then started fresh ("spawn") rather than forked, because
    libraries like LightGBM can deadlock in a forked copy of a process that
    has already used their thread pool. For a GPU model, use workers=1."""
    if (games is None) == (minutes is None):
        raise ValueError("give exactly one of games or minutes")
    seeds = range(first_seed, first_seed + games) if games else count(first_seed)
    deadline = time.time() + 60 * minutes if minutes else None
    workers = workers or os.cpu_count() or 1
    
    cols = {k: [] for k in ("window", "q", "mine", "x", "y", "position", "chosen", "game")}
    summaries = []
    t0 = time.time()
    ctx = mp.get_context("spawn" if policy is not None else None)
    with ctx.Pool(workers) as pool:
        for rows, summary in pool.imap(_job, ((s, cfg, policy) for s in seeds), chunksize=4):
            n = len(rows["q"])
            for k in rows:
                cols[k].extend(rows[k])
            cols["game"].extend([summary["seed"]] * n)
            summaries.append(summary)
            if progress and len(summaries) % 100 == 0:
                print(f"  {len(summaries)} games, {len(cols['q'])} rows, ")
                print(f"{time.time() - t0:.0f}s", flush=True)
            if deadline and time.time() >= deadline:
                pool.terminate()
                break
    
    data = dict(
        window=np.array(cols["window"], dtype=np.uint8).reshape(-1, WINDOW, WINDOW),
        q=np.array(cols["q"], dtype=np.float64),
        mine=np.array(cols["mine"], dtype=np.uint8),
        x=np.array(cols["x"], dtype=np.int32),
        y=np.array(cols["y"], dtype=np.int32),
        position=np.array(cols["position"], dtype=np.int32),
        chosen=np.array(cols["chosen"], dtype=bool),
        game=np.array(cols["game"], dtype=np.int64),
    )
    for k in summaries[0]:
        data[f"game_{k}"] = np.array([s[k] for s in summaries])
    data["config"] = np.array(json.dumps(asdict(cfg)))
    return data


def report(data: dict[str, np.ndarray]) -> str:
    """Short text summary of a dataset."""
    n_games = len(data["game_seed"])
    scored = ~np.isnan(data["q"])
    q, mine = data["q"][scored], data["mine"][scored]
    positions = len(np.unique(np.stack([data["game"], data["position"]]), axis=1).T)
    p = float(json.loads(str(data["config"]))["density"])
    cm, cs = data["game_cleared_mines"].sum(), data["game_cleared"].sum()
    lm, ls = data["game_locked_mines"].sum(), data["game_locked"].sum()
    lines = [
        f"games: {n_games}   forced-guess positions: {positions}   rows: {len(scored)}",
        f"candidates per position: {len(scored) / max(positions, 1):.1f}",
        f"mean oracle q: {q.mean():.4f}   actual mine rate: {mine.mean():.4f}",
        f"guesses per game: {data['game_guesses'].mean():.1f}   "
        f"mines hit per game: {data['game_mines_hit'].mean():.2f}",
        f"inner sectors per game: cleared {data['game_cleared'].mean():.1f}, "
        f"locked {data['game_locked'].mean():.1f} of {data['game_sectors'][0]}   "
        f"score {data['game_score'].mean():.0f}",
        f"rows without an oracle q (search budget): {(~scored).sum()} "
        f"({(~scored).mean():.1%}); their mine rate {data['mine'][~scored].mean() if (~scored).any() else float('nan'):.4f}",
        f"guessed cells: mean oracle q {data['game_pick_q_sum'].sum() / max(data['game_pick_q_n'].sum(), 1):.4f}, "
        f"mean regret {data['game_regret_sum'].sum() / max(data['game_pick_q_n'].sum(), 1):.4f}",
        f"density: true {p:.4f}   in cleared sectors {cm / (64 * cs):.4f}"
        + (f"   in locked sectors {lm / (64 * ls):.4f}" if ls else ""),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])        #type:ignore
    ap.add_argument("--games", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--out", default="data/raw/guesses.npz")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--density", type=float, default=Config.density)
    ap.add_argument("--radius", type=int, default=Config.radius)
    ap.add_argument("--epsilon", type=float, default=Config.epsilon)
    ap.add_argument("--policy", help="pickled model with a predict(windows) method")
    args = ap.parse_args()
    policy = None
    if args.policy:
        with open(args.policy, "rb") as f:
            policy = pickle.load(f).predict
    if args.games is None and args.minutes is None:
        args.games = 200
    cfg = Config(density=args.density, radius=args.radius, epsilon=args.epsilon)
    data = run(args.games, args.minutes, args.first_seed, cfg, args.workers, policy=policy)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, allow_pickle=False, **data)
    print(report(data))
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    
    
if __name__ == "__main__":
    main()
