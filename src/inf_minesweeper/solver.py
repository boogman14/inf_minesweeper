#!usr/bin/env python3
"""solver.py -- deterministic solver: safe moves, flags, detects forced guesses

The solver sees only what the player sees, never calling `Board.is_mine` or `Board.number`. It applies two rules until neither produces anything new:

1.  Single constraint. A revealed number w/ `m` mines left among its `k` hidden enighbors: m == 0 -> all safe; m == k -> all mines
2.  Pairwise. For two constraints A and B that share cells, every cell only in A is a mine and every cell only in B is safe iff: mines(A) - mines(B) == |A only|

On an infinite board, the solver needs a boundary. At moderate mine densities, deduction almost never stalls everywhere at once: some part of the frontier can 
    always be pushed further, so an unbounded solve() would run forever. Giving the solver a 'region' allows for plausible simulations.
    
When both rules run dry inside the region the solver is "stuck". Stuck positions are where the dataset rows come from. Note the rules are sound but not complete: 
    a stuck position can still contain cells that a full enumeration (the oracle) proves safe or mined. Define the "hard" subset by the oracle's probability
    (0 < q < 1), not by the solver being stuck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from inf_minesweeper.board import Board, neighbors

Cell = tuple[int, int]                      # x, y
Region = tuple[int, int, int, int]          # x0, y0, x1, y1


def sector_region(board:Board, radius:int = 2, center:Cell = (0, 0)) -> Region:
    """Box of (2*radius + 1)^2 whole sectors around the sector containing `sector`"""
    s = board.sector_size
    sx, sy = board.sector_of(*center)
    return ((sx - radius) * s,
            (sy - radius) * s,
            (sx + radius) * s - 1,
            (sy + radius) * s - 1)
    

@dataclass
class SolveResult:
    revealed: int   # cells open by deduction
    flagged: int    # miens flagged by deduction
    passes: int     # rounds that revealed or flagged cells

class Solver:
    """Wraps a board and plays every move that can be deduced."""
    
    def __init__(self, board:Board, region:Region) -> None:
        self.board = board
        self.region = region
        self.active: set[Cell] = set()
        self.sync()
        
    # ------------------------------------------------------------- tracking
    def sync(self) -> None:
        """Rebuidl the active set from scratch"""
        self._active = {c for c, n in self.board.revealed.items() if n > 0}
    
    def in_region(self, c: Cell) -> bool:
        if self.region is None:
            return True
        x0, y0, x1, y1 = self.region
        return (x0 <= c[0] <= x1) and (y0 <= c[1] <= y1)
    
    # -------------------------------------------------------------- actions
    def reveal(self, x:int, y:int) -> list[Cell]:
        """reveal a cell"""
        opened = self.board.reveal(x, y)
        for c in opened:
            if self.board.revealed[c] > 0:
                self._active.add(c)
        return opened
    
    def flag(self, x:int, y:int) -> None:
        if (x, y) not in self.board.flagged:
            self.board.toggle_flag(x, y)
            
    # ---------------------------------------------------------- constraints
    def constraints(self) -> dict[frozenset[Cell], int]:
        """
        """
        out: dict[frozenset[Cell], int] = {}
        done = []
        for c in self._active:
            unknown = []
            flags = 0
            for nb in neighbors(*c):
                if nb in self.board.flagged:
                    flags += 1
                elif nb not in self.board.revealed:
                    unknown.append(nb)
            if not unknown:
                done.append(c)
                continue
            mines = self.board.revealed[c] - flags
            if not (0 <= mines <= len(unknown)):
                raise ValueError(f"flags around {c} contradict its number")
            out[frozenset(unknown)] = mines
        self._active.difference_update(done)
        return out
    
    def frontier(self) -> set[Cell]:
        """
        """
        cells: set[Cell] = set()
        for group in self.constraints():
            cells |= group
        return {c for c in cells if self.in_region(c)}
    
    # ------------------------------------------------------------ deduction
    def deduce(self) -> tuple[set[Cell], set[Cell]]:
        """One round of both rules."""
        cons = self.constraints()
        safe: set[Cell] = set()
        mines: set[Cell] = set()
        
        # Rule 1: single constraint
        for cells, m in cons.items():
            if m == 0:
                safe |= cells
            elif m == len(cells):
                mines |= cells
        safe = {c for c in safe if self.in_region(c)}
        mines = {c for c in mines if self.in_region(c)}
        if safe or mines:
            return safe, mines
        
        # Rule 2: pairs of constraints that share a cell
        by_cell: dict[Cell, list[frozenset[Cell]]] = {}
        for cells in cons:
            for c in cells:
                by_cell.setdefault(c, []).append(cells)
        for a, ma in cons.items():
            partners = {b for c in a for b in by_cell[c] if b is not a}
            for b in partners:
                only_a = a - b
                if ma - cons[b] == len(only_a):
                    mines |= only_a
                    safe |= b - a
        safe = {c for c in safe if self.in_region(c)}
        mines = {c for c in mines if self.in_region(c)}
        return safe, mines
    
    def solve(self, max_revealed:int | None = None) -> SolveResult:
        """Play every deducible move until stuck or `max_revealed` cells on the current view is reached."""
        revealed = flagged = passes = 0
        while self.board.alive:
            if max_revealed is not None and len(self.board.revealed) >= max_revealed:
                break
            safe, mines = self.deduce()
            if not safe and not mines:
                break
            passes += 1
            for c in mines:
                self.flag(*c)
            flagged += len(mines)
            for c in safe:
                revealed += len(self.reveal(*c))
                if not self.board.alive:
                    raise RuntimeError(f"solver bug: 'safe' cell {c} was a mine")
        return SolveResult(revealed, flagged, passes)
    
    
# ------------------------------------------------------------------ autoplay
Guesser = Callable[[Solver], Cell]


def random_guesser(rng: random.Random) -> Guesser:
    """baseline policy: unformly random frontier cell"""
    def pick(solver: Solver) -> Cell:
        return rng.choice(sorted(solver.frontier()))
    return pick


@dataclass
class GameResult:
    alive: bool
    guesses: int
    revealed: int
    sectors_cleared: int


def autoplay(
    board: Board,
    guesser: Guesser | None = None,
    start: Cell = (5, 5),
    radius: int = 2,
    max_guesses: int = 1_000,
    on_stuck: Callable[[Solver], None] | None = None,
) -> GameResult:
    """Play one game inside a (2*radius + 1)^2-sector box around `start`:
    deduce, guess when stuck, repeat. The game ends on a mine, when the box
    has no frontier left (finished), or after `max_guesses`.

    `on_stuck` is called at every stuck position before the guess is made;
    simulate.py will use it to log frontier cells.
    """
    guesser = guesser or random_guesser(random.Random(board.seed))
    solver = Solver(board, sector_region(board, radius, start))
    if board.start is None:
        solver.reveal(*start)
    guesses = 0
    while board.alive:
        solver.solve()
        if guesses >= max_guesses or not solver.frontier():
            break
        if on_stuck is not None:
            on_stuck(solver)
        solver.reveal(*guesser(solver))
        guesses += 1
    return GameResult(
        alive=board.alive,
        guesses=guesses,
        revealed=len(board.revealed),
        sectors_cleared=len(board.cleared_sectors())
    )