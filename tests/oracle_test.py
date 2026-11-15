#!usr/bin/env python3

import itertools
import math

from inf_minesweeper.board import Board
from inf_minesweeper.solver import Solver, sector_region
from inf_minesweeper.oracle import components, mine_proba, frontier_proba


def brute_force(constraints, density):
    """Enumerate every layout of the cells"""
    cells = sorted({c for group in constraints for c in group})
    r = density / (1 - density)
    total, mass = 0.0, dict.fromkeys(cells, 0.0)
    for bits in itertools.product((0, 1), repeat=len(cells)):
        layout = dict(zip(cells, bits))
        if all(sum(layout[c] for c in g) == m for g, m in constraints.items()):
            w = r ** sum(bits)
            total += w
            for c in cells:
                mass[c] += w * layout[c]
    return {c: mass[c] / total for c in cells}


def stuck_solver(seed, density=0.21, radius=1):
    board = Board(seed, density=density)
    solver = Solver(board, sector_region(board, radius, (4, 4)))
    solver.reveal(4, 4)
    solver.solve()
    return solver


def test_hand_example():
    # one number seeing two cells with one mine: 50/50 at any density
    a, b = (0, 0), (1, 0)
    q = mine_proba({frozenset({a, b}): 1}, 0.2)
    assert math.isclose(q[a], 0.5) and math.isclose(q[b], 0.5)
    
    
def test_density_matters_when_mine_counts_differ():
    # Cells a, b, c. {a, b} has 1 mine, {b, c} has 1 mine
    # Layouts: b alone (1 mine) or a and c (2 mines). P(b) = r / (r + r^2)
    a, b, c = (0, 0), (1, 0), (2, 0)
    p = 0.21
    r = p / (1 - p)
    q = mine_proba({frozenset({a, b}): 1, frozenset({b, c}): 1}, p)
    assert math.isclose(q[b], 1 / (1 + r))
    assert math.isclose(q[a], r / (1 + r))
    
    
def test_matches_brute_force_on_real_positions():
    checked = 0
    for seed in range(120):
        solver = stuck_solver(seed)
        cons = solver.constraints()
        for comp in components(cons):
            n = len({c for g in comp for c in g})
            if n > 16:
                continue
            want = brute_force(comp, 0.21)
            got = mine_proba(comp, 0.21)
            for c in want:
                assert math.isclose(got[c], want[c], abs_tol=1e-12)
            checked += 1
    assert checked > 50


def test_budget_gives_nan():
    # some seeds are solved by deduction alone, so find one that needs a guess
    solver = next(s for s in map(stuck_solver, range(20)) if s.frontier())
    q = frontier_proba(solver, max_nodes=1)
    assert q and all(math.isnan(v) for v in q.values())
    