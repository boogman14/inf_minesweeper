#!usr/bin/env python3
"""oracle.py -- exact frontier probabilities (component split + enumeration)

Setting: mines are independent with density p, and the board is infinite, so there is no global mine count tying 
distant cells together. Given what the player sees, the posterior over the unknown cells next to revealed numbers is

    P(layout) proportional to r^(number of mines),  r = p / (1 - p),

restricted to layouts that satisfy every number. Two consequences:

* Groups of cells that share no number ("components") are independent, so each
  is solved on its own.
* Unknown cells that touch no revealed number keep the prior p.

Within a component, cells that touch exactly the same set of numbers are interchangeable. They are merged into one 
"group" of size g, which takes k = 0..g mines in C(g, k) ways. The search then runs over groups, not cells, which is 
what keeps it fast.

The input is the constraint dict from `Solver.constraints()`: it uses only what the player sees. Pass ALL constraints, 
including ones outside the solver's region; dropping any changes the answer.
"""

from __future__ import annotations

from math import comb, nan

from inf_minesweeper.solver import Cell, Solver

Constraints = dict[frozenset[Cell], int]


class BudgetExceeded(Exception):
    pass


def components(constraints: Constraints) -> list[Constraints]:
    """Split constraints into independent pieces, linked by shared cells"""
    by_cell: dict[Cell, list[frozenset[Cell]]] = {}
    for cells in constraints:
        for c in cells:
            by_cell.setdefault(c, []).append(cells)
    seen: set[frozenset[Cell]] = set()
    out = []
    for start in constraints:
        if start in seen:
            continue
        seen.add(start)
        stack: list[frozenset[Cell]] = [start]
        comp: Constraints = {}
        while stack:
            cur = stack.pop()
            comp[cur] = constraints[cur]
            for c in cur:
                for other in by_cell[c]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        out.append(comp)
    return out


def _solve_component(comp: Constraints, r: float, max_nodes: int) -> dict[Cell, float]:
    cons = list(comp.items())
    
    # Merge cells w/ identical constraint membership into groups
    membership: dict[Cell, list[int]] = {}
    for i, (cells, _) in enumerate(cons):
        for c in cells:
            membership.setdefault(c, []).append(i)
    grouped: dict[tuple[int, ...], list[Cell]] = {}
    for c, idx in membership.items():
        grouped.setdefault(tuple(idx), []).append(c)
    
    # Order groups s.t. neighbors in the constraint graphare assigned close together
    keys = sorted(grouped, key=lambda k: (k[0], k))
    order, placed, used_cons = [], set(), set()
    pending = [keys[0]]
    remaining = set(keys)
    while remaining:
        if not pending:
            pending = [min(remaining)]
        k = pending.pop(0)
        if k in placed:
            continue
        placed.add(k)
        remaining.discard(k)
        order.append(k)
        used_cons.update(k)
        pending += sorted(g for g in remaining if used_cons.intersection(g) and g not in pending)
        
    sizes = [len(grouped[k]) for k in order]
    members = order         # constraint indices touched by each group
    n = len(order)
    
    need = [m for _, m in cons]                         # mines still required
    room = [len(cells) for cells, _ in cons]            # unassigned cells left
    ways = [[comb(g, k) * r**k for k in range(g + 1)] for g in sizes]
    
    total = 0.0
    mine_weight = [0.0] * n
    picks = [0] * n
    nodes = 0
    
    def search(i: int, weight: float) -> None:
        nonlocal total, nodes
        nodes += 1
        if nodes > max_nodes:
            raise BudgetExceeded
        if i == n:
            total += weight
            for j in range(n):
                if picks[j]:
                    mine_weight[j] += weight * picks[j]
            return
        g, idx = sizes[i], members[i]
        lo, hi = 0, g
        for ci in idx:
            hi = min(hi, need[ci])
            lo = max(lo, need[ci] - (room[ci] - g))
        if lo > hi:
            return
        for ci in idx:
            room[ci] -= g
        for k in range(lo, hi + 1):
            for ci in idx:
                need[ci] -= k
            picks[i] = k
            search(i + 1, weight * ways[i][k])
            for ci in idx:
                need[ci] += k
        picks[i] = 0
        for ci in idx:
            room[ci] += g
    
    search(0, 1.0)
    if total == 0.0:
        raise ValueError("constarints are inconsistent: no valid mien layout")
    out = {}
    for j, key in enumerate(order):
        q = mine_weight[j] / (total * sizes[j])
        for c in grouped[key]:
            out[c] = q
    
    return out


def mine_proba(constraints: Constraints, density: float, max_nodes: int = 2_000_000,) -> dict[Cell, float]:
    """Exact P(mine) for every cell that appears in `constraints`.
    `max_nodes` bounds the search per component, returns NaN if component exceeds it
    """
    if not (0.0 < density < 1.0):
        raise ValueError("density must be in (0, 1)")
    r = density / (1.0 - density)
    out: dict[Cell, float] = {}
    for comp in components(constraints):
        try:
            out.update(_solve_component(comp, r, max_nodes))
        except BudgetExceeded:
            for cells in comp:
                for c in cells:
                    out[c] = nan
    return out


def frontier_proba(solver: Solver, max_nodes: int = 2_000_000) -> dict[Cell, float]:
    """Orcale probabilities for the solver's `region`"""
    probs = mine_proba(solver.constraints(), solver.board.density, max_nodes)
    return {c: q for c, q in probs.items() if solver.in_region(c)}
