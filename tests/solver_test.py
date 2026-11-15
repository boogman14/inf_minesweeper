#!usr/bin/env python3

from inf_minesweeper.board import Board
from inf_minesweeper.solver import Solver, autoplay, sector_region


def test_deductions_are_sound():
    """Every flag is a real mine; the solver never dies on a `safe` cell"""
    for seed in range(200):
        board = Board(seed)
        solver = Solver(board, sector_region(board, 2))
        solver.reveal(0, 0)
        solver.solve()
        assert board.alive
        assert all(board.is_mine(*c) for c in board.flagged)


def test_stuck_means_no_simple_move():
    board = Board(3)
    solver = Solver(board, sector_region(board, 2))
    solver.reveal(0, 0)
    solver.solve()
    assert solver.deduce() == (set(), set())
    
    
def test_every_step_agrees_with_ground_truth():
    """Each deduction round is checked against the true mines, and the
    constraints left at the stuck position are all satisfied by the truth."""
    for seed in range(30):
        board = Board(seed)
        solver = Solver(board, sector_region(board, 2))
        solver.reveal(0, 0)
        while True:
            safe, mines = solver.deduce()
            if not safe and not mines:
                break
            for c in safe:
                assert not board.is_mine(*c)
            for c in mines:
                assert board.is_mine(*c)
            for c in mines:
                solver.flag(*c)
            for c in safe:
                solver.reveal(*c)
        # at the stuck position every constraint is satisfied by the truth
        for cells, m in solver.constraints().items():
            assert sum(board.is_mine(*c) for c in cells) == m


def test_autoplay_runs_and_scores():
    res = autoplay(Board(1), max_guesses=50)
    assert res.guesses <= 50
    assert res.revealed > 0


def test_solver_stays_in_region():
    board = Board(5)
    region = sector_region(board, 1)
    solver = Solver(board, region)
    solver.reveal(0, 0)
    solver.solve()
    x0, y0, x1, y1 = region
    assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in board.flagged)
    
    
def test_sector_region_is_centered_and_whole():
    board = Board(0)
    assert sector_region(board, 1, (4, 4)) == (-8, -8, 15, 15)
    assert sector_region(board, 2, (4, 4)) == (-16, -16, 23, 23)
    assert sector_region(board, 0, (-1, -1)) == (-8, -8, -1, -1)