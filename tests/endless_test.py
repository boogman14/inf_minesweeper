#!usr/bin/env python3

import math

import pytest

from inf_minesweeper.board import (FLAGGED, HIDDEN, MINE_HIT, WRONG_FLAG, Board, GameOver, neighbors)
from inf_minesweeper.solver import Solver, autoplay, sector_region
from inf_minesweeper.oracle import frontier_proba


def opened_board(seed=0, **kw):
    board = Board(seed, **kw)
    board.reveal(4, 4)
    return board


def frontier_mine(board, in_sector=None):
    for c in sorted(board.frontier()):
        if board.is_mine(*c) and (in_sector is None or board.sector_of(*c) == in_sector):
            return c
    raise AssertionError("no frontier mine found")


def test_mine_locks_sector_and_game_goes_on():
    board = opened_board()
    mine = frontier_mine(board)
    sector = board.sector_of(*mine)
    revealed_before = dict(board.revealed)

    assert board.reveal(*mine) == []
    assert board.alive
    assert sector in board.locked
    assert board.state(*mine) == MINE_HIT
    assert board.revealed == revealed_before          # clues stay visible
    assert not board.sector_cleared(*sector)

    # nothing in the locked sector can be revealed or flagged
    for c in board.sector_cells(*sector):
        if board.state(*c) == HIDDEN:
            assert board.reveal(*c) == []
            board.toggle_flag(*c)
            assert board.state(*c) == HIDDEN


def test_wrong_flags_are_marked_on_lock():
    board = opened_board()
    mine = frontier_mine(board)
    sector = board.sector_of(*mine)
    safe = next(c for c in sorted(board.frontier())
                if not board.is_mine(*c) and board.sector_of(*c) == sector)
    right = next((c for c in sorted(board.frontier())
                  if board.is_mine(*c) and c != mine and board.sector_of(*c) == sector), None)
    board.toggle_flag(*safe)
    if right:
        board.toggle_flag(*right)
    board.reveal(*mine)
    assert board.state(*safe) == WRONG_FLAG
    if right:
        assert board.state(*right) == FLAGGED


def test_adjacent_only():
    board = opened_board()
    far = (500, 500)
    assert board.reveal(*far) == []
    board.toggle_flag(*far)
    assert board.state(*far) == HIDDEN
    free = opened_board(adjacent_only=False)
    free.toggle_flag(*far)
    assert free.state(*far) == FLAGGED


def test_hardcore_ends_the_game():
    board = opened_board(mode="hardcore")
    board.reveal(*frontier_mine(board))
    assert not board.alive and not board.locked
    with pytest.raises(GameOver):
        board.reveal(0, 0)


def test_solver_and_oracle_stay_sound_through_locks():
    """Play endless games with random guesses. After every guess, each solver
    flag must be a real mine, no playable cell may sit in a locked sector,
    and the oracle's certain cells must agree with the truth."""
    for seed in range(40):
        board = Board(seed)

        def check(solver):
            q = frontier_proba(solver)
            for c, v in q.items():
                assert not board.is_locked(*c)
                if v == 0.0:
                    assert not board.is_mine(*c)
                if v == 1.0:
                    assert board.is_mine(*c)

        res = autoplay(board, radius=1, on_stuck=check)
        assert res.alive
        assert all(board.is_mine(*c) for c in board.flagged)
        assert res.sectors_locked == len(board.locked)
        assert res.sectors_cleared + res.sectors_locked <= 9


def test_score_is_mines_cleared_minus_lock_penalty():
    board = opened_board(lock_penalty=4)
    assert board.score() == 0
    board.reveal(*frontier_mine(board))
    assert board.score() == -4

    board = Board(2, lock_penalty=4)
    res = autoplay(board, radius=1)
    earned = sum(len(board.sector_mines(*k)) for k in board.cleared_sectors())
    assert res.sectors_cleared > 0
    assert res.score == earned - 4 * res.sectors_locked


def test_openings_stop_at_locked_sectors():
    """No flood, from anywhere, reveals a cell inside a locked sector."""
    for seed in range(30):
        board = Board(seed)
        frozen = {}

        def check(solver):
            for sector in board.locked:
                inside = {c for c in board.sector_cells(*sector) if c in board.revealed}
                assert frozen.setdefault(sector, inside) == inside

        autoplay(board, radius=1, on_stuck=check)


def test_marks_are_display_only():
    from inf_minesweeper.board import render_labeled, tenths
    board = opened_board()
    cell = sorted(board.frontier())[0]
    before = (dict(board.revealed), set(board.flagged))
    plain = render_labeled(board, 4, 4, 10, color=False)
    marked = render_labeled(board, 4, 4, 10, color=False, marks=[cell])
    digits = render_labeled(board, 4, 4, 10, color=False, marks=tenths({cell: 0.14}))
    assert "?" not in plain and marked.count("?") == 1
    assert "[?" in marked and "[1" in digits
    assert (dict(board.revealed), set(board.flagged)) == before