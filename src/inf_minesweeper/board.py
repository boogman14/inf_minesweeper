#!usr/bin/env python3

"""board.py -- Infinite Minesweeper board

The board is unbounded. Mines are never stored: whether (x, y) holds a mine is
decided by a seeded hash of its coordinates, so a board uses memory only for
the cells the player has touched, and any game can be replayed from its seed.

Coordinates: x grows to the right, y grows downward. Sectors are
`sector_size` x `sector_size` blocks indexed by (x // size, y // size); Python's
floor division keeps negative coordinates in the right sector.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

# Cell-state codes shared with features.py (13 categories).
# 0-8 = revealed number, HIDDEN = non-revealed, FLAGGED = flagged
# The last two only appear in locked sectors (endless mode)
# MINE_HIT = mine was stepped on... BOOM!
# WRONG_FLAG = incorrectly placed flag (known to be safe but number is uncertain)
HIDDEN = 9
FLAGGED = 10
MINE_HIT = 11
WRONG_FLAG = 12
N_STATES = 13

GAME_MODES = ("endless", "hardcore")

FIRST_CLICK_RULES = ("none", "safe", "zero")

_MASK = (1 << 64) - 1
_NEIGHBOR_OFFSETS = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def _mix64(z: int) -> int:
    """SplitMix64 finalizer: 64-bit integer hash"""
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def neighbors(x: int, y: int) -> Iterator[tuple[int, int]]:
    """The 8 cells around (x, y)"""
    for dx, dy in _NEIGHBOR_OFFSETS:
        yield x + dx, y + dy


class GameOver(Exception):
    """Raised when an action is attempted after the game has ended."""


class Board:
    """One game of Infinite Minesweeper.

    Parameters
    ----------
    seed : int
        Determines the whole (infinite) mine layout.
    density : float
        Probability that any given cell is a mine, independently per cell.
        0.2 is a placeholder; set it to the value estimated from the app.
    first_click : {"none", "safe", "zero"}
        "none": no protection. "safe": the first clicked cell is never a mine.
        "zero": the first cell and its 8 neighbors are mine-free, so the first
        click always opens an area.
        Mines are removed from the protected zone rather than relocated; on an
        infinite board relocation would not change anything measurable.
    mode : {"endless", "hardcore"}
        "endless": revealing a mine locks that mine's sector for good. Nothing
        in a locked sector can be revealed or flagged again, its wrong flags
        are marked, and everything already revealed there stays visible as
        clues for neighboring sectors. An opening that spreads from another
        sector stops at a locked sector's edge. The game goes on.
        "hardcore": revealing a mine ends the game.
    adjacent_only : bool
        After the first click, only cells that touch a revealed cell can be
        revealed or flagged (the app's rule).
    sector_points, lock_penalty : float
        Scoring. Each cleared sector earns `sector_points`; each locked sector
        is lost for good and also costs `lock_penalty`. The defaults (1 and 1)
        are placeholders; set them to the app's values.
    sector_size : int
        Side length of a scoring sector (10 in the app).
    max_flood : int
        Safety cap on cells revealed by one flood fill. At very low densities
        an opening can, in principle, grow without bound.
    """

    def __init__(
        self,
        seed: int,
        density: float = 0.21,
        first_click: str = "zero",
        mode: str = "endless",
        adjacent_only: bool = True,
        lock_penalty: float = 5.0,
        sector_size: int = 8,
        max_flood: int = 1_000_000,
    ) -> None:
        if not 0.0 < density < 1.0:
            raise ValueError("density must be in (0, 1)")
        if first_click not in FIRST_CLICK_RULES:
            raise ValueError(f"first_click must be one of {FIRST_CLICK_RULES}")
        if mode not in GAME_MODES:
            raise ValueError(f"mode must be in {GAME_MODES}")

        self.seed = seed
        self.density = density
        self.first_click = first_click
        self.mode = mode
        self.adjacent_only = adjacent_only
        self.lock_penalty = lock_penalty
        self.sector_size = sector_size
        self.max_flood = max_flood

        self._threshold = int(density * (1 << 64))
        self._seed_hash = _mix64(seed & _MASK)

        self.revealed: dict[tuple[int, int], int] = {}  # cell -> number 0-8
        self.flagged: set[tuple[int, int]] = set()
        self.start: tuple[int, int] | None = None
        self._safe_zone: frozenset[tuple[int, int]] = frozenset()

        self.alive = True
        self.exploded: tuple[int, int] | None = None
        self.hits: set[tuple[int, int]] = set()
        self.wrong_flags: set[tuple[int, int]] = set()
        self.locked: set[tuple[int, int]] = set()

        # Cached ground-truth mine sets per sector.
        self._sector_mines: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

    # ------------------------------------------------------------------ truth
    def is_mine(self, x: int, y: int) -> bool:
        """Ground truth. Solvers and models must not call this."""
        if (x, y) in self._safe_zone:
            return False
        h = _mix64(_mix64(self._seed_hash ^ (x & _MASK)) ^ (y & _MASK))
        return h < self._threshold

    def number(self, x: int, y: int) -> int:
        """Count of mines among the 8 neighbors of (x, y)."""
        return sum(self.is_mine(nx, ny) for nx, ny in neighbors(x, y))

    # ---------------------------------------------------------- player view
    def state(self, x: int, y: int) -> int:
        """What the player sees at (x, y): 0-8, HIDDEN, FLAGGED, MINE_HIT, or WRONG_FLAG."""
        c = (x, y)
        if c in self.revealed:
            return self.revealed[c]
        if c in self.hits:
            return MINE_HIT
        if c in self.wrong_flags:
            return WRONG_FLAG
        if c in self.flagged:
            return FLAGGED
        return HIDDEN
    
    def known_mine(self, c: tuple[int, int]) -> bool:
        return c in self.flagged or c in self.hits
    
    def is_unknown(self, c: tuple[int, int]) -> bool:
        return (c not in self.revealed and c not in self.flagged and
                c not in self.hits and c not in self.wrong_flags)
        
    def is_locked(self, x: int, y: int) -> bool:
        return self.sector_of(x, y) in self.locked
    
    def touches_revealed(self, x: int, y: int) -> bool:
        return any(c in self.revealed for c in neighbors(x, y))
    
    def can_act(self, x: int, y: int) -> bool:
        """True if the rules allow revealing or flagging (x, y) right now."""
        if not self.alive or self.is_locked(x, y):
            return False
        if self.start is None or not self.adjacent_only:
            return True
        return self.touches_revealed(x, y)

    def frontier(self) -> set[tuple[int, int]]:
        """Hidden, unflagged cells that touch at least one revealed cell."""
        out = set()
        for x, y in self.revealed:
            for c in neighbors(x, y):
                if self.is_unknown(c) and not self.is_locked(*c):
                    out.add(c)
        return out

    # -------------------------------------------------------------- actions
    def reveal(self, x: int, y: int) -> list[tuple[int, int]]:
        """Reveal (x, y), flood-filling from zeros.

        Returns the newly revealed cells (empty if nothing changed). Hitting a
        mine ends the game: `alive` becomes False and `exploded` is set.
        """
        self._check_alive()
        if not self.is_unknown((x, y)) or not self.can_act(x, y):
            return []
        if self.start is None:
            self._place_start(x, y)
        if self.is_mine(x, y):
            self.exploded = (x, y)
            self.hits.add((x, y))
            if self.mode == "hardcore":
                self.alive = False
            else:
                self._lock(self.sector_of(x, y))
            return []
        return self._flood(x, y)

    def toggle_flag(self, x: int, y: int) -> None:
        self._check_alive()
        c = (x, y)
        if c in self.flagged:
            if not self.is_locked(x, y):
                self.flagged.discard(c)
        elif self.is_unknown(c) and self.can_act(x, y):
            self.flagged.add(c)
            
    def chord(self, x: int, y: int) -> list[tuple[int, int]]:
        """Reveal every unflagged neighbor of a revealed number whose
        flag count already matches it. A wrong flag can end the game."""
        self._check_alive()
        n = self.revealed.get((x, y))
        if n is None:
            return []
        nbrs = list(neighbors(x, y))
        if sum(self.known_mine(c) for c in nbrs) != n:
            return []
        opened: list[tuple[int, int]] = []
        hits_before = len(self.hits)
        for c in nbrs:
            if self.is_unknown(c):
                opened += self.reveal(*c)
                if len(self.hits) > hits_before:
                    break
        return opened

    # -------------------------------------------------------------- sectors
    def sector_of(self, x: int, y: int) -> tuple[int, int]:
        return x // self.sector_size, y // self.sector_size

    def sector_cells(self, sx: int, sy: int) -> Iterator[tuple[int, int]]:
        s = self.sector_size
        for y in range(sy * s, (sy + 1) * s):
            for x in range(sx * s, (sx + 1) * s):
                yield x, y

    def sector_mines(self, sx: int, sy: int) -> frozenset[tuple[int, int]]:
        """Ground-truth mine cells in a sector (cached)."""
        key = (sx, sy)
        if key not in self._sector_mines:
            self._sector_mines[key] = frozenset(
                c for c in self.sector_cells(sx, sy) if self.is_mine(*c)
            )
        return self._sector_mines[key]

    def sector_cleared(self, sx: int, sy: int) -> bool:
        """True when the sector's flags match its mines exactly: every mine
        flagged and no wrong flags. A mine-free sector (vanishingly rare at
        realistic densities) never counts as cleared."""
        mines = self.sector_mines(sx, sy)
        if not mines or (sx, sy) in self.locked:
            return False
        return all(
            (c in self.flagged) == (c in mines) for c in self.sector_cells(sx, sy)
        )

    def cleared_sectors(self) -> list[tuple[int, int]]:
        touched = {self.sector_of(*c) for c in self.flagged}
        return sorted(k for k in touched if self.sector_cleared(*k))
    
    def score(self) -> float:
        """One point per mine in each cleared sector, minus the penalty for each locked sector"""
        earned = sum(len(self.sector_mines(*k)) for k in self.cleared_sectors())
        return earned - self.lock_penalty * len(self.locked)

    # ------------------------------------------------------------ rendering
    def render(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        show_mines: bool = False,
    ) -> str:
        """Text view of the rectangle [x0, x1] x [y0, y1] (inclusive)."""
        rows = []
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                s = self.state(x, y)
                if s == MINE_HIT:
                    ch = "X"
                elif s == WRONG_FLAG:
                    ch = "!"
                elif s == FLAGGED:
                    ch = "F"
                elif s == HIDDEN:
                    if show_mines and self.is_mine(x, y):
                        ch = "*"
                    else:
                        ch = "~" if self.is_locked(x, y) else "#"
                else:
                    ch = "." if s == 0 else str(s)
                row.append(ch)
            rows.append(" ".join(row))
        return "\n".join(rows)

    # -------------------------------------------------------------- helpers
    def _check_alive(self) -> None:
        if not self.alive:
            raise GameOver(f"game over: mine at {self.exploded}")

    def _place_start(self, x: int, y: int) -> None:
        self.start = (x, y)
        if self.first_click == "safe":
            self._safe_zone = frozenset({(x, y)})
        elif self.first_click == "zero":
            self._safe_zone = frozenset({(x, y), *neighbors(x, y)})
        self._sector_mines.clear()  # the safe zone changes the mine sets
        
    def _lock(self, sector: tuple[int, int]) -> None:
        """Lock a sector and mark its incorrect flags."""
        self.locked.add(sector)
        wrong = {c for c in self.sector_cells(*sector)
                 if c in self.flagged and not self.is_mine(*c)}
        self.flagged -= wrong
        self.wrong_flags |= wrong

    def _flood(self, x: int, y: int) -> list[tuple[int, int]]:
        opened: list[tuple[int, int]] = []
        queue = deque([(x, y)])
        seen = {(x, y)}
        while queue:
            c = queue.popleft()
            if not self.is_unknown(c) or self.is_locked(*c):
                continue
            n = self.number(*c)
            self.revealed[c] = n
            opened.append(c)
            if len(opened) > self.max_flood:
                raise RuntimeError(
                    "flood fill exceeded max_flood; density may be too low"
                )
            if n == 0:
                for nb in neighbors(*c):
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        return opened


_COLORS = {
    1: "\033[94m", 2: "\033[32m", 3: "\033[91m", 4: "\033[34m",
    5: "\033[31m", 6: "\033[36m", 7: "\033[35m", 8: "\033[90m",
}
_RESET = "\033[0m"


def render_labeled(
    board: Board,
    cx: int,
    cy: int,
    view: int,
    show_mines: bool = False,
    color: bool = True,
) -> str:
    """Board view centered on (cx, cy) with x labels on top, y labels on the
    left, and `|` / `-` lines on sector boundaries."""
    xs = range(cx - view, cx + view + 1)
    ys = range(cy - view, cy + view + 1)
    s = board.sector_size

    def paint(ch: str, code: str) -> str:
        return f"{code}{ch}{_RESET}" if color else ch

    head = "     " + "".join(
        ("|" if x % s == 0 else " ") + f"{x:>3}" for x in xs
    )
    lines = [head]
    for y in ys:
        if y % s == 0:
            lines.append("     " + "".join(
                ("+" if x % s == 0 else "-") + "---" for x in xs
            ))
        row = [f"{y:>4} "]
        for x in xs:
            st = board.state(x, y)
            if st == MINE_HIT:
                ch = paint("X", "\033[41m")
            elif st == WRONG_FLAG:
                ch = paint("!", "\033[95m")
            elif st == FLAGGED:
                ch = paint("F", "\033[93m")
            elif st == HIDDEN:
                if show_mines and board.is_mine(x, y):
                    ch = paint("*", "\033[31m")
                elif board.is_locked(x, y):
                    ch = paint("~", "\033[90m")
                else:
                    ch = "#"
            elif st == 0:
                ch = "."
            else:
                ch = paint(str(st), _COLORS[st])
            row.append(("|" if x % s == 0 else " ") + "  " + ch)
        lines.append("".join(row))
    return "\n".join(lines)


HELP = """commands (x = column, y = row; y grows downward):
  r x y - reveal            f x y - flag / unflag
  c x y - chord             v x y - move the view to center on (x, y)
  m ----- show/hide mines   h ----- help        q ----- quit
  symbols: # hidden         . empty        F flag      
           X mine you hit   ! wrong flag   ~ locked cell"""


def play(seed: int = 0, view: int = 10, color: bool = True, **kwargs) -> None:
    """Terminal game for checking the rules by hand."""
    board = Board(seed, **kwargs)
    show = False
    cx, cy = 0, 0
    msg = HELP
    while True:
        print(render_labeled(board, cx, cy, view, show, color))
        print(f"score: {board.score():g}   "
              f"cleared sectors: {len(board.cleared_sectors())}   "
              f"locked: {len(board.locked)}   flags: {len(board.flagged)}   "
              f"view center: ({cx}, {cy})")
        if msg:
            print(msg)
        if not board.alive:
            print(f"BOOM: mine at {board.exploded}")
            return
        cmd = input("> ").split()
        msg = ""
        if not cmd:
            continue
        op = cmd[0]
        if op == "q":
            return
        if op == "h":
            msg = HELP
            continue
        if op == "m":
            show = not show
            continue
        try:
            x, y = int(cmd[1]), int(cmd[2])
        except (IndexError, ValueError):
            msg = "need two integers, e.g. 'r 3 -2'. Type h for help."
            continue
        if op == "v":
            cx, cy = x, y
        elif op in "rf" and board.is_locked(x, y):
            msg = f"({x}, {y}) is in a locked sector"
        elif (op in "rf" and board.start is not None and board.adjacent_only
              and (x, y) not in board.revealed
              and not board.touches_revealed(x, y)):
            msg = f"({x}, {y}) doesn't touch a revealed cell"
        elif op == "r":
            if (x, y) in board.revealed:
                msg = f"({x}, {y}) is already revealed"
            elif (x, y) in board.flagged:
                msg = f"({x}, {y}) is flagged; unflag it first"
            else:
                first = board.start is None
                hits = len(board.hits)
                opened = board.reveal(x, y)
                if first:
                    cx, cy = x, y
                if len(board.hits) > hits:
                    msg = ("" if not board.alive else
                           f"mine at ({x}, {y}): sector {board.sector_of(x, y)} is locked")
                else:
                    msg = f"revealed {len(opened)} cell(s)"
        elif op == "f":
            if (x, y) in board.revealed:
                msg = f"({x}, {y}) is revealed; only hidden cells can be flagged"
            else:
                board.toggle_flag(x, y)
                msg = f"{'flagged' if (x, y) in board.flagged else 'unflagged'} ({x}, {y})"
        elif op == "c":
            if (x, y) not in board.revealed:
                msg = f"({x}, {y}) is not a revealed number"
            else:
                opened = board.chord(x, y)
                msg = (f"chord opened {len(opened)} cell(s)" if opened
                       else "chord did nothing: flag count doesn't match the number")
        else:
            msg = f"unknown command '{op}'. Type h for help."


if __name__ == "__main__":
    import sys
    # usage: python -m inf_minesweeper.board [seed] [endless|hardcore]
    play(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
         mode=sys.argv[2] if len(sys.argv) > 2 else "endless",
         color=sys.stdout.isatty())
