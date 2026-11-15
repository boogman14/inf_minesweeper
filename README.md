# Learning to Guess in Infinite Minesweeper

Final project for a data science course. Minesweeper bots are usually
hand-coded constraint solvers. This project asks a different question: when no
safe move can be deduced, how well can a model that sees only a local window
of the board estimate the probability that a hidden cell is a mine?

The game modeled here is Infinite Minesweeper: an unbounded board, split into
8×8 sectors. A sector is cleared when all of its mines are flagged correctly.

The project models the game's endless mode. Revealing a mine locks that
mine's sector for good: nothing in it can be revealed or flagged again, its
incorrect flags are marked, and what was already revealed stays visible as
clues for neighboring sectors. Openings that spread from other sectors stop at
its edge. Play continues elsewhere. A hardcore mode, where one mine ends the
game, is also available (`mode="hardcore"`).

Scoring: a cleared sector earns one point per mine in it, as in the app. A
locked sector is lost permanently. The app has no further penalty; this
project subtracts 5 points per locked sector (`lock_penalty` on `Board`) so
that a bad guess costs more than the points it forfeits.

## Questions

1. How close can a model trained on a k×k window get to the exact mine
   probability on cells that require a guess?
2. How does that accuracy change as the window grows from 3×3 to 9×9?
3. Does a better probability estimate raise the game score (more sectors
   cleared, fewer locked)?

## Status

| Module | Purpose | State |
| --- | --- | --- |
| `board.py` | Infinite board, endless and hardcore rules, sectors, scoring, terminal game | Done |
| `solver.py` | Deterministic solver that plays every simple certain move; full-game autoplay | Done |
| `oracle.py` | Exact mine probability for every frontier cell | Done |
| `scripts/screenshot_density.py` | Mine density from app screenshots | Done |
| `simulate.py` | Plays games and logs forced-guess positions as data | Next |
| `features.py` | k×k windows, one-hot encoding, rotations and reflections | Planned |
| `models.py` | Logistic regression, tree ensembles, CNN | Planned |
| `metrics.py` | Log-loss, Brier, excess loss over the oracle, calibration | Planned |

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv --prompt logic
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

If ROS 2 is sourced in the shell, its pytest plugins break the venv's pytest.
Disable plugin auto-loading for this project:

```bash
echo 'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1' >> .venv/bin/activate
```

## Usage

Play in the terminal (optional seed and mode as arguments):

```bash
python -m inf_minesweeper.board 42
python -m inf_minesweeper.board 42 hardcore
```

Commands: `r x y` reveal, `f x y` flag or unflag, `c x y` chord, `v x y` move
the view, `m` show mines, `h` help, `q` quit. x is the column, y is the row,
and y grows downward. After the first click, only cells that touch a revealed
cell can be played, as in the app.

Solve a board and get exact probabilities at the stuck position:

```python
from inf_minesweeper.board import Board, render_labeled
from inf_minesweeper.solver import Solver, sector_region, autoplay
from inf_minesweeper.oracle import frontier_proba

board = Board(seed=0)        # density defaults to the measured 0.21
solver = Solver(board, sector_region(board, radius=1, center=(4, 4)))
solver.reveal(4, 4)          # first click
solver.solve()               # every simple certain move
print(render_labeled(board, 4, 4, view=14))

probs = frontier_proba(solver)   # {cell: P(mine)}
print(min(probs.items(), key=lambda kv: kv[1]))  # safest guess

print(autoplay(Board(seed=1)))           # full endless game, random guesses
```

Run the tests (17 tests, about 15 seconds):

```bash
pytest
```

Re-estimate the mine density from screenshots (needs Pillow):

```bash
python scripts/screenshot_density.py data/screenshots/*.png
```

Check that a copy of the code reproduces the reference board and solver:

```bash
python -c "
from inf_minesweeper.board import Board
from inf_minesweeper.solver import Solver, sector_region
b = Board(0, density=0.2)
print(sum(b.is_mine(x, y) for x in range(50) for y in range(50)))
S = Solver(b, sector_region(b, 1, (4, 4)))
S.reveal(4, 4); S.solve()
print(len(b.revealed), len(b.flagged), len(S.frontier()))"
```

Expected output: `544`, then `523 96 14`.

## How it works

**Board.** Mines are never stored. Whether a cell holds a mine is a seeded
hash of its coordinates compared with the density, so memory grows only with
the cells touched and any game replays exactly from its seed. The first-click
rule is configurable: `none`, `safe`, or `zero`. A cell shows one of 13 states
to the player: a number 0 to 8, hidden, flagged, mine hit, or wrong flag.

**Solver.** Sees only what a player sees. It applies two sound rules: a single
number whose remaining neighbors are all safe or all mines, and pairs of
overlapping numbers. It works inside a box of sectors, because on an infinite
board deduction almost never stalls everywhere at once. In a locked sector it
treats the mine that was hit as a known mine and marked wrong flags as known
safe cells, and it never plays there.

**Oracle.** Mines are independent and there is no global mine count, so
frontier cells that share no number are independent. Each connected piece is
solved exactly by enumerating the layouts consistent with the numbers,
weighted by density. Cells that touch the same numbers are merged into groups,
which keeps the search fast. Pieces that exceed a search budget return NaN.

## Mine density in the app

Estimated from four screenshots of cleared sectors, where every hidden cell is
a flagged mine (`scripts/screenshot_density.py` locates the sector lines and
classifies each cell). Sectors containing anything else, such as unfinished
cells or icons, were dropped.

- 58 distinct cleared sectors, 786 mines in 3,712 cells: density 0.212, with
  a 95% interval of about 0.20 to 0.23. The simulator uses 0.21.
- Mines per sector range from 6 to 21, with mean 13.6 and variance 9.8. A
  binomial count at this density has variance 10.7, so the data fit
  independent placement and rule out a fixed number of mines per sector.
- Caveat: only cleared sectors are counted. Dense sectors are more likely to
  end up locked, so the true density may be slightly higher.

## Findings so far

The policy comparison uses the measured density, 0.21. The other numbers were
run at 0.20 and have not been rerun.

- An unbounded solver does not stop: runs were still going past 100,000
  revealed cells. Games are therefore played in a 5×5 block of sectors (40×40
  cells).
- Guess quality shows up in the score. Over 100 endless games per policy in a
  25-sector box, random guessing clears a mean of 19.2 sectors, locks 3.9, and
  scores 227; always taking the oracle's safest cell clears 22.0, locks 2.0,
  and scores 277. The gap between those two is what the learned models
  compete for.
- Density matters. With random guessing, the mean number of locked sectors is
  0.6 at density 0.15, 3.0 at 0.20, and 8.1 at 0.25.
- The oracle is calibrated against the true mines. Over 1,500 stuck positions
  (about 49,500 cells), the mean predicted probability is 0.397 and the
  actual mine rate is 0.398.
- The solver's two rules are incomplete. In 56% of stuck positions the oracle
  finds a cell that is certainly safe. A forced guess is therefore defined by
  the oracle (no certain-safe cell left), not by the solver stalling.
- The oracle takes a median of 3 ms per position; 0.4% of cells hit the
  search budget.

## Layout

```
src/inf_minesweeper/   package code
tests/                 pytest tests (*_test.py)
notebooks/             exploration and modeling notebooks
scripts/               one-off analysis scripts
data/screenshots/      app screenshots used for the density estimate
data/                  simulated datasets (not tracked)
results/               figures and tables (not tracked)
```

## Next steps

1. Score only interior sectors: sectors on the edge of the play box can be
   impossible to finish because the solver may not act outside the box.
2. `simulate.py`: reveal oracle-certain cells, log each forced-guess position
   with its game id and seed, split train and test by game.
3. Baselines: constant density, logistic regression on simple features.
4. Tree ensembles and a CNN; evaluate by excess log-loss over the oracle,
   calibration, guess regret, and game score.

## Later ideas

Outside the scope of this project:

- Timed mode: clear as many sectors as possible before the clock runs out.
  This turns guessing into a speed trade-off, since a slow exact answer can
  score worse than a fast approximate one.
- First-click initialization: how the first-click rule biases mine density
  near the start, and whether models lose calibration there.

## References

1. Infinite Minesweeper (iOS game), App Store page:
   https://apps.apple.com/app/id6476042029
   The game this project models: sector scoring, the first-click rule, and the
   mine density used by the simulator.
2. Seeded, "Minesweeper Cheats on Your First Click" (YouTube Short):
   https://www.youtube.com/shorts/RcypVbPEprE
   How Minesweeper adjusts the board on the first click; background for the
   `first_click` option.

<!-- Add further references below: papers, videos, solver write-ups. -->
