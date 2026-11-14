# Learning to Guess in Infinite Minesweeper

Final project for a data science course. Minesweeper bots are usually
hand-coded constraint solvers. This project asks a different question: when no
safe move can be deduced, how well can a model that sees only a local window
of the board estimate the probability that a hidden cell is a mine?

The game modeled here is Infinite Minesweeper: an unbounded board, split into
10×10 sectors. A sector is cleared when all of its mines are flagged correctly.

## Questions

1. How close can a model trained on a k×k window get to the exact mine
   probability on cells that require a guess?
2. How does that accuracy change as the window grows from 3×3 to 9×9?
3. Does a better probability estimate clear more sectors per game?

## Status

| Module | Purpose | State |
| --- | --- | --- |
| `board.py` | Infinite board, game rules, sectors, terminal game | Done |
| `solver.py` | Deterministic solver that plays every simple certain move | Done |
| `oracle.py` | Exact mine probability for every frontier cell | Done |
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

Play in the terminal (optional seed as an argument):

```bash
python -m inf_minesweeper.board 42
```

Commands: `r x y` reveal, `f x y` flag or unflag, `c x y` chord, `v x y` move
the view, `m` show mines, `h` help, `q` quit. x is the column, y is the row,
and y grows downward.

Solve a board and get exact probabilities at the stuck position:

```python
from inf_minesweeper.board import Board, render_labeled
from inf_minesweeper.solver import Solver, sector_region, autoplay
from inf_minesweeper.oracle import frontier_probabilities

board = Board(seed=0, density=0.2)
solver = Solver(board, sector_region(board, radius=1, center=(5, 5)))
solver.reveal(5, 5)          # first click
solver.solve()               # every simple certain move
print(render_labeled(board, 5, 5, view=14))

probs = frontier_probabilities(solver)   # {cell: P(mine)}
print(min(probs.items(), key=lambda kv: kv[1]))  # safest guess

print(autoplay(Board(seed=1)))           # full game with random guesses
```

Run the tests:

```bash
pytest
```

## How it works

**Board.** Mines are never stored. Whether a cell holds a mine is a seeded
hash of its coordinates compared with the density, so memory grows only with
the cells touched and any game replays exactly from its seed. The first-click
rule is configurable: `none`, `safe`, or `zero`.

**Solver.** Sees only what a player sees. It applies two sound rules: a single
number whose remaining neighbors are all safe or all mines, and pairs of
overlapping numbers. It works inside a box of sectors, because on an infinite
board deduction almost never stalls everywhere at once.

**Oracle.** Mines are independent and there is no global mine count, so
frontier cells that share no number are independent. Each connected piece is
solved exactly by enumerating the layouts consistent with the numbers,
weighted by density. Cells that touch the same numbers are merged into groups,
which keeps the search fast. Pieces that exceed a search budget return NaN.

## Findings so far

All at density 0.2, which is a placeholder until the app's real density is
measured.

- An unbounded solver does not stop: runs were still going past 100,000
  revealed cells. Games are therefore played in a 5×5 block of sectors.
- Random guessing ends games quickly: a median of 1 guess per game, with 20 of
  25 sectors cleared. At density 0.25 the median drops to 5 sectors.
- The oracle is calibrated against the true mines. Over 1,500 stuck positions
  (about 57,600 cells), the mean predicted probability and the actual mine
  rate are both 0.4001.
- The solver's two rules are incomplete. In 63% of stuck positions the oracle
  finds a cell that is certainly safe. A forced guess is therefore defined by
  the oracle (no certain-safe cell left), not by the solver stalling.
- The oracle takes a median of 4 ms per position; 0.4% of cells hit the
  search budget.

## Layout

```
src/inf_minesweeper/   package code
tests/                 pytest tests
notebooks/             exploration and modeling notebooks
data/                  simulated datasets (not tracked)
results/               figures and tables (not tracked)
```

## Next steps

1. Estimate the app's mine density from screenshots of cleared sectors.
2. `simulate.py`: reveal oracle-certain cells, log each forced-guess position
   with its game id and seed, split train and test by game.
3. Baselines: constant density, logistic regression on simple features.
4. Tree ensembles and a CNN; evaluate by excess log-loss over the oracle,
   calibration, guess regret, and sectors cleared.
