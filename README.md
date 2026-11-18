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

In the app, a locked sector reopens once the sectors around it are solved.
This project leaves locked sectors locked: the aim is to measure how good each
guess is, and a lock that can be undone would blur that.

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
| `screenshots.py` | Reads cleared sectors from app screenshots; `scripts/screenshot_density.py` prints the density estimate | Done |
| `simulate.py` | Plays games and logs forced-guess positions as data | Done |
| `loader.py` | Loads the dataset, splits by game, crops windows | Done |
| `features.py` | Handcrafted features, one-hot windows, CNN channels, rotations and reflections | Done |
| `models.py` | Baselines done (constant, local ratio, logistic); tree ensembles and CNN planned | In progress |
| `plotting.py` | Shared figure style and colors for notebooks and the report | Done |
| `notebooks/01_exploration.ipynb` | Dataset exploration; saves figures to `results/figures/` | Done |
| `metrics.py` | Log-loss, Brier, excess loss over the oracle, calibration, guess quality | Done |

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv --prompt logic
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` pins the newest versions that still support Python 3.10.
PyTorch is used for the CNN. To train on an NVIDIA GPU, install the CUDA build
and check that it sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On a machine without a GPU, the much smaller CPU-only build works too:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

XGBoost can also train on the GPU (`device="cuda"`); LightGBM runs on the CPU.

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
from inf_minesweeper.board import Board, render_labeled, tenths
from inf_minesweeper.solver import Solver, sector_region, autoplay
from inf_minesweeper.oracle import frontier_proba

board = Board(seed=0)        # density defaults to the measured 0.21
solver = Solver(board, sector_region(board, radius=1, center=(4, 4)))
solver.reveal(4, 4)          # first click
solver.solve()               # every simple certain move

probs = frontier_proba(solver)   # {cell: P(mine)}
best = min(probs, key=probs.get) # safest guess
print(best, probs[best])

# Highlight cells without changing the game: one cell, or every candidate
# with its mine probability in tenths (0.14 shows as 1, 0.68 as 6).
print(render_labeled(board, 4, 4, view=14, marks=[best]))
print(render_labeled(board, 4, 4, view=14, marks=tenths(probs)))

print(autoplay(Board(seed=1)))           # full endless game, random guesses
```

Generate a dataset (a fixed number of games, or as many as fit in a time
budget; uses all CPU cores):

```bash
python -m inf_minesweeper.simulate --minutes 5 --out data/raw/guesses.npz
python -m inf_minesweeper.simulate --games 2000 --out data/raw/guesses.npz
```

Load it:

```python
from inf_minesweeper.loader import load, split_by_game, crop

d = load("data/raw/guesses.npz")             # "../data/raw/..." from notebooks/
train, test = split_by_game(d["game"])       # whole games on each side
X = crop(d["window"], 5)                     # 5x5 windows from the stored 9x9
y, q = d["mine"], d["q"]                     # label and oracle probability
```

Run the tests (26 tests, about 20 seconds):

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

## Terminology

A **forced guess** here is a position where no candidate cell is provably
safe, so some cell must be guessed to continue. The Minesweeper Wiki uses the
term more narrowly, for a region cut off from all further information, which
must be guessed no matter what else is solved. Many cells in this project's
forced-guess positions could still be resolved later by information from
elsewhere; the position as a whole cannot move without a guess.

Probabilities quoted on the wiki, such as two-thirds survival in a finned
box, assume a board with a fixed total number of mines. On an infinite board
with independent mines there is no total, so layouts with more mines are less
likely by a factor of p/(1 - p) each, and the same pattern can have different
odds.

## Mine density in the app

Estimated from four screenshots of cleared sectors, where every hidden cell is
a flagged mine (`inf_minesweeper.screenshots` locates the sector lines and
classifies each cell). Sectors containing anything else, such as unfinished
cells or icons, were dropped.

- 58 distinct cleared sectors, 786 mines in 3,712 cells: density 0.212, with
  a 95% interval of about 0.20 to 0.23. The simulator uses 0.21.
- Mines per sector range from 6 to 21, with mean 13.6 and variance 9.8. A
  binomial count at this density has variance 10.7, so the data fit
  independent placement and rule out a fixed number of mines per sector.
- Selection bias is small. Only cleared sectors are counted, and dense
  sectors are more likely to end up locked. In simulation at a true density
  of 0.21, locked sectors average 0.25 and cleared sectors 0.2076, against
  0.2085 for all sectors: a bias of about -0.001 when 2 to 3% of sectors are
  locked, growing to roughly -0.005 if 10% are. Both are inside the sampling
  error of the estimate (0.007), so 0.21 stands.

## The dataset

`simulate.py` plays endless games in a 7x7 block of sectors and scores the
inner 5x5, because sectors on the edge of the play box cannot always be
finished. In each game the solver plays its certain moves, then the oracle
flags and reveals every cell it can prove. When nothing certain is left, the
position is a forced guess and every candidate cell becomes one row: its 9x9
window of visible states, its oracle probability, and whether it is a mine.
The guess itself is the oracle's safest cell, or a random candidate 20% of
the time.

The current dataset comes from 5,000 games, seeds 0 to 4,999
(`python -m inf_minesweeper.simulate --games 5000`, about 7.5 minutes):

- 62,701 forced-guess positions and 2,063,361 rows: about 33 candidates per
  position and 12.6 guesses per game. The file is 11.0 MB.
- Mean oracle probability 0.4220, actual mine rate 0.4228.
- Log-loss on these rows: 0.587 for the oracle, 0.681 for a constant equal to
  the mine rate, 0.796 for a constant equal to the density. Learned models
  should land between the first two; the gap of about 0.094 is what they can
  win.
- Per game, 24.2 of 25 inner sectors cleared, 0.3 locked, mean score 320.
  Mines hit average 3.1 per game, most of them in the outer ring of the play
  box, which is not scored.
- 72,031 candidate cells (3.4%) were dropped because their component exceeded
  the oracle's search budget of 200,000 nodes.
- Cleared sectors have density 0.2082 and locked sectors 0.2569, against a
  true 0.21: the selection bias described above, measured on this data.
- A few percent of games end early: the first guess locks the starting sector
  before the opening reaches its edge, leaving nothing playable. These games
  are kept as they are, since locked sectors never reopen here.
- Splitting by game with a 20% test share gives 4,000 training games and
  1,000 test games.

## Baseline results

Preliminary: trained on 280 games and tested on 70 held-out games (31,303
rows) from an earlier 350-game sample. To be rerun on the current dataset.
"Excess" is the expected log-loss above the oracle's, so 0 is perfect. "Gap 
closed" is measured from the mine-rate constant to the oracle. "Hit rate" is 
the chance of a mine when the model picks the guess.

| Model | Log-loss | Excess | Gap closed | Hit rate |
| --- | --- | --- | --- | --- |
| Constant, board density 0.21 | 0.802 | 0.217 | | 0.44 |
| Constant, training mine rate | 0.683 | 0.099 | 0% | 0.44 |
| Local ratio (no fitting) | 0.654 | 0.069 | 30% | 0.31 |
| Logistic regression, 8 features | 0.646 | 0.061 | 38% | 0.30 |
| Oracle | 0.589 | 0 | 100% | 0.20 |

The local ratio reads each touching number on its own, as mines still needed
over unknown cells around it, and averages them. Constant models cannot rank
cells, so their hit rate is that of a random pick.

## Findings so far

Generated by `scripts/findings.py` (1,000 games per policy and density, seeds 0 to 999).

- An unbounded solver does not stop: runs were still going past 100,000
  revealed cells. Games are therefore played in a 7×7 block of sectors and
  scored on the inner 5×5.
- Guess quality shows up in the score. Over 1,000 endless games at density
  0.21, with every certain move played and only the forced guesses
  differing, random guessing clears a mean of 23.0 of 25 sectors,
  locks 0.62, and scores 302; always taking the oracle's safest
  cell clears 24.2, locks 0.31, and scores 321. The paired
  difference is 18.8 ± 4.7 points per game (95%). The gap between the
  two policies is what the learned models compete for.
- The oracle's lead holds across the uncertainty in the measured density.
  Scores rise with density because a cleared sector earns one point per
  mine.

  | Density | Policy | Cleared | Locked | Score |
  | --- | --- | --- | --- | --- |
  | 0.20 | random | 23.3 | 0.35 | 294 |
  | 0.20 | oracle | 24.5 | 0.19 | 310 |
  | 0.21 | random | 23.0 | 0.62 | 302 |
  | 0.21 | oracle | 24.2 | 0.31 | 321 |
  | 0.22 | random | 22.2 | 1.06 | 302 |
  | 0.22 | oracle | 24.1 | 0.43 | 333 |

- The oracle is calibrated against the true mines. Over 2,063,361 candidate
  cells, the mean predicted probability is 0.422 and the actual mine rate
  is 0.423; across 20 probability bins the largest gap is 0.011.
- The solver's two rules are incomplete. In 15% of the 7,323 positions
  where the solver is stuck, the oracle still finds a cell that is
  certainly safe. A forced guess is therefore defined by the oracle (no
  certain-safe cell left), not by the solver stalling.
- The oracle takes a median of 15 ms per stuck position (95th
  percentile 219 ms); 2.7% of candidate cells exceed its search
  budget of 200,000 nodes and are left out of the data.

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

1. Tree ensembles and a CNN; evaluate by excess log-loss over the oracle,
   calibration, guess regret, and game score.
2. Sensitivity check: rerun the main results at densities 0.20 and 0.22.

## Later ideas

Outside the scope of this project:

- Timed mode: clear as many sectors as possible before the clock runs out.
  This turns guessing into a speed trade-off, since a slow exact answer can
  score worse than a fast approximate one.
- First-click initialization: how the first-click rule biases mine density
  near the start, and whether models lose calibration there.
- Endless gameplay: circle back to locked sectors by clearing neighboring
  sectors. Requires major changes to the scoring system.
- Playable front end with a model-assist overlay: a GUI over `board.py` that
  shows the model's mine probabilities on the frontier, or lets the model
  play while you watch. infinisweeper (reference 6) is a working example of
  an infinite-scrolling board in Pygame Zero.
- Game themes: customize the colorway for the numbers, markers, etc. Let the
  models have a little fun of their own.

## License

The code is released under the Apache License 2.0; see `LICENSE`. The app
screenshots in `data/screenshots/` are from Infinite Minesweeper and are
included only as data for the density estimate; they are not covered by this
license.

## References

1. Infinite Minesweeper (iOS game), App Store page:
   https://apps.apple.com/app/id6476042029
   The game this project models: sector scoring, the first-click rule, and the
   mine density used by the simulator.
2. Seeded, "Minesweeper Cheats on Your First Click" (YouTube Short):
   https://www.youtube.com/shorts/RcypVbPEprE
   How Minesweeper adjusts the board on the first click; background for the
   `first_click` option.
3. Minesweeper Wiki, "Forced Guess":
   https://minesweeper.wiki.gg/wiki/Forced_Guess
   Definitions and examples of forced guesses (50/50, pseudo 50/50, finned
   box). Note that this project uses "forced guess" more broadly; see
   "Terminology" above.
4. Minesweeper Wiki, Guessing category:
   https://minesweeper.wiki.gg/wiki/Category:Guessing
   Pages on guessing, 50/50s, and pseudo 50/50s.
5. Minesweeper Wiki, Patterns category:
   https://minesweeper.wiki.gg/wiki/Category:Patterns
   Easy, medium, and evil deduction patterns; background for what the
   solver's two rules can and cannot catch.
6. lordmauve, infinisweeper (GPL-3.0):
   https://github.com/lordmauve/infinisweeper
   An infinite-scrolling Minesweeper in Python and Pygame Zero; reference for
   a playable front end. Its code is GPL-3.0, so a front end that copies it
   must also be released under GPL-3.0.

<!-- Add further references below: papers, videos, solver write-ups. -->
