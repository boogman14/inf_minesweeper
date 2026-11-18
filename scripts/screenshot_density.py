"""Estimate the app's mine density from screenshots of cleared sectors.

Usage: python scripts/screenshot_density.py data/screenshots/*.png

The work is done by inf_minesweeper.screenshots; this script only prints it.
"""

import sys

from inf_minesweeper.screenshots import density_summary

if __name__ == "__main__":
    s = density_summary(sys.argv[1:])
    print(f"cleared sectors: {s['sectors']}")
    print(f"mines: {s['mines']} of {64 * s['sectors']} cells")
    print(f"density: {s['density']:.3f}  (95% interval {s['low']:.3f} to {s['high']:.3f})")
    print(f"mines per sector: min {s['min']}, max {s['max']}, mean {s['mean']:.2f}")
    print(f"variance: {s['variance']:.2f}  (binomial would be {s['binomial_variance']:.2f})")
