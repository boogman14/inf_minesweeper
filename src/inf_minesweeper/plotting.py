"""plotting.py -- shared figure style for notebooks and the report.

Call `use_style()` once, then build figures w/ plain matplotlib. Colors are named by role
so every figure in the project agrees: SERIES[0] and SERIES[1] for two compared groups, 
SEQ for magnitude, DIV for signed difference
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Categorical order: identity of a group
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Sequential (magnitude): one hue, light to dark
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

# Diverging (signed): blue - neutral gray - red
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2f"]
)


def use_style() -> None:
    """Quiet, report-ready defaults: hairline axes, light grid, no top/right spines, system sans."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
        "axes.prop_cycle": cycler(color=SERIES),
    })
    
    
def no_grid(ax) -> None:
    """For image-like panels (heatmaps, board windows)."""
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def save(fig, name: str, folder: str = "../results/figures") -> str:
    """Save a figure as PNG under `folder` and return the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path)
    return path