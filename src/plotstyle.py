"""Shared matplotlib style so every report figure looks consistent."""
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files only
import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # single-series colour
ORANGE = "#eb6834"    # second series / highlight
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"    # single-hue sequential map for heatmaps


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
