"""Shared colours for renders: beige paper, wine-red material (site palette)."""
from matplotlib.colors import LinearSegmentedColormap
BEIGE = "#f4ecdf"; WINE = "#7a1f2b"; INK = "#2b2522"
DENSITY_CMAP = LinearSegmentedColormap.from_list("beige_wine", [BEIGE, "#c98b7f", WINE])
