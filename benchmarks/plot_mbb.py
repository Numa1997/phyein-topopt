import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from topopt.style import BEIGE, INK, DENSITY_CMAP
paper = {(60,1):216.81,(60,2):233.71,(150,1):219.52,(150,2):235.73}
fig, axs = plt.subplots(2, 2, figsize=(11, 4.6), facecolor=BEIGE)
for ax,(nx,ny,ft) in zip(axs.flat,[(60,20,1),(150,50,1),(60,20,2),(150,50,2)]):
    d = np.load(f"results/mbb/mbb_{nx}x{ny}_ft{ft}.npz"); x = d["xPhys"]
    full = np.hstack([x[:, ::-1], x])          # mirror the half-beam about the symmetry line
    ax.imshow(full, cmap=DENSITY_CMAP, vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
    c = float(d["c"]); ref = paper[(nx,ft)]
    ax.set_title(f"{nx}×{ny}, {'sensitivity' if ft==1 else 'density'} filter: c = {c:.2f} (paper {ref:.2f}, {100*(c/ref-1):+.3f} %)", fontsize=9, color=INK)
fig.suptitle("MBB beam, volume fraction 0.5, p = 3, r_min = 0.04·nelx  ·  approximate FEA, design proposal", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig("results/mbb/mbb_benchmark.png", dpi=160, facecolor=BEIGE)
