"""Acceptance check: the MBB half-beam must reproduce the compliances published in
Andreassen et al., Struct. Multidisc. Optim. 43 (2011), Fig. 3 (volfrac 0.5, p = 3,
r_min = 0.04 * nelx; ft = 1 sensitivity filter, ft = 2 density filter).
Writes mbb_results.json and mbb_benchmark.png into the current directory; exits non-zero on failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topopt.top2d import top2d  # noqa: E402
from topopt.style import BEIGE, INK, DENSITY_CMAP  # noqa: E402

PAPER = {(60, 20, 1): 216.81, (60, 20, 2): 233.71, (150, 50, 1): 219.52, (150, 50, 2): 235.73}
TOL = 1e-4          # relative; the paper prints 5 significant digits

rows, ok = [], True
fig, axs = plt.subplots(2, 2, figsize=(11, 4.6), facecolor=BEIGE)
for ax, (nx, ny, ft) in zip(axs.T.flat, PAPER):
    r = top2d(nx, ny, 0.5, 3.0, 0.04 * nx, ft, verbose=False)
    ref = PAPER[(nx, ny, ft)]
    err = r["c"] / ref - 1
    ok &= abs(err) < TOL
    rows.append(dict(nelx=nx, nely=ny, ft=ft, c=r["c"], paper=ref, rel_err=err, iters=r["iters"]))
    print(f"{nx}x{ny} ft={ft}: c = {r['c']:.4f}  paper {ref:.2f}  rel. err {err:+.2e}  ({r['iters']} it.)", flush=True)
    ax.imshow(np.hstack([r["xPhys"][:, ::-1], r["xPhys"]]), cmap=DENSITY_CMAP, vmin=0, vmax=1,
              interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"{nx}×{ny}, {'sensitivity' if ft == 1 else 'density'} filter: c = {r['c']:.2f} "
                 f"(paper {ref:.2f}, {100 * err:+.3f} %)", fontsize=9, color=INK)
fig.suptitle("MBB beam, V* = 0.5, p = 3, r_min = 0.04·nelx  ·  approximate FEA, design proposal",
             color=INK, fontsize=10)
fig.tight_layout(); fig.savefig("mbb_benchmark.png", dpi=160, facecolor=BEIGE)
json.dump(rows, open("mbb_results.json", "w"), indent=1)
print("MBB PASS" if ok else "MBB FAIL")
sys.exit(0 if ok else 1)
