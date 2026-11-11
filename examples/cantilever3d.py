"""3D cantilever, the classic test case for 3D topology optimisation (as in the top3d example of
Liu & Tovar 2014, Struct. Multidisc. Optim. 50:1175-1196, cited from memory; parameter values are
choices made here, not theirs).

Physical setup: a box of nx x ny x nz voxels of side a (x along the length L, y up, z across).
The face x = 0 is clamped (welded to a wall). A line load P, spread evenly over the nodes of the
bottom edge at the free end (x = L, y = 0, all z), pulls straight down. Hand statics: the wall
carries the shear force P and the bending moment P L; the top fibres near the wall are in tension,
the bottom fibres in compression, so the expected design is two flanges (top and bottom chords)
joined by diagonal webs, Michell-like, symmetric about the mid-plane z = W/2.
Usage: python examples/cantilever3d.py --nx 48 --ny 16 --nz 8 --volfrac 0.3 --out results/cant
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topopt.top3d import Grid, Problem, optimise, element_von_mises


def build(nx=48, ny=16, nz=8, a=0.01, P=1000.0):
    g = Grid(nx, ny, nz, a)
    j, k = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    wall = g.node_id(0, j.ravel(), k.ravel())
    fixed = np.concatenate([3 * wall, 3 * wall + 1, 3 * wall + 2])
    edge = g.node_id(nx, 0, np.arange(nz + 1))
    w = np.ones(nz + 1); w[[0, -1]] = 0.5; w /= w.sum()
    f = np.zeros((3 * g.nnode, 1)); f[3 * edge + 1, 0] = -P * w
    return Problem(g, f, fixed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=48); ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nz", type=int, default=8); ap.add_argument("--volfrac", type=float, default=0.3)
    ap.add_argument("--rmin", type=float, default=1.5); ap.add_argument("--E0", type=float, default=1650e6)
    ap.add_argument("--nu", type=float, default=0.43); ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--out", default="results/cantilever3d")
    A = ap.parse_args()
    p = build(A.nx, A.ny, A.nz)
    r = optimise(p, A.volfrac, rmin=A.rmin, E0=A.E0, Emin=1e-9 * A.E0, nu=A.nu, maxit=A.maxit)
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "result.npz", xPhys=r["xPhys"], shape=np.array([A.nx, A.ny, A.nz]), a=p.grid.a)
    json.dump(dict(params=vars(A), iters=r["iters"], c=r["c"], log=r["log"]), open(out / "run.json", "w"), indent=1)
    print(f"done: {r['iters']} iterations, C = {r['c']:.4e}")
