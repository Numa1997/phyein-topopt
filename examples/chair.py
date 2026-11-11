"""Chair load case ("seat and back"): which material does a chair need?

Physical setup (full chair; the code models the half x >= 0 and uses mirror symmetry):
  * design box W x D x Hc (width x depth x height), voxel side a = (W/2) / n
  * seat: a solid plate of thickness t_seat whose top is at height h_seat
  * backrest zone: the last d_back of the depth above the seat; its front face is a solid
    skin of thickness t_skin (where the sitter's back presses); behind it, design material
  * the space above the seat in front of the backrest stays empty (the sitter is there)
  * feet: square patches of side s_foot at the four floor corners, fully fixed
  * load case 1 "sit":       seat force F_seat downwards, spread uniformly over the seat top
  * load case 2 "lean back": F_seat on the seat plus F_back pushing the backrest backwards,
                             spread over the upper half of the backrest face
Objective: minimise C = C_sit + C_lean at material volume fraction V* of the design region.
Stresses are independent of E0 in linear elasticity (u ~ 1/E0, sigma = E0 B u); E0 only sets
deflections, which are reported as delta * E0 [N/m] so that no material value is assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topopt.top3d import Grid, Problem, optimise, element_von_mises  # noqa: E402


def face_nodes_weights(g: Grid, i_rng, j_rng, k_rng):
    """Nodes of a rectangular node patch and consistent weights of a uniform traction (sum 1).
    One of the three ranges must be a single index (the face normal direction)."""
    I, J, K = np.meshgrid(np.arange(*i_rng), np.arange(*j_rng), np.arange(*k_rng), indexing="ij")
    def w1(r):
        n = r[1] - r[0]; w = np.ones(n)
        if n > 1: w[[0, -1]] = 0.5
        return w
    W = w1(i_rng)[:, None, None] * w1(j_rng)[None, :, None] * w1(k_rng)[None, None, :]
    return g.node_id(I.ravel(), J.ravel(), K.ravel()), (W / W.sum()).ravel()


def build(n=12, W=0.45, D=0.45, Hc=0.90, h_seat=0.45, t_seat=0.025, d_back=0.10, t_skin=0.025,
          s_foot=0.04, F_seat=1000.0, F_back=300.0):
    a = (W / 2) / n
    nx, ny, nz = n, int(round(D / a)), int(round(Hc / a))
    g = Grid(nx, ny, nz, a)
    ks = int(round(h_seat / a))                           # seat top node layer
    kt = max(1, int(round(t_seat / a)))
    jb = ny - max(1, int(round(d_back / a)))              # first element row of the backrest zone
    js = max(1, int(round(t_skin / a)))
    f = max(1, int(round(s_foot / a)))

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    seat = (k >= ks - kt) & (k < ks)
    skin = (j >= jb) & (j < jb + js) & (k >= ks)
    void = (j < jb) & (k >= ks)
    passive_solid = np.flatnonzero(seat | skin)
    passive_void = np.flatnonzero(void)

    ndof = 3 * g.nnode
    loads = np.zeros((ndof, 2))
    # half model: half of each total force; the symmetry-plane nodes get half weight automatically
    n_s, w_s = face_nodes_weights(g, (0, nx + 1), (0, jb + 1), (ks, ks + 1))
    for L in (0, 1):
        loads[3 * n_s + 2, L] -= 0.5 * F_seat * w_s
    kb0 = ks + (nz - ks) // 2
    n_b, w_b = face_nodes_weights(g, (0, nx + 1), (jb, jb + 1), (kb0, nz + 1))
    loads[3 * n_b + 1, 1] += 0.5 * F_back * w_b

    sym = g.node_id(*[x.ravel() for x in np.meshgrid(0, np.arange(ny + 1), np.arange(nz + 1), indexing="ij")])
    feet = []
    for j0, j1 in ((0, f + 1), (ny - f, ny + 1)):
        I, J = np.meshgrid(np.arange(nx - f, nx + 1), np.arange(j0, j1), indexing="ij")
        feet.append(g.node_id(I.ravel(), J.ravel(), 0))
    feet = np.concatenate(feet)
    fixed = np.unique(np.concatenate([3 * sym, 3 * feet, 3 * feet + 1, 3 * feet + 2]))
    meta = dict(n=n, a=a, nx=nx, ny=ny, nz=nz, ks=ks, kt=kt, jb=jb, js=js, f=f, kb0=kb0, W=W, D=D, Hc=Hc,
                F_seat=F_seat, F_back=F_back, load_nodes_seat=n_s.tolist(), load_nodes_back=n_b.tolist())
    return Problem(g, loads, fixed, passive_solid, passive_void), meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="voxels across the half width")
    ap.add_argument("--volfrac", type=float, default=0.12)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--rmin", type=float, default=1.5, help="filter radius in voxels")
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--beta_max", type=float, default=16.0)
    ap.add_argument("--beta_every", type=int, default=40)
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--out", default="results/chair")
    A = ap.parse_args()
    prob, meta = build(A.n)
    g = prob.grid
    print(f"grid {g.nx}x{g.ny}x{g.nz} (half chair), a = {meta['a']*1000:.1f} mm, "
          f"{g.nel} elements, {3*g.nnode} DOFs", flush=True)
    E0 = 1.0                                                 # see docstring: stresses do not depend on E0
    r = optimise(prob, A.volfrac, A.penal, A.rmin, E0=E0, nu=A.nu, beta_max=A.beta_max,
                 beta_every=A.beta_every, maxit=A.maxit)
    vm = element_von_mises(g, r["U"], r["xPhys"], E0, A.nu)          # Pa, per load case
    # stress field of the starting (uniform-density) block, for the "stress" stage of the animation
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    U = r["U"]
    tip = {f"lc{L+1}_max_disp_times_E0_N_per_m": float(np.abs(U[:, L]).max() * E0) for L in range(2)}
    np.savez_compressed(out / "chair_result.npz", xPhys=r["xPhys"], x=r["x"], U=U, vm=vm,
                        hist=np.stack(r["hist"]), shape=np.array([g.nx, g.ny, g.nz]), a=g.a,
                        passive_solid=prob.passive_solid, passive_void=prob.passive_void)
    json.dump(dict(meta=meta, params=vars(A), iters=r["iters"], c=r["c"], log=r["log"], **tip),
              open(out / "chair_run.json", "w"), indent=1)
    print(f"done: {r['iters']} iterations, C = {r['c']:.4e}, saved to {out}")
