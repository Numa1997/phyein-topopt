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
deflections. Material: PA12 (E0 = 1650 MPa, EOS PA 2200 datasheet; nu = 0.43, HP PA12 datasheet).
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


def build(n=12, W=0.45, D=0.45, Hc=0.90, h_seat=0.45, t_seat=0.025, d_back=0.10, t_back=0.025,
          d_brace=0.05, h_brace=0.20, s_foot=0.04, d_front=0.10,
          F_service=980.0, back_ratio=0.35, F_seat_test=1600.0, F_back_test=560.0,
          F_front_test=1300.0, F_side_test=400.0):
    """Half chair (x >= 0, mirror plane x = 0). SI units. Load cases (parameter choices):
      S  service: a 100 kg sitter, F_service on the seat and back_ratio * F_service on the back
         (back_ratio 0.35 = the standard's own back/seat ratio 560/1600) -> stiffness goal
      A  seat and back static test (EN 1728, values as required by EN 1335-2 / EN 16139 L1):
         F_seat_test on the seat, F_back_test on the back                    -> strength check
      B  seat front edge static: F_front_test on a strip of depth d_front at the front edge
      C  sideways static: F_side_test along x on the seat top (antisymmetric: own support set)
    Approximations (the standard's loading pads, points and fixtures are in EN 1728, paywalled):
    loads are spread uniformly over the stated areas; the feet are fully fixed (like test stops)."""
    a = (W / 2) / n
    nx, ny, nz = n, int(round(D / a)), int(round(Hc / a))
    g = Grid(nx, ny, nz, a)
    ks = int(round(h_seat / a)); kt = max(1, int(round(t_seat / a)))
    jb = ny - max(1, int(round(d_back / a)))
    js = max(1, int(round(t_back / a)))
    jbr = min(ny, jb + js + max(1, int(round(d_brace / a))))
    kbr = min(nz, ks + int(round(h_brace / a)))
    f = max(1, int(round(s_foot / a)))
    jf = max(1, int(round(d_front / a)))

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    seat = (k >= ks - kt) & (k < ks)
    plate = (j >= jb) & (j < jb + js) & (k >= ks)
    brace = (j >= jb + js) & (j < jbr) & (k >= ks) & (k < kbr)
    passive_solid = np.flatnonzero(seat | plate)
    passive_void = np.flatnonzero((k >= ks) & ~plate & ~brace)

    ndof = 3 * g.nnode
    n_s, w_s = face_nodes_weights(g, (0, nx + 1), (0, jb + 1), (ks, ks + 1))       # seat top
    kb0 = ks + (nz - ks) // 2
    n_b, w_b = face_nodes_weights(g, (0, nx + 1), (jb, jb + 1), (kb0, nz + 1))     # upper backrest face
    n_f, w_f = face_nodes_weights(g, (0, nx + 1), (0, jf + 1), (ks, ks + 1))       # front-edge strip
    sym_loads = np.zeros((ndof, 3))                     # columns S, A, B; half model carries half
    sym_loads[3 * n_s + 2, 0] -= 0.5 * F_service * w_s
    sym_loads[3 * n_b + 1, 0] += 0.5 * back_ratio * F_service * w_b
    sym_loads[3 * n_s + 2, 1] -= 0.5 * F_seat_test * w_s
    sym_loads[3 * n_b + 1, 1] += 0.5 * F_back_test * w_b
    sym_loads[3 * n_f + 2, 2] -= 0.5 * F_front_test * w_f
    side = np.zeros((ndof, 1))
    side[3 * n_s + 0, 0] += 0.5 * F_side_test * w_s

    sym = g.node_id(*[x.ravel() for x in np.meshgrid(0, np.arange(ny + 1), np.arange(nz + 1), indexing="ij")])
    feet = []
    for j0, j1 in ((0, f + 1), (ny - f, ny + 1)):
        I, J = np.meshgrid(np.arange(nx - f, nx + 1), np.arange(j0, j1), indexing="ij")
        feet.append(g.node_id(I.ravel(), J.ravel(), 0))
    feet = np.concatenate(feet)
    feetdofs = np.concatenate([3 * feet, 3 * feet + 1, 3 * feet + 2])
    fixed_sym = np.unique(np.concatenate([3 * sym, feetdofs]))
    fixed_anti = np.unique(np.concatenate([3 * sym + 1, 3 * sym + 2, feetdofs]))
    meta = dict(n=n, a=a, nx=nx, ny=ny, nz=nz, ks=ks, kt=kt, jb=jb, js=js, f=f, kb0=kb0, jf=jf, W=W, D=D, Hc=Hc,
                F_service=F_service, back_ratio=back_ratio, F_seat_test=F_seat_test, F_back_test=F_back_test,
                F_front_test=F_front_test, F_side_test=F_side_test,
                load_cases=["S service (980 N seat + 343 N back)", "A seat+back static test (1600 + 560 N)",
                            "B seat front edge static test (1300 N)", "C sideways static test (400 N)"],
                seat_nodes=n_s.tolist())
    prob = Problem(g, sym_loads, fixed_sym, passive_solid, passive_void, extra_groups=[(fixed_anti, side)])
    return prob, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="voxels across the half width")
    ap.add_argument("--volfrac", type=float, default=0.05)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--rmin", type=float, default=2.0, help="filter radius in voxels")
    ap.add_argument("--nu", type=float, default=0.43, help="PA12, HP datasheet (Z): 0.43")
    ap.add_argument("--E0", type=float, default=1650e6, help="PA12 tensile modulus [Pa], EOS PA 2200 datasheet")
    ap.add_argument("--beta_max", type=float, default=16.0)
    ap.add_argument("--beta_every", type=int, default=40)
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--out", default="results/chair")
    A = ap.parse_args()
    prob, meta = build(A.n)
    g = prob.grid
    print("load cases:", "; ".join(meta["load_cases"]), flush=True)
    print(f"grid {g.nx}x{g.ny}x{g.nz} (half chair), a = {meta['a']*1000:.1f} mm, "
          f"{g.nel} elements, {3*g.nnode} DOFs", flush=True)
    E0 = A.E0
    r = optimise(prob, A.volfrac, A.penal, A.rmin, E0=E0, Emin=1e-9 * E0, nu=A.nu, beta_max=A.beta_max,
                 beta_every=A.beta_every, maxit=A.maxit)
    vm = element_von_mises(g, r["U"], r["xPhys"], E0, A.nu)          # Pa, per load case
    # stress field of the starting (uniform-density) block, for the "stress" stage of the animation
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    U = r["U"]
    tip = {f"lc{L+1}_max_disp_times_E0_N_per_m": float(np.abs(U[:, L]).max() * E0) for L in range(U.shape[1])}
    np.savez_compressed(out / "chair_result.npz", xPhys=r["xPhys"], x=r["x"], U=U, vm=vm,
                        hist=np.stack(r["hist"]), shape=np.array([g.nx, g.ny, g.nz]), a=g.a,
                        passive_solid=prob.passive_solid, passive_void=prob.passive_void)
    json.dump(dict(meta=meta, params=vars(A), iters=r["iters"], c=r["c"], log=r["log"], **tip),
              open(out / "chair_run.json", "w"), indent=1)
    print(f"done: {r['iters']} iterations, C = {r['c']:.4e}, saved to {out}")
