"""Acceptance check: matrix-free MGCG gives the same displacements as the PARDISO direct solver
on the chair load cases (sit, lean back, sideways: two support sets), for a uniform design and for
a finished solid/void design (contrast E_max/E_min = 1e9, the hard case for multigrid)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.chair import build
from topopt.top3d import hex8_stiffness, Solver, Assembler
from topopt.mgcg import MGCG

def compare(n, xPhys, label, nu=0.3, penal=3.0, Emin=1e-9):
    prob, meta = build(n); g = prob.grid; k0 = hex8_stiffness(nu)
    Ee = (Emin + xPhys**penal * (1 - Emin)) * g.a
    worst = 0.0
    for gi, (fixed, loads) in enumerate(prob.groups()):
        free = np.setdiff1d(np.arange(3 * g.nnode), fixed)
        t = time.time()
        Ud = np.zeros_like(loads); Ud[free], _ = Solver(g, free).solve(Assembler(g, k0, free).K(Ee), loads[free])
        td = time.time() - t
        mg = MGCG(g, k0, fixed, tol=1e-10, maxit=1000)
        t = time.time(); mg.update(Ee); Um, it = mg.solve(loads); tm = time.time() - t
        err = np.linalg.norm(Um - Ud, axis=0) / np.linalg.norm(Ud, axis=0)
        worst = max(worst, err.max())
        print(f"  {label} n={n} group {gi}: levels {len(mg.levels)}, CG its {it}, rel. diff {err.max():.2e}, "
              f"PARDISO {td:.2f} s, MGCG {tm:.2f} s", flush=True)
    return worst

if __name__ == "__main__":
    out = []
    prob, _ = build(8)
    out.append(compare(8, np.ones(prob.grid.nel) * 0.05 + 0.0, "uniform rho=0.05"))
    d = np.load(sys.argv[1]) if len(sys.argv) > 1 else None
    if d is not None:
        out.append(compare(int(round(float(d['shape'][0]))), np.clip(d["xPhys"], 0, 1), "finished design"))
    ok = max(out) < 1e-6
    print("MGCG PASS" if ok else "MGCG FAIL", f"(worst relative difference {max(out):.2e})")
    sys.exit(0 if ok else 1)
