"""Time one optimisation iteration of the chair on this machine, for a given solver backend, and
check the JAX solver against the NumPy solver on the same design.
Usage: python benchmarks/time_solver.py --n 20 --backend mgcg_jax
Prints the device, per-iteration times (the first JAX iteration includes compilation and is
reported separately) and the relative difference between the two solvers."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.chair import build
from topopt.top3d import hex8_stiffness, optimise

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20); ap.add_argument("--backend", default="mgcg_jax")
ap.add_argument("--iters", type=int, default=5); ap.add_argument("--check", type=int, default=1)
A = ap.parse_args()
import jax
print("JAX devices:", jax.devices())
prob, meta = build(A.n); g = prob.grid
print(f"grid {g.nx}x{g.ny}x{g.nz} (half chair), {g.nel} elements, {3 * g.nnode} DOFs, backend {A.backend}")
if A.check:
    from topopt.mgcg import MGCG
    from topopt.mgcg_jax import MGCGJax
    k0 = hex8_stiffness(0.43); Ee = (1e-9 + 0.05 ** 3) * g.a * np.ones(g.nel)
    fixed, loads = prob.groups()[0]
    a = MGCG(g, k0, fixed, tol=1e-10); a.update(Ee); Ua, _ = a.solve(loads)
    j = MGCGJax(g, k0, fixed, tol=1e-10); j.update(Ee); Uj, _ = j.solve(loads)
    print(f"JAX vs NumPy solver, relative difference: {np.max(np.linalg.norm(Uj - Ua, axis=0) / np.linalg.norm(Ua, axis=0)):.2e}")
r = optimise(prob, 0.05, rmin=2.0, E0=1650e6, Emin=1650e6 * 1e-9, nu=0.43, maxit=A.iters, backend=A.backend,
             record=False, verbose=False)
t = [e["t"] for e in r["log"]]
dt = np.diff([0.0] + t)
print("per-iteration times [s]:", " ".join(f"{v:.2f}" for v in dt))
print(f"steady per-iteration time (iterations 2..{A.iters}): {np.mean(dt[1:]):.2f} s")
