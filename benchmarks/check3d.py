"""Acceptance checks for the 3D voxel FE + sensitivity code.

1. Element: k0 symmetric, positive semi-definite with exactly 6 zero eigenvalues (rigid-body modes).
2. Solver: solid cantilever (square section h x h, length L = 10 h, clamped face at x = 0,
   tip load F in -z spread consistently over the tip face). Compare with
     Euler-Bernoulli   delta = F L^3 / (3 E I),            I = h^4 / 12
     Timoshenko        delta = F L^3 / (3 E I) + F L / (kappa G A),
                       kappa = 10 (1 + nu) / (12 + 11 nu)  (Cowper 1966, rectangle), A = h^2.
3. Gradient: dC/dxbar from the adjoint formula vs. automatic differentiation (JAX, float64)
   of C(xbar) = f^T K(xbar)^-1 f on a small random-density grid.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topopt.top3d import Grid, Problem, hex8_stiffness, solve_static  # noqa: E402


def check_element(nu=0.3):
    k0 = hex8_stiffness(nu)
    ev = np.linalg.eigvalsh(k0)
    nzero = int(np.sum(np.abs(ev) < 1e-10 * ev.max()))
    print(f"[1] k0: symmetric err {np.abs(k0-k0.T).max():.1e}, min eig {ev.min():+.1e}, zero modes {nzero} (expect 6)")
    return np.allclose(k0, k0.T) and nzero == 6 and ev.min() > -1e-10


def face_weights(n1, n2):
    """Consistent nodal weights of a uniform traction on an n1 x n2 bilinear face, sum = 1."""
    w1 = np.ones(n1 + 1); w1[[0, -1]] = 0.5
    w2 = np.ones(n2 + 1); w2[[0, -1]] = 0.5
    w = np.outer(w1, w2)
    return w / w.sum()


def cantilever(n_h, Lh=10, nu=0.3, E=1.0, F=1.0):
    g = Grid(Lh * n_h, n_h, n_h, a=1.0)
    ndof = 3 * g.nnode
    j, k = np.meshgrid(np.arange(n_h + 1), np.arange(n_h + 1), indexing="ij")
    root = g.node_id(0, j.ravel(), k.ravel())
    tip = g.node_id(g.nx, j.ravel(), k.ravel())
    fixed = np.concatenate([3 * root, 3 * root + 1, 3 * root + 2])
    f = np.zeros((ndof, 1)); f[3 * tip + 2, 0] = -F * face_weights(n_h, n_h).ravel()
    U = solve_static(Problem(g, f, fixed), np.ones(g.nel), E, 1e-9, 3.0, nu, tol=1e-10)
    d_fe = -np.average(U[3 * tip + 2, 0], weights=face_weights(n_h, n_h).ravel())
    h, L = n_h, g.nx
    I, A, G = h**4 / 12, h * h, E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    d_eb = F * L**3 / (3 * E * I)
    return d_fe, d_eb, d_eb + F * L / (kappa * G * A)


def check_gradient(seed=0, penal=3.0, nu=0.3, Emin=1e-3):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    g = Grid(4, 3, 2)
    ndof = 3 * g.nnode
    edof = g.edof()
    k0 = hex8_stiffness(nu)
    j, k = np.meshgrid(np.arange(4), np.arange(3), indexing="ij")
    root = g.node_id(0, j.ravel(), k.ravel())
    fixed = np.concatenate([3 * root, 3 * root + 1, 3 * root + 2])
    free = np.setdiff1d(np.arange(ndof), fixed)
    f = np.zeros(ndof); f[3 * g.node_id(4, 3, 2) + 2] = -1.0; f[3 * g.node_id(4, 0, 1) + 1] = 0.5
    iK = np.repeat(edof, 24, axis=1).ravel(); jK = np.tile(edof, (1, 24)).ravel()

    def compliance(xb):
        Ee = Emin + xb**penal * (1 - Emin)
        K = jnp.zeros((ndof, ndof)).at[iK, jK].add((Ee[:, None] * k0.ravel()[None, :]).ravel())
        Kf = K[np.ix_(free, free)]
        u = jnp.linalg.solve(Kf, f[free])
        return f[free] @ u

    xb = np.random.default_rng(seed).uniform(0.2, 1.0, g.nel)
    g_ad = np.asarray(jax.grad(compliance)(jnp.asarray(xb)))
    # adjoint formula, evaluated with the same dense solve
    Ee = Emin + xb**penal * (1 - Emin)
    K = np.zeros((ndof, ndof)); np.add.at(K, (iK, jK), (Ee[:, None] * k0.ravel()[None, :]).ravel())
    u = np.zeros(ndof); u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    ce = np.einsum("ei,ij,ej->e", u[edof], k0, u[edof])
    g_adj = -penal * xb**(penal - 1) * (1 - Emin) * ce
    rel = np.max(np.abs(g_ad - g_adj)) / np.max(np.abs(g_ad))
    print(f"[3] gradient: max |AD - adjoint| / max |AD| = {rel:.2e} over {g.nel} elements")
    return rel < 1e-10


if __name__ == "__main__":
    ok1 = check_element()
    print("[2] cantilever L/h = 10, square section")
    print("    n_h   delta_FE     delta_EB     delta_Tim   FE/EB-1   FE/Tim-1")
    rows = []
    for n_h in (2, 4, 8, 16):
        fe, eb, tim = cantilever(n_h)
        rows.append(fe / tim - 1)
        print(f"    {n_h:3d} {fe:11.2f} {eb:11.2f} {tim:11.2f} {fe/eb-1:+8.3%} {fe/tim-1:+8.3%}", flush=True)
    ok2 = abs(rows[-1]) < 0.01
    ok3 = check_gradient()
    print("ALL PASS" if (ok1 and ok2 and ok3) else "FAIL", ok1, ok2, ok3)
