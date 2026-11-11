"""Acceptance check of the 2D FE solver alone (no optimisation): a solid cantilever,
clamped at x = 0, tip load F at x = L, plane stress, thickness t.

Beam theory with shear (Timoshenko), rectangular section b = t, height h:
    delta = F L^3 / (3 E I) + F L / (kappa G A),  I = t h^3 / 12,  A = t h,
    G = E / (2 (1 + nu)),  kappa = 5/6.
The first term alone is the Euler-Bernoulli result F L^3/(3EI). The FE load is spread
uniformly over the tip edge nodes (lumped consistently), and the clamp fixes every DOF
on the left edge, which adds a small root stiffening that beam theory ignores.

Q4 elements with full 2x2 integration lock in shear under bending, so the FE tip
deflection is expected below beam theory on coarse meshes and to converge from below.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topopt.top2d import q4_stiffness, edof_matrix  # noqa: E402


def tip_deflection(L_over_h: float, n_h: int, E=1.0, nu=0.3, F=1.0, t=1.0):
    nely = n_h
    nelx = int(round(L_over_h * n_h))
    a = 1.0                                     # element side; h = nely * a, L = nelx * a
    h, L = nely * a, nelx * a
    KE = q4_stiffness(nu) * E * t               # k0 scales with E and thickness, not with element size in 2D
    edof = edof_matrix(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    iK = np.kron(edof, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).ravel()
    K = sp.coo_matrix((np.tile(KE.ravel(), nelx * nely), (iK, jK)), shape=(ndof, ndof)).tocsc()
    fixed = np.arange(0, 2 * (nely + 1))        # all DOFs of the left edge
    tip_nodes = nelx * (nely + 1) + np.arange(nely + 1)
    w = np.full(nely + 1, 1.0); w[[0, -1]] = 0.5; w /= w.sum()   # consistent load on a linear edge
    Fv = np.zeros(ndof); Fv[2 * tip_nodes + 1] = -F * w
    free = np.setdiff1d(np.arange(ndof), fixed)
    U = np.zeros(ndof); U[free] = spla.spsolve(K[free][:, free], Fv[free])
    delta_fe = -np.mean(U[2 * tip_nodes + 1])
    I, A, G = t * h**3 / 12, t * h, E / (2 * (1 + nu))
    d_eb = F * L**3 / (3 * E * I)
    d_tim = d_eb + F * L / (5 / 6 * G * A)
    return delta_fe, d_eb, d_tim


if __name__ == "__main__":
    print("L/h  n_h   delta_FE      delta_EB      delta_Tim    FE/EB-1    FE/Tim-1")
    for Lh in (10.0,):
        for n_h in (2, 4, 8, 16, 32):
            fe, eb, tim = tip_deflection(Lh, n_h)
            print(f"{Lh:4.0f} {n_h:4d} {fe:12.4f} {eb:12.4f} {tim:12.4f} {fe/eb-1:+9.4%} {fe/tim-1:+9.4%}")
