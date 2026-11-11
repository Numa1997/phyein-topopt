"""2D minimum-compliance topology optimisation (SIMP), a line-by-line Python port of
the 88-line MATLAB code:

    E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, O. Sigmund,
    "Efficient topology optimization in MATLAB using 88 lines of code",
    Struct. Multidisc. Optim. 43(1):1-16 (2011), doi:10.1007/s00158-010-0594-7

Every physical and numerical parameter is an argument; nothing is hard-wired.
Indexing follows the MATLAB code (column-major element and node numbering, nodes
numbered top to bottom, then left to right), shifted to zero-based.

Domain: nelx x nely square bilinear (Q4) elements of unit side, unit thickness,
plane stress. Default load case: the MBB half-beam (unit downward load at the
top-left node, symmetry on the left edge, roller at the bottom-right node).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def q4_stiffness(nu: float) -> np.ndarray:
    """Element stiffness matrix k0 of a unit-square Q4 element with E = 1,
    plane stress, 2x2 Gauss integration, in closed form (paper eq. 12 / code lines 5-10).
    DOF order: (x,y) of nodes bottom-left, bottom-right, top-right, top-left."""
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    A = np.block([[A11, A12], [A12.T, A11]])
    B = np.block([[B11, B12], [B12.T, B11]])
    return (A + nu * B) / (24.0 * (1.0 - nu**2))


def edof_matrix(nelx: int, nely: int) -> np.ndarray:
    """(nelx*nely, 8) global DOF numbers of each element (code lines 11-13)."""
    ix, iy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    n1 = (ix * (nely + 1) + iy).ravel()          # top-left node of each element
    base = 2 * (n1 + 1)                           # x-DOF of the bottom-left node
    off = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    return base[:, None] + off[None, :]


def filter_matrix(nelx: int, nely: int, rmin: float):
    """Linear 'hat' filter H_ei = max(0, rmin - dist(e,i)) and row sums Hs (code lines 25-42)."""
    r = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - r, 0), min(i1 + r, nelx - 1) + 1):
                for j2 in range(max(j1 - r, 0), min(j1 + r, nely - 1) + 1):
                    w = rmin - np.hypot(i1 - i2, j1 - j2)
                    if w > 0:
                        rows.append(e1); cols.append(i2 * nely + j2); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def mbb_bc(nelx: int, nely: int, force: float = -1.0):
    """MBB half-beam: load on the y-DOF of the top-left node, x fixed on the left edge,
    y fixed at the bottom-right node (code lines 15-18)."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof); F[1] = force
    fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])
    return F, fixed


def top2d(nelx, nely, volfrac, penal, rmin, ft, E0=1.0, Emin=1e-9, nu=0.3,
          move=0.2, tol=0.01, maxit=2000, bc=None, record_every=0, verbose=True):
    """Run SIMP with OC updates. ft = 1: sensitivity filter, ft = 2: density filter.
    Returns dict with final physical densities xPhys (nely x nelx), compliance c,
    iteration count, and optionally a history of xPhys snapshots."""
    KE = q4_stiffness(nu)
    edof = edof_matrix(nelx, nely)
    iK = np.kron(edof, np.ones((8, 1), dtype=int)).ravel()   # same as MATLAB kron(edofMat,ones(8,1))'
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).ravel()
    F, fixed = (bc or mbb_bc)(nelx, nely)
    ndof = F.size
    free = np.setdiff1d(np.arange(ndof), fixed)
    H, Hs = filter_matrix(nelx, nely, rmin)

    # design variables in column-major (element e = ix*nely + iy) vector form
    x = np.full(nelx * nely, volfrac)
    xPhys = x.copy()
    U = np.zeros(ndof)
    hist = []
    loop, change = 0, 1.0
    t0 = time.time()
    while change > tol and loop < maxit:
        loop += 1
        # --- FE analysis: K(x) u = f
        Ee = Emin + xPhys**penal * (E0 - Emin)
        sK = (KE.ravel()[None, :] * Ee[:, None]).ravel()          # 64 entries per element
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2
        U[:] = 0.0
        U[free] = spla.spsolve(K[free][:, free], F[free])
        # --- objective and sensitivity: c = sum_e E_e u_e^T k0 u_e ; dc/dx_e = -p x^(p-1)(E0-Emin) u_e^T k0 u_e
        Ue = U[edof]
        ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
        c = float(np.sum(Ee * ce))
        dc = -penal * (E0 - Emin) * xPhys**(penal - 1) * ce
        dv = np.ones_like(x)
        # --- filtering
        if ft == 1:
            dc = (H @ (x * dc)) / Hs / np.maximum(1e-3, x)
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)
        # --- optimality criteria update with bisection on the volume multiplier
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move,
                   np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xPhys = xnew if ft == 1 else (H @ xnew) / Hs
            if xPhys.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        if record_every and (loop == 1 or loop % record_every == 0):
            hist.append(xPhys.reshape(nelx, nely).T.copy())
        if verbose and (loop <= 3 or loop % 25 == 0):
            print(f" It.:{loop:5d} Obj.:{c:11.4f} Vol.:{xPhys.mean():7.3f} ch.:{change:7.3f}")
    if record_every:
        hist.append(xPhys.reshape(nelx, nely).T.copy())
    return dict(xPhys=xPhys.reshape(nelx, nely).T, c=c, iters=loop,
                seconds=time.time() - t0, U=U, hist=hist)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("nelx", type=int); p.add_argument("nely", type=int)
    p.add_argument("volfrac", type=float); p.add_argument("penal", type=float)
    p.add_argument("rmin", type=float); p.add_argument("ft", type=int, choices=[1, 2])
    p.add_argument("--E0", type=float, default=1.0); p.add_argument("--Emin", type=float, default=1e-9)
    p.add_argument("--nu", type=float, default=0.3); p.add_argument("--out", default="")
    a = p.parse_args()
    r = top2d(a.nelx, a.nely, a.volfrac, a.penal, a.rmin, a.ft, a.E0, a.Emin, a.nu)
    print(f"final compliance c = {r['c']:.4f} after {r['iters']} iterations ({r['seconds']:.1f} s)")
    if a.out:
        np.savez_compressed(a.out, xPhys=r["xPhys"], c=r["c"], iters=r["iters"])
