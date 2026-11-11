"""Matrix-free geometric-multigrid preconditioned conjugate gradients (MGCG) for voxel elasticity.

Method: Amir, Aage & Lazarov, "On multigrid-CG for efficient topology optimization",
Struct. Multidisc. Optim. 49(5):815-829 (2014), doi:10.1007/s00158-013-1015-5. Reference
implementation checked against: pymoto 2.0.1 (A. Delissen, MIT), pymoto.solvers.GeometricMultigrid
(trilinear interpolation, Galerkin coarse operators R^T A R, damped Jacobi w = 0.5, direct solve on
the coarsest grid).

Physics of the operator. On a regular voxel grid every element has the same stiffness matrix up to
its modulus: k_e = E_e a k0 (see top3d.py). The product K u is therefore computed without a global
matrix: gather the 24 nodal displacements of every element, multiply by k0 once (one dense BLAS
product for all elements), scale by E_e a, scatter-add back. Supports are imposed by a mask m
(1 = free DOF): A = M K M + (I - M), so fixed DOFs stay exactly zero.

Coarse grids. Grid l+1 has half the elements per direction. The prolongation R (trilinear
interpolation, the same as pymoto's weights 1, 1/2, 1/4, 1/8) carries a coarse correction to the
fine grid; on level 0 it is masked, R_0 = M R, so supports never move. The coarse operator is the
Galerkin product R^T A R, built element by element and exactly: inside one coarse cube E the fine
nodes are interpolated from E's 8 corners only, so
    K_E = sum over the 8 children c of  P_c^T k_c P_c,
with P_c the fixed 24 x 24 trilinear map from E's corners to child c's corners. For unmasked
children k_c = E_c a k0, so K_E = sum_c (E_c a) (P_c^T k0 P_c): eight precomputed matrices and one
BLAS product per update. Children touching supports are corrected individually.

Solver. Preconditioned CG on all right-hand sides of one support set at once, preconditioned by
one symmetric V-cycle (equal numbers of damped-Jacobi sweeps before and after), warm-started from
the previous design iteration's displacements.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .top3d import Grid, HEX_NODES


def _child_prolongators():
    """P_c (24 x 24) for the 8 children of a cube: maps the parent's 8 corner displacement vectors
    to the child's 8 corner displacement vectors by trilinear interpolation."""
    Ps = []
    for off in HEX_NODES:                       # child offset (a, b, c) in {0,1}^3, same order as nodes
        P = np.zeros((24, 24))
        for n, o in enumerate(HEX_NODES):       # child node n sits at t = (off + o)/2 in the parent
            t = (off + o) / 2.0
            for N, O in enumerate(HEX_NODES):   # parent node N, trilinear weight
                w = np.prod(np.where(O == 1, t, 1.0 - t))
                if w:
                    for d in range(3):
                        P[3 * n + d, 3 * N + d] = w
        Ps.append(P)
    return np.stack(Ps), HEX_NODES.copy()


def _interp_1d(nc: int) -> sp.csr_matrix:
    """1D linear interpolation from nc+1 coarse nodes to 2nc+1 fine nodes."""
    rows, cols, vals = [], [], []
    for i in range(2 * nc + 1):
        if i % 2 == 0:
            rows.append(i); cols.append(i // 2); vals.append(1.0)
        else:
            rows += [i, i]; cols += [(i - 1) // 2, (i + 1) // 2]; vals += [0.5, 0.5]
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * nc + 1, nc + 1))


def _prolongation(gc: Grid) -> sp.csr_matrix:
    """DOF prolongation from grid gc to the grid with twice the elements per direction.
    Node number n = (i*(ny+1) + j)*(nz+1) + k, DOF = 3n + d, so R = kron(Px, Py, Pz, I3)."""
    Pn = sp.kron(_interp_1d(gc.nx), sp.kron(_interp_1d(gc.ny), _interp_1d(gc.nz)))
    return sp.kron(Pn, sp.identity(3), format="csr")


def _scatter(edof_T: np.ndarray, ndof: int) -> sp.csr_matrix:
    """Sparse 0/1 matrix S with S @ f summing element vectors laid out as (24, nel) into DOFs."""
    n = edof_T.size
    return sp.csr_matrix((np.ones(n), (edof_T.ravel(), np.arange(n))), shape=(ndof, n))


def _children(gc: Grid):
    """For each coarse element and each child offset, the fine element index. (8, nel_c)"""
    I, J, K = np.meshgrid(np.arange(gc.nx), np.arange(gc.ny), np.arange(gc.nz), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    fny, fnz = 2 * gc.ny, 2 * gc.nz
    return np.stack([((2 * I + o[0]) * fny + (2 * J + o[1])) * fnz + (2 * K + o[2]) for o in HEX_NODES])


class _Level:
    def __init__(self, g: Grid, ndof_mask=None):
        self.g = g
        self.ndof = 3 * g.nnode
        self.edof = g.edof()                     # (nel, 24)
        self.edof_T = np.ascontiguousarray(self.edof.T)
        self.S = _scatter(self.edof_T, self.ndof)
        self.KE = None                           # (nel, 24, 24) element matrices (levels >= 1)
        self.diag = None
        self.zero = None                         # DOFs with an empty row (only touch supports)


class MGCG:
    """Matrix-free MGCG solver for K(E) u = f on a voxel grid with the supports `fixed`."""

    def __init__(self, g: Grid, k0: np.ndarray, fixed: np.ndarray, smooth: int = 3, omega: float = 0.5,
                 tol: float = 1e-8, maxit: int = 300, min_coarse_el: int = 200, smoother: str = "chebyshev",
                 cheb_ratio: float = 30.0):
        """smoother 'chebyshev' (default): Chebyshev polynomial of degree `smooth` in D^-1 A on the
        interval [lmax/cheb_ratio, 1.1 lmax], lmax from 15 power iterations per level and design.
        Robust for nearly incompressible materials (PA12, nu = 0.43), where damped Jacobi stalls.
        'jacobi': damped Jacobi with weight omega (pymoto's default, fine for nu <= 0.35)."""
        self.k0, self.smooth, self.omega, self.tol, self.maxit = k0, smooth, omega, tol, maxit
        self.smoother, self.cheb_ratio = smoother, cheb_ratio
        self.P, _ = _child_prolongators()
        self.M = np.einsum("cia,ij,cjb->cab", self.P, k0, self.P)       # P_c^T k0 P_c, (8, 24, 24)
        self.levels = [_Level(g)]
        self.R = []
        while True:
            gf = self.levels[-1].g
            if gf.nx % 2 or gf.ny % 2 or gf.nz % 2 or gf.nel // 8 < min_coarse_el:
                break
            gc = Grid(gf.nx // 2, gf.ny // 2, gf.nz // 2, gf.a * 2)
            self.levels.append(_Level(gc))
            self.R.append(_prolongation(gc))
        ndof = self.levels[0].ndof
        self.m = np.ones(ndof); self.m[fixed] = 0.0
        if self.R:
            self.R[0] = (sp.diags(self.m) @ self.R[0]).tocsr()
            self.RT = [R.T.tocsr() for R in self.R]
        self.children = [_children(L.g) for L in self.levels[1:]]
        L0 = self.levels[0]
        mloc = self.m[L0.edof]                                          # (nel, 24)
        self.masked_el = np.flatnonzero(mloc.min(axis=1) < 1)           # elements touching supports
        self.mloc_masked = mloc[self.masked_el]
        self.Ee = None

    # ---------------------------------------------------------------- operators
    def _A0(self, u):
        """Fine operator A = M K M + (I - M), matrix-free. u: (ndof, ncol)."""
        L = self.levels[0]
        um = u * self.m[:, None]
        ncol = u.shape[1]
        Ue = um[L.edof_T]                                               # (24, nel, ncol)
        f = (self.k0 @ Ue.reshape(24, -1)).reshape(24, -1, ncol) * self.Ee[None, :, None]
        y = L.S @ f.reshape(-1, ncol)
        return y * self.m[:, None] + u * (1.0 - self.m)[:, None]

    def _Al(self, l, u):
        L = self.levels[l]
        Ue = u[L.edof]                                                  # (nel, 24, ncol)
        f = np.matmul(L.KE, Ue)                                         # (nel, 24, ncol)
        y = L.S @ np.ascontiguousarray(f.transpose(1, 0, 2)).reshape(-1, u.shape[1])
        return y + u * L.zero[:, None]

    def _A(self, l, u):
        return self._A0(u) if l == 0 else self._Al(l, u)

    # ---------------------------------------------------------------- setup per design
    def update(self, Ee: np.ndarray):
        """Ee: element stiffness factors E_e * a (nel,). Builds all coarse operators."""
        self.Ee = np.asarray(Ee, float)
        L0 = self.levels[0]
        d0 = L0.S @ (np.diag(self.k0)[:, None] * self.Ee[None, :]).ravel()
        L0.diag = d0 * self.m + (1.0 - self.m)
        for l in range(1, len(self.levels)):
            L = self.levels[l]
            ch = self.children[l - 1]                                   # (8, nel_l)
            if l == 1:
                Ech = self.Ee[ch].T                                     # (nel_1, 8)
                KE = (Ech @ self.M.reshape(8, -1)).reshape(-1, 24, 24)
                # children touching supports: replace E_c P^T k0 P by P^T (m k0 m) P E_c
                pos = {e: i for i, e in enumerate(self.masked_el)}
                for c in range(8):
                    sel = np.flatnonzero(np.isin(ch[c], self.masked_el))
                    if sel.size == 0:
                        continue
                    fe = ch[c, sel]
                    mm = self.mloc_masked[[pos[e] for e in fe]]         # (ns, 24)
                    kc = self.k0[None] * mm[:, :, None] * mm[:, None, :] * self.Ee[fe][:, None, None]
                    KE[sel] += np.einsum("ia,eij,jb->eab", self.P[c], kc, self.P[c]) \
                        - self.Ee[fe][:, None, None] * self.M[c][None]
            else:
                KEf = self.levels[l - 1].KE
                KE = np.zeros((L.g.nel, 24, 24))
                for c in range(8):
                    KE += np.matmul(np.matmul(self.P[c].T[None], KEf[ch[c]]), self.P[c][None])
            L.KE = KE
            d = L.S @ np.ascontiguousarray(np.diagonal(KE, axis1=1, axis2=2).T).ravel()
            L.zero = (np.abs(d) <= 1e-14 * np.abs(d).max()).astype(float)
            L.diag = d + L.zero
        if self.smoother == "chebyshev":
            rng = np.random.default_rng(0)
            for l, L in enumerate(self.levels[:-1] if len(self.levels) > 1 else self.levels):
                v = rng.standard_normal((L.ndof, 1))
                lam = 1.0
                for _ in range(15):                      # power iteration on D^-1 A
                    w = self._A(l, v) / L.diag[:, None]
                    lam = float(np.linalg.norm(w) / np.linalg.norm(v)); v = w / np.linalg.norm(w)
                L.lmax = 1.1 * lam
        Lc = self.levels[-1]
        if len(self.levels) > 1:
            rows = np.repeat(Lc.edof, 24, axis=1).ravel(); cols = np.tile(Lc.edof, (1, 24)).ravel()
            Kc = sp.coo_matrix((Lc.KE.ravel(), (rows, cols)), shape=(Lc.ndof, Lc.ndof)).tocsc()
            Kc = Kc + sp.diags(Lc.zero)
            self.coarse = splu(Kc)

    # ---------------------------------------------------------------- V-cycle and PCG
    def _smooth(self, l, r, u):
        """Apply `smooth` steps of the smoother to A u = r starting from u (u may be None = 0)."""
        L = self.levels[l]
        if self.smoother == "jacobi":
            Dinv = (self.omega / L.diag)[:, None]
            if u is None:
                u = Dinv * r; n = self.smooth - 1
            else:
                n = self.smooth
            for _ in range(n):
                u = u + Dinv * (r - self._A(l, u))
            return u
        # Chebyshev iteration (Saad, Iterative Methods, alg. 12.1) on D^-1 A over [a, b]
        b = L.lmax; a = b / self.cheb_ratio
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        Dinv = (1.0 / L.diag)[:, None]
        u = np.zeros_like(r) if u is None else u
        res = Dinv * (r - self._A(l, u))
        rho = 1.0 / sigma
        d = res / theta
        for _ in range(self.smooth):
            u = u + d
            res = res - Dinv * self._A(l, d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * res
            rho = rho_new
        return u

    def _vcycle(self, l, r):
        L = self.levels[l]
        if l == len(self.levels) - 1:
            if l == 0:                                                   # single level: Jacobi only
                return r / L.diag[:, None]
            return self.coarse.solve(r)
        u = self._smooth(l, r, None)
        res = r - self._A(l, u)
        u = u + self.R[l] @ self._vcycle(l + 1, self.RT[l] @ res)
        return self._smooth(l, r, u)

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
        """b, x0: (ndof, ncol) full-length vectors (fixed DOFs are forced to 0).
        Returns u and the number of CG iterations (all columns are iterated together)."""
        b = np.atleast_2d(b.T).T * self.m[:, None]
        x = np.zeros_like(b) if x0 is None else np.array(x0, float).reshape(b.shape) * self.m[:, None]
        r = b - self._A0(x)
        bn = np.linalg.norm(b, axis=0); bn[bn == 0] = 1.0
        z = self._vcycle(0, r); p = z.copy()
        rz = np.sum(r * z, axis=0)
        it = 0
        while it < self.maxit and np.max(np.linalg.norm(r, axis=0) / bn) > self.tol:
            it += 1
            q = self._A0(p)
            alpha = rz / np.sum(p * q, axis=0)
            x += alpha * p; r -= alpha * q
            z = self._vcycle(0, r)
            rz_new = np.sum(r * z, axis=0)
            p = z + (rz_new / rz) * p
            rz = rz_new
        self.last_res = np.linalg.norm(r, axis=0) / bn
        return x, it
