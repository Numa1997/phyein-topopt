"""3D minimum-compliance topology optimisation on a voxel grid (SIMP + density filter +
Heaviside projection + optimality criteria), CPU reference implementation.

Same algorithm as the 2D 88-line code (topopt/top2d.py), lifted to 3D:
  * domain: nx * ny * nz cubic 8-node (hex8, trilinear) elements of side a
  * stiffness interpolation:   E_e = Emin + xbar_e^p (E0 - Emin)
  * objective:                 C = sum_L w_L f_L^T u_L,   K(xbar) u_L = f_L   (several load cases)
  * sensitivity (adjoint, self-adjoint for compliance):
        dC/dxbar_e = - sum_L w_L p xbar_e^(p-1) (E0 - Emin) u_Le^T k0 u_Le
  * density filter  xtil = H x / Hs,  projection  xbar = P_beta(xtil)  (tanh form, threshold eta)
  * chain rule back to x:  dC/dx = H^T ( dC/dxbar * dxbar/dxtil / Hs )
  * passive solid / void elements are pinned after projection and excluded from the volume limit
Linear solves: algebraic-multigrid-preconditioned CG (pyamg, smoothed aggregation with
the six rigid-body modes as near-null space), warm-started from the previous iterate.

Units: give a [m], E0 [Pa], forces [N]; displacements come out in m, stresses in Pa.
k0 for a cube of side a scales as E*a (derivation in the skill notes), so the code builds
k0 for a unit cube with E = 1 and multiplies by E0*a.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import pyamg

# local node offsets (i,j,k) of a hex8 element, counter-clockwise bottom face then top face
HEX_NODES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def elasticity_matrix(nu: float, E: float = 1.0) -> np.ndarray:
    """Isotropic Hooke's law in Voigt form sigma = D eps, eps = (exx,eyy,ezz,gyz,gxz,gxy)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam; D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D


def hex8_B(xi, eta, zeta, a=1.0):
    """Strain-displacement matrix B (6x24) of a cube of side a at natural point (xi,eta,zeta)."""
    s = 2 * HEX_NODES - 1                                    # node signs in [-1,1]^3
    dN = np.empty((8, 3))
    dN[:, 0] = s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta) / 8
    dN[:, 1] = s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta) / 8
    dN[:, 2] = s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) / 8
    dN *= 2.0 / a                                            # d(xi)/dx = 2/a for a cube of side a
    B = np.zeros((6, 24))
    for n in range(8):
        dx, dy, dz = dN[n]
        c = 3 * n
        B[0, c] = dx; B[1, c + 1] = dy; B[2, c + 2] = dz
        B[3, c + 1] = dz; B[3, c + 2] = dy
        B[4, c] = dz; B[4, c + 2] = dx
        B[5, c] = dy; B[5, c + 1] = dx
    return B


def hex8_stiffness(nu: float) -> np.ndarray:
    """k0 = integral over the unit cube of B^T D B dV with E = 1, by 2x2x2 Gauss quadrature
    (exact for the trilinear element). det J = (a/2)^3 = 1/8, all weights 1."""
    D = elasticity_matrix(nu)
    g = 1 / np.sqrt(3)
    k = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = hex8_B(xi, eta, zeta)
                k += B.T @ D @ B / 8.0
    return k


@dataclass
class Grid:
    nx: int; ny: int; nz: int; a: float = 1.0

    @property
    def nel(self): return self.nx * self.ny * self.nz

    @property
    def nnode(self): return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def node_id(self, i, j, k):
        return (np.asarray(i) * (self.ny + 1) + np.asarray(j)) * (self.nz + 1) + np.asarray(k)

    def edof(self) -> np.ndarray:
        """(nel, 24) global DOFs; element e = (i*ny + j)*nz + k (C order over (nx,ny,nz))."""
        i, j, k = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        nodes = np.stack([self.node_id(i + o[0], j + o[1], k + o[2]) for o in HEX_NODES], axis=1)
        return (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)

    def node_coords(self) -> np.ndarray:
        i, j, k = np.meshgrid(np.arange(self.nx + 1), np.arange(self.ny + 1), np.arange(self.nz + 1), indexing="ij")
        return self.a * np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(float)


def density_filter(g: Grid, rmin: float):
    """H_ei = max(0, rmin - |c_e - c_i|) in voxel units, built by offsets (vectorised)."""
    r = int(np.ceil(rmin)) - 1
    shape = (g.nx, g.ny, g.nz)
    idx = np.arange(g.nel).reshape(shape)
    rows, cols, vals = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                w = rmin - np.sqrt(di * di + dj * dj + dk * dk)
                if w <= 0:
                    continue
                src = idx[max(0, -di):g.nx - max(0, di), max(0, -dj):g.ny - max(0, dj), max(0, -dk):g.nz - max(0, dk)]
                dst = idx[max(0, di):g.nx - max(0, -di) or None, max(0, dj):g.ny - max(0, -dj) or None,
                          max(0, dk):g.nz - max(0, -dk) or None]
                rows.append(src.ravel()); cols.append(dst.ravel()); vals.append(np.full(src.size, w))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(g.nel, g.nel))
    return H, np.asarray(H.sum(axis=1)).ravel()


def heaviside(xt, beta, eta=0.5):
    """Smoothed Heaviside projection (Wang, Lazarov & Sigmund 2011) and its derivative."""
    if beta <= 0:
        return xt, np.ones_like(xt)
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    th = np.tanh(beta * (xt - eta))
    return (np.tanh(beta * eta) + th) / den, beta * (1 - th**2) / den


class Assembler:
    """Fast repeated assembly of K_ff (free-free block) from element moduli E_e."""

    def __init__(self, g: Grid, k0: np.ndarray, free: np.ndarray):
        self.g, self.k0 = g, k0
        self.edof = g.edof()
        ndof = 3 * g.nnode
        fmap = -np.ones(ndof, dtype=np.int64); fmap[free] = np.arange(free.size)
        self.nfree = free.size
        r = fmap[np.repeat(self.edof, 24, axis=1)].ravel()        # row DOF of each of 576 entries
        c = fmap[np.tile(self.edof, (1, 24))].ravel()             # column DOF
        keep = (r >= 0) & (c >= 0)
        self.keep = keep
        lin = r[keep] * self.nfree + c[keep]
        uniq, self.inv = np.unique(lin, return_inverse=True)
        self.indptr_rows = (uniq // self.nfree).astype(np.int64)
        self.indices = (uniq % self.nfree).astype(np.int32)
        self.indptr = np.searchsorted(self.indptr_rows, np.arange(self.nfree + 1)).astype(np.int64)
        self.k0flat = k0.ravel()

    def K(self, Ee: np.ndarray) -> sp.csr_matrix:
        vals = (Ee[:, None] * self.k0flat[None, :]).ravel()[self.keep]
        data = np.bincount(self.inv, weights=vals, minlength=self.indices.size)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.nfree, self.nfree))


def rigid_body_modes(coords: np.ndarray) -> np.ndarray:
    """Six rigid-body displacement fields (3 translations, 3 rotations) as columns, (3*nnode, 6)."""
    n = coords.shape[0]
    B = np.zeros((3 * n, 6))
    x, y, z = coords.T
    for d in range(3):
        B[d::3, d] = 1.0
    B[0::3, 3] = -y; B[1::3, 3] = x
    B[1::3, 4] = -z; B[2::3, 4] = y
    B[0::3, 5] = z;  B[2::3, 5] = -x
    return B


class Solver:
    """Linear solver for K_ff u = f (all load cases at once).

    backend 'pardiso' (default): sparse direct LU from Intel MKL via pypardiso. Exact to
        round-off, one factorisation shared by all load cases; memory grows quickly in 3D,
        practical up to a few hundred thousand DOFs on a 16 GB runner.
    backend 'amg': CG preconditioned by pyamg smoothed aggregation. Low memory, but in our
        tests (pyamg 5.4.1) a run could stall or diverge depending on earlier calls in the
        same process, so it stays opt-in and every solve is checked for convergence.
    """

    def __init__(self, g: Grid, free: np.ndarray, tol=1e-8, maxiter=500, backend="pardiso"):
        self.backend = backend
        self.tol, self.maxiter = tol, maxiter
        if backend == "amg":
            self.B = rigid_body_modes(g.node_coords())[free]

    def solve(self, K, rhs, x0=None):
        if self.backend == "pardiso":
            import pypardiso
            u = pypardiso.spsolve(K.tocsr(), np.ascontiguousarray(rhs))
            u = u.reshape(rhs.shape)
            res = np.linalg.norm(K @ u - rhs) / np.linalg.norm(rhs)
            if not np.isfinite(u).all() or res > 1e-6:
                raise RuntimeError(f"PARDISO solve inaccurate: relative residual {res:.1e}")
            return u, [0] * rhs.shape[1]
        ml = pyamg.smoothed_aggregation_solver(K, B=self.B, max_coarse=500)
        out, res = [], []
        for L in range(rhs.shape[1]):
            r = []
            # pyamg's compiled smoothers need contiguous vectors; a column slice is strided
            b = np.ascontiguousarray(rhs[:, L])
            xs = None if x0 is None else np.ascontiguousarray(x0[:, L])
            u = ml.solve(b, x0=xs, tol=self.tol, accel="cg", maxiter=self.maxiter, residuals=r)
            if not np.isfinite(u).all() or r[-1] > 1e3 * self.tol * r[0]:
                raise RuntimeError(f"CG/AMG did not converge: rel. residual {r[-1]/r[0]:.1e} after {len(r)} its")
            out.append(u); res.append(len(r))
        return np.stack(out, axis=1), res


@dataclass
class Problem:
    grid: Grid
    loads: np.ndarray                      # (3*nnode, nL) nodal forces [N]
    fixed: np.ndarray                      # constrained DOF numbers
    passive_solid: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    passive_void: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    weights: np.ndarray | None = None      # load-case weights w_L


def solve_static(p: Problem, xPhys, E0, Emin, penal, nu, tol=1e-8):
    """One FE analysis at fixed densities. Returns U (3*nnode, nL) and per-element u^T k0 u."""
    g = p.grid
    ndof = 3 * g.nnode
    free = np.setdiff1d(np.arange(ndof), p.fixed)
    asm = Assembler(g, hex8_stiffness(nu), free)
    Ee = (Emin + xPhys**penal * (E0 - Emin)) * g.a
    U = np.zeros((ndof, p.loads.shape[1]))
    U[free], _ = Solver(g, free, tol).solve(asm.K(Ee), p.loads[free])
    return U


def optimise(p: Problem, volfrac, penal=3.0, rmin=1.5, E0=1.0, Emin=1e-9, nu=0.3, move=0.2,
             beta0=1.0, beta_max=16.0, beta_every=40, eta=0.5, tol=0.01, ctol=1e-3, maxit=300,
             record=True, verbose=True, cg_tol=1e-8):
    g = p.grid
    ndof = 3 * g.nnode
    nL = p.loads.shape[1]
    w = np.ones(nL) if p.weights is None else np.asarray(p.weights, float)
    free = np.setdiff1d(np.arange(ndof), p.fixed)
    k0 = hex8_stiffness(nu)
    asm = Assembler(g, k0, free)
    solver = Solver(g, free, cg_tol)
    H, Hs = density_filter(g, rmin)
    design = np.ones(g.nel, bool); design[p.passive_solid] = False; design[p.passive_void] = False
    nd = design.sum()

    x = np.full(g.nel, volfrac); x[p.passive_solid] = 1.0; x[p.passive_void] = 0.0
    beta = beta0 if beta0 > 0 else 0.0

    def physical(xd):
        xt = H @ xd / Hs
        xb, dxb = heaviside(xt, beta, eta)
        xb = np.clip(xb, 0.0, 1.0)
        xb[p.passive_solid] = 1.0; xb[p.passive_void] = 0.0
        dxb[~design] = 0.0
        return xb, dxb

    xPhys, dxb = physical(x)
    U = np.zeros((ndof, nL)); Uf = None
    hist, log = [], []
    loop, change, since_beta = 0, 1.0, 0
    c_hist = []
    t0 = time.time()
    while loop < maxit:
        loop += 1; since_beta += 1
        Ee = (Emin + xPhys**penal * (E0 - Emin)) * g.a
        Uf, its = solver.solve(asm.K(Ee), p.loads[free], Uf)
        U[free] = Uf
        Ue = U[asm.edof]                                           # (nel, 24, nL)
        ce = np.einsum("eil,ij,ejl->el", Ue, k0, Ue) * g.a           # u_e^T (a k0) u_e, per load case
        c = float(np.sum(w * np.sum(Ee[:, None] / g.a * ce, axis=0)))
        dc_xb = -(penal * (E0 - Emin) * xPhys**(penal - 1))[:, None] * ce @ w
        dc = H.T @ (dc_xb * dxb / Hs)                                # chain rule through projection and filter
        dv = H.T @ (design * dxb / Hs)
        dc = np.minimum(dc, 0.0)                                     # compliance can only drop with added material
        # OC update on design variables only, bisection on the volume multiplier
        l1, l2 = 0.0, 1e9
        xd = x.copy()
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            ratio = np.sqrt(np.maximum(-dc, 1e-30) / np.maximum(dv, 1e-30) / lmid)
            xnew = np.clip(np.clip(x * ratio, x - move, x + move), 0.0, 1.0)
            xnew[~design] = x[~design]
            xb, _ = physical(xnew)
            if xb[design].sum() > volfrac * nd:
                l1 = lmid
            else:
                l2 = lmid
        x = xnew
        xPhys_old = xPhys
        xPhys, dxb = physical(x)
        # convergence is judged on the PHYSICAL design: under a steep projection, raw variables
        # deep in solid or void have ~zero sensitivity and may wander without changing xPhys
        change = float(np.max(np.abs(xPhys - xPhys_old)))
        c_hist.append(c)
        c_rel = (max(c_hist[-10:]) - min(c_hist[-10:])) / c if len(c_hist) >= 10 else 1.0
        grey = float(np.mean(4 * xPhys[design] * (1 - xPhys[design])))   # 0 = crisp, 1 = all grey
        log.append(dict(it=loop, c=c, vol=float(xPhys[design].mean()), change=change, c_rel10=c_rel, beta=beta,
                        grey=grey, cg_its=its, t=time.time() - t0))
        if record:
            hist.append(xPhys.astype(np.float16))
        if verbose and (loop <= 2 or loop % 10 == 0):
            print(f" it {loop:4d}  C {c:.5e}  vol {xPhys[design].mean():.3f}  ch {change:.3f}  "
                  f"beta {beta:4.0f}  grey {grey:.3f}  cg {its}  {time.time()-t0:6.0f}s", flush=True)
        # continuation: raise beta when converged at this beta or every beta_every iterations
        settled = change < tol or c_rel < ctol
        if beta0 > 0 and beta < beta_max and (since_beta >= beta_every or settled):
            beta *= 2; since_beta = 0; c_hist = []
            xPhys, dxb = physical(x)
            continue
        if settled and (beta0 <= 0 or beta >= beta_max):
            break
    return dict(xPhys=xPhys, x=x, U=U, c=c, iters=loop, log=log, hist=hist, design=design)


def element_von_mises(g: Grid, U: np.ndarray, xPhys: np.ndarray, E0: float, nu: float, q: float = 0.5):
    """Von Mises stress at element centres per load case, (nel, nL). For SIMP designs the
    'relaxed' stress sigma_e = xbar_e^q * sigma_solid_e (q = 0.5, Le et al. 2010) avoids
    spurious stress in near-void elements."""
    D = elasticity_matrix(nu, E0)
    B = hex8_B(0.0, 0.0, 0.0, g.a)
    Ue = U[g.edof()]                                   # (nel, 24, nL)
    s = np.einsum("ij,jk,ekl->eil", D, B, Ue)          # (nel, 6, nL)
    sx, sy, sz, tyz, txz, txy = (s[:, i] for i in range(6))
    vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 3 * (tyz**2 + txz**2 + txy**2))
    return vm * (np.clip(xPhys, 0.0, 1.0)[:, None] ** q)
