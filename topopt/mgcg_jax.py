"""JAX port of the matrix-free MGCG solver (topopt/mgcg.py) for GPUs.

Same mathematics, same data layout: the fine operator is gather -> k0 product -> scale by E_e a ->
scatter-add; coarse operators are the exact element-by-element Galerkin products; Chebyshev
smoother; dense Cholesky on the coarsest grid; preconditioned CG in a jax.lax.while_loop. All arrays
are float64 (the solver needs 1e-8 relative residuals): on Kaggle use the P100 GPU, whose float64
throughput is about half its float32 throughput; the T4 runs float64 at about 1/32 of float32.

The index structures (element DOF maps, prolongations, children, support masks) are built once on
the CPU by reusing topopt.mgcg.MGCG and then moved to the device. Interface as in MGCG:
update(Ee) then solve(b, x0) -> (u as numpy array, CG iterations); last_res holds the residuals.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

jax.config.update("jax_enable_x64", True)

from .mgcg import MGCG  # noqa: E402


def _bcoo(m):
    m = m.tocoo()
    return jsparse.BCOO((jnp.asarray(m.data), jnp.asarray(np.stack([m.row, m.col], 1))), shape=m.shape)


class MGCGJax:
    def __init__(self, g, k0, fixed, smooth=3, tol=1e-8, maxit=500, min_coarse_el=200, cheb_ratio=30.0):
        ref = MGCG(g, k0, fixed, smooth=smooth, tol=tol, maxit=maxit, min_coarse_el=min_coarse_el,
                   smoother="chebyshev", cheb_ratio=cheb_ratio)
        self.ref = ref
        self.smooth, self.tol, self.maxit, self.cheb_ratio = smooth, tol, maxit, cheb_ratio
        self.nl = len(ref.levels)
        self.k0 = jnp.asarray(k0)
        self.m = jnp.asarray(ref.m)
        self.edof = [jnp.asarray(L.edof) for L in ref.levels]
        self.edof_T = [jnp.asarray(L.edof_T) for L in ref.levels]
        self.ndof = [L.ndof for L in ref.levels]
        self.R = [_bcoo(R) for R in ref.R] if ref.R else []
        self.RT = [_bcoo(R) for R in ref.RT] if ref.R else []
        self.children = [jnp.asarray(c) for c in ref.children]
        self.P = jnp.asarray(ref.P)
        self.M = jnp.asarray(ref.M)
        # level-1 correction data for children touching supports (fixed for the whole run)
        self.corr = []
        if self.nl > 1:
            pos = {e: i for i, e in enumerate(ref.masked_el)}
            ch = ref.children[0]
            for c in range(8):
                sel = np.flatnonzero(np.isin(ch[c], ref.masked_el))
                if sel.size:
                    fe = ch[c, sel]
                    mm = ref.mloc_masked[[pos[e] for e in fe]]
                    self.corr.append((c, jnp.asarray(sel), jnp.asarray(fe), jnp.asarray(mm)))
        self._pcg = jax.jit(self._pcg_impl)
        self.state = None

    # ------------------------------------------------------------ operators (pure functions)
    def _A0(self, Ee, u):
        um = u * self.m[:, None]
        nc = u.shape[1]
        Ue = um[self.edof_T[0]]                                         # (24, nel, nc)
        f = (self.k0 @ Ue.reshape(24, -1)).reshape(24, -1, nc) * Ee[None, :, None]
        y = jnp.zeros((self.ndof[0], nc)).at[self.edof_T[0].reshape(-1)].add(f.reshape(-1, nc))
        return y * self.m[:, None] + u * (1.0 - self.m)[:, None]

    def _Al(self, st, l, u):
        nc = u.shape[1]
        Ue = u[self.edof[l]]                                            # (nel, 24, nc)
        f = jnp.matmul(st["KE"][l - 1], Ue)
        y = jnp.zeros((self.ndof[l], nc)).at[self.edof[l].reshape(-1)].add(f.reshape(-1, nc))
        return y + u * st["zero"][l][:, None]

    def _A(self, st, l, u):
        return self._A0(st["Ee"], u) if l == 0 else self._Al(st, l, u)

    # ------------------------------------------------------------ setup per design (traced)
    def _setup(self, Ee):
        st = {"Ee": Ee, "KE": [], "diag": [], "zero": [None], "lmax": []}
        d0 = jnp.zeros(self.ndof[0]).at[self.edof_T[0].reshape(-1)].add(
            (jnp.diag(self.k0)[:, None] * Ee[None, :]).reshape(-1))
        st["diag"].append(d0 * self.m + (1.0 - self.m))
        for l in range(1, self.nl):
            ch = self.children[l - 1]
            if l == 1:
                KE = (Ee[ch].T @ self.M.reshape(8, -1)).reshape(-1, 24, 24)
                for c, sel, fe, mm in self.corr:
                    kc = self.k0[None] * mm[:, :, None] * mm[:, None, :] * Ee[fe][:, None, None]
                    KE = KE.at[sel].add(jnp.einsum("ia,eij,jb->eab", self.P[c], kc, self.P[c])
                                        - Ee[fe][:, None, None] * self.M[c][None])
            else:
                KEf = st["KE"][l - 2]
                KE = sum(jnp.matmul(jnp.matmul(self.P[c].T[None], KEf[ch[c]]), self.P[c][None])
                         for c in range(8))
            st["KE"].append(KE)
            d = jnp.zeros(self.ndof[l]).at[self.edof_T[l].reshape(-1)].add(
                jnp.diagonal(KE, axis1=1, axis2=2).T.reshape(-1))
            zero = (jnp.abs(d) <= 1e-14 * jnp.abs(d).max()).astype(d.dtype)
            st["zero"].append(zero)
            st["diag"].append(d + zero)
        for l in range(self.nl - 1 if self.nl > 1 else 1):            # power iteration on D^-1 A
            v = jax.random.normal(jax.random.PRNGKey(l), (self.ndof[l], 1))
            def body(_, carry):
                v, lam = carry
                w = self._A(st, l, v) / st["diag"][l][:, None]
                n = jnp.linalg.norm(w)
                return w / n, n / jnp.linalg.norm(v)
            v, lam = jax.lax.fori_loop(0, 15, body, (v / jnp.linalg.norm(v), 1.0))
            st["lmax"].append(1.1 * lam)
        if self.nl > 1:
            Lc, KE = self.nl - 1, st["KE"][-1]
            n = self.ndof[Lc]
            rows = jnp.repeat(self.edof[Lc], 24, axis=1).reshape(-1)
            cols = jnp.tile(self.edof[Lc], (1, 24)).reshape(-1)
            Kc = jnp.zeros((n, n)).at[rows, cols].add(KE.reshape(-1)) + jnp.diag(st["zero"][Lc])
            st["chol"] = jnp.linalg.cholesky(Kc)
        return st

    def _smooth(self, st, l, r, u):
        b = st["lmax"][l]; a = b / self.cheb_ratio
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        Dinv = (1.0 / st["diag"][l])[:, None]
        u = jnp.zeros_like(r) if u is None else u
        res = Dinv * (r - self._A(st, l, u))
        rho = 1.0 / sigma
        d = res / theta
        for _ in range(self.smooth):
            u = u + d
            res = res - Dinv * self._A(st, l, d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * res
            rho = rho_new
        return u

    def _vcycle(self, st, l, r):
        if l == self.nl - 1:
            if l == 0:
                return r / st["diag"][0][:, None]
            return jax.scipy.linalg.cho_solve((st["chol"], True), r)
        u = self._smooth(st, l, r, None)
        res = r - self._A(st, l, u)
        u = u + self.R[l] @ self._vcycle(st, l + 1, self.RT[l] @ res)
        return self._smooth(st, l, r, u)

    def _pcg_impl(self, Ee, b, x0):
        st = self._setup(Ee)
        b = b * self.m[:, None]
        x = x0 * self.m[:, None]
        r = b - self._A0(Ee, x)
        bn = jnp.linalg.norm(b, axis=0); bn = jnp.where(bn == 0, 1.0, bn)
        z = self._vcycle(st, 0, r)
        p = z
        rz = jnp.sum(r * z, axis=0)

        def cond(c):
            it, x, r, p, rz = c
            return (it < self.maxit) & (jnp.max(jnp.linalg.norm(r, axis=0) / bn) > self.tol)

        def body(c):
            it, x, r, p, rz = c
            q = self._A0(Ee, p)
            alpha = rz / jnp.sum(p * q, axis=0)
            x = x + alpha * p; r = r - alpha * q
            z = self._vcycle(st, 0, r)
            rz_new = jnp.sum(r * z, axis=0)
            p = z + (rz_new / rz) * p
            return it + 1, x, r, p, rz_new

        it, x, r, p, rz = jax.lax.while_loop(cond, body, (0, x, r, p, rz))
        return x, it, jnp.linalg.norm(r, axis=0) / bn

    # ------------------------------------------------------------ public interface (numpy in/out)
    def update(self, Ee):
        self.Ee = jnp.asarray(Ee, dtype=jnp.float64)

    def solve(self, b, x0=None):
        b = np.atleast_2d(np.asarray(b, float).T).T
        x0 = np.zeros_like(b) if x0 is None else np.asarray(x0, float).reshape(b.shape)
        x, it, res = self._pcg(self.Ee, jnp.asarray(b), jnp.asarray(x0))
        self.last_res = np.asarray(res)
        return np.asarray(x), int(it)
