import contextlib, io, re
import numpy as np
from topopt.top2d import top2d
from topopt.top3d import hex8_stiffness
import cantilever2d, check3d, check_mgcg      # the repository's acceptance checks, used as a library
from examples.chair import build


def captured(fn, *a, **kw):
    """Run fn, return (its result, what it printed); the printed lines also go to this notebook."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    print(buf.getvalue(), end="")
    return out, buf.getvalue()


R = {}

with stage("A1 cantilever 2D"):
    R["cantilever2d"] = []
    for n in N_H_2D:
        fe, eb, tim = cantilever2d.tip_deflection(L_OVER_H, n, E=E0, nu=NU, F=F, t=T)
        R["cantilever2d"].append(dict(n_h=n, delta_fe=fe, delta_eb=eb, delta_tim=tim, rel_eb=fe / eb - 1, rel_tim=fe / tim - 1))
        log(f"2D cantilever n_h={n:3d}: FE {fe:.4f}, Euler-Bernoulli {eb:.4f}, Timoshenko {tim:.4f}, FE/Tim-1 {fe / tim - 1:+.4%}")

with stage("A2 cantilever 3D"):
    R["cantilever3d"] = []
    for n in N_H_3D:
        fe, eb, tim = check3d.cantilever(n, Lh=L_OVER_H, nu=NU, E=E0, F=F)
        R["cantilever3d"].append(dict(n_h=n, delta_fe=fe, delta_eb=eb, delta_tim=tim, rel_eb=fe / eb - 1, rel_tim=fe / tim - 1))
        log(f"3D cantilever n_h={n:3d}: FE {fe:.4f}, Euler-Bernoulli {eb:.4f}, Timoshenko {tim:.4f}, FE/Tim-1 {fe / tim - 1:+.4%}")

with stage("B element modes"):
    k0 = hex8_stiffness(NU)
    ev = np.linalg.eigvalsh(k0)
    ok_repo, _ = captured(check3d.check_element, NU)          # the repository's own verdict
    R["element3d"] = dict(sym_err=float(np.abs(k0 - k0.T).max()), min_eig=float(ev.min()), max_eig=float(ev.max()),
                          zero_modes=int(np.sum(np.abs(ev) < 1e-10 * ev.max())), eigenvalues=ev.tolist(),
                          repo_check_pass=bool(ok_repo))
    log(f"hex8 element: {R['element3d']['zero_modes']} zero modes, symmetry error {R['element3d']['sym_err']:.1e}")

with stage("C gradient"):
    ok_repo, text = captured(check3d.check_gradient, seed=GRAD_SEED, penal=GRAD_PENAL, nu=NU, Emin=GRAD_EMIN)
    rel = float(re.search(r"= ([0-9.eE+-]+) over (\d+) elements", text).group(1))
    R["gradient"] = dict(rel_diff=rel, repo_check_pass=bool(ok_repo))
    log(f"adjoint vs automatic differentiation: max relative difference {rel:.2e}")

with stage("D multigrid vs direct"):
    prob, _ = build(MG_N)
    worst, text = captured(check_mgcg.compare, MG_N, np.full(prob.grid.nel, MG_RHO), f"uniform rho={MG_RHO}",
                           nu=NU, penal=MG_PENAL, Emin=E_MIN)
    groups = [dict(group=int(g), cg_iterations=int(it), rel_diff=float(d))
              for g, it, d in re.findall(r"group (\d+): levels \d+, CG its (\d+), rel\. diff ([0-9.eE+-]+)", text)]
    R["mgcg"] = dict(worst_rel_diff=float(worst), groups=groups, nel=int(prob.grid.nel))
    log(f"multigrid vs PARDISO: worst relative difference {worst:.2e} over {len(groups)} support sets")

with stage("E MBB beam"):
    R["mbb"] = []
    for nx, ny, ft in MBB_CASES:
        t = time.time()
        r = top2d(nx, ny, VOLFRAC, PENAL, RMIN_FACTOR * nx, ft, E0=E0, Emin=E_MIN, nu=NU,
                  move=MOVE, tol=OC_TOL, maxit=MAXIT, verbose=False)
        key = f"{nx}x{ny} ft{ft}"
        ref = MBB_PAPER[key]
        np.savez_compressed(WORK / f"mbb_{nx}x{ny}_ft{ft}.npz", xPhys=r["xPhys"], c=r["c"], iters=r["iters"])
        R["mbb"].append(dict(case=key, nelx=nx, nely=ny, ft=ft, c=r["c"], paper=ref, rel_err=r["c"] / ref - 1,
                             iters=r["iters"], volume=float(r["xPhys"].mean()), seconds=round(time.time() - t, 2)))
        log(f"MBB {key}: c = {r['c']:.4f} (paper {ref:.2f}, rel. err {r['c'] / ref - 1:+.2e}), {r['iters']} iterations")
