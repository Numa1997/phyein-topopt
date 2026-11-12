def check(name, value, limit, ok, note=""):
    CHECKS[name] = {"value": value, "limit": limit, "pass": bool(ok), "note": note}
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {value!s:>24}   limit {limit}   {note}")


print("1) Against theory, automatic differentiation, the direct solver and the paper")
e = max(abs(r["rel_err"]) for r in R["mbb"])
check("MBB: largest |C / C_paper - 1| over 4 cases", f"{e:.2e}", MBB_TOL, e < MBB_TOL)
r2, r3 = R["cantilever2d"][-1], R["cantilever3d"][-1]
check(f"2D cantilever: FE / Timoshenko - 1 at n_h = {r2['n_h']}", f"{r2['rel_tim']:+.4%}", f"±{CANTILEVER_TOL:.0%}",
      abs(r2["rel_tim"]) < CANTILEVER_TOL)
check(f"3D cantilever: FE / Timoshenko - 1 at n_h = {r3['n_h']}", f"{r3['rel_tim']:+.4%}", f"±{CANTILEVER_TOL:.0%}",
      abs(r3["rel_tim"]) < CANTILEVER_TOL)
el = R["element3d"]
check("hex8 element: zero eigenvalues (rigid-body modes)", el["zero_modes"], ZERO_MODES_EXPECTED,
      el["zero_modes"] == ZERO_MODES_EXPECTED and el["sym_err"] < 1e-12 and el["min_eig"] > -1e-10 * el["max_eig"],
      f"symmetry error {el['sym_err']:.1e}, smallest eigenvalue {el['min_eig']:+.1e}")
check("gradient: max |AD - adjoint| / max |AD|", f"{R['gradient']['rel_diff']:.2e}", GRAD_TOL,
      R["gradient"]["rel_diff"] < GRAD_TOL)
check("multigrid: relative difference to PARDISO", f"{R['mgcg']['worst_rel_diff']:.2e}", MG_TOL,
      R["mgcg"]["worst_rel_diff"] < MG_TOL)

print("\n2) Against the numbers the repository records, at the precision they are printed")
RECORDED = [   # (name, value as printed, where, this run's value, how to round it)
    ("MBB largest relative error", "2e-05", "HANDOVER.md, 'Verified'", e, lambda v: f"{v:.0e}"),
    ("2D cantilever, % from Timoshenko at 32", "-0.23", "README.md, 'Verified'", 100 * R["cantilever2d"][-1]["rel_tim"],
     lambda v: f"{v:.2f}"),
    ("3D cantilever, % from Timoshenko at 16", "-0.74", "HANDOVER.md, 'Verified' (README.md says -0.93, from before "
     "selective reduced integration)", 100 * R["cantilever3d"][-1]["rel_tim"], lambda v: f"{v:.2f}"),
]
for name, printed, where, v, fmt in RECORDED:
    check(f"recorded: {name}", fmt(v), f"recorded {printed}", fmt(v) == printed, where)

print("\n3) Against a reference run of this notebook (see the text above)")
REFERENCE = {{REFERENCE}}


def deterministic(res):
    """Every number that must come out the same on any machine, up to rounding: name -> value."""
    out = {}
    for key in ("cantilever2d", "cantilever3d"):
        for r in res[key]:
            for q in ("delta_fe", "delta_eb", "delta_tim"):
                out[f"{key} n_h={r['n_h']} {q}"] = r[q]
    for r in res["mbb"]:
        out[f"mbb {r['case']} C"] = r["c"]
        out[f"mbb {r['case']} iterations"] = r["iters"]
        out[f"mbb {r['case']} volume"] = r["volume"]
    ev = res["element3d"]["eigenvalues"]
    for i, v in enumerate(ev):
        if abs(v) > 1e-10 * max(ev):      # the six zero eigenvalues are rounding noise, compared by count instead
            out[f"hex8 eigenvalue {i}"] = v
    return out


if REFERENCE is None:
    print("no reference stored yet: this run is the reference")
    CHECKS_REFERENCE = None
else:
    a, b = deterministic(R), deterministic(REFERENCE["results"])
    assert a.keys() == b.keys(), "the reference holds different quantities"
    diffs = {k: abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in a}
    worst = max(diffs, key=diffs.get)
    print(f"reference made {REFERENCE['made']} on: {REFERENCE['machine']}")
    print(f"{len(diffs)} numbers compared; {sum(d == 0 for d in diffs.values())} identical to the last bit")
    check("reference run: largest relative difference", f"{diffs[worst]:.2e}", 1e-8, diffs[worst] < 1e-8,
          f"in '{worst}'")
    CHECKS["reference run: largest relative difference"]["all"] = diffs

json.dump(CHECKS, open(WORK / "verification.json", "w"), indent=1, default=str)
log(f"verification: {sum(c['pass'] for c in CHECKS.values())} of {len(CHECKS)} checks passed")
