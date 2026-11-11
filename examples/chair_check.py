"""Design check of a finished chair against the stated design goal (parameter choices):
  stiffness: seat deflection under load case S (100 kg sitter) <= delta_max = 5 mm
  strength:  von Mises stress under the standard's test loads A, B, C <= sigma_y / 2
             sigma_y = 42 MPa: PA12 tensile strength at max load, weakest build direction
             (EOS PA 2200 CarbonReduced, Z); PA12 yields at about max load (HP PA12: elongation
             at yield 8-10 %), so the datasheet strength is used as yield.
Stress measure: 99th percentile over solid voxels (xbar >= 0.5), because single voxels at the
clamped feet and at load-area edges carry singular peaks that no mesh refinement makes finite;
the maximum is reported too. Writes design_check.json; exit code 0 if both goals are met."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.chair import build
from topopt.top3d import solve_static, element_von_mises
from topopt.checks import connectivity

def check(run_dir, delta_max=5e-3, sigma_y=42e6, sf=2.0):
    run = Path(run_dir)
    J = json.load(open(run / "chair_run.json")); P = J["params"]; meta = J["meta"]
    d = np.load(run / "chair_result.npz"); x = np.clip(d["xPhys"], 0, 1)
    prob, m = build(meta["n"])
    E0, nu, p = P["E0"], P["nu"], P["penal"]
    U = solve_static(prob, x, E0, 1e-9 * E0, p, nu)
    seat = np.array(m["seat_nodes"])
    delta = float(-U[3 * seat + 2, 0].min())                         # largest downward seat displacement
    vm = element_von_mises(prob.grid, U, x, E0, nu)
    solid = x >= 0.5
    res = dict(volfrac=P["volfrac"], n=meta["n"], voxel_mm=meta["a"] * 1e3, delta_service_mm=delta * 1e3,
               delta_max_mm=delta_max * 1e3, sigma_allow_MPa=sigma_y / sf / 1e6, stress={})
    ok = delta <= delta_max
    for L, name in ((1, "A seat+back"), (2, "B front edge"), (3, "C sideways")):
        s = vm[solid, L]
        p99, smax = float(np.percentile(s, 99)) / 1e6, float(s.max()) / 1e6
        res["stress"][name] = dict(p99_MPa=p99, max_MPa=smax)
        ok &= p99 <= sigma_y / sf / 1e6
    # connectivity of the half model (the mirror copy joins it across the plane x = 0)
    g = prob.grid
    x3 = x.reshape(g.nx, g.ny, g.nz)
    ncomp, sizes, lab = connectivity(x3)
    main = int(np.argmax(sizes)) + 1 if ncomp else 0
    f = m["f"]
    feet = {"front": bool((lab[g.nx - f:, :f, 0] == main).any()), "back": bool((lab[g.nx - f:, g.ny - f:, 0] == main).any())}
    res["connectivity"] = dict(components=int(ncomp), sizes=[int(v) for v in sizes[:5]],
                               main_share=float(sizes.max() / sizes.sum()) if ncomp else 0.0,
                               reaches_feet=feet)
    ok &= ncomp >= 1 and all(feet.values())

    # Buckling hand estimate (an addition to the user's goals, an engineering choice; labelled):
    # slice the legs at mid-height (half model: legs near the outer side x = W/2), take each leg
    # section's area and smallest principal second moment, Euler load with pinned ends (K = 1,
    # conservative), compare with the axial force of the most loaded leg in test A from statics:
    # back legs carry R_back = (F_s y_s + F_b h_b) / D, i.e. R_back / 2 per leg.
    from scipy import ndimage
    a = g.a
    kmid = max(1, (m["ks"] - m["kt"]) // 2)
    sec = x3[:, :, kmid] >= 0.5
    lab2, n2 = ndimage.label(sec)
    legs = []
    for c in range(1, n2 + 1):
        ii, jj = np.nonzero(lab2 == c)
        A = ii.size * a * a
        xc, yc = (ii + 0.5) * a, (jj + 0.5) * a
        Ixx = np.sum((yc - yc.mean()) ** 2) * a * a + ii.size * a ** 4 / 12
        Iyy = np.sum((xc - xc.mean()) ** 2) * a * a + ii.size * a ** 4 / 12
        Ixy = np.sum((xc - xc.mean()) * (yc - yc.mean())) * a * a
        Imin = 0.5 * (Ixx + Iyy) - np.sqrt(0.25 * (Ixx - Iyy) ** 2 + Ixy ** 2)
        legs.append(dict(area_mm2=A * 1e6, Imin_mm4=Imin * 1e12, y_mm=float(yc.mean() * 1e3)))
    Lleg = (m["ks"] - m["kt"]) * a
    y_s = 0.5 * m["jb"] * a
    h_b = 0.5 * (m["kb0"] + g.nz) * a
    F_leg = (m["F_seat_test"] * y_s + m["F_back_test"] * h_b) / m["D"] / 2
    if legs:
        weakest = min(legs, key=lambda d: d["Imin_mm4"])
        Pcr = np.pi ** 2 * E0 * weakest["Imin_mm4"] * 1e-12 / Lleg ** 2
        res["buckling_estimate"] = dict(note="hand estimate, pinned-pinned Euler, mid-height section; "
                                             "an added check, not part of the stated goal",
                                        sections_at_mid_height=legs, leg_length_m=Lleg,
                                        leg_force_testA_N=F_leg, Pcr_weakest_N=float(Pcr),
                                        factor=float(Pcr / F_leg), passes_factor_2=bool(Pcr >= 2 * F_leg))
    res["pass"] = bool(ok)
    json.dump(res, open(run / "design_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return ok

if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1]) else 1)
