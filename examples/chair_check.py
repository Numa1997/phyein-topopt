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
    res["pass"] = bool(ok)
    json.dump(res, open(run / "design_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return ok

if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1]) else 1)
