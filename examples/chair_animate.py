"""Turn a finished chair run into: STL + GLB (stress-coloured) + stills + MP4 animation
"loads -> stress -> only the load path stays".

Stage 1  the design block (everything that may hold material) with the loads and feet
Stage 2  von Mises stress in the full solid block under load case 2 (sit + lean back)
Stage 3  the optimisation history: iso-surface of xbar = 0.5 at selected iterations,
         coloured by the stress recomputed for that intermediate design
Stage 4  the final smoothed shape, one full turn
One fixed colour scale (98th percentile of the final design's stress) for all stages, so the
viewer sees stress rise as material is removed: less material carries the same load.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.chair import build  # noqa: E402
from topopt.top3d import solve_static, element_von_mises  # noqa: E402
from topopt.render import (mirror_x, density_to_mesh, vertex_field, plot_scene,  # noqa: E402
                           STRESS_CMAP)

LABEL = "approximate voxel FEA · design proposal, not a structural verification"


def stress_field(prob, xPhys, nu):
    """Stress envelope: per element, the largest von Mises stress over all load cases. For the
    antisymmetric case the mirrored half carries the same magnitude, so mirroring stays valid."""
    U = solve_static(prob, xPhys, 1.0, 1e-9, 3.0, nu)
    return element_von_mises(prob.grid, U, xPhys, 1.0, nu).max(axis=1)


def camera(meta, azim_deg, elev_deg=18.0, r=2.15):
    W, D, H = meta["W"], meta["D"], meta["Hc"]
    c = np.array([W / 2, D / 2, H * 0.45])
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    pos = c + r * np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])
    return [tuple(pos), tuple(c), (0, 0, 1)]


def load_arrows(meta):
    a, W = meta["a"], meta["W"]
    zs, yb = meta["ks"] * a, meta["jb"] * a
    arrows = []
    for x in np.linspace(0.12, W - 0.12, 3):
        for y in np.linspace(0.07, yb - 0.07, 3):
            arrows.append(((x, y, zs + 0.17), (0, 0, -1), 0.15))
    zs_side = zs + 0.02
    for y in (0.08, yb - 0.08):
        arrows.append(((-0.22, y, zs_side), (1, 0, 0), 0.15))      # sideways shove
    zb0 = meta["kb0"] * a
    for x in np.linspace(0.12, W - 0.12, 3):
        for z in np.linspace(zb0 + 0.04, meta["Hc"] - 0.04, 2):
            arrows.append(((x, yb - 0.17, z), (0, 1, 0), 0.15))
    return arrows


def feet_meshes(meta):
    import pyvista as pv
    a, f, W, D = meta["a"], meta["f"], meta["W"], meta["D"]
    s = f * a
    out = []
    for x0 in (0.0, W - s):
        for y0 in (0.0, D - s):
            out.append((pv.Box(bounds=(x0 - 0.01, x0 + s + 0.01, y0 - 0.01, y0 + s + 0.01, -0.02, 0.0)),
                        dict(color="#2b2522")))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames_history", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--cam_r", type=float, default=2.75, help="camera distance for film frames [m]")
    A = ap.parse_args()
    run = Path(A.run_dir)
    meta_all = json.load(open(run / "chair_run.json"))
    meta, nu = meta_all["meta"], meta_all["params"]["nu"]
    d = np.load(run / "chair_result.npz")
    nx, ny, nz = (int(v) for v in d["shape"]); a = float(d["a"])
    prob, _ = build(meta["n"])
    hist = d["hist"].astype(float)
    frames = run / "frames"; frames.mkdir(exist_ok=True)
    for old in frames.glob("*.png"):
        old.unlink()
    win = tuple(A.size)

    def to_full(v):
        return mirror_x(v.reshape(nx, ny, nz))

    # ---- final design, its stress, colour scale, STL + GLB + stills
    xf = np.clip(d["xPhys"], 0, 1)
    vm_f = stress_field(prob, xf, nu) / 1e6
    mesh_f = density_to_mesh(to_full(xf), a)
    s_f = vertex_field(to_full(vm_f), a, mesh_f.vertices)
    clim = (0.0, float(np.percentile(s_f, 98)))
    mesh_f.export(run / "chair.stl")
    rgba = (STRESS_CMAP(np.clip(s_f / clim[1], 0, 1)) * 255).astype(np.uint8)
    m_col = mesh_f.copy(); m_col.visual.vertex_colors = rgba
    m_col.export(run / "chair_stress.glb")
    for name, az in (("still_front", -40), ("still_back", 140)):
        plot_scene(mesh_f, s_f, clim, str(run / f"{name}.png"), title=LABEL, window=(1600, 1600),
                   camera=camera(meta, az, r=2.3))

    k = 0
    def shot(mesh, scal, az, title, arrows=(), extra=()):
        global k
        plot_scene(mesh, scal, clim, str(frames / f"f{k:05d}.png"), title=title, arrows=arrows,
                   window=win, camera=camera(meta, az, r=A.cam_r), extra_meshes=extra)
        k += 1

    arrows, feet = load_arrows(meta), feet_meshes(meta)
    az = -40.0
    # ---- stage 1: the design block and the loads (density 1 wherever material is allowed)
    block = np.ones(nx * ny * nz); block[prob.passive_void] = 0.0
    mesh_b = density_to_mesh(to_full(block), a, smooth_iters=0)
    n1 = int(2.0 * A.fps)
    for _ in range(n1):
        shot(mesh_b, None, az, "1  the design space, the loads (arrows) and the feet\n" + LABEL, arrows, feet)
        az += 0.25
    # ---- stage 2: stress in the full block
    vm_b = stress_field(prob, block, nu) / 1e6
    s_b = vertex_field(to_full(vm_b), a, mesh_b.vertices)
    for _ in range(int(2.5 * A.fps)):
        shot(mesh_b, s_b, az, "2  stress in the solid block: most of it carries almost nothing\n" + LABEL, arrows, feet)
        az += 0.25
    # ---- stage 3: history, denser sampling early where the design changes fastest
    nh = hist.shape[0]
    idx = np.unique(np.round(np.geomspace(1, nh, A.frames_history)).astype(int) - 1)
    hold = max(2, int(6.0 * A.fps / len(idx)))
    for t, i in enumerate(idx):
        xi = np.clip(hist[i], 0, 1)
        mi = density_to_mesh(to_full(xi), a)
        si = vertex_field(to_full(stress_field(prob, xi, nu) / 1e6), a, mi.vertices)
        for _ in range(hold):
            shot(mi, si, az, f"3  removing material where stress is low  (iteration {i+1} of {nh})\n" + LABEL,
                 arrows, feet)
            az += 0.25
        print(f"history frame {t+1}/{len(idx)}", flush=True)
    # ---- stage 4: final shape, one turn
    n4 = int(5.0 * A.fps)
    for j in range(n4):
        shot(mesh_f, s_f, az + 360.0 * j / n4, "4  what stays is the load path\n" + LABEL, arrows, feet)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(A.fps), "-i", str(frames / "f%05d.png"),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(run / "chair_load_path.mp4")]
    subprocess.run(cmd, check=True)
    json.dump(dict(clim_MPa=clim, frames=k, stress_final_p98_MPa=clim[1],
                   stress_block_p98_MPa=float(np.percentile(s_b, 98))),
              open(run / "render_summary.json", "w"), indent=1)
    print(f"wrote {k} frames, STL, GLB, stills and MP4 to {run}")
