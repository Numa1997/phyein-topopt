"""Densities -> smooth surface -> STL, stress-coloured stills and the animation frames.

Pipeline:
  1. mirror the half model about the symmetry plane x = 0 (full chair)
  2. pad with void and extract the iso-surface xbar = level by marching cubes
     (Lorensen & Cline 1987; skimage.measure.marching_cubes), vertices in metres
  3. Taubin smoothing (lambda/mu, Taubin 1995): removes voxel steps without shrinking the part
  4. per-vertex stress: trilinear interpolation of the element von Mises field at the vertices
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes
import trimesh

BEIGE = "#f4ecdf"; INK = "#2b2522"; WINE = "#7a1f2b"
STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress_beige_wine", ["#efe3cf", "#e3b27a", "#c8553d", "#7a1f2b", "#2b0b10"])


def mirror_x(field3d: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) half field with symmetry plane at i = 0 -> (2nx, ny, nz) full field."""
    return np.concatenate([field3d[::-1], field3d], axis=0)


def density_to_mesh(x: np.ndarray, a: float, level: float = 0.5, smooth_iters: int = 12) -> trimesh.Trimesh:
    """x: element densities on a (Nx, Ny, Nz) grid of voxel side a. Element centres sit at
    (i + 1/2) a; padding by one void layer closes the surface at the domain boundary."""
    xp = np.pad(x.astype(float), 1, constant_values=0.0)
    verts, faces, _, _ = marching_cubes(xp, level=level, spacing=(a, a, a))
    verts -= a * 0.5                                   # undo pad (1 voxel) and centre offset (1/2)
    m = trimesh.Trimesh(verts, faces, process=True)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth_iters)
    return m


def vertex_field(field3d: np.ndarray, a: float, verts: np.ndarray) -> np.ndarray:
    """Interpolate an element-centred field (values at (i+1/2)a, ...) to arbitrary points."""
    axes = [a * (np.arange(n) + 0.5) for n in field3d.shape]
    f = RegularGridInterpolator(axes, field3d, bounds_error=False, fill_value=None)
    return f(np.clip(verts, [ax[0] for ax in axes], [ax[-1] for ax in axes]))


def plot_scene(mesh: trimesh.Trimesh, scalars, clim, out_png, title="", arrows=(), window=(1400, 1400),
               camera=None, label="von Mises stress [MPa]", show_bar=True, extra_meshes=()):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pm = pv.wrap(mesh)
    p = pv.Plotter(off_screen=True, window_size=window)
    p.set_background(BEIGE)
    if scalars is None:
        p.add_mesh(pm, color=WINE, smooth_shading=True, specular=0.3)
    else:
        pm["s"] = scalars
        p.add_mesh(pm, scalars="s", cmap=STRESS_CMAP, clim=clim, smooth_shading=True, specular=0.25,
                   show_scalar_bar=show_bar,
                   scalar_bar_args=dict(title=label, color=INK, fmt="%.2f", vertical=False,
                                        position_x=0.3, position_y=0.03, width=0.4, height=0.06, title_font_size=14, label_font_size=12))
    for em, kw in extra_meshes:
        p.add_mesh(pv.wrap(em) if not isinstance(em, pv.DataSet) else em, **kw)
    for start, vec, mag in arrows:
        p.add_mesh(pv.Arrow(start=start, direction=vec, scale=mag, tip_length=0.3, tip_radius=0.08,
                            shaft_radius=0.03), color=INK)
    if title:
        p.add_text(title, position="upper_left", font_size=12, color=INK)
    if camera is not None:
        p.camera_position = camera
    else:
        p.view_isometric(); p.camera.azimuth = -35; p.camera.elevation = 8
    p.screenshot(out_png)
    cam = p.camera_position
    p.close()
    return cam
