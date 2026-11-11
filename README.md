# phyein-topopt

> The general physics toolchain (OpenFOAM, finite elements, JAX, Kaggle, website delivery) now lives in [phyein-sim](https://github.com/Numa1997/phyein-sim); this repository keeps the structural topology-optimisation code and its image.

Structural topology optimisation, from a block of material and its loads to the shape that carries
them: voxel finite elements, SIMP with a density filter and Heaviside projection, optimality-criteria
updates, then marching cubes, smoothing, a stress-coloured render and an animation.

**Results are design proposals from approximate voxel FEA, not structural verification.**

## Verified
| check | command | result |
|---|---|---|
| MBB beam vs. Andreassen et al. 2011, Fig. 3 (4 cases) | `python benchmarks/mbb_check.py` | max. relative error 2·10⁻⁵ |
| 2D FE cantilever vs. Timoshenko beam theory | `python benchmarks/cantilever2d.py` | −0.23 % at 32 elements through the height |
| 3D element: symmetry, 6 rigid-body modes | `python benchmarks/check3d.py` | pass |
| 3D FE cantilever vs. Timoshenko (Cowper κ) | same | −0.93 % at 16 elements through the section |
| adjoint sensitivity vs. JAX automatic differentiation | same | 3·10⁻¹⁵ relative |

## Layout
- `topopt/top2d.py`: 2D SIMP, a line-by-line port of the 88-line code
- `topopt/top3d.py`: 3D hex8 voxel SIMP (density filter, projection, passive regions, several load cases)
- `topopt/render.py`: densities → STL/GLB, stress per vertex, PyVista renders
- `examples/chair.py`, `examples/chair_animate.py`: the seat-and-back load case and its film
- `benchmarks/`: the acceptance checks above
- `Dockerfile`, `requirements.txt`: the pinned toolchain image
- `.github/workflows/image.yml`: build → run all checks in the image → push to `ghcr.io/<owner>/phyein-topopt`
- `.github/workflows/run.yml`: run a case on demand (Actions tab → run-case → Run workflow)

## Run
```sh
docker pull ghcr.io/<owner>/phyein-topopt:latest                      # <owner> in lower case
docker run --rm ghcr.io/<owner>/phyein-topopt:latest                  # acceptance checks
docker run --rm -v "$PWD/out:/work" ghcr.io/<owner>/phyein-topopt:latest \
  sh -c "cd /opt/phyein-topopt && python examples/chair.py --n 16 --out /work/chair && python examples/chair_animate.py /work/chair"
```
All parameters (volume fraction, penalty, filter radius, Poisson ratio, loads, dimensions) are
command-line arguments or function arguments; nothing is fixed in the code.
