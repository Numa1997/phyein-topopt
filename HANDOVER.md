# HANDOVER: phyein-topopt

State on 11 November 2026, written for whoever continues (a person or a Claude session). For the
method, load the skill `topology-optimisation`; for running things, load `topopt-toolchain`.

## What this is

A reusable structural topology-optimisation capability: give it a design space, loads and supports;
it computes the stresses with finite elements and removes material until only the stiffest load
path for the chosen amount of material remains, then delivers a watertight STL, stress-coloured
stills, a 3D file with stress colours (GLB, for three.js) and a film ("loads → stress → the load
path stays"). Every result is a **design proposal from approximate voxel FEA, not a structural
verification**, and is labelled so.

## What exists

- **Repository** `phyein-topopt` (public, on the user's GitHub account). Solver code in `topopt/`:
  `top2d.py` (2D, port of the 88-line code), `top3d.py` (3D voxel SIMP, element with selective
  reduced integration, density filter, Heaviside projection, passive regions, several load cases
  and support sets), `mgcg.py` (matrix-free multigrid-preconditioned CG solver), `render.py`
  (STL/GLB, renders). Load cases in `examples/`: `chair.py` (the chair), `chair_check.py` (design
  check), `chair_animate.py` (STL, GLB, stills, film). Acceptance checks in `benchmarks/`.
- **Container image** `ghcr.io/<owner>/phyein-topopt:latest`, and `:<commit sha>` for every build
  (`<owner>` = the GitHub login in lower case). About 4 GB: Python 3.13 with pinned NumPy, SciPy,
  pypardiso/MKL, scikit-image, trimesh, PyVista/VTK, JAX (CPU), ffmpeg. Free to store and pull.
- **Workflow `build-test-publish-image`** (`.github/workflows/image.yml`): on every push that touches
  code or dependencies, builds the image, runs all acceptance checks inside it, and only then
  publishes it.
- **Workflow `run-case`** (`.github/workflows/run.yml`), started by hand. Inputs: `case` (chair or
  mbb), `n` (voxels across the chair's half width), `volfrac`, `rmin` (filter radius in voxels),
  `penal`, `label` (result name), `animate` (yes/no: STL, stills, film), `maxit` (iteration limit;
  3 is enough to time a grid). Results come back as a downloadable artifact named
  `<case>-<label>`, kept 90 days.
- **CONNECTORS.md**: MCP servers and connectors explained, freecad-mcp audited and set up step by
  step, how repositories are attached to sessions.

## Verified (all run inside the image on every build)

- 2D MBB beam against Andreassen et al. 2011 (Fig. 3): all four published compliances reproduced,
  largest relative error 2·10⁻⁵.
- Cantilever tip deflection against beam theory (L/h = 10, 16 elements through the section): within
  0.02 % of F L³/(3 E I) and 0.74 % below Timoshenko's value, which includes shear.
- 3D element: symmetric, exactly 6 zero-energy modes. Adjoint sensitivity against JAX automatic
  differentiation: 6·10⁻¹⁵ relative.
- Multigrid solver against the direct solver (PARDISO): displacements agree to 4·10⁻¹² relative.

## How to run a new case, step by step

1. Write `examples/<case>.py` in the style of `chair.py`: a docstring with the physical setup, every
   dimension, load and material value as an argument with its source, loads on passive solid pads,
   supports that stop all rigid-body motion, a symmetric half model only if loads allow it
   (antisymmetric loads get their own support set, see `chair.py`).
2. Do the hand statics first (reactions, tipping) and write them into the docstring.
3. Time it locally with `--maxit 3`, then add the case to the `case` choice in `run.yml`, push, and
   wait for the image workflow to pass (about 7 minutes).
4. Run a short material-fraction sweep at medium resolution with `animate=no`, read each
   `design_check.json`, pick the smallest fraction meeting the design goal.
5. Launch the fine run with `animate=yes`, download the artifact, check convergence, connectivity
   and the re-analysis before showing anything.

## How long it takes and what it costs

Measured on a free GitHub runner (4 vCPU, 16 GB): medium chair (12 voxels across the half width,
19 mm voxels, 13 824 elements) about 8.7 s per iteration, 30 to 45 minutes for a full run; fine chair
(20 voxels, 11 mm, 64 000 elements) about 27 s per iteration, 1.5 to 2.3 hours plus 20 to 30 minutes
for the film. Runs need 160 to 300 iterations. A job is killed after 6 hours or at 16 GB. Cost:
nothing (public repository: free unlimited runners and free image storage). The session's own
2-core machine is about twice as fast per iteration as the runner but cannot hold a run beyond the
end of a turn.

## The chair: what has been found so far

Material-fraction sweep on the medium grid (19 mm voxels, PA12, all four load cases, fresh starts),
re-checked with `examples/chair_check.py`:

- 3 %: seat deflection 0.26 mm; 99th-percentile stress 2.3 MPa (test A), 1.4 (B), 1.6 (C); one
  piece reaching all feet; buckling factor 2.24.
- 5 %: 0.40 mm; 1.2 / 0.9 / 0.9 MPa; one piece; buckling factor 2.24.
- 8 %: 0.20 mm; 0.8 / 0.6 / 0.5 MPa; one piece; buckling factor 3.35.
- 12 %: 0.09 mm; 0.5 / 0.4 / 0.3 MPa; one piece; buckling factor 14.5.

Limits: deflection 5 mm, stress 21 MPa (yield 42 MPa / 2), buckling factor ≥ 2 (an added check).
All fractions meet the stated goals by wide margins; the binding check is buckling of the back legs.
On this grid the leg sections are only about two voxels (identical factor at 3 % and 5 %), the seat
is one to two voxels thick (deflections indicative), and the 3–8 % runs stopped at the
300-iteration limit with grey fractions of 1.2–2.9 %. Shape at 3 %: four slender, slightly splayed
legs, a small fork under the front seat edge, three ribs behind the backrest; unlike the reference
chair the legs barely branch. Fine runs (11 mm voxels) at 1.5 %, 2 % and 3 % were launched to settle
the fraction by the same checks.

## Known limitations

- Voxel FEA: surfaces are staircases, so surface stresses are indicative only; the stress check uses
  the 99th percentile over solid voxels and also reports the maximum. Members thinner than about 8
  voxels are stiffer in the model than in reality.
- The chair's loads follow EN 1728 tests with forces from EN 16139 Level 1 / EN 1335-2 (1600 N seat
  + 560 N back from a test-lab summary; 1300 N front edge and 400 N sideways from a manufacturer's
  certificate). The exact loading pads, points and fixtures are in the paywalled standard; the model
  spreads loads over areas and fixes the feet. With these load points the back-load cases sit at the
  rearward tipping edge (front feet would need about −1 to −2 N).
- PA12 values: E = 1650 MPa (EOS PA 2200 datasheet), ν = 0.43 (HP PA12 datasheet), strength 42 MPa
  in the weakest build direction (EOS PA 2200 CarbonReduced, Z), used as yield; printed parts vary
  with direction and process.
- No GPU path yet; the Kaggle/JAX solver is the next step.
