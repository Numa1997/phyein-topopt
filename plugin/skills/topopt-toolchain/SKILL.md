---
name: topopt-toolchain
description: "Run the phyein-topopt toolchain: pull and run the pinned image, launch and read runs on free GitHub runners, time and cost estimates, Kaggle, storing results and publishing to phyeinusul.com."
---

# The phyein-topopt toolchain: one image, three places to run, one place to publish

Use this skill whenever a topology-optimisation job has to be run, repeated, scaled up, stored or
published. The method itself (physics, decisions, benchmarks, failure modes) is in the companion
skill `topology-optimisation`; load it first if the task involves any modelling choice. The
repository's `HANDOVER.md` lists everything that exists.

## 1. What lives where, and why

- **The code**: the public GitHub repository `phyein-topopt` on the user's account. It is the single
  source of truth: solver (`topopt/`), load cases (`examples/`), acceptance checks (`benchmarks/`),
  the `Dockerfile` with pinned versions, and two workflows. Public, because GitHub's standard runners
  are free and unlimited only for public repositories (4 vCPU, 16 GB RAM, 14 GB SSD, 6 h per job,
  20 jobs at once on the Free plan; no GPU) and public images on the GitHub Container Registry
  (GHCR) are free.
- **The tools**: one Docker image, `ghcr.io/<owner>/phyein-topopt:latest` (and `:<commit-sha>` for
  every build), about 4 GB: Python 3.13, NumPy/SciPy, pypardiso + Intel MKL (direct solver), pyamg,
  scikit-image, trimesh, PyVista/VTK (off-screen rendering), JAX (CPU), ffmpeg. The optimiser uses
  the matrix-free multigrid solver (`topopt/mgcg.py`); PARDISO stays only as the reference. `<owner>`
  is the GitHub login in lower case (`gh api user --jq .login | tr A-Z a-z`). Docker image names must
  be lower case. Never write the user's name into files; use `${{ github.repository_owner }}` in
  workflows and `<owner>` in docs.
- **The results**: GitHub Actions artifacts (kept 90 days by default) for every run; the keepers are
  copied into the website repository `SiteTest`, which Cloudflare (Workers Builds) publishes from
  `main` to phyeinusul.com. Cloudflare is for serving files and the viewer only; no heavy compute
  there.
- **Cost**: nothing. Public repository, free runners, free image storage.

Two things the first build taught: (1) a new GHCR package can only be written by the workflow's
built-in token once the image carries the label `org.opencontainers.image.source=<repo URL>`
(otherwise the registry answers "denied" after uploading every layer); the workflow adds it from
`${{ github.server_url }}/${{ github.repository }}`. (2) Pushing from a session needs the Claude
GitHub App to have access to the repository (GitHub → Settings → Applications → Installed GitHub
Apps → Claude → Configure → Repository access); attaching the repository alone is not enough.

The image is built by `.github/workflows/image.yml` on every push to `main` that touches code or
dependencies: build, run every acceptance check inside the new image, and only if all pass push
`:latest` and `:<sha>` to GHCR. A broken toolchain can therefore never become `latest`.

## 2. Run it in a Claude session (cloud workspace)

1. Attach the repository with push access if you will commit (`add_repo`, owner = the user's login,
   repo `phyein-topopt`), then clone it. Read-only work needs no attachment for a public repo.
2. Docker: if `docker info` fails, start the daemon: `nohup dockerd > /tmp/dockerd.log 2>&1 &`,
   wait about 8 s. Then `docker pull ghcr.io/<owner>/phyein-topopt:latest` (about 20 s).
3. Acceptance checks: `docker run --rm ghcr.io/<owner>/phyein-topopt:latest` (about 1 minute;
   prints `ALL PASS`).
4. A case, output to the current folder:
   ```sh
   docker run --rm -v "$PWD/out:/work" ghcr.io/<owner>/phyein-topopt:latest sh -c \
     "cd /opt/phyein-topopt && python examples/chair.py --n 12 --volfrac 0.03 --out /work/chair12 \
      && python examples/chair_check.py /work/chair12 && python examples/chair_animate.py /work/chair12"
   ```
5. Building the image locally inside a session: the session's outbound HTTPS goes through a proxy
   with its own certificate authority, which the build container does not trust. Pass it as a
   build secret (used for the pip step only, never stored in the image):
   `docker build --secret id=ca,src=$HOME/.ccr/ca-bundle.crt -t phyein-topopt:local .`
6. Without Docker: `pip install --break-system-packages -r requirements.txt`, then run with
   `PYTHONPATH=$PWD`. The session's own CPUs are few (2); use it for small runs and tests, and
   GitHub runners for real ones.

Long runs inside a session: start them with `nohup … > log 2>&1 &` and poll the log with short
progress messages to the user. A background process may be stopped a few minutes after a reply ends,
so anything longer than the turn goes to GitHub Actions.

## 3. Run it on GitHub Actions (free CPU, the default for real runs)

- From the browser: repository → Actions → `run-case` → Run workflow → choose `case` (chair or mbb),
  `n` (voxels across the chair's half width), `volfrac`, `rmin`, `penal`, `label`, `animate`
  (yes/no), `maxit` (3 to time a grid) → Run.
- From a Claude session: `gh workflow run` fails there (it uses GitHub's GraphQL API, which
  sessions cannot reach). Use the REST endpoint, which works everywhere:
  ```sh
  R=repos/<owner>/phyein-topopt
  gh api -X POST $R/actions/workflows/run.yml/dispatches -f ref=main \
    -f "inputs[case]=chair" -f "inputs[n]=20" -f "inputs[volfrac]=0.03" -f "inputs[rmin]=2.0" \
    -f "inputs[penal]=3.0" -f "inputs[label]=chair20" -f "inputs[animate]=yes" -f "inputs[maxit]=300"
  gh api "$R/actions/workflows/run.yml/runs?per_page=1" --jq '.workflow_runs[0] | "\(.id) \(.status) \(.created_at)"'
  gh api $R/actions/runs/<id>/jobs --jq '.jobs[0].steps[] | "\(.name): \(.status)"'   # which step
  gh run view <id> -R <owner>/phyein-topopt --log-failed                              # errors
  gh run download <id> -R <owner>/phyein-topopt -D out/                              # results
  ```
  On the user's own terminal `gh workflow run run-case -R <owner>/phyein-topopt -f case=chair …`
  also works.
- Starting a run acts on the user's account: state the time estimate and get an explicit yes first.
- **GitHub timestamps are UTC.** Convert before saying how long a run has been going (the user
  is in Europe/Berlin, UTC+1 in winter). Misreading UTC as local time once made a healthy run look
  an hour late and got it cancelled for nothing. Elapsed time = `date -u` minus `created_at`.
- A running job's log is not readable until it ends; the jobs endpoint shows which step it is in.
  Do not cancel a job to read its log unless the measured rate says it is stuck.
- Poll with short checks (one call, no long sleeps) and tell the user what you see; the user gets
  GitHub's e-mails for every failed run, so check for failures in all repositories you touched
  before saying a build is fine, and say which repository a failure came from.
- Size before you launch: time the grid first (`maxit=3` on the runner, or `--maxit 3` in a
  session), multiply by 200–300 iterations, add 20–30 min for the film, and tell the user the total.
  Measured (MGCG, PA12, ν = 0.43): the free GitHub runner is about half as fast per iteration as
  a session's 2-core machine. Chair at 13 824 voxels: 8.7 s per iteration on the runner (4.7 s in
  a session), 30–45 min for a full run. Chair at 64 000 voxels: 27 s per iteration on the runner,
  1.5–2.3 h plus 20–30 min for the film. Runs take 160–300 iterations. A job that hits 16 GB
  or 6 h is killed and returns nothing.
- `animate=no` skips STL, stills and film (for parameter sweeps); every chair run writes
  `design_check.json` (seat deflection, stress percentiles, connectivity, buckling estimate,
  pass/fail against the design goal). Re-run `examples/chair_check.py` on downloaded results when
  the check has been improved since the run.
- Parameter sweeps: start several `run-case` runs with different labels (up to 20 run at once).
- To add a new load case: write `examples/<case>.py` in the style of `chair.py` (all parameters as
  arguments, passive pads under loads, a docstring describing the physical setup and the hand
  statics), add it to the `case` choice in `run.yml`, push. The image workflow re-tests and
  republishes (about 7 minutes).

## 4. Run it on Kaggle (more memory, or GPU)

Kaggle notebooks cannot run Docker, so they install from the repository instead:
```python
!git clone https://github.com/<owner>/phyein-topopt
%cd phyein-topopt
!pip install -q -r requirements.txt
!python examples/chair.py --n 24 --out /kaggle/working/chair24
!python examples/chair_animate.py /kaggle/working/chair24
```
Files in `/kaggle/working` are kept as the notebook's output. Check `python --version` and the
current GPU quota on the Kaggle settings page before planning.

**GPU route (status: being built; record the measurement here).** The matrix-free MGCG solver is
made of gathers, dense products and scatters, which map directly onto a GPU. A JAX port of
`topopt/mgcg.py` is the planned GPU path; it must match the CPU solver on `check_mgcg.py` before use,
and its per-iteration time on Kaggle's GPU must be measured against the runner's 27 s at 64 000
voxels. For other physics or unstructured CAD meshes, JAX-FEM (`pip install jax-fem`, Python ≥ 3.12)
is the alternative; validate it on the same benchmarks first.

## 5. On the user's PC (optional)

Install Docker Desktop, then the same `docker pull` / `docker run` commands as in section 2. The
image is built for x86-64 (Intel MKL, used only by the reference direct solver in the checks); on an
ARM machine run from source with `pip install -r requirements.txt`, skip `pypardiso`/`mkl`, and the
default MGCG solver still works. For CAD and FreeCAD's own FEM on the PC, see the repository's
CONNECTORS.md (freecad-mcp, audited).

## 6. Publish to phyeinusul.com

Work in the website repository `SiteTest` (attach it with push access). Follow the site's own rules
first: `CLAUDE.md` and `AGENTS.md` in that repository, and the skills `phyein-usul-physics-content`,
`physics-simulation-study` (structure, voice, palette) and `phyein-usul-data-panel` (every number on
a page is stored and checked).

What to publish for one study:
- `chair_load_path.mp4` (H.264, faststart, a few MB): the film, as a ready-made animation, never a
  live 3D optimisation in the visitor's browser.
- `still_front.png`, `still_back.png`: stress-coloured stills with the honesty label.
- `chair_stress.glb`: the smoothed shape with per-vertex stress colours, for an interactive viewer.
- `chair.stl`: the printable shape, as a download.
- `chair_run.json`, `render_summary.json`, `design_check.json`: parameters, iteration log, stress
  scale, checks; the page's numbers come from these files, not from retyping.

Where: source in `portfolios-src/topopt/`, served copy in `public/portfolios/topopt/` (follow the
layout of an existing portfolio such as `portfolios-src/hydraulics/`). The 3D viewer is three.js with
GLTFLoader and OrbitControls from a pinned copy stored in the repository (not a live CDN link), the
site palette, a label on the canvas, and no reaction to the pointer merely crossing the figure:
rotate on drag, a readout only when the reader switches it on. Check every file's size against the
host's per-file limit before committing.

Publishing: commit on a branch, check the page in a real browser at 360, 390, 1024 and 1440 px,
check it against the other sessions' work on `main`, then merge to `main`; Cloudflare publishes it by
itself. Confirm on the live URL. Follow the user's current rule on whether merging needs a yes; any
other outward action (creating repositories, connecting accounts, paid services) always needs the
user's explicit yes for that exact action.

## 7. Keeping the toolchain healthy

- Change a dependency only in `requirements.txt`, with an exact version, then push and let the image
  workflow prove it. Never `pip install` an unpinned package into the image.
- When a run must be reproducible, use the image by commit, `ghcr.io/<owner>/phyein-topopt:<sha>`,
  and write the sha into the results.
- GHCR is currently free for images; GitHub promises at least a month's notice before that changes.
  If it changes, the image can be rebuilt anywhere from the `Dockerfile`.
- After any change to the solver, the acceptance checks must still pass (MBB within 10⁻⁴ of the
  paper, cantilever within 1 % of Timoshenko at 16 elements, gradient within 10⁻¹⁰, MGCG within
  10⁻⁶ of the direct solve). The workflow enforces this; when you test locally, run
  `docker run --rm <image>` before reporting success.
