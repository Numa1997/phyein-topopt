# Kaggle copies of the computed simulations

This folder publishes the computed (not interactive) simulations of this repository and of the website repository
SiteTest as Kaggle notebooks that anyone can read and rerun with **Copy & Edit**. It is a copy. GitHub stays the source
of every line of code, and phyeinusul.com (served by Cloudflare from SiteTest) is unchanged. Each notebook clones its
code from a public GitHub repository at a pinned commit, so a notebook always runs exactly the code it describes.

Status on 12 November 2026: one notebook built and tested locally, not yet on Kaggle. This file is updated as each one
goes up.

## What is on Kaggle, and where its code comes from

| notebook (folder `projects/`) | what it computes | code: repository @ commit | machine | status |
|---|---|---|---|---|
| `topopt-benchmarks` | the five acceptance checks of the topology-optimisation solver: clamped beam in 2D and 3D against Timoshenko, rigid-body modes of the hex8 element, adjoint gradient against JAX autodiff, multigrid against PARDISO, MBB beam against Andreassen et al. 2011 | phyein-topopt @ `ef2d27b` | CPU | built, ran locally in 113 s, 10/10 checks |
| `topopt-chair` (planned) | the topology-optimised PA12 chair, its design check and film | phyein-topopt | CPU, then JAX on GPU after a run card | not started |
| `fig13-chaos` (planned) | Fig. 13: chaos map, flower hunt, butterfly orbit, its physics and its control | SiteTest `tools/fig13` | CPU | not started |
| `fig12-stout` (planned) | Fig. 12: OpenFOAM stout in a swung glass, the 8 Phase 1 gate cases. **Work in progress, not yet verified** | SiteTest branch `ccr-76841c15-le08bl` | CPU, OpenFOAM 13 | not started |

`index.json` holds, for every notebook: its Kaggle reference and version, the source repository and commit, the commit
of this folder that generated it, and the SHA-256 checksums of its output files. It never holds the data itself.

## How every notebook is built

`build_notebook.py` writes each notebook from its folder in `projects/<name>/`, so a notebook can always be regenerated
from git and nobody edits `.ipynb` files by hand:

```sh
python3 kaggle/build_notebook.py              # rebuild all notebooks into kaggle/notebooks/
python3 kaggle/build_notebook.py --check      # fail if a committed notebook differs from its sources
```

Every notebook has the same parts, in this order:

1. title, summary, link to the live page on phyeinusul.com where one exists, and the line "Computed results; this page
   is not interactive. Press Copy & Edit to run it yourself."
2. the physics (`physics.md`), built from the project's existing lecture note, in Markdown with LaTeX, every parameter
   a symbol;
3. the machine (`common/env_check.py`): CPU, cores, RAM, OS, GPU; stops at once if a requested GPU is missing;
4. the code (`common/clone_install.py`): `git clone` at the pinned commit, then `pip install` of only the packages the
   Kaggle image lacks, at the versions pinned in the repository's `requirements.txt`; versions used go to `env.json`;
5. parameters (`params.py`): every input a named variable, echoed into `params.json`;
6. the run (`run.py`): each stage logged with its elapsed time in `progress.log`; checkpoints for runs over 30 minutes;
7. results (`results.py`): figures, `results.json`;
8. verification (`verify.py`): against theory and published values, against the numbers the repositories record, and
   against a stored reference run (`reference.json`, inlined into the notebook); the largest difference is printed;
9. files (`common/finish.py`): `timing.json` (machine, precision, wall time per stage), `env.json`, `SHA256SUMS.txt`;
   the notebook ends with an error if any check failed.

## How to rerun

- On Kaggle: open the notebook, press **Copy & Edit**, change any value in section 4, then **Run All**. Internet must be
  on (the notebook clones its code). CPU notebooks need no accelerator.
- Anywhere else (Python 3.11 or newer, git):
  ```sh
  pip install nbclient ipykernel
  NB_WORK=$PWD/out NB_SRC=$PWD/src python3 -c "import nbformat, nbclient; nb = nbformat.read('kaggle/notebooks/topopt-benchmarks.ipynb', 4); nbclient.NotebookClient(nb, timeout=3600).execute(); nbformat.write(nb, 'out.ipynb')"
  ```
  `NB_WORK` replaces `/kaggle/working` (outputs), `NB_SRC` the folder the code is cloned into.

## What is not on Kaggle, and why

- **Browser simulations** (Fig. 1 to 10, the live controllers of Project 05 and Fig. 11, the Fig. 12 pint and its
  Node checks, the live Fig. 13 pages): they are JavaScript that runs in the reader's browser on phyeinusul.com. A
  Kaggle notebook could only show a frozen copy of something that is already interactive, so they stay on the site and
  the project write-up links to them.
- **Rendering and site tools** (`SiteTest/tools/fig11` Blender scenes and figure layout, `tools/fig13/build_page.py`,
  `map_png.py`, `hunt_export.py`, `tools/i18n`, `tools/perf`, `redirects.py`, `inventory.py`): they build pictures and
  pages, not physics results.
- **Kaggle datasets, write-up and collection**: the Kaggle connector used here has no tool that creates a dataset, a
  dataset version, a write-up or a collection (checked against its live tool list on 11 November 2026). Until one
  exists, each notebook's output files (data, `params.json`, `timing.json`, `SHA256SUMS.txt`) are published with the
  notebook itself, and the project page text is kept in `writeup.md` to be created on kaggle.com.

## Licence

Not yet chosen; the repositories carry no licence file. Nothing here goes public before the licence is decided.
