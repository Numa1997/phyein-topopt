"""Build the Kaggle notebooks from their specs, so every notebook can be regenerated from git.

Every notebook has the same eight parts, in this order:
  a  title, summary, link to the live page (if any), "computed results" sentence
  b  the physics            projects/<name>/physics.md
  c  the machine            common/env_check.py       (nproc, RAM, OS, GPU; stops if a requested GPU is missing)
  d  the code               common/clone_install.py   (git clone at a pinned commit, pip install only what is missing)
  e  parameters             projects/<name>/params.py + common/params_echo.py (writes params.json)
  f  the run                projects/<name>/run.py    (progress.log, checkpoints for long runs)
  g  results                projects/<name>/results.py
  h  verification           projects/<name>/verify.py (+ reference.json inlined as REFERENCE)
  and a last cell           common/finish.py          (timing.json, env.json, SHA256SUMS.txt, stop if a check failed)

Placeholders {{NAME}} in the cell sources are filled from projects/<name>/spec.json.

    python3 kaggle/build_notebook.py                 # rebuild every notebook into kaggle/notebooks/
    python3 kaggle/build_notebook.py topopt-benchmarks
    python3 kaggle/build_notebook.py --check         # fail if a committed notebook differs from its spec
"""
import json
import pprint
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = HERE / "projects"
COMMON = HERE / "common"
OUT = HERE / "notebooks"

STANDARD_LINE = ("**Computed results; this page is not interactive. "
                 "Press Copy & Edit to run it yourself.**")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


def fill(text, spec):
    for key, val in spec["_fill"].items():
        text = text.replace("{{" + key + "}}", val)
    if "{{" in text and "}}" in text.split("{{", 1)[1]:
        bad = text.split("{{", 1)[1].split("}}", 1)[0]
        raise SystemExit(f"unfilled placeholder {{{{{bad}}}}}")
    return text


def read(path, spec):
    return fill(path.read_text(), spec)


def title_cell(spec):
    live = (f"Live page: <{spec['live_url']}>" if spec.get("live_url")
            else "This project has no page on <https://phyeinusul.com> yet.")
    src = spec["source"]
    lines = [
        f"# {spec['title']}",
        "",
        spec["summary"],
        "",
        live,
        "",
        f"Code: <{src['repo_url']}> at commit `{src['commit'][:7]}`"
        + (f" (branch `{src['branch']}`)" if src.get("branch") else "") + ".",
        "",
        STANDARD_LINE,
        "",
        "Contents: 1 The physics · 2 The machine · 3 The code · 4 Parameters · 5 The run · "
        "6 Results · 7 Verification · 8 Files",
    ]
    if spec.get("status_note"):
        lines[2:2] = [f"> **{spec['status_note']}**", ""]
    return md("\n".join(lines))


def build(name):
    pdir = PROJECTS / name
    spec = json.loads((pdir / "spec.json").read_text())
    ref_path = pdir / "reference.json"
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else None
    src = spec["source"]
    spec["_fill"] = {
        "PROJECT": name,
        "TITLE": spec["title"],
        "REPO_URL": src["repo_url"],
        "COMMIT": src["commit"],
        "NEED_GPU": "True" if spec.get("gpu") else "False",
        "REQUIRES": json.dumps(spec["requires"]),
        "REFERENCE": pprint.pformat(reference, width=110, sort_dicts=True) if reference else "None",
    }
    cells = [
        title_cell(spec),
        md("## 1. The physics\n\n" + read(pdir / "physics.md", spec)),
        md("## 2. The machine\n\nWhat this run had to work with. The same lines go into `env.json` and "
           "`timing.json`." + (" A GPU was requested, so the notebook stops here if none is present."
                               if spec.get("gpu") else "")),
        code(read(COMMON / "env_check.py", spec)),
        md("## 3. The code\n\nThe solver is not copied into this notebook: it is fetched from GitHub at one "
           "fixed commit, so this page always runs the code it describes. Only packages the Kaggle image "
           "lacks are installed, at the versions pinned in the repository's `requirements.txt`; every "
           "version actually used is written to `env.json`."),
        code(read(COMMON / "clone_install.py", spec)),
        md("## 4. Parameters\n\nEvery input of the run, as a named variable. Change one and run the "
           "notebook again; the values used are written to `params.json`."),
        code(read(pdir / "params.py", spec) + "\n\n" + read(COMMON / "params_echo.py", spec)),
        md("## 5. The run\n\nEach stage appends a line to `progress.log` with the elapsed time."),
        code(read(pdir / "run.py", spec)),
        md("## 6. Results"),
        code(read(pdir / "results.py", spec)),
        md("## 7. Verification\n\n" + spec["verification_text"]),
        code(read(pdir / "verify.py", spec)),
        md("## 8. Files\n\nThe files this run wrote, with their SHA-256 checksums (also in `SHA256SUMS.txt`). "
           "The last line stops the notebook with an error if any verification check failed."),
        code(read(COMMON / "finish.py", spec)),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"{name}-{i:02d}"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "phyein": {"project": name, "source_repo": src["repo_url"], "source_commit": src["commit"],
                       "kaggle_slug": spec["kaggle_slug"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Compact, one string per cell: this exact text is what is sent to Kaggle (save_notebook "text"), so the
    # SHA-256 of the committed file identifies the notebook version on Kaggle.
    return json.dumps(nb, ensure_ascii=False, separators=(",", ":"))


def main(argv):
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")] or sorted(p.name for p in PROJECTS.iterdir()
                                                                  if (p / "spec.json").exists())
    OUT.mkdir(exist_ok=True)
    bad = []
    for name in names:
        text = build(name)
        path = OUT / f"{name}.ipynb"
        if check:
            if not path.exists() or path.read_text() != text:
                bad.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(HERE.parent)} ({len(text)} bytes)")
    if check:
        print("notebooks match their specs" if not bad else f"out of date: {', '.join(bad)}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
