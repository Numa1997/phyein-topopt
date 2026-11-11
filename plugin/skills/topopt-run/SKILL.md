---
name: topopt-run
description: "Launch a phyein-topopt optimisation run on a free GitHub runner (case, resolution, material fraction, filter radius, film or not), after stating the time estimate and getting the user's explicit yes. Use when the user says run the chair, launch a sweep, start a topology-optimisation run."
---

# Launch a topology-optimisation run

Load `topopt-toolchain` and `topology-optimisation` first if any modelling choice is involved.

1. Settle the inputs: `case` (chair, mbb, or a case added to `run.yml`), `n` (voxels across the
   half width), `volfrac`, `rmin` (voxels), `penal`, `label`, `animate` (yes/no), `maxit`.
   Every load, material value and volume fraction is a parameter choice; say so.
2. Give the time estimate before anything else, from the measured runner rates in
   `topopt-toolchain` (for example 8.7 s per iteration at 13 824 voxels, 27 s at 64 000; 160–300
   iterations; 20–30 min extra for the film). For an unmeasured grid, launch with `maxit=3` first
   and scale.
3. Starting a run acts on the user's GitHub account: ask for an explicit yes for this exact run.
4. Launch through the REST API (sessions cannot use `gh workflow run`):
   ```sh
   R=repos/<owner>/phyein-topopt
   gh api -X POST $R/actions/workflows/run.yml/dispatches -f ref=main \
     -f "inputs[case]=chair" -f "inputs[n]=20" -f "inputs[volfrac]=0.05" -f "inputs[rmin]=2.0" \
     -f "inputs[penal]=3.0" -f "inputs[label]=chair20" -f "inputs[animate]=yes" -f "inputs[maxit]=300"
   gh api "$R/actions/workflows/run.yml/runs?per_page=1" --jq '.workflow_runs[0] | "\(.id) \(.created_at)"'
   ```
5. Report the run id, the start time converted from UTC to the user's time zone, and the
   expected finish time. Poll with single short checks; never cancel a run without a measured
   reason.
