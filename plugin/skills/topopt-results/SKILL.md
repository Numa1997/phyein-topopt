---
name: topopt-results
description: "Fetch and check the results of a finished phyein-topopt run: download the artifact, verify convergence, connectivity and the design check, and present STL, stills and film with the honesty label. Use when the user asks for the result of a run, the chair film, or whether a design meets its goal."
---

# Fetch and check a run's results

1. Find the run: `gh api "repos/<owner>/phyein-topopt/actions/workflows/run.yml/runs?per_page=5"`;
   if it failed, `gh run view <id> -R <owner>/phyein-topopt --log-failed` and report the cause.
2. Download: `gh run download <id> -R <owner>/phyein-topopt -D out/` (artifact `<case>-<label>`).
3. Check before showing anything:
   - convergence: `chair_run.json` log (compliance settled at the final β, grey fraction < 0.01;
     a few voxels flipping at β = 16 is normal, say so);
   - connectivity: one face-connected solid touching every support and load pad
     (`topopt/checks.py`);
   - design goal: `design_check.json` (seat deflection against its limit; 99th-percentile and
     maximum von Mises stress against yield/2, labelled indicative because voxel surfaces are
     staircases);
   - the STL is watertight.
4. Present: one paragraph of what the shape is and how the load travels, the check numbers, the
   stills and the film (`SendUserFile`), and the label "approximate voxel FEA · design proposal,
   not a structural verification". If the shape differs from a reference the user showed, say how,
   plainly; never re-tune loads or volume fraction to make it match.
