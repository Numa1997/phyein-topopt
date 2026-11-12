import matplotlib
import matplotlib.pyplot as plt
from topopt.style import BEIGE, INK, WINE, DENSITY_CMAP

json.dump(R, open(WORK / "results.json", "w"), indent=1)

# Figure 1: the four MBB designs, mirrored to show the whole beam
fig, axs = plt.subplots(2, 2, figsize=(11, 3.7), facecolor=BEIGE)
for ax, row in zip(axs.T.flat, R["mbb"]):
    x = np.load(WORK / f"mbb_{row['nelx']}x{row['nely']}_ft{row['ft']}.npz")["xPhys"]
    ax.imshow(np.hstack([x[:, ::-1], x]), cmap=DENSITY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"{row['nelx']}×{row['nely']}, {'sensitivity' if row['ft'] == 1 else 'density'} filter: "
                 f"C = {row['c']:.2f} (paper {row['paper']:.2f}, {100 * row['rel_err']:+.4f} %)", fontsize=9, color=INK)
fig.suptitle("Figure 1. MBB beam, V* = 0.5, p = 3, r_min = 0.04·nelx (whole beam: the computed left half and its mirror image)",
             color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(WORK / "fig1_mbb_designs.png", dpi=150, facecolor=BEIGE); plt.show()

# Figure 2: convergence of the cantilever tip deflection towards Timoshenko's value (8)
fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=BEIGE)
ax.set_facecolor(BEIGE)
for key, label, mk in (("cantilever2d", "2D, Q4 plane stress, κ = 5/6", "o"),
                       ("cantilever3d", "3D, hex8 (selective reduced integration), Cowper κ", "s")):
    n = [r["n_h"] for r in R[key]]; e = [abs(r["rel_tim"]) for r in R[key]]
    ax.loglog(n, e, marker=mk, color=WINE if key == "cantilever3d" else INK, label=label)
ax.axhline(CANTILEVER_TOL, color="#8a7f76", ls="--", lw=1, label=f"pass limit {CANTILEVER_TOL:.0%}")
ax.set_xticks(N_H_2D); ax.set_xticklabels([str(n) for n in N_H_2D]); ax.minorticks_off()
ax.set_xlabel("elements across the section, $n_h$"); ax.set_ylabel(r"$|\delta_{FE}/\delta_T - 1|$")
ax.set_title(f"Figure 2. Clamped beam, L/h = {L_OVER_H}: finite elements against beam theory", fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False); fig.tight_layout()
fig.savefig(WORK / "fig2_cantilever_convergence.png", dpi=150, facecolor=BEIGE); plt.show()

print("Test A  2D cantilever:   " + ", ".join(f"n_h={r['n_h']}: {r['rel_tim']:+.4%}" for r in R["cantilever2d"]))
print("Test A  3D cantilever:   " + ", ".join(f"n_h={r['n_h']}: {r['rel_tim']:+.4%}" for r in R["cantilever3d"]))
print(f"Test B  hex8 element:    {R['element3d']['zero_modes']} zero eigenvalues, smallest {R['element3d']['min_eig']:+.1e}, "
      f"symmetry error {R['element3d']['sym_err']:.1e}")
print(f"Test C  gradient:        max |AD - adjoint| / max |AD| = {R['gradient']['rel_diff']:.2e}")
print(f"Test D  multigrid:       worst relative difference to PARDISO {R['mgcg']['worst_rel_diff']:.2e}, "
      f"CG iterations {[g['cg_iterations'] for g in R['mgcg']['groups']]}")
for r in R["mbb"]:
    print(f"Test E  MBB {r['case']:11s} C = {r['c']:.4f}  paper {r['paper']:.2f}  rel. err {r['rel_err']:+.2e}  "
          f"{r['iters']} iterations, {r['seconds']:.1f} s")
log("results.json and figures written")
