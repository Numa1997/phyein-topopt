---
name: topology-optimisation
description: "Structural topology optimisation done properly: derive FEA and SIMP from first principles, set loads from physics and standards, pick 2D/3D, resolution and hardware, run, check and label results."
---

# Structural topology optimisation: from a block and its loads to the load path

Use this skill when the user wants material placed only where a load needs it: "what shape should this
bracket/chair/beam be", "remove material until only the load path stays", "make a film like the
'Nobody drew this chair' clip", generative or lightweight design, or a study of SIMP/BESO/level sets.
For running jobs, images, GitHub Actions, Kaggle and publishing, also load the companion skill
`topopt-toolchain`.

The reference implementation lives in the repository `phyein-topopt` (owner: the user's GitHub
account). Read its `README.md`, `HANDOVER.md` and `topopt/top3d.py` before writing new code. Everything
below was built and verified there on 11 November 2026; the numbers quoted are from those runs.

---

## 1. What the result is, and what it is not

A topology-optimised shape is a **design proposal**: the stiffest arrangement of a fixed amount of
linear-elastic material for the loads and supports you assumed, computed on a coarse voxel mesh.
It is not a strength check. It says nothing about yield, buckling, fatigue, joints, manufacturing
tolerances or load cases you did not include. Every render, film and page carries the label
**"approximate voxel FEA · design proposal, not a structural verification"**. To verify a proposal,
re-mesh the smoothed STL with tetrahedra (Gmsh), analyse it with a validated solver (CalculiX) using
real material data and safety factors, and say that this is a separate step.

---

## 2. The physics, derived step by step

Keep every parameter as a symbol here and as an argument in the code: E0, ν, p, V*, r_min, β, η,
loads, dimensions. Never hard-wire a number.

### 2.1 Physical setup
A body occupies the region Ω. Picture a sponge-like block standing on the floor: some faces are held
(supports, the set Γ_D where the displacement u = 0), some faces are pushed (the set Γ_N carrying a
traction t, a force per area in N/m²), and gravity or other body forces b (N/m³) act inside. The
block deforms a little: each material point x moves to x + u(x). We want u, the strains and the
stresses, and then we want to decide where the material should be.

### 2.2 Equilibrium (Cauchy)
Cut out any piece V of the body. The forces on it are the tractions on its surface, σn per unit
area (σ is the stress tensor, n the outward normal), and the body force b. At rest they balance:
∮_∂V σ n dA + ∫_V b dV = 0. The divergence theorem turns the surface integral into ∫_V ∇·σ dV, and
since V is arbitrary the integrand vanishes everywhere:

  ∇·σ + b = 0 in Ω,  σn = t on Γ_N,  u = 0 on Γ_D.

The balance of moments on the same piece makes σ symmetric. (Any continuum mechanics text, e.g.
Gurtin, *An Introduction to Continuum Mechanics*, ch. 14.)

### 2.3 Kinematics (small strain)
For displacement gradients much smaller than 1, the change of length and angle of a small fibre is
measured by the symmetric part of ∇u: ε = ½(∇u + ∇uᵀ). The antisymmetric part is a rigid rotation
and stores no energy.

### 2.4 Hooke's law (isotropic, linear)
Picture a rubber eraser: squeeze it and it widens sideways; shear it and it skews. Two constants
describe this for an isotropic material: σ = λ tr(ε) I + 2μ ε, with the Lamé constants
λ = Eν/((1+ν)(1−2ν)) and μ = E/(2(1+ν)) (E Young's modulus in Pa, ν Poisson's ratio). In Voigt
notation, with ε = (ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy) and engineering shears γ = 2ε, this is
σ = D ε with a 6×6 matrix D (`elasticity_matrix` in `top3d.py`).

### 2.5 Weak form = virtual work = minimum potential energy
Multiply the equilibrium equation by any smooth test displacement v that vanishes on Γ_D and integrate
over Ω. Integrate the stress term by parts (divergence theorem again); the boundary term becomes
∫_Γ_N t·v dA because v = 0 on Γ_D and σn = t on Γ_N. Using the symmetry of σ:

  ∫_Ω σ(u) : ε(v) dV = ∫_Ω b·v dV + ∫_Γ_N t·v dA  for all admissible v.

Read it as virtual work: internal work of the stresses in a virtual deformation equals external work
of the loads. It is also the stationarity condition of the potential energy
Π(u) = ½∫ ε:D:ε dV − ∫ b·u dV − ∫ t·u dA, which is a minimum because D is positive definite.

### 2.6 Discretisation: voxels, shape functions, the stiffness matrix
Fill the design box with cubes of side a (voxels). Each cube is an 8-node "hex8" element. Inside one
cube, in natural coordinates ξ, η, ζ ∈ [−1, 1], the displacement is interpolated from its 8 corner
displacements with trilinear shape functions N_A = ⅛(1+ξξ_A)(1+ηη_A)(1+ζζ_A), (ξ_A, η_A, ζ_A) the
corner signs. Then ε = B u_e, where u_e (24 numbers) are the corner displacements and B (6×24)
contains the derivatives ∂N_A/∂x. For a cube x = (a/2)(ξ+1)+const, so ∂/∂x = (2/a)∂/∂ξ and
dV = (a/2)³ dξ dη dζ.

Insert u ≈ Σ N_A u_A and v ≈ Σ N_A v_A into the weak form. Because it must hold for every choice of
the v_A, it becomes one linear equation per degree of freedom (DOF):

  K u = f,  K = Σ_e A_eᵀ k_e A_e,  k_e = ∫_Ωe Bᵀ D B dV,

where A_e picks the element's 24 DOFs out of the global vector ("assembly") and f collects the
loads (a uniform traction on a face is spread over its nodes with weights ¼, ½, 1 for corner, edge
and interior nodes before normalising; see `face_nodes_weights`). Two facts used everywhere:
- **Scaling.** k_e = (2/a)² (a/2)³ ∫ B̂ᵀ D B̂ dξ = (a/2) E ∫ B̂ᵀ D̂ B̂ dξ, so k_e = E·a·k̂ with k̂ fixed:
  build k̂ once for a unit cube with E = 1. Consequence: stresses do not depend on E0 at all
  (u ∝ 1/E0, σ = E0·B·u), only deflections do. Report deflection as δ·E0 when no material is chosen.
- **Integration.** The entries of Bᵀ D B for a cube are polynomials of degree ≤ 2 in each natural
  coordinate, so 2×2×2 Gauss points (exact to degree 3) integrate k̂ exactly. The code nevertheless
  integrates the volumetric part with one point (selective reduced integration, section 3),
  because exact integration of the trilinear element locks for nearly incompressible materials.
Acceptance of the element: k̂ symmetric, positive semi-definite, exactly 6 zero eigenvalues (3
translations, 3 rotations). More zeros mean a spurious mechanism, fewer mean a coding error.

### 2.7 The design problem (SIMP)
Give each element a density ρ_e ∈ [0, 1] and let its Young's modulus be

  E_e = E_min + ρ_e^p (E0 − E_min),  p ≈ 3, E_min ≈ 10⁻⁹ E0 (keeps K invertible in voids).

Minimise the **compliance** C = fᵀu (the work done by the loads; at equilibrium C = uᵀKu = twice the
strain energy). Picture: press on the structure; the stiffer it is, the less the load sinks and the
less work it does. The problem is

  min_ρ C(ρ) = fᵀ u(ρ)  subject to  K(ρ) u = f,  Σ_e ρ_e v_e ≤ V* Σ_e v_e,  0 ≤ ρ_e ≤ 1.

Why the power p > 1: with p = 1, a half-dense element gives half the stiffness for half the volume,
so grey material is a perfectly good answer (the variable-thickness sheet). With p = 3 it gives one
eighth of the stiffness for half the volume, so grey is a bad buy and the optimiser drives elements
to 0 or 1. (Bendsøe 1989, Struct. Optim. 1:193–202; Sigmund 2001, Struct. Multidisc. Optim.
21:120–127.) Several load cases L with weights w_L: C = Σ_L w_L f_Lᵀ u_L.

### 2.8 Sensitivities by the adjoint method
We need ∂C/∂ρ_e for thousands of elements, at the cost of one solve. Differentiate K u = f with
respect to ρ_e (f does not depend on ρ): (∂K/∂ρ_e) u + K ∂u/∂ρ_e = 0, so
∂u/∂ρ_e = −K⁻¹ (∂K/∂ρ_e) u. Then ∂C/∂ρ_e = fᵀ ∂u/∂ρ_e = −(K⁻¹f)ᵀ (∂K/∂ρ_e) u = −uᵀ (∂K/∂ρ_e) u,
using K symmetric. In general the adjoint vector λ solves Kλ = ∂C/∂u; for compliance ∂C/∂u = f, so
λ = u and no second solve is needed ("self-adjoint"). Only element e's block of K depends on ρ_e:

  ∂C/∂ρ_e = − p ρ_e^(p−1) (E0 − E_min) · a · u_eᵀ k̂ u_e ≤ 0.

In words: **the gradient of each element is minus its own strain energy (up to the SIMP factor)**,
and it is never positive, because adding material can only stiffen. Acceptance: compare with
automatic differentiation of C(ρ) = fᵀK(ρ)⁻¹f in JAX (float64) on a small grid; the reference code
agrees to about 10⁻¹⁴ relative (`benchmarks/check3d.py`, part 3).

### 2.9 Two numerical diseases and their cures
- **Checkerboards.** Low-order elements make a chessboard of solid and void look artificially stiff,
  so the optimiser produces it (Díaz & Sigmund 1995; Sigmund & Petersson 1998, Struct. Optim.
  16:68–75).
- **Mesh dependence.** The continuum problem has no length scale: finer meshes give ever finer
  members and no converged design.
Cure: a **density filter** with radius r_min (a length scale): ρ̃_e = Σ_i H_ei ρ_i / Σ_i H_ei,
H_ei = max(0, r_min − |x_e − x_i|) (Bruns & Tortorelli 2001; Bourdin 2001, Int. J. Numer. Meth. Eng.
50:2143–2158). Members cannot be thinner than about 2 r_min. Sensitivities follow by the chain rule:
∂C/∂ρ_j = Σ_e (∂C/∂ρ̃_e) H_ej / Σ_i H_ei. The older **sensitivity filter** (Sigmund 1997) filters the
gradient instead; it is heuristic but works and is what the 88-line code calls ft = 1. For large
grids the **PDE filter** −r² ∇²ρ̃ + ρ̃ = ρ with r = r_min/(2√3) does the same job with one extra sparse
solve (Lazarov & Sigmund 2011).

### 2.10 Projection: from grey to crisp
The filter smears every edge over about 2 r_min of grey. A smoothed Heaviside step sharpens it:

  ρ̄ = [tanh(βη) + tanh(β(ρ̃ − η))] / [tanh(βη) + tanh(β(1 − η))],  η = 0.5,

with derivative dρ̄/dρ̃ = β(1 − tanh²(β(ρ̃ − η))) / [tanh(βη) + tanh(β(1 − η))] (Wang, Lazarov &
Sigmund 2011, Struct. Multidisc. Optim. 43:767–784). Start at β = 1 and double it every 40
iterations or when the design settles, up to β = 16 (continuation); a steep step from the start
locks the optimiser into poor local minima. The chain is ρ → ρ̃ (filter) → ρ̄ (projection) → E_e,
and the gradient runs back through it: ∂C/∂ρ = Hᵀ[(∂C/∂ρ̄)(dρ̄/dρ̃)/Hs].

### 2.11 The update: optimality criteria (OC) or MMA
At an optimum with only the volume constraint, every element that is neither 0 nor 1 satisfies
the Karush-Kuhn-Tucker condition ∂C/∂ρ_e + Λ ∂V/∂ρ_e = 0, i.e. B_e = −(∂C/∂ρ_e)/(Λ ∂V/∂ρ_e) = 1.
OC turns this into a fixed-point step: ρ_e ← ρ_e B_e^½, clipped to a move limit m = 0.2 and to [0, 1],
with Λ found by bisection so that the volume is exactly V*. Picture a market: Λ is the price of
material, and elements whose strain energy (their usefulness) is above the price get more material,
the others less. OC is the right tool for one volume constraint. With several constraints (stress,
displacement, frequency) use MMA, the Method of Moving Asymptotes (Svanberg 1987, Int. J. Numer. Meth.
Eng. 24:359–373); JAX-FEM ships an implementation (`jax_fem.mma`).

In practice (first chair run, ν = 0.3) every β stage used its full 40 iterations and the last
stage stopped after 10 (compliance spread 0.06 %). Compliance fell at every β step, so the sharper
design was also the stiffer one; most of the gain came in the first stage. Record the compliance at
the start of each β stage for every run; it shows where the design was made.

At β = 16 a few voxels may keep flipping between 0 and 1 (max change 0.5–0.8) while the compliance
is settled to four digits; the run then stops on the compliance criterion. Report that, and judge
the design by compliance history, grey fraction and connectivity, not by the max change.

**Convergence** is judged on the physical design ρ̄ (max change < 0.01) or on the compliance
(spread over the last 10 iterations < 10⁻³ relative), at the final β. Do not judge it on the raw
variables ρ: under a steep projection, raw variables deep inside solid or void have near-zero
gradient and wander at the move limit forever without changing the design (seen in the first chair
run: "change" stuck at 0.200 while C had settled to four digits).

### 2.12 Stress for colouring
Von Mises stress at the element centre, σ_vm = √(½[(σx−σy)² + (σy−σz)² + (σz−σx)²] + 3(τyz²+τxz²+τxy²)),
with the relaxed form σ_e = ρ̄_e^q σ_solid,e, q = 0.5 (Le et al. 2010, Struct. Multidisc. Optim.
41:605–620), so near-void elements do not show fake stress. It is good for colour and for "where is
the load going"; it is not a strength result (staircase boundaries, one point per voxel).

### 2.13 From densities to a shape you can print
Mirror symmetric halves, pad the field with one layer of void, extract the iso-surface ρ̄ = 0.5 by
marching cubes (Lorensen & Cline 1987, Computer Graphics 21(4):163–169; `skimage.measure`), smooth it
with Taubin's λ/μ filter (Taubin 1995, SIGGRAPH), which removes voxel steps without shrinking the part
the way plain Laplacian smoothing does, check it is watertight, export STL (geometry) and GLB (with
per-vertex stress colours, for three.js).

### 2.14 Alternatives, and when they win
- **BESO** (bi-directional evolutionary structural optimisation; Xie & Steven 1993 for ESO, Huang &
  Xie 2007 for the convergent BESO): elements are strictly 0 or 1. Each iteration ranks elements by
  their strain-energy density, removes the least useful and adds the most useful in their
  neighbourhood, shrinking the volume step by step at an evolution rate until V* is reached; filtered
  and history-averaged sensitivity numbers stabilise it. Wins when you must drive an external,
  non-differentiable solver (the `beso` package drives CalculiX this way) and want crisp designs
  without projection. Loses on theory (heuristic convergence) and on extra constraints.
- **Level-set methods** (Allaire, Jouve & Toader 2004, J. Comput. Phys. 194:363–393): the boundary
  is the zero contour of a function φ, moved with a normal velocity from the shape derivative by a
  Hamilton-Jacobi equation. Wins on crisp, smooth boundaries. Loses because new holes need a
  topological derivative and the result depends on the initial holes.
- **SIMP** (this skill's default): general, gradient-based, mature, easy to extend to many
  constraints and physics; needs filtering and projection to be crisp.

Primary sources verified in the building session: Bendsøe 1989, Sigmund 2001, Bourdin 2001,
Andreassen et al. 2011 (and its numbers), Amir, Aage & Lazarov 2014. The others are standard
citations; check the page numbers before quoting them in a publication.

---

## 3. Decision guide

**2D or 3D.** Use 2D to explain and to teach (MBB beam, cantilever, bridge): it runs in seconds and
the picture is clear. Use 3D for objects (chair, bracket, stool, frame): only 3D shows how load
paths twist and branch.

**Which tool.**
- Box-shaped or voxelised design space, compliance objective: the reference code (`top2d.py`,
  `top3d.py`, `mgcg.py`). Every line is derived above and checked.
- Real CAD geometry with curved boundaries, other physics (thermal, hyperelastic), MMA with several
  constraints: JAX-FEM (github.com/deepmodeling/jax-fem, GPL-3.0, Python ≥ 3.12, release
  0.0.12 of June 2026; young API, pin the version). Validate it on the same benchmarks first.
- A FreeCAD/CalculiX workflow on the user's PC: `beso` (LGPLv3, CalculiX ≥ 2.17, FreeCAD ≥ 0.18;
  last commit April 2025), BESO method; or FreeCAD's own FEM through freecad-mcp (see the
  repository's CONNECTORS.md).
- Independent cross-check of a result: topoptlab (Python, GPL-3.0, JOSS 2026; ships the MBB case).
- Parallel 2D/3D with PETSc: the FEniCS 55-line code (Gupta et al., arXiv 2012.08208).

**The linear solver is the cost of everything.** Use the established method for voxel grids:
matrix-free FEA (every voxel shares one k0, so K·u is one gather, one dense product, one scatter)
with geometric-multigrid-preconditioned CG (Amir, Aage & Lazarov 2014, Struct. Multidisc. Optim.
49:815–829; reference implementation pymoto.solvers.GeometricMultigrid, MIT). This is
`topopt/mgcg.py`: coarse operators are exact Galerkin products R^T A R, built element by element
(each coarse cube's matrix is Σ over its 8 children of P_cᵀ k_c P_c), a Chebyshev smoother, a
direct solve on the coarsest grid, warm starts, support groups solved in parallel threads.
Measured on the chair's fine grid (209 000 unknowns): about 10 s per optimisation iteration on 2
cores against about 40 s with the sparse direct solver (PARDISO), which refactorises from scratch
every solve and whose memory grows faster than linearly in 3D. Keep PARDISO only as the reference
that every new solver must match (`benchmarks/check_mgcg.py`: agreement 4·10⁻¹² relative).
Never test a multigrid solver on independent random densities per voxel (10⁹ contrast between
neighbours): it may not converge there, and no filtered design looks like that; test on uniform
and on finished designs.

**Nearly incompressible materials (ν ≳ 0.38) need two changes.** Trilinear voxels lock
volumetrically: too stiff (unconservative deflections) and ill-conditioned (CG iterations rose
19 → 60 → 218 for ν = 0.35, 0.40, 0.43). (1) Selective reduced integration: deviatoric part with
2×2×2 points, volumetric part K m mᵀ with one point (Hughes 1980, Int. J. Numer. Meth. Eng.
15:1413–1418); k0 keeps exactly 6 zero modes, and the cantilever error at 2 elements per section
fell from −13 % to −2.4 %. (2) Chebyshev smoother instead of damped Jacobi (Jacobi stalled at
ν = 0.43; Chebyshev of degree 3 on [λmax/30, 1.1 λmax] of D⁻¹A took 24 iterations).

**Where it runs.** With MGCG, a free GitHub runner (4 vCPU, 16 GB, 6 h per job) handles the
fine chair (64 000 voxels). Before launching anything long, time one iteration and give the user
the total estimate. Never discover a limit by crashing a runner.

**Loads and material fraction come from a stated design goal, never from the look you want.**
Take load cases from the governing standard (furniture seating: EN 1728 test methods with the
forces of EN 16139 / EN 12520 / EN 1335-2; the standards are paywalled, so record which values
were verified from which source and which are secondhand). Fix a material with sourced values.
State a design goal (e.g. stiffness: seat deflection ≤ 5 mm for a 100 kg sitter; strength:
stress under the test loads ≤ yield/2). Sweep V*, pick the smallest that meets the goal
(`examples/chair_check.py`), report whatever shape results, and say plainly when it differs from
a reference picture. Every load value, ratio and V* is a parameter choice: call it that.

**Linear compliance optimisation does not see buckling; check it yourself.** In the PA12 chair
the stated goals (seat deflection ≤ 5 mm, stress ≤ yield/2) were met 20× and 9× over already at
3 % material on the medium grid, so they did not limit the material. The limit came from Euler
buckling of the slender back legs, P_cr = π² E I_min / L² (pinned ends): at 3 % the weakest leg had
P_cr / F_leg = 2.2 against the test-A leg force F_leg = (F_s y_s + F_b h_b)/(2D).
`examples/chair_check.py` reports this estimate from the mid-height leg sections. Whenever thin
compression members appear, add this check, label it as yours, and pick the fraction by the most
critical of all checks. Also state when the grid cannot resolve the members a lower fraction
would need (members thinner than about 2 voxels).

**Load cases decide the topology more than anything else.** Before running, do the statics of the
whole object by hand: which reaction each support must carry, and when it tips. For the chair
(seat load F_s at depth y_s, back push F_b at height h_b, feet at y = 0 and y = D):
R_back = (F_s y_s + F_b h_b)/D, R_front = F_s − R_back, tipping when F_b > F_s (D − y_s)/h_b
(with the chair's approximated load points the back-load cases sit right at this edge: the
front feet would need −1 to −2 N; label that, since the true loading points are in EN 1728).
Then check that the load cases span every direction the object must resist. If all loads lie in
one plane, the optimiser builds walls in that plane (a shear panel is the ideal in-plane carrier,
and nothing tests its weak out-of-plane bending ∝ t³); that is what the first chair did. A sideways
case makes legs splay and triangulate.

**Why struts at all (state this to the user).** For a fixed volume V carrying F over length L,
axial loading gives C = F²L²/(EV); bending a square section of side b gives C = 4F²L⁵/(EV²);
the ratio is 4(L/b)², 400 for a leg ten times longer than thick. So optimal structures carry
force along members in triangles (Michell 1904, Phil. Mag. 8:589–597), and as V* → 0 SIMP results
approach such strut networks; at generous V* a solid web is the better answer.

**Antisymmetric loads on a half model.** A sideways push on a mirror-symmetric chair is
antisymmetric about the mirror plane (its x-component is even in x). Solve it on the half model
with u_y = u_z = 0 on the mirror plane instead of u_x = 0, as its own boundary-condition group
(`Problem.extra_groups`). For a symmetric design, symmetric and antisymmetric displacement fields
are K-orthogonal, so the compliances add and the half model stays exact for both.

**Resolution and what it buys.** The filter makes members at least about 2 r_min voxels thick. The
shape is meaningful at 3 to 5 voxels per member; deflections are trustworthy only from about 8
elements through a member (cantilever with the current element: −0.9 % at 8, −2.4 % at 2; with
plain full integration it was −13 % at 2, because trilinear elements lock in bending and, near
ν = 0.5, in volume). Say which of the two the user is getting.

---

## 4. The standard pipeline

1. **Domain.** Box size, voxel side a, which elements are design, passive solid (seat plates, load
   pads, bolt bosses) or passive void (where a person sits, where a shaft passes). Use symmetry to
   halve the model, with u_normal = 0 on the symmetry plane.
2. **Loads and supports.** Every load sits on a passive solid pad (a force on a void node does
   nothing physical). Spread loads over faces with consistent weights. Supports remove all six
   rigid-body motions, or K is singular. Several load cases if the real object meets several.
3. **FEA.** Matrix-free MGCG, warm-started; check the residual of every solve.
4. **Optimise.** V*, p = 3, r_min ≈ 1.5–2.5 voxels, β continuation, OC with bisection. Record the
   physical densities every iteration (float16) for the film.
5. **Check.** Convergence, connectivity, re-analysis of the final design under every load case
   (deflection, stress percentiles), buckling of slender compression members.
6. **Shape.** Mirror, marching cubes at 0.5, Taubin smoothing, STL + GLB, watertight check.
7. **Render.** Stress on the final shape with one fixed colour scale (98th percentile of the final
   stress), so a film shows stress rising as material leaves.
8. **Animate.** (1) the design space with load arrows and feet, (2) stress in the full block,
   (3) the history, with the stress recomputed for each shown iteration, sampled densely at the start
   where the design changes fastest, (4) one turn of the final shape. Label every frame.
Commands for all of this are in `topopt-toolchain`.

---

## 5. Benchmarks and acceptance checks (run before trusting anything new)

- **MBB beam** against Andreassen et al. 2011, Fig. 3 (V* = 0.5, p = 3, r_min = 0.04·nelx, which is
  what the paper's numbers use, not the 1.5 in its example call): 216.81 / 233.71 (60×20,
  sensitivity / density filter), 219.52 / 235.73 (150×50). Pass: relative error < 10⁻⁴. The reference
  code reaches 2·10⁻⁵.
- **Cantilever tip deflection** against beam theory, δ = F L³/(3 E I) + F L/(κ G A), with Cowper's
  κ = 10(1+ν)/(12+11ν) for a rectangle (use the Timoshenko form: at L/h = 10 shear adds 0.8 %).
  Pass: within 1 % of Timoshenko at 16 elements through the section (measured with the
  selective-reduced-integration element, ν = 0.3: −2.4 % at 2, −0.91 % at 8, −0.74 % at 16; +0.02 %
  against F L³/(3EI) alone). The remainder is the fully clamped root, which also stops Poisson
  contraction.
- **Element**: symmetric, 6 zero modes. **Gradient**: AD vs adjoint < 10⁻¹⁰.
- **Solver**: MGCG against the direct solve, < 10⁻⁶ relative (measured 4·10⁻¹²).
- **Connectivity** (`topopt/checks.py`): the solid (ρ̄ ≥ 0.5) must be one face-connected piece
  that touches every support and every load pad; report the voxel counts. Skill test case
  `examples/cantilever3d.py` (48×16×8, V* = 0.3, 2 min in a session): one piece, solid at the wall
  and along the whole loaded edge, top and bottom flanges joined by perforated webs, as the hand
  statics predict.
- **Physics sanity of every design**: symmetric loads give a symmetric design; members run from load
  pads to supports; compliance falls monotonically at fixed β; final volume equals V*; grey fraction
  4ρ̄(1−ρ̄) averaged below about 0.01.

---

## 6. Failure modes and fixes

- **Checkerboards** → filter (density or sensitivity), r_min ≥ 1.5 voxels.
- **Grey regions** that stay → projection with β continuation; check p = 3; check the volume isn't
  too high for the load (then grey is genuinely "use some everywhere").
- **Mesh dependence** → fix r_min in metres, not in voxels, when comparing resolutions.
- **"Change" never falls below tol** → you are measuring raw variables under projection; measure
  the physical design or the compliance (2.11).
- **Floating islands or a design that ignores a load** → the load sits on design material instead
  of a passive pad, or a support is missing.
- **Solver diverges or depends on previous calls** → seen with pyamg 5.4.1 smoothed aggregation
  (algebraic multigrid) in 3D elasticity: the same system solved in one process and stalled or blew
  up in another. Do not use it. The default is the geometric MGCG in `topopt/mgcg.py`; every solve
  checks its residual and stops with an error rather than return a wrong field; any change to a
  solver must again match the direct PARDISO solve (`benchmarks/check_mgcg.py`).
- **CG iterations climb (hundreds per solve)** → nearly incompressible material with damped Jacobi
  or full integration; use selective reduced integration and the Chebyshev smoother (section 3).
- **Singular K** → rigid-body motion not restrained (count the supports) or E_min = 0.
- **E_min not scaled with E0** → with a physical modulus (Pa) the void stiffness must be 10⁻⁹·E0, not
  10⁻⁹ Pa, or the contrast becomes 10¹⁸.
- **Poor local minimum** → slower β continuation, move limit 0.1, several r_min values; report that
  the result is a local optimum.
- **Dimpled, rippled walls in the render** → members or walls only about 2 voxels thick; marching
  cubes on a thin sheet plus smoothing leaves a regular ripple. It is a resolution artefact, not
  physics. Fix with r_min ≥ 2 voxels or a finer grid; say so if you show the coarse result.
- **Plates instead of struts** → in 3D, thin walls are often the true compliance optimum at
  moderate volume fractions (the first chair, V* = 0.12 with loads in one plane, grew arched side
  walls). Struts appear at lower V*, larger r_min or with several load cases that need different
  paths. Do not "fix" a plate that the physics prefers; explain it.

---

## 7. How to present it

Lead with the physical picture (where the load wants to go), then the formal mechanism (the gradient
is each element's strain energy). Give parameters as symbols with their values and say which are
parameter choices. Show the benchmark numbers next to the result. Put the honesty label on every
image, video and page. Describe the shape honestly, including how it differs from any reference the
user showed. For the user's website use the `phyein-usul-*` skills for style and the data panel, and
`topopt-toolchain` for publishing.
