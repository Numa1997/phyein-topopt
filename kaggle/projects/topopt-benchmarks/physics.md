*Source of this section: the derivation in `skills/topology-optimisation/SKILL.md`, sections 2.1 to 2.11, and the docstrings of the four scripts in `benchmarks/`, all in the repository at the commit named above. The text follows them; the beam-theory steps in 1.4 are written out here in full. Phyein Usul has no lecture note on topology optimisation yet, so this repository text is the note.*

### 1.1 What the code must get right, and why these five tests

Clamp a steel ruler to a table edge and press its free end. It bends, and the end sinks by an amount you can predict with a pencil: that is beam theory. A topology-optimisation code does the same calculation millions of times, on a block cut into small cubes, and after each one it moves a little material from where it does little to where it carries load. If the calculation is wrong, or the direction in which it moves material is wrong, the shape it draws means nothing, however good it looks.

So the code is tested in the order it is built. (A) Does the finite-element model of a clamped beam sag as much as beam theory says, in 2D and in 3D? (B) Is the 3D building block, one cube, free to move rigidly and stiff against every deformation? (C) Is the gradient that steers the material the true derivative of the objective? (D) Does the fast iterative solver return the same displacements as a slow exact one? (E) Does the whole optimiser reproduce the four MBB-beam designs published with the 88-line code?

### 1.2 Elastic equilibrium

A body occupies the region $\Omega$. Some faces are held ($\Gamma_D$, displacement $\mathbf u = 0$), some are pushed by a traction $\mathbf t$ (force per area, N/m², on $\Gamma_N$), and a body force $\mathbf b$ (N/m³) acts inside. Each material point $\mathbf x$ moves to $\mathbf x + \mathbf u(\mathbf x)$.

Cut out any piece $V$. At rest the surface forces $\boldsymbol\sigma\mathbf n$ and the body force balance: $\oint_{\partial V}\boldsymbol\sigma\mathbf n\,dA + \int_V \mathbf b\,dV = 0$, with $\boldsymbol\sigma$ the stress tensor and $\mathbf n$ the outward normal. The divergence theorem turns the surface integral into a volume integral, and because $V$ is arbitrary the integrand vanishes:

$$\nabla\cdot\boldsymbol\sigma + \mathbf b = 0 \ \text{in}\ \Omega,\qquad \boldsymbol\sigma\mathbf n = \mathbf t \ \text{on}\ \Gamma_N,\qquad \mathbf u = 0 \ \text{on}\ \Gamma_D. \tag{1}$$

The balance of moments on the same piece makes $\boldsymbol\sigma$ symmetric.

For displacement gradients much smaller than 1, stretching and shearing are measured by the symmetric part of $\nabla\mathbf u$; the antisymmetric part is a rigid rotation and stores no energy:

$$\boldsymbol\varepsilon = \tfrac12\left(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T}\right). \tag{2}$$

Squeeze a rubber eraser and it widens sideways; shear it and it skews. For an isotropic linear material two constants describe both, Young's modulus $E$ (Pa) and Poisson's ratio $\nu$:

$$\boldsymbol\sigma = \lambda\,\mathrm{tr}(\boldsymbol\varepsilon)\,\mathbf I + 2\mu\,\boldsymbol\varepsilon,\qquad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)},\qquad \mu = G = \frac{E}{2(1+\nu)}. \tag{3}$$

In Voigt notation, $\boldsymbol\sigma = \mathbf D\boldsymbol\varepsilon$ with a $6\times6$ matrix $\mathbf D$ in 3D; in 2D plane stress ($\sigma_{zz} = 0$, a thin plate) $\mathbf D$ is $3\times3$.

Multiply (1) by any test displacement $\mathbf v$ that vanishes on $\Gamma_D$, integrate over $\Omega$, and integrate the stress term by parts. The boundary term becomes $\int_{\Gamma_N}\mathbf t\cdot\mathbf v\,dA$, and with the symmetry of $\boldsymbol\sigma$:

$$\int_\Omega \boldsymbol\sigma(\mathbf u):\boldsymbol\varepsilon(\mathbf v)\,dV = \int_\Omega \mathbf b\cdot\mathbf v\,dV + \int_{\Gamma_N}\mathbf t\cdot\mathbf v\,dA\qquad\text{for all admissible }\mathbf v. \tag{4}$$

This is virtual work: the internal work of the stresses in a virtual deformation equals the external work of the loads. It is also the condition that the potential energy $\Pi(\mathbf u) = \tfrac12\int\boldsymbol\varepsilon:\mathbf D:\boldsymbol\varepsilon\,dV - \int\mathbf b\cdot\mathbf u\,dV - \int\mathbf t\cdot\mathbf u\,dA$ is stationary, and a minimum because $\mathbf D$ is positive definite.

### 1.3 Finite elements

Fill the domain with squares (2D, four-node "Q4" elements) or cubes (3D, eight-node "hex8") of side $a$. Inside one element, in natural coordinates $\xi,\eta,\zeta\in[-1,1]$, the displacement is interpolated from the corner displacements with the shape functions $N_A = \tfrac18(1+\xi\xi_A)(1+\eta\eta_A)(1+\zeta\zeta_A)$ in 3D (the 2D ones drop $\zeta$ and the factor becomes $\tfrac14$). Then $\boldsymbol\varepsilon = \mathbf B\mathbf u_e$, where $\mathbf u_e$ holds the element's corner displacements (8 numbers in 2D, 24 in 3D) and $\mathbf B$ the derivatives $\partial N_A/\partial x$. Insert $\mathbf u\approx\sum N_A\mathbf u_A$ and $\mathbf v\approx\sum N_A\mathbf v_A$ into (4); because it holds for every choice of the $\mathbf v_A$, it becomes one linear equation per degree of freedom:

$$\mathbf K\mathbf u = \mathbf f,\qquad \mathbf K = \sum_e \mathbf A_e^{\mathsf T}\mathbf k_e\mathbf A_e,\qquad \mathbf k_e = \int_{\Omega_e}\mathbf B^{\mathsf T}\mathbf D\,\mathbf B\,dV, \tag{5}$$

where $\mathbf A_e$ picks the element's degrees of freedom out of the global vector and $\mathbf f$ collects the loads. For a cube, $\partial/\partial x = (2/a)\,\partial/\partial\xi$ and $dV = (a/2)^3 d\xi\,d\eta\,d\zeta$, so

$$\mathbf k_e = E\,a\,\hat{\mathbf k}\quad(\text{3D}),\qquad \mathbf k_e = E\,t\,\hat{\mathbf k}\quad(\text{2D, thickness } t), \tag{6}$$

with $\hat{\mathbf k}$ built once for a unit element with $E = 1$. In 2D the element size drops out entirely. The 2D code integrates $\hat{\mathbf k}$ with $2\times2$ Gauss points (the closed form of the 88-line paper, its eq. 12). The 3D code splits $\mathbf D$ into a deviatoric part, integrated with $2\times2\times2$ points, and a volumetric part, integrated with one point at the centre ("selective reduced integration", Hughes 1980, *Int. J. Numer. Meth. Eng.* 15:1413). The reason: full integration of the trilinear cube cannot change shape at constant volume, so it is far too stiff for a material like PA12 with $\nu = 0.43$.

### 1.4 Test A: the clamped beam against beam theory

**Picture.** A beam of length $L$, clamped at $x = 0$, with a load $F$ pulling its free end down. Look at a cut at position $x$: everything to the right of the cut is held in equilibrium by a shear force $V(x)$ and a bending moment $M(x)$ carried across the cut. The end sinks for two reasons. The beam curves (bending), and each slice skews slightly (shear).

**Bending (Euler–Bernoulli).** Balance the piece between $x$ and the tip: $V(x) = F$ and $M(x) = F\,(L - x)$. Plane sections stay plane and normal to the axis, so the curvature is $w''(x) = M(x)/(E I)$, with $w$ the downward deflection and $I$ the second moment of area of the cross-section. Integrate twice with the clamp conditions $w(0) = 0$, $w'(0) = 0$:

$$w'(x) = \frac{F}{EI}\Big(Lx - \frac{x^2}{2}\Big),\qquad w(x) = \frac{F}{EI}\Big(\frac{Lx^2}{2} - \frac{x^3}{6}\Big),\qquad \delta_{EB} = w(L) = \frac{F L^3}{3EI}. \tag{7}$$

**Shear (Timoshenko).** The shear force $V = F$ skews every slice by an average angle $\gamma = V/(\kappa G A)$, where $A$ is the area of the section and $\kappa$ a shear coefficient that accounts for the shear stress being uneven over the section. The slices add up along the length, $w_s(L) = \int_0^L \gamma\,dx$:

$$\delta_{T} = \frac{F L^3}{3EI} + \frac{F L}{\kappa G A}. \tag{8}$$

The two sections tested, with their $\kappa$:

* 2D (`benchmarks/cantilever2d.py`): a plate of height $h$ and thickness $t$ in plane stress, $I = t h^3/12$, $A = t h$, $\kappa = 5/6$.
* 3D (`benchmarks/check3d.py`): a square section $h\times h$, $I = h^4/12$, $A = h^2$, and Cowper's coefficient for a rectangle, $\kappa = 10(1+\nu)/(12+11\nu)$ (Cowper 1966, *J. Appl. Mech.* 33:335).

**Check.** The ratio of the shear term to the bending term in (8) is $3EI/(\kappa G A L^2) = (E/G)\,h^2/(4\kappa L^2)$ for a rectangle. It vanishes like $(h/L)^2$ for a slender beam, as it should. With $L/h = 10$, $E/G = 2(1+\nu) = 2.6$ and $\kappa = 5/6$ it is $2.6\times0.01/3.33 = 0.78\,\%$: shear adds less than 1 % here, which is why the test compares with (8) and not with (7) alone.

**What to expect.** The test uses $E = F = t = 1$ and elements of unit side, so $h$ equals the number of elements $n_h$ across the section and $L = 10\,n_h$. In 2D, $\delta_{EB} = F L^3/(3EI) = 4000$ for every $n_h$. Bilinear elements are too stiff in bending (they cannot bend without also shearing, "shear locking"), so the finite-element deflection approaches (8) from below as $n_h$ grows. The fully clamped root also stops the Poisson contraction there, which beam theory ignores. Pass: within 1 % of (8) at the finest mesh.

### 1.5 Test B: the rigid-body modes of one cube

**Picture.** Push a loose dice across the table, or spin it: it moves, but nothing in it is strained, so it stores no energy. A free cube can do this in 6 independent ways (3 translations, 3 rotations), and no others. In the language of (5) this means $\hat{\mathbf k}$ must be symmetric, have no negative eigenvalue, and have exactly 6 zero eigenvalues. A seventh zero would be a deformation that costs nothing (an "hourglass mode", which makes the model floppy); fewer than 6 would mean a coding error. Selective reduced integration is exactly the kind of change that can create hourglass modes, which is why this test exists.

### 1.6 Test C: the design problem and its gradient

Give every element a density $\rho_e\in[0,1]$ and let its Young's modulus be (SIMP, Bendsøe 1989, *Struct. Optim.* 1:193)

$$E_e = E_{\min} + \rho_e^{\,p}\,(E_0 - E_{\min}),\qquad p \approx 3,\quad E_{\min} = 10^{-9}E_0. \tag{9}$$

The objective is the compliance $C = \mathbf f^{\mathsf T}\mathbf u$, the work done by the loads. Press on a structure: the stiffer it is, the less the load sinks and the less work it does. The design problem is

$$\min_{\boldsymbol\rho}\ C(\boldsymbol\rho) = \mathbf f^{\mathsf T}\mathbf u(\boldsymbol\rho)\quad\text{subject to}\quad \mathbf K(\boldsymbol\rho)\,\mathbf u = \mathbf f,\quad \sum_e\rho_e v_e \le V^*\sum_e v_e,\quad 0\le\rho_e\le1. \tag{10}$$

Why $p > 1$: with $p = 1$ a half-dense element gives half the stiffness for half the material, so grey is a fair buy; with $p = 3$ it gives one eighth of the stiffness, so the optimiser drives elements to 0 or 1.

The optimiser needs $\partial C/\partial\rho_e$ for every element, at the cost of one solve. Differentiate $\mathbf K\mathbf u = \mathbf f$ ($\mathbf f$ does not depend on $\boldsymbol\rho$): $(\partial\mathbf K/\partial\rho_e)\mathbf u + \mathbf K\,\partial\mathbf u/\partial\rho_e = 0$. Then, with $\mathbf K$ symmetric and $\mathbf K^{-1}\mathbf f = \mathbf u$,

$$\frac{\partial C}{\partial\rho_e} = \mathbf f^{\mathsf T}\frac{\partial\mathbf u}{\partial\rho_e} = -\mathbf u^{\mathsf T}\frac{\partial\mathbf K}{\partial\rho_e}\mathbf u = -\,p\,\rho_e^{\,p-1}(E_0 - E_{\min})\;\mathbf u_e^{\mathsf T}\hat{\mathbf k}\,\mathbf u_e\ \le 0. \tag{11}$$

In words: the gradient for each element is minus its own strain energy, up to the SIMP factor, and never positive, because adding material can only stiffen. This is the adjoint method; for compliance the adjoint vector equals $\mathbf u$ itself, so no second solve is needed.

**The test.** On a small 3D grid ($4\times3\times2$ cubes, random densities between 0.2 and 1, $E_{\min} = 10^{-3}$, the face $x = 0$ clamped, two point loads at the free end) the code computes (11), and JAX computes the derivative of $C(\boldsymbol\rho) = \mathbf f^{\mathsf T}\mathbf K(\boldsymbol\rho)^{-1}\mathbf f$ by automatic differentiation in 64-bit arithmetic, without knowing (11). Pass: the largest difference, relative to the largest gradient, below $10^{-10}$.

### 1.7 Test D: the fast solver against the exact one

Solving (5) is the cost of everything. The code solves it without ever building $\mathbf K$: every cube shares one $\hat{\mathbf k}$, so $\mathbf K\mathbf u$ is one gather, one small dense product and one scatter. Conjugate gradients converge fast on this when they are preconditioned by geometric multigrid (Amir, Aage & Lazarov 2014, *Struct. Multidisc. Optim.* 49:815): the error is smoothed on the fine grid, and its smooth part is removed on coarser grids, where it looks rough again. Picture smoothing a crumpled sheet: first iron the small creases, then step back and pull out the big waves.

**The test.** The chair's load cases, at 8 voxels across the half width, with a uniform density 0.05, are solved by this multigrid solver to a tolerance of $10^{-10}$ and by the direct sparse solver PARDISO, which factorises $\mathbf K$ exactly. Pass: relative difference of the displacement fields below $10^{-6}$.

### 1.8 Test E: the MBB beam against the 88-line paper

**Picture.** A long bridge deck on two supports, loaded at mid-span from above. By symmetry only the left half is modelled: the left edge may slide up and down but not sideways (the mirror plane), the bottom-right corner rests on a roller, and a unit load pushes down at the top-left corner. The task is to place half the material ($V^* = 0.5$) so the half-beam is as stiff as possible.

Two diseases need cures (Sigmund & Petersson 1998, *Struct. Optim.* 16:68). Low-order elements make a chessboard of solid and void look artificially stiff, so the optimiser draws one; and the continuum problem has no length scale, so finer meshes give ever finer members. Both are cured by a filter of radius $r_{\min}$, which makes members at least about $2r_{\min}$ wide. The paper uses two:

* sensitivity filter ($f_t = 1$, Sigmund 1997): $\widetilde{\dfrac{\partial C}{\partial\rho_e}} = \dfrac{\sum_i H_{ei}\,\rho_i\,\partial C/\partial\rho_i}{\max(\gamma,\rho_e)\sum_i H_{ei}}$, with $\gamma = 10^{-3}$;
* density filter ($f_t = 2$, Bruns & Tortorelli 2001; Bourdin 2001): $\tilde\rho_e = \dfrac{\sum_i H_{ei}\rho_i}{\sum_i H_{ei}}$, gradients by the chain rule;

both with the weights $H_{ei} = \max(0,\ r_{\min} - |\mathbf x_e - \mathbf x_i|)$. The density update is the optimality-criteria step: at an optimum every grey element satisfies $\partial C/\partial\rho_e + \Lambda\,\partial V/\partial\rho_e = 0$, with $\Lambda$ the price of material. The step

$$\rho_e \leftarrow \rho_e\,B_e^{1/2},\qquad B_e = \frac{-\partial C/\partial\rho_e}{\Lambda\,\partial V/\partial\rho_e}, \tag{12}$$

clipped to a move limit $m = 0.2$ and to $[0,1]$, with $\Lambda$ found by bisection so that the volume is exactly $V^*$. Picture a market: elements whose strain energy is worth more than the price of material get more material, the others less. The loop stops when no design variable changes by more than 0.01.

**The test.** The paper's Fig. 3 gives the final compliance for $60\times20$ and $150\times50$ elements with each filter, at $p = 3$, $\nu = 0.3$ and $r_{\min} = 0.04\,n_{elx}$ (the value its numbers use, not the 1.5 of its example call): 216.81 and 233.71 ($60\times20$, $f_t = 1, 2$), 219.52 and 235.73 ($150\times50$) (Andreassen, Clausen, Schevenels, Lazarov & Sigmund 2011, *Struct. Multidisc. Optim.* 43:1, doi:10.1007/s00158-010-0594-7). The Python port in `topopt/top2d.py` follows the MATLAB code line by line. Pass: relative error below $10^{-4}$ for all four.
