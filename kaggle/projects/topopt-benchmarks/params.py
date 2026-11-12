# Material and load. The tests are dimensionless: deflections scale as F / (E t), compliances as F^2 / E.
E0 = 1.0             # Young's modulus of solid material, E_0 in (9)
E_MIN = 1e-9         # Young's modulus of void, E_min in (9)
NU = 0.3             # Poisson's ratio, nu in (3)
F = 1.0              # tip load of the cantilevers, F in (7)-(8)
T = 1.0              # plate thickness of the 2D cantilever, t in (6)

# Test A: cantilever against beam theory, (7)-(8)
L_OVER_H = 10                    # length / height of the beam
N_H_2D = [2, 4, 8, 16, 32]       # elements across the height, 2D
N_H_3D = [2, 4, 8, 16]           # elements across the square section, 3D
CANTILEVER_TOL = 0.01            # pass: |FE / Timoshenko - 1| below this at the finest mesh

# Test B: rigid-body modes of the hex8 element (uses NU)
ZERO_MODES_EXPECTED = 6

# Test C: adjoint gradient (11) against JAX automatic differentiation
GRAD_SEED = 0                    # random densities in [0.2, 1] from this seed
GRAD_PENAL = 3.0                 # p in (9)
GRAD_EMIN = 1e-3                 # E_min in (9) for this test (a stiffer void keeps the dense solve well conditioned)
GRAD_TOL = 1e-10

# Test D: multigrid-preconditioned CG against the direct solver PARDISO
MG_N = 8                         # chair grid: voxels across the half width
MG_RHO = 0.05                    # uniform density
MG_PENAL = 3.0
MG_TOL = 1e-6                    # pass: relative difference of the displacement fields

# Test E: MBB beam against Andreassen et al. (2011), Fig. 3
MBB_CASES = [(60, 20, 1), (60, 20, 2), (150, 50, 1), (150, 50, 2)]   # (nelx, nely, filter type)
VOLFRAC = 0.5                    # V* in (10)
PENAL = 3.0                      # p in (9)
RMIN_FACTOR = 0.04               # r_min = RMIN_FACTOR * nelx, as in the paper's Fig. 3
MOVE = 0.2                       # move limit m of the update (12)
OC_TOL = 0.01                    # stop when no design variable changes by more than this
MAXIT = 2000
MBB_PAPER = {"60x20 ft1": 216.81, "60x20 ft2": 233.71, "150x50 ft1": 219.52, "150x50 ft2": 235.73}
MBB_TOL = 1e-4                   # relative; the paper prints five significant digits

PRECISION = "float64 throughout (NumPy, SciPy, PARDISO); JAX with jax_enable_x64 for the gradient check"

PARAMS = {k: globals()[k] for k in [
    "E0", "E_MIN", "NU", "F", "T", "L_OVER_H", "N_H_2D", "N_H_3D", "CANTILEVER_TOL", "ZERO_MODES_EXPECTED",
    "GRAD_SEED", "GRAD_PENAL", "GRAD_EMIN", "GRAD_TOL", "MG_N", "MG_RHO", "MG_PENAL", "MG_TOL",
    "MBB_CASES", "VOLFRAC", "PENAL", "RMIN_FACTOR", "MOVE", "OC_TOL", "MAXIT", "MBB_PAPER", "MBB_TOL", "PRECISION"]}
