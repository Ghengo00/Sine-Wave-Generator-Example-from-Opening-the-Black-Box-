"""
Configuration file for RNN sine wave generation experiment.
Contains all hyperparameters, network dimensions, and experimental settings.
"""


import numpy as np
import jax
import jax.numpy as jnp

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


# =============================================================================
# RANDOM SEEDS
# =============================================================================
NUMPY_SEED = 42
JAX_SEED = 42


# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================
N = 200         # number of neurons (STANDARD VALUE IS 200)
I = 1           # input dimension (scalar input)
num_tasks = 51  # number of different sine-wave tasks (STANDARD VALUE IS 51)

# Sparsity parameter
s = 0.0         # sparsity of the recurrent connections

# L1 regularization parameter
L1_REG_STRENGTH = 0.0  # L1 regularization strength for connectivity matrix (default: no regularization)


# =============================================================================
# TASK PARAMETERS
# =============================================================================
# Frequencies: equally spaced between 0.1 and 0.6 rad/s
omegas = jnp.linspace(0.1, 0.6, num_tasks, dtype=jnp.float64)

# Static input offset for each task: j/51 + 0.25, j=0,...,50
static_inputs = jnp.linspace(0, num_tasks - 1, num_tasks, dtype=jnp.float64) / num_tasks + 0.25


# =============================================================================
# TIME PARAMETERS
# =============================================================================
dt = 0.02        # integration time step (seconds)
T_drive = 8.0    # driving phase duration (seconds)
T_train = 64.0   # training phase duration (seconds)

# Computed time parameters
num_steps_drive = int(T_drive / dt)
num_steps_train = int(T_train / dt)
time_drive = jnp.arange(0, T_drive, dt, dtype=jnp.float64)
time_train = jnp.arange(0, T_train, dt, dtype=jnp.float64)
time_full = jnp.concatenate([time_drive, T_drive + time_train])


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
# Optimization parameters
NUM_EPOCHS_ADAM = 1000
NUM_EPOCHS_LBFGS = 4000
LOSS_THRESHOLD = 1e-4

# Adam optimizer parameters
ADAM_LR = 1e-3

# L-BFGS optimizer parameters
LEARNING_RATE = None        # "None" means use the built-in zoom linesearch
MEMORY_SIZE = 10            # store the last 10 (s,y) pairs
SCALE_INIT_PRECOND = True   # scale initial inverse Hessian to identity


# =============================================================================
# FIXED (AND SLOW) POINT AND FREQUENCY ANALYSIS PARAMETERS
# =============================================================================
# Fixed point search parameters
NUM_ATTEMPTS = 50
TOL = 1e-2
MAXITER = 1000
GAUSSIAN_STD = 0.5         # standard deviation for Gaussian noise in perturbed initial conditions

# Slow point search parameters
SLOW_POINT_SEARCH = False  # whether to search for slow points
NUM_ATTEMPTS_SLOW = 50
TOL_SLOW = 1e-2
MAXITER_SLOW = 1000
GAUSSIAN_STD_SLOW = 1.0    # standard deviation for Gaussian noise in perturbed initial conditions for slow points
SLOW_POINT_CUT_OFF = 1e-1  # threshold for slow points to be accepted

# Jacobian and slow point analysis parameters
JACOBIAN_TOL = 1e-2

# Eigenvalue analysis parameters
EIGENVALUE_SAMPLE_FREQUENCY = 20  # Sample eigenvalues every N iterations during training


# =============================================================================
# PCA ANALYSIS PARAMETERS
# =============================================================================
# Number of principal components to compute
PCA_N_COMPONENTS = 10

# Number of steps to skip when computing PCA
PCA_SKIP_OPTIONS = [0, 200, 400, 600]
# Whether or not to apply tanh transformation to the data before PCA
PCA_TANH_OPTIONS = [False, True]


# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================
# Test indices for plotting
TEST_INDICES = np.linspace(0, len(omegas) - 1, 5, dtype=int)
TEST_INDICES_EXTREMES = [0, len(omegas) - 1]

# Colors and markers for plotting
COLORS = [
    'b', 'g', 'r', 'c', 'm',
    'y', 'k',
    'orange', 'purple', 'brown',
    'pink', 'gray', 'olive', 'teal',
    'navy', 'gold', 'lime', 'coral',
    'slateblue', 'darkgreen', 'cyan'
]

MARKERS = [
    'o', 's', '^', 'v', '<',
    '>',
    'd', 'D',
    'p',
    'h', 'H',
    '*',
    '+', 'x',
    '|', '_',
    '1', '2', '3', '4'
]