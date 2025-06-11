# -------------------------------
# -------------------------------
# 0. IMPORTS
# -------------------------------
# -------------------------------

import os
from tqdm import tqdm
import time
from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap, lax
import optax

from sklearn.decomposition import PCA
from scipy.optimize import root, least_squares
from scipy.linalg import eig
from functools import partial




# -------------------------------
# EXECUTION TIMING
# -------------------------------

# Start timing the full execution
EXECUTION_START_TIME = time.time()
print(f"Starting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)




# -------------------------------
# -------------------------------
# 1. RNN SETUP
# -------------------------------
# -------------------------------

# -------------------------------
# 1.1. Set Random Seeds & RNN, Driving, and Training Parameters
# -------------------------------

np.random.seed(42)
key = random.PRNGKey(42)
# key is for random noise during training; subkey is for initialization of parameters
key, subkey = random.split(key)


# Network dimensions and task parameters (CHANGE THESE FOR TEST RUNS)
N = 200         # number of neurons (STANDARD VALUE IS 200)
I = 1           # input dimension (scalar input)
num_tasks = 51  # number of different sine-wave tasks (STANDARD VALUE IS 51)

# Frequencies: equally spaced between 0.1 and 0.6 rad/s
omegas = jnp.linspace(0.1, 0.6, num_tasks, dtype=jnp.float32)

# Static input offset for each task: j/51 + 0.25, j=0,...,50
static_inputs = jnp.linspace(0, num_tasks - 1, num_tasks, dtype=jnp.float32) / num_tasks + 0.25


# Sparsity parameter
s = 0.0         # sparsity of the recurrent connections


# Time parameters (in seconds)
dt = 0.02        # integration time step
T_drive = 8.0    # driving phase duration
T_train = 64.0   # training phase duration
num_steps_drive = int(T_drive / dt)
num_steps_train = int(T_train / dt)
time_drive = jnp.arange(0, T_drive, dt, dtype=jnp.float32)
time_train = jnp.arange(0, T_train, dt, dtype=jnp.float32)
time_full = jnp.concatenate([time_drive, T_drive + time_train])


# Build the driving phase and training phase inputs for all tasks, as well as the training phase targets
# Driving phase inputs: shape (num_tasks, num_steps_drive, I)
drive_inputs = (
    jnp.sin(omegas[:, None] * time_drive[None, :]) # omegas[:, None] is (num_tasks, 1); time_drive[None, :] is (1, num_steps_drive)
                                                   # jnp.sin(...) is (num_tasks, num_steps_drive)
    + static_inputs[:, None]                       # static_inputs[:, None] is (num_tasks, 1); this broadcasts to (num_tasks, num_steps_drive)
)[..., None]                                       # Add a new axis to make it (num_tasks, num_steps_drive, I)

# Training phase inputs: shape (num_tasks, num_steps_train, I)
train_inputs = jnp.broadcast_to(
    static_inputs[:, None, None],                  # static_inputs[:, None, None] is (num_tasks, 1, I); this broadcasts to (num_tasks, num_steps_train, I)
    (num_tasks, num_steps_train, I)
).astype(jnp.float32)

# Training phase targets: shape (num_tasks, num_steps_train)
train_targets = jnp.sin(omegas[:, None] * time_train[None, :])  # omegas[:, None] is (num_tasks, 1); time_train[None, :] is (1, num_steps_train)


def get_drive_input(task):
    """
    Get the driving phase input for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        drive_input: driving phase input, shape (num_steps_drive, I)
    """
    return (jnp.sin(omegas[task] * time_drive) + static_inputs[task])[:, None]  # shape (num_steps_drive, I)


# Input function
def get_train_input(task):
    """
    Get the training phase input for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        train_input: training phase input, shape (num_steps_train, I)
    """
    return jnp.full((num_steps_train, I), static_inputs[task], dtype=jnp.float32)  # shape (num_steps_train, I)


# (Target) output function
def get_train_target(task):
    """
    Get the training phase target for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        train_target: training phase target, shape (num_steps_train,)
    """
    return (jnp.sin(omegas[task] * time_train)) # shape (num_steps_train,)




# -------------------------------
# 1.2. Define RNN Weights
# -------------------------------

# Initialize parameters with scaling ~1/sqrt(N)
def init_params(key):
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    mask        = random.bernoulli(k_mask, p = 1.0 - s, shape=(N,N)).astype(jnp.float32)
    J_unscaled  = random.normal(k1, (N, N)) / jnp.sqrt(N) * mask
    # By Girko's law, we know that, in the limit N -> infinity, the eigenvalues of J will converge to a uniform distribution
    # within a disk centred at the origin of radius sqrt(Var * N) = sqrt((1 / N) * N) = 1.
    # After masking, the elements remain independently distributed with mean zero,
    # but their variance becomes Var_m = (1 - s) / N, meaning the spectral radius becomes sqrt(1 - s).
    # Thus, to maintain the same spectral radius as the full matrix, we scale J by 1 / sqrt(1 - s).
    J           = J_unscaled / jnp.sqrt(1 - s)

    B           = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x         = jnp.zeros((N,))
    w           = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z         = jnp.array(0.0, dtype=jnp.float32)
    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}

mask, params = init_params(subkey)




# -------------------------------
# 1.3. Define Useful Global Variables
# -------------------------------

BATCH_SIZE = 10


# Optimization parameters
NUM_EPOCHS_ADAM = 2000
NUM_EPOCHS_LBFGS = 1000
LOSS_THRESHOLD = 1e-4

# Adam optimizer parameters
ADAM_LR = 1e-3

# L-BFGS optimizer parameters
LEARNING_RATE = None        # "None" means use the built-in zoom linesearch
MEMORY_SIZE = 10            # store the last 10 (s,y) pairs
SCALE_INIT_PRECOND = True   # scale initial inverse Hessian to identity


# Plotting parameters
TEST_INDICES = np.linspace(0, omegas.shape[0] - 1, 5, dtype=int)
TEST_INDICES_EXTREMES = [0, omegas.shape[0] - 1]
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


# Fixed point search parameters
NUM_ATTEMPTS = 50
TOL = 1e-2
MAXITER = 1000
GAUSSIAN_STD = 0.5         # standard deviation for Gaussian noise in perturbed initial conditions

# Slow point search parameters
NUM_ATTEMPTS_SLOW = 50
TOL_SLOW = 1e-2
MAXITER_SLOW = 1000
GAUSSIAN_STD_SLOW = 1.0    # standard deviation for Gaussian noise in perturbed initial conditions for slow points
SLOW_POINT_CUT_OFF = 1e-1  # threshold for slow points to be accepted


# Jacobian and slow point analysis parameters
JACOBIAN_TOL = 1e-2




# -------------------------------
# 1.4. Define Saving Functions
# -------------------------------

# Create timestamp for output filenames - this ensures all files from a run will have the same timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train):
    """
    Generate a filename with timestamp and parameters.
    """
    return f"{variable_name}_{RUN_TIMESTAMP}_Neuron_Number_{N}_Task_Number_{num_tasks}_Time_Steps_{dt}_Driving_Time_{T_drive}_Training_Time_{T_train}.pkl"


def save_variable(variable, variable_name, N, num_tasks, dt, T_drive, T_train):
    """
    Save a variable to a pickle file with a descriptive filename in the Outputs folder.
    """
    # Define the output directory
    output_dir = os.path.join(os.getcwd(), 'Outputs', f'JAX_Outputs_{RUN_TIMESTAMP}')

    # Create the output directory if it doesn't exist (exist_ok avoids error if it does)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train)
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")


def save_figure(fig, figure_name, N, num_tasks, dt, T_drive, T_train):
    """
    Save a matplotlib figure with timestamp and parameters.
    """
    # Define the output directory
    output_dir = os.path.join(os.getcwd(), 'Outputs', f'JAX_Outputs_{RUN_TIMESTAMP}')

    # Create the output directory if it doesn't exist (exist_ok avoids error if it does)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(figure_name, N, num_tasks, dt, T_drive, T_train) + '.png'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        fig.savefig(filepath)
        print(f"Saved figure {figure_name} to {filepath}")
    except Exception as e:
        print(f"Error saving figure {figure_name} to {filepath}: {e}")




# -------------------------------
# -------------------------------
# 2. RNN TRAINING
# -------------------------------
# -------------------------------

# -------------------------------
# 2.1. Trajectory Simulation Function
# -------------------------------

# Dynamics
@jit
def rnn_step(x, inputs, params):
    """
    Single time-step update of the RNN state.
    
    Arguments:
    x     : current state (at time t), dimensions (N,)
    inputs: scalar input at this time-step (time t), dimensions (I,)
    params: dict containing J, B, b_x, w, b_z

    Returns:
    x_new: new state (at time t+1), dimensions (N,)
    z    : output at this time-step (time t), scalar
    """
    J, B, b_x, w, b_z = params["J"], params["B"], params["b_x"], params["w"], params["b_z"]
    
    r = jnp.tanh(x)            # (N,)
    z = jnp.dot(w, r) + b_z    # scalar
    
    # dx/dt = -x + J r + B u + b_x
    # Note that (B * inputs).sum(axis=1) gives a vector of shape (N,), where each element is the dot product between the corresponding row of B and the vector inputs
    # This is because B is shape (N, I) and inputs is shape (I,), so B * inputs is broadcast to (N,I); .sum(axis=1) then collpases each row to a single element
    dx = -x + J @ r + (B * inputs).sum(axis=1) + b_x
    x_new = x + dt * dx
    
    return x_new, z


# Solving
@jit
def simulate_trajectory(x0, u_seq, params):
    """
    Vectorized simulation over T time-steps using lax.scan.

    Arguments:
    x0    : initial state, dimensions (N,)
    u_seq : input sequence over the T time steps, dimensions (T, I)
    params: dict containing J, B, b_x, w, b_z

    Returns
      xs : system state sequence over the T time steps, dimensions (T+1, N)
      zs : output sequence over the T time steps, dimensions (T,)
    """
    
    def scan_func(carry, u_t):
        """
        Single step function for lax.scan.

        Arguments:
            carry : current state (at time t), dimensions (N,)
            u_t   : input at this time-step (time t), dimensions (I,)
            params: dict containing J, B, b_x, w, b_z
        
        Returns:
            x_tp1 : new state (at time t+1), dimensions (N,)
            (x_tp1, z_t) : tuple containing new state and output at this time-step (time t), dimensions (N,), scalar
        """
        x_t        = carry
        x_tp1, z_t = rnn_step(x_t, u_t.squeeze(), params)
        
        return x_tp1, (x_tp1, z_t)


    # Run scan: initial carry is x0, iterate over u_seq
    # lax.scan collects the tuples returned by scan_func into a tuple of arrays
    x_final, (xs_all, zs_all) = lax.scan(scan_func, x0, u_seq)
    
    # xs_all has shape (T, N), but we want (T+1, N) including x0 at index 0, so we stack this
    xs = jnp.vstack([x0[None, :], xs_all])
    
    return xs, zs_all


# Solving
@jit
def solve_task(params, task):
    x0        = jnp.zeros((N,), dtype=jnp.float32)      # Initial state for the task
    
    # Drive phase
    d_u       = get_drive_input(task)                   # shape (num_steps_drive, I)
    xs_d, _   = simulate_trajectory(x0, d_u, params)
    x_final   = xs_d[-1]
    
    # Train phase
    t_u       = get_train_input(task)                   # shape (num_steps_train, I)
    _, zs_tr  = simulate_trajectory(x_final, t_u, params)

    return zs_tr    # shape (num_steps_train,)


# Vectorize solve_task over all task values
vmap_solve_task = vmap(solve_task, in_axes=(None, 0))   # vmap_solve_task gives an array of shape (num_tasks, num_steps_train)




# -------------------------------
# 2.2. Full-Scale Simulation Functions
# -------------------------------




# -------------------------------
# 2.2.1. Diagnostics Simulation Function
# -------------------------------

def run_single_task_diagnostics(params, omega, u_off):
    """
    Run exactly one (omega, u_off) through the drive and train phases

    Arguments:
        params: dict containing J, B, b_x, w, b_z
        omega : frequency of the sine wave for this task
        u_off : static input offset for this task

    Returns:
        task_loss    : scalar loss value for this task
        xs_train     : the training trajectory for the task, for the given parameters (jnp.ndarray, shape (num_steps_train+1, N))
        x_drive_final: the final state of the system after the drive phase, for the given parameters (jnp.ndarray, shape (N,))
    """
    # Set the initial hidden state
    x0 = jnp.zeros((N,), dtype=jnp.float32)

    # Build the driving‐phase input
    u_drive = jnp.sin(omega * time_drive).reshape(-1, 1) + u_off        # (num_steps_drive, I)

    # Drive phase
    xs_drive, _ = simulate_trajectory(x0, u_drive, params)
    x_drive_final = xs_drive[-1]  # (N,)

    # Build the training‐phase input and target
    u_train = jnp.full((num_steps_train, 1), u_off, dtype=jnp.float32)  # (num_steps_train, I)
    target_train = jnp.sin(omega * time_train)                          # (num_steps_train,)

    # Train phase
    xs_train, zs_train = simulate_trajectory(x_drive_final, u_train, params)   # zs_train: (num_steps_train,)

    # Compute MSE over the training phase
    task_loss = jnp.mean((zs_train - target_train) ** 2)
    
    return task_loss, xs_train, x_drive_final


def run_batch_diagnostics(params, key):
    """
    Run all tasks in a vectorised fashion, compute loss, and collect trajectories.
    Only for post-training diagnostics and analysis.

    Arguments:
        params: dict containing J, B, b_x, w, b_z
        key   : JAX random key for reproducibility

    Returns:
      avg_loss         : total loss, scalar
      traj_states      : list of states at each time step during training for each task, so a list of num_tasks elements, each of shape (num_steps_train+1, N)
      fixed_point_inits: list of drive phase final states for each task, so a list of num_tasks elements, each of shape (N,)
    """
    # vmap over the two task‐specific arrays: omegas and static_inputs
    all_task_loss, all_xs_train, all_x_drive_final = jax.vmap(
        lambda omega, u0: run_single_task_diagnostics(params, omega, u0),
        in_axes=(0, 0)  # both omega and u_off vary along the leading dimension
    )(omegas, static_inputs)
    # all_task_loss shape    : (num_tasks,)
    # all_xs_train shape     : (num_tasks, num_steps_train+1, N)
    # all_x_drive_final shape: (num_tasks, N)

    avg_loss = jnp.mean(all_task_loss)

    # Converty to NumPy arrays
    traj_states = np.array(all_xs_train)                # shape (num_tasks, num_steps_train+1, N)
    fixed_point_inits = np.array(all_x_drive_final)     # shape (num_tasks, N)

    return float(avg_loss), traj_states, fixed_point_inits




# -------------------------------
# 2.2.2. Loss Simulation Function
# -------------------------------

# Loss
@jit
def batched_loss(params):
    """
    Compute the loss for all tasks.

    Arguments:
        params: dictionary of RNN weights
    
    Returns:
        loss  : scalar loss value averaged over all tasks
    """
    tasks = jnp.arange(num_tasks)

    # Vectorized simulation over all tasks
    zs_all = vmap_solve_task(params, tasks)      # shape: (num_tasks, num_steps_train)

    # Get the training targets for all tasks
    targets_all = vmap(get_train_target)(tasks)  # shape: (num_tasks, num_steps_train)

    # Compute the MSE over tasks and time
    return jnp.mean((zs_all - targets_all) ** 2)




# -------------------------------
# 2.3. Optimizer Setups
# -------------------------------

# Find the value and gradient of the loss function
value_and_grad_fn = jax.value_and_grad(batched_loss)




# -------------------------------
# 2.3.1. Adam Optimizer Setup
# -------------------------------

adam_opt = optax.adam(ADAM_LR)

# Initialize the optimizer state with our initial params
adam_state = adam_opt.init(params)


# Training
@jax.jit
def adam_step(params, opt_state):
    """
    Perform one step of the Adam optimizer.
    """
    # (1) Evaluate current loss and gradient
    loss, grads        = value_and_grad_fn(params)

    # (2) One optimizer update
    updates, new_state = adam_opt.update(grads, opt_state, params)

    # (3) Apply updates to your parameters
    new_params         = optax.apply_updates(params, updates)

    # (4) Enforce exact sparsity on the J matrix
    J1                 = new_params["J"] * mask

    # # (5) Scale the J matrix to maintain the same spectral radius
    # J2               = J1 / jnp.sqrt(1 - s)

    # (6) Reconstruct the new parameters with the updated J matrix
    new_params = {**new_params, "J": J1}

    return new_params, new_state, loss




# -------------------------------
# 2.3.2. L-BFGS Optimizer Setup
# -------------------------------

lbfgs_opt = optax.lbfgs(
    learning_rate       = LEARNING_RATE,
    memory_size         = MEMORY_SIZE,
    scale_init_precond  = SCALE_INIT_PRECOND
    )

# Initialize the optimizer state with our initial params
lbfgs_state = lbfgs_opt.init(params)


# Training
@jax.jit
def lbfgs_step(params, opt_state):
    """
    Perform one step of the L-BFGS optimizer.
    """
    # (1) Evaluate current loss and gradient
    loss, grads        = value_and_grad_fn(params)

    # (2) One optimizer update
    updates, new_state = lbfgs_opt.update(
        grads, opt_state, params,
        value = loss, grad = grads,
        value_fn = lambda p: batched_loss(p)
    )

    # (3) Apply updates to your parameters
    new_params         = optax.apply_updates(params, updates)

    # (4) Enforce exact sparsity on the J matrix
    J1                 = new_params["J"] * mask

    # # (5) Scale the J matrix to maintain the same spectral radius
    # J2               = J1 / jnp.sqrt(1 - s)

    # (6) Reconstruct the new parameters with the updated J matrix
    new_params = {**new_params, "J": J1}

    return new_params, new_state, loss




# -------------------------------
# 2.4. Training Procedure
# -------------------------------

# Training
@partial(jax.jit, static_argnames=("step_fn","num_steps","tag"))
def scan_with_history(params, opt_state, step_fn, num_steps, tag):
    """
    Function to run multiple steps of the chosen optimization method via lax.scan.
    Run `num_steps` of `step_fn` starting from (params, opt_state), but only keep the best-loss params, rather than the whole history.

    Arguments:
        params   : current parameters of the model
        opt_state: current state of the optimizer
        step_fn  : function to perform a single optimization step (adam_step or lbfgs_step)
        steps    : number of steps to run
        tag      : string tag for debug printing (e.g., "ADAM" or "L-BFGS")

    Returns:
        best_params     : parameters corresponding to best_loss
        best_loss       : lowest loss out of all the iterations
        final_params    : parameters after the last optimization step
        final_opt_state : optimizer state after the last optimization step
    """
    # Initialize best‐so‐far
    init_best_loss   = jnp.inf
    init_best_params = params

    def body(carry, idx_unused):
        params, opt_state, best_loss, best_params = carry

        # One optimization step
        params, opt_state, loss = step_fn(params, opt_state)

        # (Optional) debug print
        jax.debug.print("[{}] step {}: loss {:.6e}", tag, idx_unused+1, loss)

        # If this step is better, update best_loss and best_params
        better = loss < best_loss
        best_loss   = jax.lax.select(better, loss, best_loss)
        best_params = jax.tree_util.tree_map(
            lambda p_new, p_best: jax.lax.select(better, p_new, p_best),
            params, best_params
        )

        return (params, opt_state, best_loss, best_params), None

    # Run the scan; we feed a dummy sequence of length `num_steps` so body is called that many times
    init_carry = (params, opt_state, init_best_loss, init_best_params)
    (final_p, final_s, best_loss, best_params), _ = jax.lax.scan(
        body,
        init_carry,
        jnp.arange(num_steps),
        length = num_steps
    )
    return best_params, best_loss, final_p, final_s




# -------------------------------
# 2.4.1. Execute Adam Phase of Training Procedure
# -------------------------------

best_params_adam, best_loss_adam, params_after_adam, adam_state_after = scan_with_history(
    params, adam_state, adam_step,
    NUM_EPOCHS_ADAM,
    tag = "ADAM"
)

print(f"Lowest Adam Loss: {best_loss_adam:.6e}")

# Restore the best parameters
params = best_params_adam
best_loss_total = best_loss_adam




# -------------------------------
# 2.4.2. Execute L-BFGS Phase of Training Procedure
# -------------------------------

if best_loss_adam > LOSS_THRESHOLD:
    # Run L-BFGS optimization phase
    best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after = scan_with_history(
        params, lbfgs_state, lbfgs_step,
        NUM_EPOCHS_LBFGS,
        tag = "L-BFGS"
    )

    print(f"Lowest L-BFGS Loss: {best_loss_lbfgs:.6e}")

    # Restore the best parameters
    if best_loss_lbfgs < best_loss_adam:
        params = best_params_lbfgs
        best_loss_total = best_loss_lbfgs

print(f"Overall best loss = {best_loss_total:.6e}")




# -------------------------------
# 2.5. Obtain and Save Results from Training Procedure
# -------------------------------




# -------------------------------
# 2.5.1. Store Final Results After Training Procedure
# -------------------------------

# After training, collect final trajectories and fixed-point inits
final_loss, state_traj_states, state_fixed_point_inits = run_batch_diagnostics(params, key)
print(f"Final loss after training: {final_loss:.6e}")




# -------------------------------
# 2.5.2. Save Results from Training Procedure
# -------------------------------

print("\nSaving training results...")
save_variable(np.array(params["J"]), "J_param", N, num_tasks, dt, T_drive, T_train)
save_variable(np.array(params["B"]), "B_param", N, num_tasks, dt, T_drive, T_train)
save_variable(np.array(params["b_x"]), "b_x_param", N, num_tasks, dt, T_drive, T_train)
save_variable(np.array(params["w"]), "w_param", N, num_tasks, dt, T_drive, T_train)
save_variable(np.array(params["b_z"]), "b_z_param", N, num_tasks, dt, T_drive, T_train)

state_dict = {
    "traj_states": state_traj_states,
    "fixed_point_inits": state_fixed_point_inits
}
save_variable(state_dict, "state", N, num_tasks, dt, T_drive, T_train)




# -------------------------------
# -------------------------------
# 3. DYNAMICS RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 3.1. Produced Trajectories vs. Target Signals Plots
# -------------------------------

print("\nPlotting produced trajectories vs. target signals for selected tasks...")
x0_test = jnp.zeros((N,))

for idx, j in enumerate(TEST_INDICES):
    omega = omegas[j]
    u_off = static_inputs[j]

    # Build input sequences
    u_drive_np = (np.sin(omega * time_drive) + u_off).astype(np.float32).reshape(-1, 1)
    u_train_np = np.full((num_steps_train, 1), u_off, dtype=np.float32)
    target_train = np.sin(omega * time_train)

    u_drive = jnp.array(u_drive_np)
    u_train = jnp.array(u_train_np)

    xs_drive_test, _ = simulate_trajectory(x0_test, u_drive, params)
    x_drive_final_test = xs_drive_test[-1]

    _, zs_train_test = simulate_trajectory(x_drive_final_test, u_train, params)
    produced_traj = np.array(zs_train_test)

    plt.figure(figsize=(10, 3))
    plt.plot(time_train, produced_traj, label="Produced Output", linewidth=2)
    plt.plot(time_train, target_train, 'k--', label="Target Signal", linewidth=1.5)
    plt.title(f"Task {j}: ω={omega:.3f}, u_offset={u_off:.3f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()




# -------------------------------
# 3.2. Parameter Matrices Plots
# -------------------------------

def plot_parameter_matrices(J, B, b_x, j, omega, u_offset):
    """
    Plot the parameter matrices J, B, and b_x for a given task.

    Arguments:
        J        : Jacobian matrix, shape (N, N)
        B        : Input matrix, shape (N, I)
        b_x      : Bias vector, shape (N,)
        j        : Task index
        omega    : Frequency for the task
        u_offset : Static input offset for the task
    """
    # Create the figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Set the title for the figure
    fig.suptitle(f'Parameter Matrices (Task {j}, ω={omega:.3f}, u_offset={u_offset:.3f})', fontsize=16)

    # Plot J matrix
    im = axes[0].imshow(np.array(J), cmap='viridis')
    axes[0].set_title('J Matrix')
    plt.colorbar(im, ax=axes[0])

    # Plot B matrix
    im = axes[1].imshow(np.array(B), cmap='viridis')
    axes[1].set_title('B Matrix')
    plt.colorbar(im, ax=axes[1])

    # Plot b_x vector
    im = axes[2].imshow(np.array(b_x).reshape(-1, 1), cmap='viridis')
    axes[2].set_title('b_x Vector')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.show()
    plt.close()


print("\nPlotting parameter matrices for selected tasks...")
for j in TEST_INDICES_EXTREMES:
    omega = omegas[j]
    u_off = static_inputs[j]
    plot_parameter_matrices(params["J"], params["B"], params["b_x"], j, omega, u_off)




# -------------------------------
# EXECUTION TIMING
# -------------------------------

# End timing the full execution
EXECUTION_END_TIME = time.time()
TOTAL_EXECUTION_TIME = EXECUTION_END_TIME - EXECUTION_START_TIME

print("\n" + "=" * 60)
print(f"Total execution time: {TOTAL_EXECUTION_TIME:.2f} seconds")

# Convert to more readable format
hours = int(TOTAL_EXECUTION_TIME // 3600)
minutes = int((TOTAL_EXECUTION_TIME % 3600) // 60)
seconds = TOTAL_EXECUTION_TIME % 60

if hours > 0:
    print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f} (HH:MM:SS.mmm)")
elif minutes > 0:
    print(f"Total execution time: {minutes:02d}:{seconds:06.3f} (MM:SS.mmm)")
else:
    print(f"Total execution time: {seconds:.3f} seconds")




# -------------------------------
# -------------------------------
# 4. FIXED POINTS, SLOW POINTS, AND UNSTABLE MODES RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 4.1.1. Jacobians, Fixed Points, Slow Points, and Unstable Mode Frequencies Functions
# -------------------------------

def compute_F(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute the evolutiion equation for the RNN, F(x) = -x + J * tanh(x) + B * u + b_x.

    Notable arguments:
        x_np   : current state, shape (N,)
        u_val  : scalar input value
    """
    return -x_np + J_np @ np.tanh(x_np) + (B_np * np.array([u_val])).sum(axis=1) + b_x_np


def compute_jacobian(x_np, J_np):
    """
    Compute the Jacobian of the evolution equation F(x), J_F(x) = -I + J * diag(1 - tanh(x)^2).
    """
    # diag(1 - tanh(x)^2) is a diagonal matrix with the derivative of tanh on the diagonal
    diag_term = 1 - np.tanh(x_np) ** 2

    # Compute the Jacobian J_F(x) = -I + J * diag(1 - tanh(x)^2)
    return -np.eye(len(x_np)) + J_np * diag_term[np.newaxis, :]


def compute_grad_q(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute the gradient of q(x) = |F(x)|^2, grad_q = F(x) @ J_F(x).
    """
    # Compute F(x) = -x + J * tanh(x) + B * u + b_x
    F_x = compute_F(x_np, u_val, J_np, B_np, b_x_np)

    # Compute the Jacobian J_F(x) = -I + J * diag(1 - tanh(x)^2)
    J_F = compute_jacobian(x_np, J_np)

    # Compute the gradient of q(x) = |F(x)|^2, which is grad_q = F(x) @ J_F
    grad_q = F_x @ J_F

    return np.array(grad_q)


def find_points(
    x0_guess, u_const, J_trained, B_trained, b_x_trained,
    mode: str,
    num_attempts=None, tol=None, maxiter=None, alt_inits=None, gaussian_std=None):
    """
    Unified finder for fixed points and slow points.

    Notable arguments:
        mode        : 'fixed' for fixed point search or 'slow' for slow point search
        num_attempts: number of attempts to find a point (default: NUM_ATTEMPTS or NUM_ATTEMPTS_SLOW)
        tol         : tolerance for two fixed/slow points to be considered distinct (default: TOL or TOL_SLOW)
        maxiter     : maximum iterations for the solver (default: MAXITER or MAXITER_SLOW)
        alt_inits   : alternative initial conditions from training trajectories
        gaussian_std: standard deviation for Gaussian noise in perturbed initial conditions (default: GAUSSIAN_STD or GAUSSIAN_STD_SLOW)

    Returns:
        points      : list of found fixed or slow points, each point is a NumPy array of shape (N,)
    """
    mode = mode.lower()
    
    if mode == 'fixed':
        # Use the provided parameters or fixed point search defaults
        num_attempts = num_attempts if num_attempts is not None else NUM_ATTEMPTS
        tol = tol if tol is not None else TOL
        maxiter = maxiter if maxiter is not None else MAXITER
        gaussian_std = gaussian_std if gaussian_std is not None else GAUSSIAN_STD
        # Define the root solver for fixed points
        solver = lambda x: root(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                x, method='lm', options={'maxiter': maxiter})
        success_cond = lambda sol: sol.success
        extract = lambda sol: sol.x

    elif mode == 'slow':
        # Use the provided parameters or slow point search defaults
        num_attempts = num_attempts if num_attempts is not None else NUM_ATTEMPTS_SLOW
        tol = tol if tol is not None else TOL_SLOW
        maxiter = maxiter if maxiter is not None else MAXITER_SLOW
        gaussian_std = gaussian_std if gaussian_std is not None else GAUSSIAN_STD_SLOW
        # Define the least squares solver for slow points
        solver = lambda x: least_squares(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                         x, method='lm', max_nfev=maxiter)
        success_cond = lambda sol: sol.success and np.linalg.norm(sol.fun) < SLOW_POINT_CUT_OFF
        extract = lambda sol: sol.x
    
    else:
        raise ValueError("Argument 'fixed_or_slow' must be either 'fixed' or 'slow'.")

    points = []

    def _attempt(x_init):
        try:
            sol = solver(x_init)
            if success_cond(sol):
                return extract(sol)
        except Exception as e:
            print(f"{mode.capitalize()}-point search failed: {e}")
        return None

    # (I) Original guess
    pt = _attempt(x0_guess)
    if pt is not None:
        points.append(pt)

    # (II) Gaussian perturbations of the original guess
    for _ in tqdm(range(num_attempts - 1), desc=f"Finding {mode} points (perturbations)"):
        pert = x0_guess + np.random.normal(0, gaussian_std, size=x0_guess.shape)
        pt = _attempt(pert)
        if pt is not None and all(np.linalg.norm(pt - p) >= tol for p in points):
            points.append(pt)

    # (III) Alternative initial conditions from training trajectories
    if alt_inits is not None:
        for cand in tqdm(alt_inits, desc=f"Finding {mode} points (initial conditions from training)"):
            pt = _attempt(cand)
            if pt is not None and all(np.linalg.norm(pt - p) >= tol for p in points):
                points.append(pt)

    return points


def analyze_points(fixed_points, J_trained):
    """
    For each fixed point, compute the Jacobian and its unstable eigenvalues.

    Returns:
        jacobians      : list of Jacobian matrices at each fixed point, each matrix is a NumPy array of shape (N, N)
        unstable_freqs : list of lists, where each inner list contains the unstable eigenvalues (with positive real part) for the corresponding fixed point
    """
    jacobians = []
    unstable_freqs = []

    for x_star in fixed_points:
        # Compute Jacobians at the fixed point
        J_eff = np.array(compute_jacobian(x_star, J_trained))
        jacobians.append(J_eff)

        # Compute eigenvalues of the Jacobian
        eigenvals, _ = eig(J_eff)

        # Find all the unstable eigenvalues for the given fixed point (i.e., those with positive real part)
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        if len(unstable_idx) > 0:
            # Store all unstable eigenvalues for this fixed point
            unstable_freqs.append(list(eigenvals[unstable_idx]))
        else:
            # If there are no unstable eigenvalues, append an empty list
            unstable_freqs.append([])
    
    return jacobians, unstable_freqs




# -------------------------------
# 4.1.2. Plotting Functions
# -------------------------------

def plot_jacobian_matrices(jacobians, j):
    """
    Plot the Jacobian matrices for a given task.
    """
    # Create the figure and subplots
    fig, axes = plt.subplots(len(jacobians), 1, figsize=(10, 5 * len(jacobians)))
    
    # Ensure axes is iterable when there's only one Jacobian (and hence a single subplot)
    if len(jacobians) == 1:
        axes = [axes]
    # Iterate over the subplots
    for i, J_eff in enumerate(jacobians):
        im = axes[i].imshow(J_eff, cmap='viridis')
        axes[i].set_title(f'Jacobian Matrix (Task {j}, ω={omegas[j]:.3f}, FP{i+1})')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, f"Jacobian_Matrices_for_Task_{j}", N, num_tasks, dt, T_drive, T_train)
    
    plt.show()
    plt.close()


def plot_unstable_eigenvalues(unstable_freqs, j):
    """
    Plot the unstable eigenvalues for a given task on the complex plane.
    """
    # Create the figure
    fig = plt.figure(figsize=(10, 8))

    # Plot each fixed point's unstable eigenvalues
    for i, freqs in enumerate(unstable_freqs):
        if freqs:   # check that there are unstable frequencies
            freqs_array = np.array(freqs)
            plt.scatter(np.real(freqs_array), np.imag(freqs_array),
                        color=COLORS[i], marker=MARKERS[i], label=f'FP{i+1}')
    
    # Add decorations
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Unstable Eigenvalues for Task {j} (ω={omegas[j]:.3f})')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    save_figure(fig, f"Unstable_Eigenvalues_for_Task_{j}", N, num_tasks, dt, T_drive, T_train)
    
    plt.show()
    plt.close()


def plot_frequency_comparison(all_unstable_eig_freq, omegas):
    """
    Plot a comparison of target frequencies and maximum unstable mode frequencies (imaginary components) for all tasks.

    Notable arguments:
        all_unstable_eig_freq: list of lists of lists, where the outer list is over tasks,
            the middle list is over fixed points,
            and the inner list contains unstable eigenvalues for each fixed point
    """
    # Create the figure and axes
    fig = plt.figure(figsize=(12, 6))

    # Initialize lists to store data for plotting
    task_numbers = []
    target_frequencies = []
    unstable_frequencies = []

    # Collect data for all tasks
    for j, task_freqs in enumerate(all_unstable_eig_freq):
        target_freq = omegas[j]
        # Find the maximum imaginary component across all fixed points in this task
        max_imag = 0
        for fp_freqs in task_freqs:
            if fp_freqs:
                current_max = max(abs(np.imag(ev)) for ev in fp_freqs)
                max_imag = max(max_imag, current_max)
        if max_imag > 0:    # only consider tasks with unstable frequencies
            task_numbers.append(j)
            target_frequencies.append(target_freq)
            unstable_frequencies.append(max_imag)
    
    # Plot the data
    plt.scatter(task_numbers, target_frequencies, color='black', label='Target Frequency', s=100)
    plt.scatter(task_numbers, unstable_frequencies, color='red', label='Max Unstable Mode Frequency', s=100)

    # Add decorations
    plt.xlabel('Task #')
    plt.ylabel('Frequency')
    plt.title('Comparison of Target Frequencies and Maximum Unstable Mode Frequencies')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    save_figure(fig, "Frequency_Comparison", N, num_tasks, dt, T_drive, T_train)
    
    plt.show()
    plt.close()




# -------------------------------
# 4.2.0 Unified Point Search Function: Find Points, Jacobians, and Unstable Mode Frequencies
# -------------------------------

def find_and_analyze_points(point_type="fixed"):
    """
    Unified function to find and analyze either fixed points or slow points for all tasks.
    
    Arguments:
        point_type: str, either "fixed" or "slow" to specify which type of points to find
    
    Returns:
        all_points              : list of lists of points, one list per task
        all_jacobians           : list of lists of Jacobians, one list per task  
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies: outer list is over tasks,
                                middle list is over fixed points, and inner list contains unstable eigenvalues for each fixed point
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set parameters based on point type
    if point_type == "fixed":
        num_attempts = NUM_ATTEMPTS
        tol = TOL
        maxiter = MAXITER
        gaussian_std = GAUSSIAN_STD
        search_name = "fixed point"
    else:  # point_type == "slow"
        num_attempts = NUM_ATTEMPTS_SLOW
        tol = TOL_SLOW
        maxiter = MAXITER_SLOW
        gaussian_std = GAUSSIAN_STD_SLOW
        search_name = "slow point"
    
    # Extract trained parameters
    J_trained = np.array(params["J"])
    B_trained = np.array(params["B"])
    b_x_trained = np.array(params["b_x"])
    
    # Initialize lists to store results
    all_points = []              # List of lists of points, one list per task
    all_jacobians = []           # List of lists of Jacobians, one list per task
    all_unstable_eig_freq = []   # List of lists of lists of unstable frequencies
    
    # Track search time
    start_time = time.time()
    print(f"\nStarting {search_name} search...")
    
    # Iterate over each task to find points
    for j in range(num_tasks):
        if j % 5 == 0:
            print(f"\n Processing Task {j} of {num_tasks}...")

        u_const = static_inputs[j]
        x0_guess = state_fixed_point_inits[j]
        
        # Build fallback initializations from training trajectories
        idxs = np.random.choice(len(state_traj_states),
                                size=min(num_tasks, num_attempts), replace=False)
        fallback_inits = []
        for idx in idxs:
            traj = state_traj_states[idx]                   # shape (num_steps_train+1, N)
            row = np.random.randint(0, traj.shape[0])
            fallback_inits.append(traj[row])                # shape (N,)
        
        # Find the points
        task_points = find_points(
            x0_guess, u_const, J_trained, B_trained, b_x_trained,
            mode=point_type,
            num_attempts=num_attempts, 
            tol=tol, 
            maxiter=maxiter, 
            alt_inits=fallback_inits, 
            gaussian_std=gaussian_std
        )
        all_points.append(task_points)
        
        # Analyze the points to compute Jacobians and unstable eigenvalues
        task_jacobians, task_unstable_freqs = analyze_points(task_points, J_trained)
        all_jacobians.append(task_jacobians)
        all_unstable_eig_freq.append(task_unstable_freqs)
    
    # Calculate search time
    search_time = time.time() - start_time
    print(f"{search_name.capitalize()} search completed in {search_time:.2f} seconds")
    
    # Save results
    print(f"\nSaving {search_name} analysis results...")
    save_variable(all_points, f"all_{point_type}_points", N, num_tasks, dt, T_drive, T_train)
    save_variable(all_jacobians, f"all_{point_type}_jacobians", N, num_tasks, dt, T_drive, T_train)
    save_variable(all_unstable_eig_freq, f"all_{point_type}_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train)
    
    return all_points, all_jacobians, all_unstable_eig_freq
    



# -------------------------------
# 4.2.1. Fixed Points: Find Fixed Points, Jacobians, and Unstable Mode Frequencies
# -------------------------------

# Execute fixed point search using unified function
all_fixed_points, all_jacobians, all_unstable_eig_freq = find_and_analyze_points(point_type="fixed")




# -------------------------------
# 4.2.2. Slow Points: Find Slow Points, Jacobians, and Unstable Mode Frequencies
# -------------------------------

# Execute slow point search using unified function
all_slow_points, all_slow_jacobians, all_slow_unstable_eig_freq = find_and_analyze_points(point_type="slow")




# -------------------------------
# 4.3.0. Unified Point Summary Function
# -------------------------------

def generate_point_summaries(point_type="fixed", all_points=None, all_unstable_eig_freq=None):
    """
    Unified function to generate summaries for either fixed points or slow points.
    
    Arguments:
        point_type              : str, either "fixed" or "slow" to specify which type of points
        all_points              : list of lists of points, one list per task
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies: outer list is over tasks,
                                middle list is over fixed points, and inner list contains unstable eigenvalues for each fixed point
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point names based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    point_name_lower = "fixed points" if point_type == "fixed" else "slow points"
    point_name_singular = "fixed point" if point_type == "fixed" else "slow point"
    
    print(f"\nSummary of {point_name} Found:\n")
    points_count = {}
    unstable_eigs_count = {}

    print(f"1. {point_name} per Task:")
    for j in range(num_tasks):
        # Count the number of points found for this task
        num_points = len(all_points[j])
        print(f"Task {j}: {num_points} {point_name_lower} found")
        # Update the count of tasks with this number of points
        points_count[num_points] = points_count.get(num_points, 0) + 1
        
        for point_freqs in all_unstable_eig_freq[j]:
            # Count the number of unstable eigenvalues for this point
            num_unstable = len(point_freqs)
            # Update the count of points with this number of unstable eigenvalues
            unstable_eigs_count[num_unstable] = unstable_eigs_count.get(num_unstable, 0) + 1

    print(f"\n2. Tasks Grouped by Number of {point_name}:")
    for num_points in sorted(points_count.keys()):
        count = points_count[num_points]
        print(f"  {count} tasks have {num_points} {point_name_singular}{'s' if num_points != 1 else ''}")

    print(f"\n3. {point_name} Grouped by Number of Unstable Eigenvalues:")
    for num_unstable in sorted(unstable_eigs_count.keys()):
        count = unstable_eigs_count[num_unstable]
        print(f"  {count} {point_name_singular}{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")

    print("\nDetailed Information for Selected Tasks:")
    for j in TEST_INDICES:
        print(f"\nTask {j}:")
        print(f"Number of distinct {point_name_lower} found: {len(all_points[j])}")
        for i, (point, freqs) in enumerate(zip(all_points[j], all_unstable_eig_freq[j])):
            print(f"  {point_name_singular.capitalize()} {i+1}:")
            print(f"    Number of unstable eigenvalues: {len(freqs)}")
            if len(freqs) > 0:
                print("    Unstable eigenvalues:")
                for ev in freqs:
                    print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
            print(f"    Norm: {np.linalg.norm(point):.4f}")




# -------------------------------
# 4.3.1. Fixed Points: Fixed Point Summaries
# -------------------------------

# Execute fixed point summary using unified function
generate_point_summaries(point_type="fixed", all_points=all_fixed_points, all_unstable_eig_freq=all_unstable_eig_freq)




# -------------------------------
# 4.3.2. Slow Points: Slow Point Summaries
# -------------------------------

# Execute slow point summary using unified function
generate_point_summaries(point_type="slow", all_points=all_slow_points, all_unstable_eig_freq=all_slow_unstable_eig_freq)




# -------------------------------
# 4.4.0. Unified Jacobian Analysis Function: Jacobian Plots and Equal Jacobians Check
# -------------------------------

def analyze_jacobians(point_type="fixed", all_jacobians=None):
    """
    Unified function to plot Jacobian matrices and check for equal Jacobians for either fixed points or slow points.
    
    Arguments:
        point_type      : str, either "fixed" or "slow" to specify which type of points
        all_jacobians   : list of lists of Jacobians, one list per task
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    print(f"\n{point_name}: Plotting Jacobian matrices for selected tasks...")
    for j in TEST_INDICES:
        omega_j = omegas[j]
        if all_jacobians[j]:    # Check if there are any Jacobians for this task
            plot_jacobian_matrices(all_jacobians[j], j)

    print(f"\n{point_name}: Checking for Equal Jacobians within Tasks:")
    for j in range(num_tasks):
        task_jacobians = all_jacobians[j]
        num_jacobians = len(task_jacobians)
        if num_jacobians > 1:   # Only check this if there are multiple Jacobians for this task
            # Track which Jacobians have been matched
            matched = [False] * num_jacobians
            equal_groups = []
            for i in range(num_jacobians):
                if not matched[i]:  # Skip if this Jacobian has already been matched
                    current_group = [i]
                    matched[i] = True
                    for k in range(i + 1, num_jacobians):
                        if not matched[k]:
                            # Check if the Jacobians are equal within the tolerance
                            if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
                                current_group.append(k)
                                matched[k] = True
                    if len(current_group) > 1:  # Only add groups with more than one equal Jacobian
                        equal_groups.append(current_group)
            if equal_groups:
                print(f"\nTask {j}:")
                print(f"  Total number of Jacobians: {num_jacobians}")
                for group_idx, group in enumerate(equal_groups):
                    print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")




# -------------------------------
# 4.4.1. Fixed Points: Jacobian Plots and Equal Jacobians Check
# -------------------------------

# Execute fixed point Jacobian analysis using unified function
analyze_jacobians(point_type="fixed", all_jacobians=all_jacobians)




# -------------------------------
# 4.4.2. Slow Points: Jacobian Plots and Equal Jacobians Check
# -------------------------------

# Execute slow point Jacobian analysis using unified function
analyze_jacobians(point_type="slow", all_jacobians=all_slow_jacobians)




# -------------------------------
# 4.5.0. Unified Unstable Mode Frequency Analysis Function
# -------------------------------

def analyze_unstable_frequencies(point_type="fixed", all_unstable_eig_freq=None):
    """
    Unified function to plot unstable eigenvalues and frequency comparisons for either fixed points or slow points.
    
    Arguments:
        point_type              : str, either "fixed" or "slow" to specify which type of points
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies: outer list is over tasks,
                                middle list is over fixed points, and inner list contains unstable eigenvalues for each fixed point
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    print(f"\n{point_name}: Plotting unstable eigenvalues for selected tasks...")
    for j in TEST_INDICES:
        omega_j = omegas[j]
        if all_unstable_eig_freq[j]:
            plot_unstable_eigenvalues(all_unstable_eig_freq[j], j)

    print(f"\n{point_name}: Plotting comparisons between target frequencies and unstable mode frequencies...")
    plot_frequency_comparison(all_unstable_eig_freq, omegas)




# -------------------------------
# 4.5.1. Fixed Points: Unstable Mode Frequency Summaries and Plots
# -------------------------------

analyze_unstable_frequencies(point_type="fixed", all_unstable_eig_freq=all_unstable_eig_freq)




# -------------------------------
# 4.5.2. Slow Points: Unstable Mode Frequency Summaries and Plots
# -------------------------------

analyze_unstable_frequencies(point_type="slow", all_unstable_eig_freq=all_slow_unstable_eig_freq)




# -------------------------------
# -------------------------------
# 5. PCA RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 5.1.1. Fixed Points: Perform PCA on Trajectories and Fixed Points
# -------------------------------

# Track PCA computation time
start_time = time.time()
print("\nFixed Points: Starting PCA computation...")

# Concatenate all state trajectories from all tasks
all_states = np.concatenate(state_traj_states, axis=0)  # shape is (num_tasks * (num_steps_train+1), N)

# Perform PCA with 3 components
pca = PCA(n_components=3)
proj_all = pca.fit_transform(all_states)

# Project each trajectory into the PCA space
proj_trajs = []
start = 0
for traj in tqdm(state_traj_states, desc="Projecting trajectories"):
    T = traj.shape[0]   # Number of time steps in the trajectory
    proj_traj = proj_all[start:start + T]   # Extracts the projected trajectory from the PCA space, resulting shape is (T, 3)
    proj_trajs.append(proj_traj)
    start += T

# Project all fixed points from all tasks into the PCA space
all_fixed_points_flat = []
for task_fps in all_fixed_points:
    all_fixed_points_flat.extend(task_fps)
if all_fixed_points_flat:   # Check if there are any fixed points
    all_fixed_points_flat = np.array(all_fixed_points_flat)
    if all_fixed_points_flat.ndim == 1: # If it's a 1D array, reshape it to 2D
        all_fixed_points_flat = all_fixed_points_flat.reshape(1, -1)
    proj_fixed = pca.transform(all_fixed_points_flat)   # Projects all fixed points from all tasks into the PCA space, resulting shape is (num_total_fixed_points, 3)
else:
    proj_fixed = np.empty((0, 3))   # Empty array with correct shape for plotting
# Note that np.array(all_fixed_points_flat) has shape (num_total_fixed_points, N)

# Calculate PCA computation time
pca_time = time.time() - start_time
print(f"\nPCA computation completed in {pca_time:.2f} seconds")




# -------------------------------
# 5.1.2. Slow Points: Perform PCA on Trajectories and Slow Points
# -------------------------------

# Track PCA computation time
start_time_slow = time.time()
print("\nSlow Points: Starting PCA computation...")

# Project all slow points from all tasks into the PCA space
all_slow_points_flat = []
for task_sps in all_slow_points:
    all_slow_points_flat.extend(task_sps)
if all_slow_points_flat:    # Check if there are any slow points
    all_slow_points_flat = np.array(all_slow_points_flat)
    if all_slow_points_flat.ndim == 1:  # If it's a 1D array, reshape it to 2D
        all_slow_points_flat = all_slow_points_flat.reshape(1, -1)
    proj_slow = pca.transform(all_slow_points_flat) # Projects all slow points from all tasks into the PCA space, resulting shape is (num_total_slow_points, 3)
else:
    proj_slow = np.empty((0, 3))    # Empty array with correct shape for plotting
# Note that np.array(all_slow_points_flat) has shape (num_total_slow_points, N)

# Calculate PCA computation time
pca_time_slow = time.time() - start_time_slow
print(f"\nPCA computation completed in {pca_time_slow:.2f} seconds")




# -------------------------------
# 5.2.0. Unified PCA Plotting Function
# -------------------------------

def plot_pca_trajectories_and_points(point_type="fixed", all_points=None, proj_points=None):
    """
    Unified function to plot PCA trajectories with either fixed points or slow points.
    
    Arguments:
        point_type   : str, either "fixed" or "slow" to specify which type of points to plot
        all_points   : list of lists of points, one list per task
        proj_points  : projected points in PCA space, shape (num_total_points, 3)
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    J_trained = np.array(params["J"])

    # Create the figure and axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the projected trajectories in blue
    for traj in proj_trajs:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.5)

    # Plot the projected points as green scatter points
    if proj_points.size > 0:
        ax.scatter(proj_points[:, 0], proj_points[:, 1], proj_points[:, 2], color='green', s=50, label=point_name)

    # For each point, plot all unstable modes as red lines
    point_idx = 0
    for j, task_points in enumerate(all_points):
        for x_star in task_points:
            # Compute the Jacobian and find the eigenvalues at the point
            J_eff = np.array(compute_jacobian(x_star, J_trained))
            eigenvals, eigenvecs = eig(J_eff)
            # Find all the unstable eigenvalues for the given point
            unstable_idx = np.where(np.real(eigenvals) > 0)[0]
            if len(unstable_idx) > 0:
                # For each unstable eigenvalue, plot the corresponding eigenvector direction
                for idx in unstable_idx:
                    # Take the real part for the plotting direction
                    v = np.real(eigenvecs[:, idx])
                    scale = 0.5
                    # Project the unstable eigenvector direction into PCA space
                    v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_points[point_idx]
                    # Plot a line centred on the point in PCA space
                    line = np.vstack([
                        proj_points[point_idx] - v_proj,
                        proj_points[point_idx] + v_proj
                    ])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red', linewidth=2, alpha=0.5)
            point_idx += 1

    # Add decorations
    ax.set_title(f'PCA of Network Trajectories and {point_name}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()

    # Save the figure
    figure_name = f"pca_plot_{point_type}"
    save_figure(fig, figure_name, N, num_tasks, dt, T_drive, T_train)

    plt.show()
    plt.close()




# -------------------------------
# 5.2.1. Fixed Points: PCA Trajectory and Fixed Points Plot
# -------------------------------

# Use the unified function for fixed points
plot_pca_trajectories_and_points(
    point_type="fixed",
    all_points=all_fixed_points,
    proj_points=proj_fixed,
)




# -------------------------------
# 5.2.2. Slow Points: PCA Trajectory and Slow Points Plot
# -------------------------------

# Use the unified function for slow points
plot_pca_trajectories_and_points(
    point_type="slow",
    all_points=all_slow_points,
    proj_points=proj_slow,
)




# -------------------------------
# 5.3. Save Results from PCA
# -------------------------------

print("\nSaving PCA results...")
pca_results = {
    "proj_trajs": proj_trajs,
    "proj_fixed": proj_fixed,
    "proj_slow": proj_slow
}
save_variable(pca_results, "pca_results", N, num_tasks, dt, T_drive, T_train)