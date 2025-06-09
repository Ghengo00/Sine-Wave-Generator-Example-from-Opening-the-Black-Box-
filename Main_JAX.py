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


# Time parameters (in seconds)
dt = 0.02        # integration time step
T_drive = 4.0    # driving phase duration
T_train = 32.0   # training phase duration
num_steps_drive = int(T_drive / dt)
num_steps_train = int(T_train / dt)
time_drive = jnp.arange(0, T_drive, dt, dtype=jnp.float32)
time_train = jnp.arange(0, T_train, dt, dtype=jnp.float32)
time_full = np.concatenate([time_drive, T_drive + time_train])


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




# -------------------------------
# 1.2. Define RNN Weights
# -------------------------------

# Initialize parameters with scaling ~1/sqrt(N)
def init_params(key):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    J = random.normal(k1, (N, N)) / jnp.sqrt(N)
    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)
    return {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}

params = init_params(subkey)




# -------------------------------
# 1.3. Define Useful Global Variables
# -------------------------------

BATCH_SIZE = 10


# Optimization parameters
NUM_EPOCHS_ADAM = 500
NUM_EPOCHS_LBFGS = 200
CHUNK_SIZE_ADAM = 50
CHUNK_SIZE_LBFGS = 25
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
MAXITER = 5000

# Slow point search parameters
NUM_ATTEMPTS_SLOW = 50
TOL_SLOW = 1e-2
MAXITER_SLOW = 5000


# Jacobian and slow point analysis parameters
JACOBIAN_TOL = 1e-2
SEARCH_BATCH_SIZE = 5




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




# -------------------------------
# -------------------------------
# 2. RNN TRAINING
# -------------------------------
# -------------------------------

# -------------------------------
# 2.1. Trajectory Simulation Function
# -------------------------------

@jit
def rnn_step(x, inputs, params):
    """
    Single time-step update of the RNN state.
    
    Arguments:
    x     : current state (at time t), dimensions (N,)
    u_t   : scalar input at this time-step (time t), dimensions (I,)
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
            u_t   : scalar input at this time-step (time t), dimensions (I,)
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
        x_drive_final: the final state of the system during training, for the given parameters (jnp.ndarray, shape (N,))
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

@jit
def batched_loss(params, drive_u, train_u, targets):
    """
    Compute the loss for a batch of tasks.

    Arguments:
        params   : dictionary of RNN weights
        drive_u  : list of lists of lists, where each inner list contains the input for the given task for each drive step (num_tasks, num_steps_drive, I)
        train_u  : list of lists of lists, where each inner list contains the input for the given task for each train step (num_tasks, num_steps_train, I)
        targets  : list of lists, where each inner list contains the target for the given task for each train step (num_tasks, num_steps_train)
    
    Returns:
        loss     : scalar loss value averaged over all tasks
    """
    def one_task(d_u, t_u, tgt):
        """
        Compute the loss for a single task.
        Arguments:
            d_u  : driving phase input for the task, shape (num_steps_drive, I)
            t_u  : training phase input for the task, shape (num_steps_train, I)
            tgt  : target output for the task during training, shape (num_steps_train,)
        
        Returns:
            loss : scalar loss value for this task
        """
        # Drive phase
        xs_d, _   = simulate_trajectory(jnp.zeros(N), d_u, params)
        x_final   = xs_d[-1]
        
        # Train phase
        _, zs_tr  = simulate_trajectory(x_final, t_u, params)
        
        return jnp.mean((zs_tr - tgt) ** 2)


    losses = jax.vmap(one_task)(drive_u, train_u, targets)

    return jnp.mean(losses)




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


@jax.jit
def adam_step(params, opt_state, drive_u, train_u, targets):
    # (1) Evaluate current loss and gradient
    loss, grads        = value_and_grad_fn(params, drive_u, train_u, targets)

    # (2) One optimizer update
    updates, new_state = adam_opt.update(grads, opt_state, params)

    # (3) Apply updates to your parameters
    new_params         = optax.apply_updates(params, updates)

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


@jax.jit
def lbfgs_step(params, opt_state, drive_u, train_u, targets):
    # (1) Evaluate current loss and gradient
    loss, grads        = value_and_grad_fn(params, drive_u, train_u, targets)

    # (2) One optimizer update
    updates, new_state = lbfgs_opt.update(
        grads, opt_state, params,
        value = loss, grad = grads,
        value_fn = lambda p: batched_loss(p, drive_u, train_u, targets)
    )

    # (3) Apply updates to your parameters
    new_params         = optax.apply_updates(params, updates)

    return new_params, new_state, loss




# -------------------------------
# 2.4. Training Procedure
# -------------------------------

loss_history_total = []
best_params_total = None


def scan_with_history(params, opt_state, step_fn, steps, drive_u, train_u, targets, tag):
    """
    Function to run multiple steps of the chosen optimization method via lax.scan.

    Arguments:
        params   : current parameters of the model
        opt_state: current state of the optimizer
        step_fn  : function to perform a single optimization step (adam_step or lbfgs_step)
        steps    : number of steps to run
        drive_u  : driving phase inputs for all tasks (num_tasks, num_steps_drive, I)
        train_u  : training phase inputs for all tasks (num_tasks, num_steps_train, I)
        targets  : training targets for all tasks (num_tasks, num_steps_train)

    Returns:
        final_params : parameters after all steps
        final_state  : optimizer state after all steps
        losses       : array of loss values for each step (length = steps)
    """
    def body(carry, _):
        """
        Single step function to pass to lax.scan - runs one optimization step.

        Arguments:
            carry : tuple (params, opt_state, loss)
            _     : unused placeholder for lax.scan
        
        Returns:
            carry : updated tuple (params, opt_state, loss)
            loss  : loss value for this step
        """
        p, s, _, i = carry
        p, s, l = step_fn(p, s, drive_u, train_u, targets)

        # This will emit at host-side for each iteration
        jax.debug.print("[{tag}] step {i}: loss {loss:.6e}", 
                        tag=tag, i=i+1, loss=l)
        
        return (p, s, l, i+1), (p, l)


    # Run lax.scan to perform multiple optimization steps
    (p_final, s_final, _, _), (param_hist, loss_hist) = lax.scan(
        body,
        (params, opt_state, 0.0, 0),
        None,                       # No input to the scan function, we just need to iterate `steps` times
        length = steps
    )

    return param_hist, loss_hist




# -------------------------------
# 2.4.1. Execute Adam Phase of Training Procedure
# -------------------------------

param_hist_adam, losses_adam = scan_with_history(
    params, adam_state, adam_step,
    NUM_EPOCHS_ADAM,
    drive_inputs, train_inputs, train_targets, tag = "ADAM"
)

losses_adam = np.array(losses_adam)                # shape (NUM_EPOCHS_ADAM,)
# Extract the index of the best loss from history
i_best_adam = int(losses_adam.argmin())
# Extract best loss from history
best_loss_adam = float(losses_adam[i_best_adam])
# Extract best paramater values from history
best_params_adam = jax.tree_util.tree_map(lambda x: x[i_best_adam], param_hist_adam)

print(f"Lowest-Loss Adam Iteration: {i_best_adam+1}/{NUM_EPOCHS_ADAM} → Loss {best_loss_adam:.6e}")

# Restore the best parameters
params = best_params_adam




# -------------------------------
# 2.4.2. Execute L-BFGS Phase of Training Procedure
# -------------------------------

if best_loss_adam > LOSS_THRESHOLD:
    # Run L-BFGS optimization phase
    param_hist_lbfgs, losses_lbfgs = scan_with_history(
        params, lbfgs_state, lbfgs_step,
        NUM_EPOCHS_LBFGS,
        drive_inputs, train_inputs, train_targets, tag = "L-BFGS"
    )

    losses_lbfgs = np.array(losses_lbfgs)         # shape (NUM_EPOCHS_LBFGS,)
    # Extract the index of the best loss from history
    i_best_lbfgs = int(losses_lbfgs.argmin())
    # Extract best loss from history
    best_loss_lbfgs = float(losses_lbfgs[i_best_lbfgs])
    # Extract best parameter values from history
    best_params_lbfgs = jax.tree_util.tree_map(lambda x: x[i_best_lbfgs], param_hist_lbfgs)

    print(f"Lowest-Loss L-BFGS Iteration: {i_best_lbfgs+1}/{NUM_EPOCHS_LBFGS} → loss {best_loss_lbfgs:.6e}")

    if best_loss_lbfgs < best_loss_adam:
        params = best_params_lbfgs
        best_loss_total = best_loss_lbfgs
        best_params_total = best_params_lbfgs
    else:
        params = best_params_adam
        best_loss_total = best_loss_adam
        best_params_total = best_params_adam
else:
    best_loss_total = best_loss_adam
    best_params_total = best_params_adam

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




# # -------------------------------
# # -------------------------------
# # 4. FIXED POINTS, SLOW POINTS, AND UNSTABLE MODES RESULTS
# # -------------------------------
# # -------------------------------

# # -------------------------------
# # 4.1. Jacobians, Fixed Points, Slow Points, and Unstable Mode Frequencies Functions
# # -------------------------------

# def fixed_point_func(x_np, u_val, J_np, B_np, b_x_np):
#     return -x_np + J_np @ jnp.tanh(x_np) + (B_np * jnp.array([u_val])).sum(axis=1) + b_x_np

# def slow_point_func(x_np, u_val, J_np, B_np, b_x_np):
#     F_x = -x_np + J_np @ jnp.tanh(x_np) + (B_np * jnp.array([u_val])).sum(axis=1) + b_x_np
#     diag_term = 1 - jnp.tanh(x_np) ** 2
#     J_F = -jnp.eye(len(x_np)) + J_np * diag_term[jnp.newaxis, :]
#     grad_q = F_x @ J_F
#     return jnp.array(grad_q)

# def jacobian_fixed_point(x_star, J_np):
#     diag_term = 1 - jnp.tanh(x_star) ** 2
#     return -jnp.eye(len(x_star)) + J_np * diag_term[jnp.newaxis, :]

# def find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS, tol=TOL):
#     fixed_points = []
#     try:
#         sol = root(lambda x: np.array(fixed_point_func(x, u_const, J_trained, B_trained, b_x_trained)),
#                    x0_guess, method='lm', options={'maxiter': MAXITER})
#         if sol.success:
#             fixed_points.append(sol.x)
#     except Exception as e:
#         print(f"Fixed point search failed for initial guess: {e}")

#     for attempt in range(num_attempts - 1):
#         x0_perturbed = x0_guess + np.random.normal(0, 0.5, size=x0_guess.shape)
#         try:
#             sol = root(lambda x: np.array(fixed_point_func(x, u_const, J_trained, B_trained, b_x_trained)),
#                        x0_perturbed, method='lm', options={'maxiter': MAXITER})
#             if sol.success:
#                 is_distinct = True
#                 for fp in fixed_points:
#                     if np.linalg.norm(sol.x - fp) < tol:
#                         is_distinct = False
#                         break
#                 if is_distinct:
#                     fixed_points.append(sol.x)
#         except Exception as e:
#             print(f"Fixed point search failed for attempt {attempt+1}: {e}")
#     return fixed_points

# def find_slow_points_gn(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS_SLOW, tol=TOL_SLOW, alt_inits=None):
#     def _try(initial_x):
#         try:
#             res = least_squares(lambda x: np.array(slow_point_func(x, u_const, J_trained, B_trained, b_x_trained)),
#                                 initial_x, method='lm', max_nfev=MAXITER_SLOW)
#             if res.success and np.linalg.norm(res.fun) < tol:
#                 return res.x
#         except Exception as e:
#             print(f"Gauss-Newton slow-point search failed: {e}")
#         return None

#     slow_points = []
#     fp = _try(x0_guess)
#     if fp is not None:
#         slow_points.append(fp)

#     for _ in range(num_attempts - 1):
#         cand = x0_guess + np.random.normal(0.0, 0.5, size=x0_guess.shape)
#         fp = _try(cand)
#         if fp is not None and all(np.linalg.norm(fp - sp) >= tol for sp in slow_points):
#             slow_points.append(fp)

#     if alt_inits is not None:
#         for cand in alt_inits:
#             fp = _try(cand)
#             if fp is not None and all(np.linalg.norm(fp - sp) >= tol for sp in slow_points):
#                 slow_points.append(fp)
#     return slow_points

# def analyze_fixed_points(fixed_points, J_trained):
#     jacobians = []
#     unstable_freqs = []
#     for x_star in fixed_points:
#         J_eff = np.array(jacobian_fixed_point(jnp.array(x_star), J_trained))
#         jacobians.append(J_eff)
#         eigenvals, _ = eig(J_eff)
#         unstable_idx = np.where(np.real(eigenvals) > 0)[0]
#         if len(unstable_idx) > 0:
#             unstable_freqs.append(list(eigenvals[unstable_idx]))
#         else:
#             unstable_freqs.append([])
#     return jacobians, unstable_freqs

# def plot_unstable_eigenvalues(unstable_freqs, j, omega, save_dir=None):
#     plt.figure(figsize=(10, 8))
#     for i, freqs in enumerate(unstable_freqs):
#         if freqs:
#             freqs_array = np.array(freqs)
#             plt.scatter(np.real(freqs_array), np.imag(freqs_array),
#                         color=COLORS[i], marker=MARKERS[i], label=f'FP{i+1}')
#     plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
#     plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.title(f'Unstable Eigenvalues for Task {j} (ω={omega:.3f})')
#     plt.legend()
#     plt.grid(True)
#     if save_dir is not None:
#         filename = generate_filename(f"unstable_eigenvalues_task_{j}", N, num_tasks, dt, T_drive, T_train)
#         filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
#         plt.savefig(filepath)
#         print(f"Saved unstable eigenvalues plot for task {j} to {filepath}")
#     plt.show()
#     plt.close()

# def plot_jacobian_matrices(jacobians, j, omega, save_dir=None):
#     fig, axes = plt.subplots(len(jacobians), 1, figsize=(10, 5 * len(jacobians)))
#     if len(jacobians) == 1:
#         axes = [axes]
#     for i, J_eff in enumerate(jacobians):
#         im = axes[i].imshow(J_eff, cmap='viridis')
#         axes[i].set_title(f'Jacobian Matrix (Task {j}, ω={omega:.3f}, FP{i+1})')
#         plt.colorbar(im, ax=axes[i])
#     plt.tight_layout()
#     if save_dir is not None:
#         filename = generate_filename(f"jacobian_matrices_task_{j}", N, num_tasks, dt, T_drive, T_train)
#         filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
#         plt.savefig(filepath)
#         print(f"Saved Jacobian matrices plot for task {j} to {filepath}")
#     plt.show()
#     plt.close()

# def plot_frequency_comparison(all_unstable_eig_freq, omegas, save_dir=None):
#     plt.figure(figsize=(12, 6))
#     task_numbers = []
#     target_frequencies = []
#     unstable_frequencies = []
#     for j, task_freqs in enumerate(all_unstable_eig_freq):
#         target_freq = omegas[j]
#         max_imag = 0
#         for fp_freqs in task_freqs:
#             if fp_freqs:
#                 current_max = max(abs(np.imag(ev)) for ev in fp_freqs)
#                 max_imag = max(max_imag, current_max)
#         if max_imag > 0:
#             task_numbers.append(j)
#             target_frequencies.append(target_freq)
#             unstable_frequencies.append(max_imag)
#     plt.scatter(task_numbers, target_frequencies, color='black', label='Target Frequency', s=100)
#     plt.scatter(task_numbers, unstable_frequencies, color='red', label='Max Unstable Mode Frequency', s=100)
#     plt.xlabel('Task #')
#     plt.ylabel('Frequency (radians)')
#     plt.title('Comparison of Target Frequencies and Maximum Unstable Mode Frequencies')
#     plt.legend()
#     plt.grid(True)
#     if save_dir is not None:
#         filename = generate_filename("frequency_comparison", N, num_tasks, dt, T_drive, T_train)
#         filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
#         plt.savefig(filepath)
#         print(f"Saved frequency comparison plot to {filepath}")
#     plt.show()
#     plt.close()

# # -------------------------------
# # 4.2.1. Fixed Points: Find Jacobians, Fixed Points, and Unstable Mode Frequencies
# # -------------------------------

# print("\nStarting fixed point search...")
# J_trained = np.array(params["J"])
# B_trained = np.array(params["B"])
# b_x_trained = np.array(params["b_x"])

# all_fixed_points = []
# all_jacobians = []
# all_unstable_eig_freq = []

# start_time = time.time()
# for batch_start in range(0, num_tasks, BATCH_SIZE):
#     batch_end = min(batch_start + BATCH_SIZE, num_tasks)
#     for j in range(batch_start, batch_end):
#         u_const = static_inputs[j]
#         x0_guess = state_fixed_point_inits[j]

#         # Build fallback inits from trajectories
#         idxs = np.random.choice(len(state_traj_states),
#                                 size=min(num_tasks, NUM_ATTEMPTS), replace=False)
#         fallback_inits = []
#         for idx in idxs:
#             traj = state_traj_states[idx]
#             row = np.random.randint(0, traj.shape[0])
#             fallback_inits.append(traj[row])

#         task_fixed_points = find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS, tol=TOL)
#         all_fixed_points.append(task_fixed_points)

#         task_jacobians, task_unstable_freqs = analyze_fixed_points(task_fixed_points, J_trained)
#         all_jacobians.append(task_jacobians)
#         all_unstable_eig_freq.append(task_unstable_freqs)

# fixed_point_time = time.time() - start_time
# print(f"Fixed point search completed in {fixed_point_time:.2f} seconds")

# print("\nSaving fixed point analysis results...")
# save_variable(all_fixed_points, "all_fixed_points", N, num_tasks, dt, T_drive, T_train)
# save_variable(all_jacobians, "all_jacobians", N, num_tasks, dt, T_drive, T_train)
# save_variable(all_unstable_eig_freq, "all_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train)

# # -------------------------------
# # 4.2.2. Slow Points: Find Jacobians, Slow Points, and Unstable Mode Frequencies
# # -------------------------------

# print("\nStarting slow point search...")
# all_slow_points = []
# all_slow_jacobians = []
# all_slow_unstable_eig_freq = []

# start_time = time.time()
# for batch_start in range(0, num_tasks, BATCH_SIZE):
#     batch_end = min(batch_start + BATCH_SIZE, num_tasks)
#     for j in range(batch_start, batch_end):
#         u_const = static_inputs[j]
#         x0_guess = state_fixed_point_inits[j]

#         idxs = np.random.choice(len(state_traj_states),
#                                 size=min(num_tasks, NUM_ATTEMPTS_SLOW), replace=False)
#         fallback_inits = []
#         for idx in idxs:
#             traj = state_traj_states[idx]
#             row = np.random.randint(0, traj.shape[0])
#             fallback_inits.append(traj[row])

#         task_slow_points = find_slow_points_gn(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS_SLOW, tol=TOL_SLOW, alt_inits=fallback_inits)
#         all_slow_points.append(task_slow_points)

#         task_jacobians, task_unstable_freqs = analyze_fixed_points(task_slow_points, J_trained)
#         all_slow_jacobians.append(task_jacobians)
#         all_slow_unstable_eig_freq.append(task_unstable_freqs)

# slow_point_time = time.time() - start_time
# print(f"Slow point search completed in {slow_point_time:.2f} seconds")

# print("\nSaving slow point analysis results...")
# save_variable(all_slow_points, "all_slow_points", N, num_tasks, dt, T_drive, T_train)
# save_variable(all_slow_jacobians, "all_slow_jacobians", N, num_tasks, dt, T_drive, T_train)
# save_variable(all_slow_unstable_eig_freq, "all_slow_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train)

# # -------------------------------
# # 4.3.1. Fixed Points: Fixed Point Summaries
# # -------------------------------

# print("\nSummary of Fixed Points Found:\n")
# fixed_points_count = {}
# unstable_eigs_count = {}

# print("1. Fixed Points per Task:")
# for j in range(num_tasks):
#     num_fps = len(all_fixed_points[j])
#     print(f"Task {j}: {num_fps} fixed points found")
#     fixed_points_count[num_fps] = fixed_points_count.get(num_fps, 0) + 1
#     for fp_freqs in all_unstable_eig_freq[j]:
#         num_unstable = len(fp_freqs)
#         unstable_eigs_count[num_unstable] = unstable_eigs_count.get(num_unstable, 0) + 1

# print("\n2. Tasks Grouped by Number of Fixed Points:")
# for num_fps in sorted(fixed_points_count.keys()):
#     count = fixed_points_count[num_fps]
#     print(f"  {count} tasks have {num_fps} fixed point{'s' if num_fps != 1 else ''}")

# print("\n3. Fixed Points Grouped by Number of Unstable Eigenvalues:")
# for num_unstable in sorted(unstable_eigs_count.keys()):
#     count = unstable_eigs_count[num_unstable]
#     print(f"  {count} fixed point{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")

# print("\nDetailed Information for Selected Tasks:")
# for j in TEST_INDICES:
#     print(f"\nTask {j}:")
#     print(f"Number of distinct fixed points found: {len(all_fixed_points[j])}")
#     for i, (fp, freqs) in enumerate(zip(all_fixed_points[j], all_unstable_eig_freq[j])):
#         print(f"  Fixed point {i+1}:")
#         print(f"    Number of unstable eigenvalues: {len(freqs)}")
#         if len(freqs) > 0:
#             print("    Unstable eigenvalues:")
#             for ev in freqs:
#                 print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
#         print(f"    Norm: {np.linalg.norm(fp):.4f}")

# # -------------------------------
# # 4.3.2. Slow Points: Slow Point Summaries
# # -------------------------------

# print("\nSummary of Slow Points Found:\n")
# slow_points_count = {}
# slow_unstable_eigs_count = {}

# print("1. Slow Points per Task:")
# for j in range(num_tasks):
#     num_sps = len(all_slow_points[j])
#     print(f"Task {j}: {num_sps} slow points found")
#     slow_points_count[num_sps] = slow_points_count.get(num_sps, 0) + 1
#     for sp_freqs in all_slow_unstable_eig_freq[j]:
#         num_unstable = len(sp_freqs)
#         slow_unstable_eigs_count[num_unstable] = slow_unstable_eigs_count.get(num_unstable, 0) + 1

# print("\n2. Tasks Grouped by Number of Slow Points:")
# for num_sps in sorted(slow_points_count.keys()):
#     count = slow_points_count[num_sps]
#     print(f"  {count} tasks have {num_sps} slow point{'s' if num_sps != 1 else ''}")

# print("\n3. Slow Points Grouped by Number of Unstable Eigenvalues:")
# for num_unstable in sorted(slow_unstable_eigs_count.keys()):
#     count = slow_unstable_eigs_count[num_unstable]
#     print(f"  {count} slow point{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")

# print("\nDetailed Information for Selected Tasks:")
# for j in TEST_INDICES:
#     print(f"\nTask {j}:")
#     print(f"Number of distinct slow points found: {len(all_slow_points[j])}")
#     for i, (sp, freqs) in enumerate(zip(all_slow_points[j], all_slow_unstable_eig_freq[j])):
#         print(f"  Slow point {i+1}:")
#         print(f"    Number of unstable eigenvalues: {len(freqs)}")
#         if len(freqs) > 0:
#             print("    Unstable eigenvalues:")
#             for ev in freqs:
#                 print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
#         print(f"    Norm: {np.linalg.norm(sp):.4f}")

# # -------------------------------
# # 4.4.1. Fixed Points: Jacobian Plots and Equal Jacobians Check
# # -------------------------------

# print("\nFixed Points: Plotting Jacobian matrices for selected tasks...")
# for j in TEST_INDICES:
#     omega_j = omegas[j]
#     if all_jacobians[j]:
#         plot_jacobian_matrices(all_jacobians[j], j, omega_j, save_dir=output_dir)

# print("\nFixed Points: Checking for Equal Jacobians within Tasks:")
# for j in range(num_tasks):
#     task_jacobians = all_jacobians[j]
#     num_jacobians = len(task_jacobians)
#     if num_jacobians > 1:
#         matched = [False] * num_jacobians
#         equal_groups = []
#         for i in range(num_jacobians):
#             if not matched[i]:
#                 current_group = [i]
#                 matched[i] = True
#                 for k in range(i + 1, num_jacobians):
#                     if not matched[k]:
#                         if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
#                             current_group.append(k)
#                             matched[k] = True
#                 if len(current_group) > 1:
#                     equal_groups.append(current_group)
#         if equal_groups:
#             print(f"\nTask {j}:")
#             print(f"  Total number of Jacobians: {num_jacobians}")
#             for group_idx, group in enumerate(equal_groups):
#                 print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")

# # -------------------------------
# # 4.4.2. Slow Points: Jacobian Plots and Equal Jacobians Check
# # -------------------------------

# print("\nSlow Points: Plotting Jacobian matrices for selected tasks...")
# for j in TEST_INDICES:
#     omega_j = omegas[j]
#     if all_slow_jacobians[j]:
#         plot_jacobian_matrices(all_slow_jacobians[j], j, omega_j, save_dir=output_dir)

# print("\nSlow Points: Checking for Equal Jacobians within Tasks:")
# for j in range(num_tasks):
#     task_jacobians = all_slow_jacobians[j]
#     num_jacobians = len(task_jacobians)
#     if num_jacobians > 1:
#         matched = [False] * num_jacobians
#         equal_groups = []
#         for i in range(num_jacobians):
#             if not matched[i]:
#                 current_group = [i]
#                 matched[i] = True
#                 for k in range(i + 1, num_jacobians):
#                     if not matched[k]:
#                         if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
#                             current_group.append(k)
#                             matched[k] = True
#                 if len(current_group) > 1:
#                     equal_groups.append(current_group)
#         if equal_groups:
#             print(f"\nTask {j}:")
#             print(f"  Total number of Jacobians: {num_jacobians}")
#             for group_idx, group in enumerate(equal_groups):
#                 print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")

# # -------------------------------
# # 4.5.1. Fixed Points: Unstable Mode Frequency Summaries and Plots
# # -------------------------------

# print("\nFixed Points: Plotting unstable eigenvalues for selected tasks...")
# for j in TEST_INDICES:
#     omega_j = omegas[j]
#     if all_unstable_eig_freq[j]:
#         plot_unstable_eigenvalues(all_unstable_eig_freq[j], j, omega_j, save_dir=output_dir)

# print("\nFixed Points: Plotting comparisons between target frequencies and unstable mode frequencies...")
# plot_frequency_comparison(all_unstable_eig_freq, omegas, save_dir=output_dir)

# # -------------------------------
# # 4.5.2. Slow Points: Unstable Mode Frequency Summaries and Plots
# # -------------------------------

# print("\nSlow Points: Plotting unstable eigenvalues for selected tasks...")
# for j in TEST_INDICES:
#     omega_j = omegas[j]
#     if all_slow_unstable_eig_freq[j]:
#         plot_unstable_eigenvalues(all_slow_unstable_eig_freq[j], j, omega_j, save_dir=output_dir)

# print("\nSlow Points: Plotting comparisons between target frequencies and unstable mode frequencies...")
# plot_frequency_comparison(all_slow_unstable_eig_freq, omegas, save_dir=output_dir)

# # -------------------------------
# # -------------------------------
# # 5. PCA RESULTS
# # -------------------------------
# # -------------------------------

# # -------------------------------
# # 5.1.1. Fixed Points: Perform PCA on Trajectories and Fixed Points
# # -------------------------------

# print("\nFixed Points: Starting PCA computation...")
# all_states = np.concatenate(state_traj_states, axis=0)  # (num_tasks*num_steps_train, N)
# pca = PCA(n_components=3)
# proj_all = pca.fit_transform(all_states)

# proj_trajs = []
# start = 0
# for traj in tqdm(state_traj_states, desc="Projecting trajectories"):
#     T = traj.shape[0]
#     proj_traj = proj_all[start:start + T]
#     proj_trajs.append(proj_traj)
#     start += T

# all_fixed_points_flat = []
# for task_fps in all_fixed_points:
#     all_fixed_points_flat.extend(task_fps)
# if all_fixed_points_flat:
#     all_fixed_points_flat = np.array(all_fixed_points_flat)
#     if all_fixed_points_flat.ndim == 1:
#         all_fixed_points_flat = all_fixed_points_flat.reshape(1, -1)
#     proj_fixed = pca.transform(all_fixed_points_flat)
# else:
#     proj_fixed = np.empty((0, 3))

# print("\nFixed Points: PCA complete.")

# # -------------------------------
# # 5.1.2. Slow Points: Perform PCA on Trajectories and Slow Points
# # -------------------------------

# print("\nSlow Points: Starting PCA computation...")
# all_slow_points_flat = []
# for task_sps in all_slow_points:
#     all_slow_points_flat.extend(task_sps)
# if all_slow_points_flat:
#     all_slow_points_flat = np.array(all_slow_points_flat)
#     if all_slow_points_flat.ndim == 1:
#         all_slow_points_flat = all_slow_points_flat.reshape(1, -1)
#     proj_slow = pca.transform(all_slow_points_flat)
# else:
#     proj_slow = np.empty((0, 3))

# print("\nSlow Points: PCA complete.")

# # -------------------------------
# # 5.2.1. Fixed Points: PCA Trajectory and Fixed Points Plot
# # -------------------------------

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# for traj in proj_trajs:
#     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.5)
# if proj_fixed.size > 0:
#     ax.scatter(proj_fixed[:, 0], proj_fixed[:, 1], proj_fixed[:, 2], color='green', s=50, label="Fixed Points")
# fixed_point_idx = 0
# for j, task_fps in enumerate(all_fixed_points):
#     for x_star in task_fps:
#         J_eff = np.array(jacobian_fixed_point(jnp.array(x_star), J_trained))
#         eigenvals, eigenvecs = eig(J_eff)
#         unstable_idx = np.where(np.real(eigenvals) > 0)[0]
#         if len(unstable_idx) > 0:
#             for idx in unstable_idx:
#                 v = np.real(eigenvecs[:, idx])
#                 scale = 0.5
#                 v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_fixed[fixed_point_idx]
#                 line = np.vstack([
#                     proj_fixed[fixed_point_idx] - v_proj,
#                     proj_fixed[fixed_point_idx] + v_proj
#                 ])
#                 ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red', linewidth=2, alpha=0.5)
#         fixed_point_idx += 1

# ax.set_title('PCA of Network Trajectories and Fixed Points')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.legend()
# filename = generate_filename("pca_plot", N, num_tasks, dt, T_drive, T_train)
# filepath = os.path.join(output_dir, filename.replace('.pkl', '.png'))
# plt.savefig(filepath)
# print(f"Saved PCA plot to {filepath}")
# plt.show()
# plt.close()

# # -------------------------------
# # 5.2.2. Slow Points: PCA Trajectory and Slow Points Plot
# # -------------------------------

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# for traj in proj_trajs:
#     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.5)
# if proj_slow.size > 0:
#     ax.scatter(proj_slow[:, 0], proj_slow[:, 1], proj_slow[:, 2], color='green', s=50, label="Slow Points")
# slow_point_idx = 0
# for j, task_sps in enumerate(all_slow_points):
#     for x_star in task_sps:
#         J_eff = np.array(jacobian_fixed_point(jnp.array(x_star), J_trained))
#         eigenvals, eigenvecs = eig(J_eff)
#         unstable_idx = np.where(np.real(eigenvals) > 0)[0]
#         if len(unstable_idx) > 0:
#             for idx in unstable_idx:
#                 v = np.real(eigenvecs[:, idx])
#                 scale = 0.5
#                 v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_slow[slow_point_idx]
#                 line = np.vstack([
#                     proj_slow[slow_point_idx] - v_proj,
#                     proj_slow[slow_point_idx] + v_proj
#                 ])
#                 ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red', linewidth=2, alpha=0.5)
#         slow_point_idx += 1

# ax.set_title('PCA of Network Trajectories and Slow Points')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.legend()
# filename = generate_filename("pca_plot_slow", N, num_tasks, dt, T_drive, T_train)
# filepath = os.path.join(output_dir, filename.replace('.pkl', '.png'))
# plt.savefig(filepath)
# print(f"Saved PCA plot to {filepath}")
# plt.show()
# plt.close()

# # -------------------------------
# # 5.3. Save Results from PCA
# # -------------------------------

# print("\nSaving PCA results...")
# pca_results = {
#     "proj_trajs": proj_trajs,
#     "proj_fixed": proj_fixed,
#     "proj_slow": proj_slow
# }
# save_variable(pca_results, "pca_results", N, num_tasks, dt, T_drive, T_train)