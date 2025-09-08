"""
RNN model initialisation, dynamics, simulation, and loss functions.
"""


import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
import numpy as np
from functools import partial
from _1_config import N, I, dt, s, num_tasks, omegas, static_inputs, time_drive, time_train, num_steps_train
from _3_data_generation import get_drive_input, get_train_input, get_train_target


# =============================================================================
# INITIALISE RNN WEIGHTS AND BIASES
# =============================================================================
def init_params(key):
    """
    Initialize RNN parameters with proper scaling.
    
    Arguments:
        key: JAX random key
        
    Returns:
        mask: sparsity mask for connections
        params: dictionary containing J, B, b_x, w, b_z
    """
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    mask = random.bernoulli(k_mask, p=1.0 - s, shape=(N, N)).astype(jnp.float32)
    J_unscaled = random.normal(k1, (N, N)) / jnp.sqrt(N) * mask
    # By Girko's law, we know that, in the limit N -> infinity, the eigenvalues of J will converge to a uniform distribution
    # within a disk centred at the origin of radius sqrt(Var * N) = sqrt((1 / N) * N) = 1.
    # After masking, the elements remain independently distributed with mean zero,
    # but their variance becomes Var_m = (1 - s) / N, meaning the spectral radius becomes sqrt(1 - s).
    # Thus, to maintain the same spectral radius as the full matrix, we scale J by 1 / sqrt(1 - s).
    J = J_unscaled / jnp.sqrt(1 - s)

    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)
    
    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}


# =============================================================================
# TRAJECTORY SIMULATION
# =============================================================================
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
        x_t = carry
        x_tp1, z_t = rnn_step(x_t, u_t.squeeze(), params)
        
        return x_tp1, (x_tp1, z_t)

    # Run scan: initial carry is x0, iterate over u_seq
    # lax.scan collects the tuples returned by scan_func into a tuple of arrays
    x_final, (xs_all, zs_all) = lax.scan(scan_func, x0, u_seq)
    
    # xs_all has shape (T, N), but we want (T+1, N) including x0 at index 0, so we stack this
    xs = jnp.vstack([x0[None, :], xs_all])
    
    return xs, zs_all


# =============================================================================
# SOLVE TASK OPTION 1 - DRIVING PHASE AT EACH STEP
# =============================================================================
# Solving
@jit
def solve_task(params, task):
    """
    Run the drive and train phases all the way through for a single task.
    """
    x0 = jnp.zeros((N,), dtype=jnp.float32)      # Initial state for the task
    
    # Drive phase
    d_u = get_drive_input(task)                   # shape (num_steps_drive, I)
    xs_d, _ = simulate_trajectory(x0, d_u, params)
    x_final = xs_d[-1]
    
    # Train phase
    t_u = get_train_input(task)                   # shape (num_steps_train, I)
    _, zs_tr = simulate_trajectory(x_final, t_u, params)

    return zs_tr    # shape (num_steps_train,)


# Vectorize solve_task over all task values
vmap_solve_task = vmap(solve_task, in_axes=(None, 0))   # vmap_solve_task gives an array of shape (num_tasks, num_steps_train)


# =============================================================================
# SOLVE TASK OPTION 2 - USE FIRST DRIVE PHASE AS INITIALISATION FOR ALL OPTIMISATION STEPS
# =============================================================================
# Solving
@jit
def compute_driving_final_states(params):
    """
    Compute the final states after the driving phase for all tasks.
    This should be called once at the beginning of training.
    
    Arguments:
        params: dict containing J, B, b_x, w, b_z
    
    Returns:
        driving_final_states: array of shape (num_tasks, N) containing final states after driving phase
    """
    def single_task_driving(task):
        x0 = jnp.zeros((N,), dtype=jnp.float32)
        d_u = get_drive_input(task)
        xs_d, _ = simulate_trajectory(x0, d_u, params)
        
        return xs_d[-1]  # Final state after driving phase
    
    tasks = jnp.arange(num_tasks)
    driving_final_states = vmap(single_task_driving)(tasks)
    
    return driving_final_states


# Solving
@jit
def solve_task_from_state(params, task, initial_state):
    """
    Run only the training phase for a single task, starting from a given initial state.
    
    Arguments:
        params: dict containing J, B, b_x, w, b_z
        task: task index
        initial_state: initial state for the training phase, shape (N,)
    
    Returns:
        zs_tr: training phase outputs, shape (num_steps_train,)
    """
    # Train phase only
    t_u = get_train_input(task)
    _, zs_tr = simulate_trajectory(initial_state, t_u, params)

    return zs_tr


# Vectorize solve_task_from_state over all tasks
vmap_solve_task_from_state = vmap(solve_task_from_state, in_axes=(None, 0, 0))


# =============================================================================
# SOLVE TASK OPTION 3 - USE LAST HIDDEN STATE IN PREVIOUS OPTIMISATION STEP AS INITIALISATION IN NEXT OPTIMISATION STEP
# =============================================================================
@jit
def solve_task_from_state_with_final_state(params, task, initial_state):
    """
    Run only the training phase for a single task, starting from a given initial state.
    Returns both outputs and final hidden state.
    
    Arguments:
        params: dict containing J, B, b_x, w, b_z
        task: task index
        initial_state: initial state for the training phase, shape (N,)
    
    Returns:
        zs_tr: training phase outputs, shape (num_steps_train,)
        final_state: final hidden state after training, shape (N,)
    """
    # Train phase only
    t_u = get_train_input(task)
    xs_tr, zs_tr = simulate_trajectory(initial_state, t_u, params)
    
    return zs_tr, xs_tr[-1]  # Return outputs and final state


# Vectorize solve_task_from_state_with_final_state over all tasks
vmap_solve_task_from_state_with_final_state = vmap(solve_task_from_state_with_final_state, in_axes=(None, 0, 0))


# =============================================================================
# DIAGNOSTICS LOSS
# =============================================================================
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
      avg_loss         : average loss over all tasks and all time steps, scalar
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

    # Convert to NumPy arrays
    traj_states = np.array(all_xs_train)                # shape (num_tasks, num_steps_train+1, N)
    fixed_point_inits = np.array(all_x_drive_final)     # shape (num_tasks, N)

    return float(avg_loss), traj_states, fixed_point_inits


# =============================================================================
# TRAINING LOSS
# =============================================================================
# Loss (OPTION 1)
@jit
def batched_loss(params):
    """
    Compute the loss for for num_steps_train time steps, averaged over all tasks and all time steps.
    This is the original version that includes the driving phase.

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


# Loss (OPTION 2)
@jit
def batched_loss_from_states(params, driving_final_states):
    """
    Compute the loss using pre-computed driving phase final states.
    This version skips the driving phase and starts directly from the given states.

    Arguments:
        params: dictionary of RNN weights
        driving_final_states: array of shape (num_tasks, N) containing final states after driving phase
    
    Returns:
        loss  : scalar loss value averaged over all tasks
    """
    tasks = jnp.arange(num_tasks)

    # Vectorized simulation over all tasks, starting from pre-computed states
    zs_all = vmap_solve_task_from_state(params, tasks, driving_final_states)  # shape: (num_tasks, num_steps_train)

    # Get the training targets for all tasks
    targets_all = vmap(get_train_target)(tasks)  # shape: (num_tasks, num_steps_train)

    # Compute the MSE over tasks and time
    return jnp.mean((zs_all - targets_all) ** 2)


# Loss (OPTION 3)
@jit
def batched_loss_from_states_with_final_states(params, current_states):
    """
    Compute the loss using current hidden states and return both loss and final states.
    Used for chaining optimization steps.

    Arguments:
        params: dictionary of RNN weights
        current_states: array of shape (num_tasks, N) containing current initial states
    
    Returns:
        loss: scalar loss value averaged over all tasks
        final_states: array of shape (num_tasks, N) containing final states after training
    """
    tasks = jnp.arange(num_tasks)

    # Vectorized simulation over all tasks, starting from current states
    zs_all, final_states = vmap_solve_task_from_state_with_final_state(params, tasks, current_states)

    # Get the training targets for all tasks
    targets_all = vmap(get_train_target)(tasks)  # shape: (num_tasks, num_steps_train)

    # Compute the MSE over tasks and time
    loss = jnp.mean((zs_all - targets_all) ** 2)
    
    return loss, final_states


# =============================================================================
# TRAINING LOSS (WITH REGULARIZATION)
# =============================================================================
# Loss function factory for creating custom loss functions with regularization
def create_loss_function(l1_reg_strength=0.0, use_precomputed_states=False):
    """
    Factory function that returns a loss function with optional L1 regularization.
    
    Arguments:
        l1_reg_strength: L1 regularization strength for the connectivity matrix J
        use_precomputed_states: If True, returns a loss function that uses precomputed driving states
        
    Returns:
        loss_function: JIT-compiled loss function with the specified regularization
    """
    if use_precomputed_states:
        @jit
        def regularized_batched_loss_from_states(params, driving_final_states): # Loss (OPTION 2)
            """Loss function with L1 regularization using precomputed driving states."""
            # Original MSE loss
            tasks = jnp.arange(num_tasks)
            zs_all = vmap_solve_task_from_state(params, tasks, driving_final_states)
            targets_all = vmap(get_train_target)(tasks)
            mse_loss = jnp.mean((zs_all - targets_all) ** 2)
            
            # L1 regularization term
            if l1_reg_strength > 0.0:
                l1_term = l1_reg_strength * jnp.mean(jnp.abs(params["J"]))
                return mse_loss + l1_term
            return mse_loss
        
        return regularized_batched_loss_from_states
    
    else:
        @jit
        def regularized_batched_loss(params):   # Loss (OPTION 1)
            """Loss function with L1 regularization including driving phase."""
            # Original MSE loss
            tasks = jnp.arange(num_tasks)
            zs_all = vmap_solve_task(params, tasks)
            targets_all = vmap(get_train_target)(tasks)
            mse_loss = jnp.mean((zs_all - targets_all) ** 2)
            
            # L1 regularization term
            if l1_reg_strength > 0.0:
                l1_term = l1_reg_strength * jnp.mean(jnp.abs(params["J"]))
                return mse_loss + l1_term
            return mse_loss
        
        return regularized_batched_loss


# =============================================================================
# LOW-RANK RNN FUNCTIONS
# =============================================================================
@jit
def rnn_step_low_rank(x, inputs, params):
    """
    Single time-step update of the RNN state using low-rank connectivity.
    
    Arguments:
    x     : current state (at time t), dimensions (N,)
    inputs: scalar input at this time-step (time t), dimensions (I,)
    params: dict containing U, V, B, b_x, w, b_z

    Returns:
    x_new: new state (at time t+1), dimensions (N,)
    z    : output at this time-step (time t), scalar
    """
    U, V, B, b_x, w, b_z = params["U"], params["V"], params["B"], params["b_x"], params["w"], params["b_z"]
    
    r = jnp.tanh(x)            # (N,)
    z = jnp.dot(w, r) + b_z    # scalar
    
    # Efficient computation: U @ (V.T @ r) instead of (U @ V.T) @ r
    Vt_r = V.T @ r             # (R,)
    J_r = U @ Vt_r             # (N,)
    
    # dx/dt = -x + J r + B u + b_x
    dx = -x + J_r + (B * inputs).sum(axis=1) + b_x
    x_new = x + dt * dx
    
    return x_new, z


@jit
def simulate_trajectory_low_rank(x0, u_seq, params):
    """
    Vectorized simulation over T time-steps using lax.scan with low-rank connectivity.

    Arguments:
    x0    : initial state, dimensions (N,)
    u_seq : input sequence over the T time steps, dimensions (T, I)
    params: dict containing U, V, B, b_x, w, b_z

    Returns
      xs : system state sequence over the T time steps, dimensions (T+1, N)
      zs : output sequence over the T time steps, dimensions (T,)
    """
    
    def scan_func(carry, u_t):
        """
        Single step function for lax.scan.
        """
        x_t = carry
        x_tp1, z_t = rnn_step_low_rank(x_t, u_t.squeeze(), params)
        
        return x_tp1, (x_tp1, z_t)

    # Run scan: initial carry is x0, iterate over u_seq
    x_final, (xs_all, zs_all) = lax.scan(scan_func, x0, u_seq)
    
    # xs_all has shape (T, N), but we want (T+1, N) including x0 at index 0
    xs = jnp.vstack([x0[None, :], xs_all])
    
    return xs, zs_all


@jit
def solve_task_low_rank(params, task):
    """
    Run the drive and train phases for a single task using low-rank connectivity.
    """
    x0 = jnp.zeros((N,), dtype=jnp.float32)      # Initial state for the task
    
    # Drive phase
    d_u = get_drive_input(task)                   # shape (num_steps_drive, I)
    xs_d, _ = simulate_trajectory_low_rank(x0, d_u, params)
    x_final = xs_d[-1]
    
    # Train phase
    t_u = get_train_input(task)                   # shape (num_steps_train, I)
    _, zs_tr = simulate_trajectory_low_rank(x_final, t_u, params)

    return zs_tr    # shape (num_steps_train,)


# Vectorize solve_task_low_rank over all task values
vmap_solve_task_low_rank = vmap(solve_task_low_rank, in_axes=(None, 0))


@jit
def batched_loss_low_rank(params):
    """
    Compute the batched loss for low-rank connectivity RNN.
    """
    tasks = jnp.arange(num_tasks)

    # Vectorized simulation over all tasks
    zs_all = vmap_solve_task_low_rank(params, tasks)  # shape: (num_tasks, num_steps_train)

    # Get the training targets for all tasks
    targets_all = vmap(get_train_target)(tasks)       # shape: (num_tasks, num_steps_train)

    # Compute the MSE over tasks and time
    return jnp.mean((zs_all - targets_all) ** 2)