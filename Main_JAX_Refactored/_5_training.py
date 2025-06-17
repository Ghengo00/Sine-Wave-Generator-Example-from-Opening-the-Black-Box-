"""
RNN model optimizers and training procedures.
"""


import jax
import jax.numpy as jnp
import optax
from functools import partial
from _4_rnn_model import compute_driving_final_states, batched_loss_from_states_with_final_states


# =============================================================================
# OPTIMIZER SETUP AND STEPS
# =============================================================================
def setup_optimizers():
    """
    Set up Adam and L-BFGS optimizers with configurations from config.
    
    Returns:
        adam_opt: Adam optimizer
        lbfgs_opt: L-BFGS optimizer
    """
    from _1_config import ADAM_LR, LEARNING_RATE, MEMORY_SIZE, SCALE_INIT_PRECOND
    
    adam_opt = optax.adam(ADAM_LR)
    
    lbfgs_opt = optax.lbfgs(
        learning_rate=LEARNING_RATE,
        memory_size=MEMORY_SIZE,
        scale_init_precond=SCALE_INIT_PRECOND
    )
    
    return adam_opt, lbfgs_opt


def create_training_functions_with_state_chaining(adam_opt, lbfgs_opt, mask):
    """
    Create JIT-compiled training step functions that chain hidden states between steps.
    
    Arguments:
        adam_opt: Adam optimizer
        lbfgs_opt: L-BFGS optimizer  
        mask: sparsity mask for J matrix
        
    Returns:
        adam_step: JIT-compiled Adam training step function
        lbfgs_step: JIT-compiled L-BFGS training step function
    """
    from _1_config import s

    @jax.jit
    def adam_step(params, opt_state, current_states):
        """
        Perform one step of the Adam optimizer with state chaining.
        
        Arguments:
            params: model parameters
            opt_state: optimizer state
            current_states: current hidden states for all tasks, shape (num_tasks, N)
            
        Returns:
            new_params: updated parameters
            new_opt_state: updated optimizer state
            loss: current loss value
            final_states: final states after this step, shape (num_tasks, N)
        """
        # Define loss function that returns both loss and final states
        def loss_fn(p):
            return batched_loss_from_states_with_final_states(p, current_states)

        # 1) Get loss, final states, and gradients
        (loss, final_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # (2) One optimizer update
        updates, new_state = adam_opt.update(grads, opt_state, params)

        # (3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # (4) Enforce exact sparsity on the J matrix
        J1 = new_params["J"] * mask

        # (5) Reconstruct the new parameters with the updated J matrix
        new_params = {**new_params, "J": J1}

        return new_params, new_state, loss, final_states


    @jax.jit
    def lbfgs_step(params, opt_state, current_states):
        """
        Perform one step of the L-BFGS optimizer with state chaining.
        
        Arguments:
            params: model parameters
            opt_state: optimizer state
            current_states: current hidden states for all tasks, shape (num_tasks, N)
            
        Returns:
            new_params: updated parameters
            new_opt_state: updated optimizer state
            loss: current loss value
            final_states: final states after this step, shape (num_tasks, N)
        """
        # Define loss function that returns both loss and final states
        def loss_fn(p):
            return batched_loss_from_states_with_final_states(p, current_states)

        # 1) Get loss, final states, and gradients
        (loss, final_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # (2) One optimizer update
        updates, new_state = lbfgs_opt.update(
            grads, opt_state, params,
            value=loss, grad=grads,
            value_fn=lambda p: batched_loss_from_states_with_final_states(p, current_states)[0]  # Only return loss for value_fn
        )

        # (3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # (4) Enforce exact sparsity on the J matrix
        J1 = new_params["J"] * mask

        # (5) Reconstruct the new parameters with the updated J matrix
        new_params = {**new_params, "J": J1}

        return new_params, new_state, loss, final_states
    

    return adam_step, lbfgs_step


# =============================================================================
# TRAINING PROCEDURE
# =============================================================================
# Training
@partial(jax.jit, static_argnames=("step_fn", "num_steps", "tag"))
def scan_with_history_and_states(params, opt_state, current_states, step_fn, num_steps, tag):
    """
    Function to run multiple steps of the chosen optimization method via lax.scan with state chaining.
    Run `num_steps` of `step_fn` starting from (params, opt_state, current_states), but only keep the best-loss params, rather than the whole history.

    Arguments:
        params: current parameters of the model
        opt_state: current state of the optimizer
        current_states: current hidden states for all tasks, shape (num_tasks, N)
        step_fn: function to perform a single optimization step (adam_step or lbfgs_step)
        num_steps: number of steps to run
        tag: string tag for debug printing (e.g., "ADAM" or "L-BFGS")

    Returns:
        best_params: parameters corresponding to best_loss
        best_loss: lowest loss out of all the iterations
        final_params: parameters after the last optimization step
        final_opt_state: optimizer state after the last optimization step
        final_states: hidden states after the last optimization step
    """
    # Initialize best‐so‐far
    init_best_loss = jnp.inf
    init_best_params = params

    def body(carry, idx_unused):
        params, opt_state, current_states, best_loss, best_params = carry

        # One optimization step
        params, opt_state, loss, new_states = step_fn(params, opt_state, current_states)

        # (Optional) debug print
        jax.debug.print("[{}] step {}: loss {:.6e}", tag, idx_unused+1, loss)

        # If this step is better, update best_loss and best_params
        better = loss < best_loss
        best_loss = jax.lax.select(better, loss, best_loss)
        best_params = jax.tree_util.tree_map(
            lambda p_new, p_best: jax.lax.select(better, p_new, p_best),
            params, best_params
        )

        return (params, opt_state, new_states, best_loss, best_params), None

    # Run the scan; we feed a dummy sequence of length `num_steps` so body is called that many times
    init_carry = (params, opt_state, current_states, init_best_loss, init_best_params)
    (final_p, final_s, final_states, best_loss, best_params), _ = jax.lax.scan(
        body,
        init_carry,
        jnp.arange(num_steps),
        length=num_steps
    )

    return best_params, best_loss, final_p, final_s, final_states


# =============================================================================
# TRAINING EXECUTION
# =============================================================================
def train_model(params, mask):
    """
    Execute the full training procedure with Adam followed by L-BFGS using chained hidden states.
    
    Arguments:
        params: initial parameters
        mask: sparsity mask for J matrix
        
    Returns:
        trained_params: best parameters after training
        final_loss: final loss value
    """
    from _1_config import NUM_EPOCHS_ADAM, NUM_EPOCHS_LBFGS, LOSS_THRESHOLD
    
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Compute driving phase final states once at the beginning to initialize the chain
    print("Computing driving phase final states...")
    initial_states = compute_driving_final_states(params)
    print(f"Driving phase completed. Initial states shape: {initial_states.shape}")
    
    # Create training functions with state chaining
    adam_step, lbfgs_step = create_training_functions_with_state_chaining(adam_opt, lbfgs_opt, mask)
    
    # Adam training phase
    print("Starting Adam optimization with chained hidden states...")
    best_params_adam, best_loss_adam, params_after_adam, adam_state_after, states_after_adam = scan_with_history_and_states(
        params, adam_state, initial_states, adam_step,
        NUM_EPOCHS_ADAM,
        tag="ADAM"
    )

    print(f"Lowest Adam Loss: {best_loss_adam:.6e}")

    # Restore the best parameters
    params = best_params_adam
    best_loss_total = best_loss_adam

    # L-BFGS training phase (if needed)
    if best_loss_adam > LOSS_THRESHOLD:
        print("Starting L-BFGS optimization with chained hidden states...")
        # Use the final states from Adam as initial states for L-BFGS
        best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after, states_after_lbfgs = scan_with_history_and_states(
            params, lbfgs_state, states_after_adam, lbfgs_step,
            NUM_EPOCHS_LBFGS,
            tag="L-BFGS"
        )

        print(f"Lowest L-BFGS Loss: {best_loss_lbfgs:.6e}")

        # Restore the best parameters
        if best_loss_lbfgs < best_loss_adam:
            params = best_params_lbfgs
            best_loss_total = best_loss_lbfgs

    print(f"Overall best loss = {best_loss_total:.6e}")
    
    return params, best_loss_total