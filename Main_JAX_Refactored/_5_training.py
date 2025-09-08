"""
RNN model optimizers and training procedures.
"""


import jax
import jax.numpy as jnp
import optax
from functools import partial

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from _1_config import s, NUM_EPOCHS_ADAM, NUM_EPOCHS_LBFGS, LOSS_THRESHOLD, ADAM_LR, LEARNING_RATE, MEMORY_SIZE, SCALE_INIT_PRECOND
from _4_rnn_model import batched_loss, batched_loss_low_rank


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
    adam_opt = optax.adam(ADAM_LR)
    
    lbfgs_opt = optax.lbfgs(
        learning_rate=LEARNING_RATE,
        memory_size=MEMORY_SIZE,
        scale_init_precond=SCALE_INIT_PRECOND
    )
    
    return adam_opt, lbfgs_opt


def create_training_functions(adam_opt, lbfgs_opt, mask, loss_fn=None):
    """
    Create JIT-compiled training step functions using pre-computed driving phase final states.
    
    Arguments:
        adam_opt: Adam optimizer
        lbfgs_opt: L-BFGS optimizer  
        mask: sparsity mask for J matrix
        loss_fn: optional custom loss function (defaults to batched_loss)
        
    Returns:
        adam_step: JIT-compiled Adam training step function
        lbfgs_step: JIT-compiled L-BFGS training step function
        loss_fn: JIT-compiled loss function using pre-computed states
    """
    # Use provided loss function or default to batched_loss
    if loss_fn is None:
        loss_fn = batched_loss

    # Find the value and gradient of the loss function
    value_and_grad_fn = jax.value_and_grad(loss_fn)


    @jax.jit
    def adam_step(params, opt_state):
        """
        Perform one step of the Adam optimizer.
        
        Arguments:
            params: model parameters
            opt_state: optimizer state
            
        Returns:
            new_params: updated parameters
            new_opt_state: updated optimizer state
            loss: current loss value
        """
        # 1) Get loss and gradients
        loss, grads = value_and_grad_fn(params)

        # 2) One optimizer update
        updates, new_state = adam_opt.update(grads, opt_state, params)

        # 3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # 4) Enforce exact sparsity on the J matrix
        J1 = new_params["J"] * mask

        # 5) Reconstruct the new parameters with the updated J matrix
        new_params = {**new_params, "J": J1}

        return new_params, new_state, loss


    @jax.jit
    def lbfgs_step(params, opt_state):
        """
        Perform one step of the L-BFGS optimizer.
        
        Arguments:
            params: model parameters
            opt_state: optimizer state
            
        Returns:
            new_params: updated parameters
            new_opt_state: updated optimizer state
            loss: current loss value
        """
        # 1) Get loss and gradients
        loss, grads = value_and_grad_fn(params)

        # 2) One optimizer update
        updates, new_state = lbfgs_opt.update(
            grads, opt_state, params,
            value=loss, grad=grads,
            value_fn=loss_fn
        )

        # 3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # 4) Enforce exact sparsity on the J matrix
        J1 = new_params["J"] * mask

        # 5) Reconstruct the new parameters with the updated J matrix
        new_params = {**new_params, "J": J1}

        return new_params, new_state, loss
    

    return adam_step, lbfgs_step, value_and_grad_fn


# =============================================================================
# TRAINING PROCEDURE
# =============================================================================
# Training
@partial(jax.jit, static_argnames=("step_fn", "num_steps", "tag"))
def scan_with_history(params, opt_state, step_fn, num_steps, tag):
    """
    Function to run multiple steps of the chosen optimization method via lax.scan.
    Run `num_steps` of `step_fn` starting from (params, opt_state), but only keep the best-loss params, rather than the whole history.

    Arguments:
        params: current parameters of the model
        opt_state: current state of the optimizer
        step_fn: function to perform a single optimization step (adam_step or lbfgs_step)
        num_steps: number of steps to run
        tag: string tag for debug printing (e.g., "ADAM" or "L-BFGS")

    Returns:
        best_params: parameters corresponding to best_loss
        best_loss: lowest loss out of all the iterations
        final_params: parameters after the last optimization step
        final_opt_state: optimizer state after the last optimization step
    """
    # Initialize best‐so‐far
    init_best_loss = jnp.inf
    init_best_params = params

    def body(carry, idx_unused):
        params, opt_state, best_loss, best_params = carry

        # One optimization step
        params, opt_state, loss = step_fn(params, opt_state)

        # (Optional) debug print - simplified to avoid tracer formatting issues
        # jax.debug.print("[{}] step {}: loss {:.6e}", tag, idx_unused+1, loss)

        # If this step is better, update best_loss and best_params
        better = loss < best_loss
        best_loss = jax.lax.select(better, loss, best_loss)
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
        length=num_steps
    )

    return best_params, best_loss, final_p, final_s


# =============================================================================
# TRAINING EXECUTION
# =============================================================================
def train_model(params, mask, loss_fn=None):
    """
    Execute the full training procedure with Adam followed by L-BFGS.
    
    Arguments:
        params: initial parameters
        mask: sparsity mask for J matrix
        loss_fn: optional custom loss function (defaults to batched_loss)
        
    Returns:
        trained_params: best parameters after training
        final_loss: final loss value
    """
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Create training functions with pre-computed driving states
    adam_step, lbfgs_step, _ = create_training_functions(adam_opt, lbfgs_opt, mask, loss_fn)
    
    # Adam training phase
    print("Starting Adam optimization...")
    best_params_adam, best_loss_adam, params_after_adam, adam_state_after = scan_with_history(
        params, adam_state, adam_step,
        NUM_EPOCHS_ADAM,
        tag="ADAM"
    )

    print(f"Lowest Adam Loss: {best_loss_adam:.6e}")

    # Restore the best parameters
    params = best_params_adam
    best_loss_total = best_loss_adam

    # L-BFGS training phase (if needed)
    if best_loss_adam > LOSS_THRESHOLD:
        print("Starting L-BFGS optimization...")
        # Run L-BFGS optimization phase
        best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after = scan_with_history(
            params, lbfgs_state, lbfgs_step,
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


# =============================================================================
# LOW-RANK RNN FUNCTIONS
# =============================================================================
def apply_column_balancing(U: jnp.ndarray, V: jnp.ndarray, eps: float = 1e-8) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply column balancing to equalize norms of corresponding columns in U and V.
    
    For each column k, compute c_k = sqrt(||v_k|| / ||u_k||) and apply:
    U <- U * C, V <- V * C^(-1)
    
    Args:
        U: left factors of shape (N, R)
        V: right factors of shape (N, R)
        eps: small value for numerical stability
        
    Returns:
        U_balanced: balanced left factors
        V_balanced: balanced right factors
    """
    # Compute norms of each column
    u_norms = jnp.linalg.norm(U, axis=0)  # shape (R,)
    v_norms = jnp.linalg.norm(V, axis=0)  # shape (R,)
    
    # Compute balancing factors with numerical stability
    c_k = jnp.sqrt(v_norms / (u_norms + eps))  # shape (R,)
    
    # Apply balancing
    U_balanced = U * c_k[None, :]  # broadcast to (N, R)
    V_balanced = V / c_k[None, :]  # broadcast to (N, R)
    
    return U_balanced, V_balanced


def create_training_functions_low_rank(adam_opt, lbfgs_opt, loss_fn=None):
    """
    Create JIT-compiled training step functions for low-rank connectivity.
    
    Arguments:
        adam_opt: Adam optimizer
        lbfgs_opt: L-BFGS optimizer  
        loss_fn: loss function (defaults to batched_loss_low_rank)
        
    Returns:
        adam_step: JIT-compiled Adam training step function
        lbfgs_step: JIT-compiled L-BFGS training step function
        loss_fn: JIT-compiled loss function
    """
    # Use provided loss function or default to batched_loss_low_rank
    if loss_fn is None:
        loss_fn = batched_loss_low_rank

    # Find the value and gradient of the loss function
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def adam_step_low_rank(params, opt_state):
        """
        Perform one step of the Adam optimizer with low-rank connectivity.
        
        Arguments:
            params: model parameters containing U, V
            opt_state: optimizer state
            
        Returns:
            new_params: updated parameters with column balancing applied
            new_opt_state: updated optimizer state
            loss: current loss value
        """
        # 1) Get loss and gradients
        loss, grads = value_and_grad_fn(params)

        # 2) One optimizer update
        updates, new_state = adam_opt.update(grads, opt_state, params)

        # 3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # 4) Apply column balancing to U and V
        U_balanced, V_balanced = apply_column_balancing(new_params["U"], new_params["V"])
        
        # 5) Reconstruct the parameters with balanced U and V
        new_params = {**new_params, "U": U_balanced, "V": V_balanced}

        return new_params, new_state, loss

    @jax.jit
    def lbfgs_step_low_rank(params, opt_state):
        """
        Perform one step of the L-BFGS optimizer with low-rank connectivity.
        
        Arguments:
            params: model parameters containing U, V
            opt_state: optimizer state
            
        Returns:
            new_params: updated parameters with column balancing applied
            new_opt_state: updated optimizer state
            loss: current loss value
        """
        # 1) Get loss and gradients
        loss, grads = value_and_grad_fn(params)

        # 2) One optimizer update
        updates, new_state = lbfgs_opt.update(
            grads, opt_state, params,
            value=loss, grad=grads,
            value_fn=loss_fn
        )

        # 3) Apply updates to your parameters
        new_params = optax.apply_updates(params, updates)

        # 4) Apply column balancing to U and V
        U_balanced, V_balanced = apply_column_balancing(new_params["U"], new_params["V"])
        
        # 5) Reconstruct the parameters with balanced U and V
        new_params = {**new_params, "U": U_balanced, "V": V_balanced}

        return new_params, new_state, loss
    
    return adam_step_low_rank, lbfgs_step_low_rank, value_and_grad_fn


def train_model_low_rank(params, loss_fn=None):
    """
    Execute the full training procedure with Adam followed by L-BFGS for low-rank connectivity.
    
    Arguments:
        params: initial parameters containing U, V
        loss_fn: optional custom loss function
        
    Returns:
        trained_params: best parameters after training
        final_loss: final loss value
    """
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Create training functions
    adam_step, lbfgs_step, _ = create_training_functions_low_rank(adam_opt, lbfgs_opt, loss_fn)
    
    # Adam training phase
    print("Starting Adam optimization...")
    best_params_adam, best_loss_adam, params_after_adam, adam_state_after = scan_with_history(
        params, adam_state, adam_step,
        NUM_EPOCHS_ADAM,
        tag="ADAM"
    )

    print(f"Lowest Adam Loss: {best_loss_adam:.6e}")

    # Restore the best parameters
    params = best_params_adam
    best_loss_total = best_loss_adam

    # L-BFGS training phase (if needed)
    if best_loss_adam > LOSS_THRESHOLD:
        print("Starting L-BFGS optimization...")
        # Run L-BFGS optimization phase
        best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after = scan_with_history(
            params, lbfgs_state, lbfgs_step,
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