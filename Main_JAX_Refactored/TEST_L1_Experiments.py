"""
L1 Regularization experiments: Train RNN with different L1 regularization strengths and track Jacobian eigenvalue evolution.

This script trains RNNs at fixed sparsity (s=0.0) with different L1 regularization strengths 
and visualizes how the eigenvalues of the Jacobian matrix at fixed points evolve during the optimization process.

DIFFERENCES FROM TEST_Sparsity_Experiments.py:
1. Fixed sparsity at s=0.0 instead of varying sparsity
2. Varies L1 regularization strength instead of sparsity
3. Uses create_loss_function() with l1_reg_strength parameter
4. Saves results with L1 regularization naming convention instead of sparsity
5. All other functionality (eigenvalue tracking, analysis, visualization) is identical
"""


import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import matplotlib
# Set matplotlib to non-interactive backend to prevent plots from displaying
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from functools import partial
from tqdm import tqdm
import time
import os
from contextlib import contextmanager

# Import all modules
from _1_config import (
    N, I, num_tasks, s, dt, omegas, static_inputs,
    NUM_EPOCHS_ADAM, NUM_EPOCHS_LBFGS, LOSS_THRESHOLD,
    NUMPY_SEED, JAX_SEED, EIGENVALUE_SAMPLE_FREQUENCY,
    TOL, MAXITER, GAUSSIAN_STD, NUM_ATTEMPTS, SLOW_POINT_SEARCH, TEST_INDICES
)
from _2_utils import (Timer, save_variable, set_custom_output_dir, get_output_dir, 
                     save_variable_with_l1_reg, save_figure_with_l1_reg, get_l1_output_dir)
from _3_data_generation import generate_all_inputs_and_targets
from _4_rnn_model import init_params, run_batch_diagnostics, create_loss_function
from _5_training import setup_optimizers, create_training_functions, scan_with_history
from _6_analysis import find_points, compute_jacobian, find_and_analyze_points, generate_point_summaries
from _7_visualization import (save_figure, plot_trajectories_vs_targets, plot_parameter_matrices_for_tasks, 
                          analyze_jacobians_visualization, analyze_unstable_frequencies_visualization)
from _8_pca_analysis import run_pca_analysis




# =============================================================================
# =============================================================================
# USER INPUT FUNCTIONS
# =============================================================================
# =============================================================================
def get_l1_reg_values():
    """
    Get L1 regularization strength values from user input.
    
    Returns:
        list: List of L1 regularization strength values to test
    """
    print("=" * 60)
    print("L1 REGULARIZATION EIGENVALUE EXPERIMENT SETUP")
    print("=" * 60)
    
    print("\nEnter L1 regularization strength values to test (≥ 0.0)")
    print("Examples: 0.0, 1e-5, 1e-4, 1e-3, 0.01")
    print("Enter values separated by commas (e.g., '0.0, 1e-5, 1e-4')")
    
    while True:
        try:
            user_input = input("\nL1 regularization strengths: ").strip()
            l1_reg_values = [float(x.strip()) for x in user_input.split(',')]
            
            # Validate L1 regularization values
            for l1_val in l1_reg_values:
                if l1_val < 0.0:
                    raise ValueError(f"L1 regularization strength {l1_val} must be ≥ 0.0")
            
            print(f"\nWill test L1 regularization strengths: {l1_reg_values}")
            confirm = input("Proceed? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return l1_reg_values
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter valid comma-separated numbers ≥ 0.0")


def get_eigenvalue_options():
    """
    Get user preferences for which eigenvalue types to compute.
    
    Returns:
        tuple: (compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
    """
    print("\n" + "=" * 60)
    print("EIGENVALUE COMPUTATION OPTIONS")
    print("=" * 60)
    
    print("\nChoose which eigenvalue types to compute during training:")
    print("1. Jacobian eigenvalues - Computed at fixed points (more expensive)")
    print("2. Connectivity matrix eigenvalues - Direct computation (faster)")
    print("3. Both types")
    print("4. Neither (skip eigenvalue tracking)")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                return True, False, get_frequency_selection()
            elif choice == '2':
                return False, True, None
            elif choice == '3':
                return True, True, get_frequency_selection()
            elif choice == '4':
                return False, False, None
            else:
                print("Please enter a number between 1 and 4")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(1)


def get_frequency_selection():
    """
    Get user selection of frequencies for Jacobian eigenvalue computation.
    
    Returns:
        list: List of frequency indices (0-based) to analyze
    """
    print("\n" + "=" * 60)
    print("FREQUENCY SELECTION FOR JACOBIAN EIGENVALUES")
    print("=" * 60)
    
    print(f"\nAvailable frequencies (total: {num_tasks}):")
    print(f"Index range: 0 to {num_tasks-1}")
    print(f"Frequency range: {float(omegas[0]):.3f} to {float(omegas[-1]):.3f} rad/s")
    
    # Show some example frequencies
    print("\nExample frequencies:")
    for i in [0, num_tasks//4, num_tasks//2, 3*num_tasks//4, num_tasks-1]:
        print(f"  Index {i:2d}: ω = {float(omegas[i]):.3f} rad/s, static_input = {float(static_inputs[i]):.3f}")
    
    print("\nOptions:")
    print("1. Select specific frequency indices (e.g., '0, 25, 50')")
    print("2. Select all frequencies")
    print("3. Select frequency range (e.g., '0-10' for indices 0 through 10)")
    
    while True:
        try:
            choice = input("\nEnter option (1-3): ").strip()
            
            if choice == '1':
                freq_input = input("Enter frequency indices separated by commas: ").strip()
                freq_indices = [int(x.strip()) for x in freq_input.split(',')]
                
                # Validate indices
                for idx in freq_indices:
                    if not (0 <= idx < num_tasks):
                        raise ValueError(f"Frequency index {idx} must be between 0 and {num_tasks-1}")
                
                print(f"\nSelected frequencies:")
                for idx in freq_indices:
                    print(f"  Index {idx}: ω = {float(omegas[idx]):.3f} rad/s, static_input = {float(static_inputs[idx]):.3f}")
                
                confirm = input("Proceed with these frequencies? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return freq_indices
                    
            elif choice == '2':
                print(f"\nSelected all {num_tasks} frequencies")
                confirm = input("Proceed? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return list(range(num_tasks))
                    
            elif choice == '3':
                range_input = input("Enter range (e.g., '0-10'): ").strip()
                start_str, end_str = range_input.split('-')
                start_idx = int(start_str.strip())
                end_idx = int(end_str.strip())
                
                if not (0 <= start_idx <= end_idx < num_tasks):
                    raise ValueError(f"Range must be within 0 to {num_tasks-1}")
                
                freq_indices = list(range(start_idx, end_idx + 1))
                print(f"\nSelected frequency range:")
                print(f"  Indices {start_idx} to {end_idx} ({len(freq_indices)} frequencies)")
                print(f"  ω range: {float(omegas[start_idx]):.3f} to {float(omegas[end_idx]):.3f} rad/s")
                
                confirm = input("Proceed with this range? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return freq_indices
                    
            else:
                print("Please enter a number between 1 and 3")
                
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            print("Please enter valid frequency indices or range")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(1)




# =============================================================================
# =============================================================================
# CONTEXT MANAGER
# =============================================================================
# =============================================================================
@contextmanager
def l1_output_context(l1_reg_value):
    """
    Context manager to temporarily set output directory to L_1-specific folder.
    """
    from _2_utils import _CUSTOM_OUTPUT_DIR, set_custom_output_dir
    
    # Save current custom output directory
    original_custom_dir = _CUSTOM_OUTPUT_DIR
    
    # Set to L_1-specific directory  
    l1_dir = get_l1_output_dir(l1_reg_value)
    set_custom_output_dir(l1_dir)
    
    try:
        yield l1_dir
    finally:
        # Restore original directory
        if original_custom_dir:
            set_custom_output_dir(original_custom_dir)
        else:
            set_custom_output_dir(None)




# =============================================================================
# =============================================================================
# EIGENVALUE COMPUTATION FUNCTIONS
# =============================================================================
# =============================================================================
def compute_connectivity_eigenvalues(params):
    """
    Compute eigenvalues of the connectivity matrix J.
    
    Arguments:
        params: RNN parameters
        
    Returns:
        eigenvalues: complex array of eigenvalues of the connectivity matrix J
    """
    try:
        J = np.array(params["J"])
        eigenvals = np.linalg.eigvals(J)
        
        return eigenvals
    
    except Exception as e:
        print(f"Warning: Could not compute connectivity eigenvalues - {e}")
        return np.array([])


def compute_jacobian_eigenvalues_at_fixed_points(params, frequency_indices):
    """
    Find fixed points and compute all Jacobian eigenvalues at those points for specified frequencies.
    
    Arguments:
        params: RNN parameters
        frequency_indices: list of frequency indices for which to compute eigenvalues
        
    Returns:
        eigenvalue_data_by_freq: dictionary mapping frequency index to eigenvalue array
    """
    # Convert JAX arrays to numpy for analysis functions
    J_trained = np.array(params["J"])
    B_trained = np.array(params["B"])
    b_x_trained = np.array(params["b_x"])
    
    # Initial guess for fixed point search (zero state)
    x0_guess = np.zeros(N)
    
    eigenvalue_data_by_freq = {}
    
    for freq_idx in frequency_indices:
        # Get the corresponding static input for this frequency
        u_const = float(static_inputs[freq_idx])
        
        try:
            # Find fixed points using existing analysis code
            fixed_points = find_points(
                x0_guess=x0_guess,
                u_const=u_const,
                J_trained=J_trained,
                B_trained=B_trained,
                b_x_trained=b_x_trained,
                mode='fixed',
                num_attempts=20,  # Reduce attempts for speed during training
                tol=TOL,
                maxiter=500,  # Reduce max iterations for speed
                gaussian_std=GAUSSIAN_STD
            )
            
            all_eigenvalues = []
            
            # Compute Jacobian eigenvalues at each fixed point
            for fp in fixed_points:
                jacobian = compute_jacobian(fp, J_trained)
                eigenvals = np.linalg.eigvals(jacobian)
                all_eigenvalues.extend(eigenvals)
            
            eigenvalue_data_by_freq[freq_idx] = np.array(all_eigenvalues)
        
        except Exception as e:
            print(f"Warning: Could not compute Jacobian eigenvalues for frequency index {freq_idx} - {e}")
            eigenvalue_data_by_freq[freq_idx] = np.array([])
    
    return eigenvalue_data_by_freq


def compute_eigenvalues_during_training(params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Compute the requested eigenvalue types during training.
    
    Arguments:
        params: RNN parameters
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        eigenvalue_data: dictionary with requested eigenvalue types
    """
    eigenvalue_data = {}
    
    if compute_jacobian_eigenvals and jacobian_frequencies is not None:
        jacobian_eigenvals = compute_jacobian_eigenvalues_at_fixed_points(params, jacobian_frequencies)
        eigenvalue_data['jacobian'] = jacobian_eigenvals
    
    if compute_connectivity_eigenvals:
        connectivity_eigenvals = compute_connectivity_eigenvalues(params)
        eigenvalue_data['connectivity'] = connectivity_eigenvals
    
    return eigenvalue_data




# =============================================================================
# =============================================================================
# TRAINING WITH EIGENVALUE TRACKING FUNCTION
# =============================================================================
# =============================================================================
def modified_scan_with_eigenvalues(params, opt_state, step_fn, num_steps, tag, sample_frequency, 
                                   compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Optimized training loop that uses lax.scan for chunks between eigenvalue sampling.
    
    This hybrid approach uses JAX's lax.scan for fast optimization steps while keeping
    eigenvalue sampling in Python for flexibility and debugging.
    
    Arguments:
        params: initial parameters
        opt_state: initial optimizer state
        step_fn: optimization step function
        num_steps: total number of steps
        tag: string tag for logging
        sample_frequency: how often to sample eigenvalues
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        best_params: parameters with best loss
        best_loss: best loss achieved
        final_params: final parameters
        final_opt_state: final optimizer state
        eigenvalue_history: list of (iteration, eigenvalue_data) tuples
    """
    # Check if any eigenvalue computation is requested
    compute_any_eigenvals = compute_jacobian_eigenvals or compute_connectivity_eigenvals
    
    if compute_any_eigenvals:
        eigenval_types = []
        if compute_jacobian_eigenvals:
            if jacobian_frequencies is not None:
                eigenval_types.append(f"Jacobian (frequencies: {jacobian_frequencies})")
            else:
                eigenval_types.append("Jacobian")
        if compute_connectivity_eigenvals:
            eigenval_types.append("Connectivity")
        print(f"Starting {tag} optimization with {', '.join(eigenval_types)} eigenvalue sampling every {sample_frequency} steps...")
        print(f"Using optimized lax.scan for {sample_frequency}-step chunks...")
    else:
        print(f"Starting {tag} optimization (no eigenvalue sampling)...")
        print(f"Using optimized lax.scan for full {num_steps}-step run...")
    
    # NO EIGENVALUE SAMPLING CASE
    # If no eigenvalue sampling, use the fast scan_with_history directly
    if not compute_any_eigenvals:
        best_params, best_loss, final_params, final_opt_state = scan_with_history(
            params, opt_state, step_fn, num_steps, tag
        )

        return best_params, best_loss, final_params, final_opt_state, []
    
    # EIGENVALUE SAMPLING CASE
    # Initialize tracking variables
    best_loss = jnp.inf
    best_params = params
    eigenvalue_history = []
    current_params = params
    current_opt_state = opt_state
    
    # Sample eigenvalues at iteration 0
    print(f"[{tag}] Sampling eigenvalues at iteration 0...")
    initial_eigenval_data = compute_eigenvalues_during_training(
        params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
    )
    eigenvalue_history.append((0, initial_eigenval_data))
    
    # Calculate chunking strategy
    chunk_size = sample_frequency
    num_chunks = num_steps // chunk_size
    remaining_steps = num_steps % chunk_size
    
    print(f"[{tag}] Processing {num_chunks} chunks of {chunk_size} steps each" + 
          (f" + {remaining_steps} remaining steps" if remaining_steps > 0 else ""))
    
    # Process in chunks using optimized scan_with_history
    for chunk_idx in tqdm(range(num_chunks), desc=f"{tag} Training (Chunks)"):
        # Use optimized scan_with_history for this chunk
        chunk_best_params, chunk_best_loss, current_params, current_opt_state = scan_with_history(
            current_params, current_opt_state, step_fn, chunk_size, f"{tag}-Chunk{chunk_idx+1}"
        )
        
        # Update global best if chunk improved
        if chunk_best_loss < best_loss:
            best_loss = chunk_best_loss
            best_params = chunk_best_params
        
        # Sample eigenvalues at end of chunk
        step_num = (chunk_idx + 1) * chunk_size
        print(f"[{tag}] step {step_num}: best_loss {best_loss:.6e} - Sampling eigenvalues...")
        eigenval_data = compute_eigenvalues_during_training(
            current_params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
        )
        eigenvalue_history.append((step_num, eigenval_data))
    
    # Handle remaining steps if any
    if remaining_steps > 0:
        print(f"[{tag}] Processing final {remaining_steps} steps...")
        final_best_params, final_best_loss, current_params, current_opt_state = scan_with_history(
            current_params, current_opt_state, step_fn, remaining_steps, f"{tag}-Final"
        )
        
        # Update global best if final chunk improved
        if final_best_loss < best_loss:
            best_loss = final_best_loss
            best_params = final_best_params
    
    print(f"[{tag}] Optimization completed. Final best loss: {best_loss:.6e}")
    
    return best_params, best_loss, current_params, current_opt_state, eigenvalue_history




# =============================================================================
# =============================================================================
# FULL RUN FUNCTION
# =============================================================================
# =============================================================================
def train_with_full_analysis(params, mask, key, compute_jacobian_eigenvals, compute_connectivity_eigenvals, sparsity_value=0.0, l1_reg_strength=0.0, jacobian_frequencies=None):
    """
    Train model with full analysis pipeline (replicating _0_main_refactored.py functionality).
    
    Arguments:
        params: initial parameters
        mask: sparsity mask (always all True for L1 experiments)
        sparsity_value: current sparsity level being tested (always 0.0 for L1 experiments)
        l1_reg_strength: L1 regularization strength being tested
        key: JAX random key for analysis
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues during training
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues during training
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        results_dict: comprehensive results dictionary containing all analysis outputs
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WITH L1 REGULARIZATION = {l1_reg_strength}")
    print(f"{'='*60}")
    
    
    # ========================================
    # 1. SETUP AND INITIALIZATION
    # ========================================
    results = {}
    results['sparsity'] = sparsity_value  # Keep for compatibility, always 0.0 for L1 experiments
    results['l1_reg_strength'] = l1_reg_strength
    
    # Create L1 regularized loss function
    loss_fn = create_loss_function(l1_reg_strength=l1_reg_strength, use_precomputed_states=False)

    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Create training functions
    adam_step, lbfgs_step, value_and_grad_fn = create_training_functions(adam_opt, lbfgs_opt, mask, loss_fn)
    
    # Track eigenvalues during training
    eigenvalue_data = {}
    

    # ========================================
    # 2. TRAINING WITH EIGENVALUE TRACKING
    # ========================================
    print("\n2. TRAINING WITH EIGENVALUE TRACKING")
    print("-" * 40)
    
    with Timer(f"Training with L1 regularization {l1_reg_strength}"):
        # Adam training phase with eigenvalue tracking
        best_params_adam, best_loss_adam, params_after_adam, adam_state_after, adam_eigenvals = \
            modified_scan_with_eigenvalues(
                params, adam_state, adam_step,
                NUM_EPOCHS_ADAM, "ADAM", EIGENVALUE_SAMPLE_FREQUENCY,
                compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
            )
        
        eigenvalue_data['adam'] = adam_eigenvals
        print(f"Adam phase completed. Best loss: {best_loss_adam:.6e}")
        
        # Continue with best parameters from Adam
        trained_params = best_params_adam
        final_loss = best_loss_adam
        
        # L-BFGS training phase (if needed)
        if best_loss_adam > LOSS_THRESHOLD:
            print("Loss threshold not met, starting L-BFGS phase...")
            best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after, lbfgs_eigenvals = \
                modified_scan_with_eigenvalues(
                    trained_params, lbfgs_state, lbfgs_step,
                    NUM_EPOCHS_LBFGS, "L-BFGS", EIGENVALUE_SAMPLE_FREQUENCY,
                    compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
                )
            
            eigenvalue_data['lbfgs'] = lbfgs_eigenvals
            
            # Use L-BFGS results if better
            if best_loss_lbfgs < best_loss_adam:
                trained_params = best_params_lbfgs
                final_loss = best_loss_lbfgs
                print(f"L-BFGS improved results. Best loss: {best_loss_lbfgs:.6e}")
        else:
            eigenvalue_data['lbfgs'] = []
            print("Loss threshold met, skipping L-BFGS phase.")
    
    results['trained_params'] = trained_params
    results['final_loss'] = final_loss
    results['eigenvalue_data'] = eigenvalue_data
    

    # ========================================
    # 3. POST-TRAINING DIAGNOSTICS
    # ========================================
    print("\n3. POST-TRAINING DIAGNOSTICS")
    print("-" * 40)
    
    # Collect final trajectories and fixed-point initializations
    final_loss_check, state_traj_states, state_fixed_point_inits = run_batch_diagnostics(trained_params, key)
    print(f"Final loss after training: {final_loss_check:.6e}")
    
    results['state_traj_states'] = state_traj_states
    results['state_fixed_point_inits'] = state_fixed_point_inits
    

    # ========================================
    # 4. SAVE TRAINING RESULTS
    # ========================================
    print("\n4. SAVING TRAINING RESULTS")
    print("-" * 40)
    
    print("Saving training results...")
    save_variable_with_l1_reg(np.array(trained_params["J"]), f"J_param_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    save_variable_with_l1_reg(np.array(trained_params["B"]), f"B_param_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    save_variable_with_l1_reg(np.array(trained_params["b_x"]), f"b_x_param_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    save_variable_with_l1_reg(np.array(trained_params["w"]), f"w_param_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    save_variable_with_l1_reg(np.array(trained_params["b_z"]), f"b_z_param_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)

    state_dict = {
        "traj_states": state_traj_states,
        "fixed_point_inits": state_fixed_point_inits
    }
    save_variable_with_l1_reg(state_dict, f"state_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    

    # ========================================
    # 5. VISUALIZATION OF BASIC RESULTS
    # ========================================
    print("\n5. VISUALIZATION OF BASIC RESULTS")
    print("-" * 40)
    
    # Plot trajectories vs targets (save in L1-specific folder)
    print("Plotting produced trajectories vs. target signals...")
    plot_trajectories_vs_targets(trained_params, test_indices=TEST_INDICES, l1_reg_value=l1_reg_strength)
    
    # Plot parameter matrices (save in L1-specific folder)
    plot_parameter_matrices_for_tasks(trained_params, test_indices=TEST_INDICES, l1_reg_value=l1_reg_strength)
    

    # ========================================
    # 6. FIXED POINT ANALYSIS
    # ========================================
    print("\n6. FIXED POINT ANALYSIS")
    print("-" * 40)
    
    with Timer("Fixed Point Analysis"):
        # Find and analyze fixed points
        all_fixed_points, all_jacobians, all_unstable_eig_freq = find_and_analyze_points(
            state_traj_states, state_fixed_point_inits, trained_params, point_type="fixed"
        )
        
        # Save fixed point results
        save_variable_with_l1_reg(all_fixed_points, f"all_fixed_points_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
        save_variable_with_l1_reg(all_jacobians, f"all_fixed_jacobians_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
        save_variable_with_l1_reg(all_unstable_eig_freq, f"all_fixed_unstable_eig_freq_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
    
    results['all_fixed_points'] = all_fixed_points
    results['all_jacobians'] = all_jacobians
    results['all_unstable_eig_freq'] = all_unstable_eig_freq
    
    # Generate fixed point summaries
    generate_point_summaries(point_type="fixed", all_points=all_fixed_points, 
                            all_unstable_eig_freq=all_unstable_eig_freq)
    
    # Visualize fixed point analysis (save in L1-specific directory)
    with l1_output_context(l1_reg_strength):
        # Temporarily update the config sparsity for plot titles (keep at 0.0 for L1 experiments)
        import _1_config
        original_s = _1_config.s
        _1_config.s = 0.0  # Fixed sparsity for L1 experiments
        
        try:
            analyze_jacobians_visualization(point_type="fixed", all_jacobians=all_jacobians)
            analyze_unstable_frequencies_visualization(point_type="fixed", 
                                                      all_unstable_eig_freq=all_unstable_eig_freq)
        finally:
            # Restore original sparsity value
            _1_config.s = original_s
    

    # ========================================
    # 7. SLOW POINT ANALYSIS (OPTIONAL)
    # ========================================
    all_slow_points = None
    all_slow_jacobians = None
    all_slow_unstable_eig_freq = None
    
    if SLOW_POINT_SEARCH:
        print("\n7. SLOW POINT ANALYSIS")
        print("-" * 40)
        
        with Timer("Slow Point Analysis"):
            # Find and analyze slow points
            all_slow_points, all_slow_jacobians, all_slow_unstable_eig_freq = find_and_analyze_points(
                state_traj_states, state_fixed_point_inits, trained_params, point_type="slow"
            )
            
            # Save slow point results
            save_variable_with_l1_reg(all_slow_points, f"all_slow_points_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
            save_variable_with_l1_reg(all_slow_jacobians, f"all_slow_jacobians_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
            save_variable_with_l1_reg(all_slow_unstable_eig_freq, f"all_slow_unstable_eig_freq_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
        
        # Generate slow point summaries
        generate_point_summaries(point_type="slow", all_points=all_slow_points, 
                                all_unstable_eig_freq=all_slow_unstable_eig_freq)
        
        # Visualize slow point analysis (save in L1-specific directory)
        with l1_output_context(l1_reg_strength):
            # Temporarily update the config sparsity for plot titles (keep at 0.0 for L1 experiments)
            import _1_config
            original_s = _1_config.s
            _1_config.s = 0.0  # Fixed sparsity for L1 experiments
            
            try:
                analyze_jacobians_visualization(point_type="slow", all_jacobians=all_slow_jacobians)
                analyze_unstable_frequencies_visualization(point_type="slow", 
                                                          all_unstable_eig_freq=all_slow_unstable_eig_freq)
            finally:
                # Restore original sparsity value
                _1_config.s = original_s
    
    results['all_slow_points'] = all_slow_points
    results['all_slow_jacobians'] = all_slow_jacobians
    results['all_slow_unstable_eig_freq'] = all_slow_unstable_eig_freq
    

    # ========================================
    # 8. PCA ANALYSIS WITH MULTIPLE TRUNCATIONS
    # ========================================
    print("\n8. PCA ANALYSIS WITH MULTIPLE TRUNCATIONS")
    print("-" * 40)
    
    # Define truncation levels
    truncation_levels = [0, 200, 400, 600]
    pca_results_all_truncations = {}
    
    with Timer("PCA Analysis (Multiple Truncations)"):
        for truncation in truncation_levels:
            print(f"\nRunning PCA with {truncation} truncations...")
            
            # Run PCA analysis (save plots in L1-specific directory)
            with l1_output_context(l1_reg_strength):
                # Temporarily update the config sparsity for plot titles (keep at 0.0 for L1 experiments)
                import _1_config
                original_s = _1_config.s
                _1_config.s = 0.0  # Fixed sparsity for L1 experiments
                
                try:
                    pca_results = run_pca_analysis(
                        state_traj_states, 
                        all_fixed_points=all_fixed_points,
                        all_slow_points=all_slow_points,
                        params=trained_params,
                        slow_point_search=SLOW_POINT_SEARCH,
                        skip_initial_steps=truncation
                    )
                finally:
                    # Restore original sparsity value
                    _1_config.s = original_s
            
            # Save PCA results for this truncation level
            pca_results_all_truncations[f"truncation_{truncation}"] = pca_results
            save_variable_with_l1_reg(pca_results, f"pca_results_truncation_{truncation}_l1_reg_{l1_reg_strength}", l1_reg_strength, s=0.0)
            
            print(f"Completed PCA analysis with {truncation} truncations")
    
    results['pca_results'] = pca_results_all_truncations
    
    print(f"\nCompleted PCA analysis with truncations: {truncation_levels}")
    print(f"Completed full analysis for L1 regularization {l1_reg_strength}: final loss = {final_loss:.6e}")
    
    return results




# =============================================================================
# =============================================================================
# EIGENVALUE EVOLUTION VISUALIZATION
# =============================================================================
# =============================================================================
def plot_eigenvalue_evolution_comparison(all_results, l1_reg_values):
    """
    Create visualization showing eigenvalue evolution during training for all L1 regularization levels.
    Handles both Jacobian and connectivity eigenvalues, with separate plots for each Jacobian frequency.
    
    Arguments:
        all_results: list of results dictionaries
        l1_reg_values: list of L1 regularization values tested
    """
    # Determine which eigenvalue types are present in the data
    has_jacobian = False
    has_connectivity = False
    jacobian_frequencies = []
    
    for result in all_results:
        if result['eigenvalue_data']['adam']:
            sample_data = result['eigenvalue_data']['adam'][0][1]  # First sample's eigenvalue data
            if 'jacobian' in sample_data:
                has_jacobian = True
                # Extract frequency indices from Jacobian data
                if isinstance(sample_data['jacobian'], dict):
                    jacobian_frequencies = list(sample_data['jacobian'].keys())
                    break
            if 'connectivity' in sample_data:
                has_connectivity = True
    
    if not has_jacobian and not has_connectivity:
        print("No eigenvalue data found to plot.")
        return
    
    # Color map for showing temporal evolution
    cmap = plt.get_cmap('viridis')
    
    # Plot connectivity eigenvalues if present
    if has_connectivity:
        plot_connectivity_eigenvalue_evolution(all_results, l1_reg_values, cmap)
    
    # Plot Jacobian eigenvalues for each frequency if present
    if has_jacobian and jacobian_frequencies:
        for freq_idx in jacobian_frequencies:
            # Create separate plot for each L1 regularization level
            for l1_reg_idx, result in enumerate(all_results):
                plot_jacobian_eigenvalue_evolution_for_single_l1_reg(result, freq_idx, cmap)


def plot_connectivity_eigenvalue_evolution(all_results, l1_reg_values, cmap):
    """
    Plot connectivity matrix eigenvalue evolution for all L1 regularization levels.
    """
    n_l1_reg = len(l1_reg_values)
    fig, axes = plt.subplots(1, n_l1_reg, figsize=(5*n_l1_reg, 5))
    if n_l1_reg == 1:
        axes = [axes]
    
    # Store scatter plot for colorbar
    scatter_plot = None
    
    for l1_reg_idx, result in enumerate(all_results):
        l1_reg = result['l1_reg_strength']
        eigenvalue_data = result['eigenvalue_data']
        final_loss = result['final_loss']
        
        ax = axes[l1_reg_idx]
        
        # Combine eigenvalue data from both training phases
        all_iterations = []
        all_eigenvals = []
        
        # Add Adam phase data
        for iteration, eigenval_dict in eigenvalue_data['adam']:
            if 'connectivity' in eigenval_dict and len(eigenval_dict['connectivity']) > 0:
                eigenvals = eigenval_dict['connectivity']
                all_iterations.extend([iteration] * len(eigenvals))
                all_eigenvals.extend(eigenvals)
        
        # Add L-BFGS phase data (offset iterations)
        adam_max_iter = NUM_EPOCHS_ADAM if eigenvalue_data['adam'] else 0
        for iteration, eigenval_dict in eigenvalue_data['lbfgs']:
            if 'connectivity' in eigenval_dict and len(eigenval_dict['connectivity']) > 0:
                eigenvals = eigenval_dict['connectivity']
                all_iterations.extend([adam_max_iter + iteration] * len(eigenvals))
                all_eigenvals.extend(eigenvals)
        
        if len(all_eigenvals) == 0:
            ax.text(0.5, 0.5, 'No connectivity\neigenvalue data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'L1 reg = {l1_reg:.1e}')
            continue
        
        # Convert to numpy arrays
        all_iterations = np.array(all_iterations)
        all_eigenvals = np.array(all_eigenvals)
        
        # Normalize iterations for color mapping
        if len(np.unique(all_iterations)) > 1:
            norm_iterations = (all_iterations - all_iterations.min()) / (all_iterations.max() - all_iterations.min())
        else:
            norm_iterations = np.zeros_like(all_iterations)
        
        # Plot eigenvalues with color representing training progress
        scatter = ax.scatter(
            all_eigenvals.real, 
            all_eigenvals.imag,
            c=norm_iterations,
            cmap=cmap,
            alpha=0.7,
            s=20
        )
        
        # Store the first valid scatter plot for colorbar
        if scatter_plot is None and len(all_eigenvals) > 0:
            scatter_plot = scatter
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'L1 reg = {l1_reg:.1e}\nFinal loss: {final_loss:.2e}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Add colorbar to the right of all subplots
    if scatter_plot is not None:
        plt.tight_layout()
        # Create space for colorbar
        fig.subplots_adjust(right=0.85)
        # Add colorbar
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Training Progress')
    
    plt.suptitle('Connectivity Matrix Eigenvalue Evolution During Training', fontsize=16)
    
    # Create L1-specific filename
    l1_reg_str = '_'.join([f'{l1:.1e}'.replace('.', 'p').replace('-', 'm').replace('+', 'p') for l1 in l1_reg_values])
    save_figure(fig, f'connectivity_eigenvalue_evolution_l1_reg_{l1_reg_str}')
    # plt.show()  # Commented out to prevent interactive display


def plot_jacobian_eigenvalue_evolution_for_single_l1_reg(result, freq_idx, cmap):
    """
    Plot Jacobian eigenvalue evolution for a specific frequency and single L1 regularization level.
    """
    l1_reg = result['l1_reg_strength']
    eigenvalue_data = result['eigenvalue_data']
    final_loss = result['final_loss']
    
    # Get frequency information for plot title
    freq_value = float(omegas[freq_idx])
    static_input_value = float(static_inputs[freq_idx])
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Combine eigenvalue data from both training phases
    all_iterations = []
    all_eigenvals = []
    
    # Add Adam phase data
    for iteration, eigenval_dict in eigenvalue_data['adam']:
        if ('jacobian' in eigenval_dict and 
            isinstance(eigenval_dict['jacobian'], dict) and 
            freq_idx in eigenval_dict['jacobian'] and
            len(eigenval_dict['jacobian'][freq_idx]) > 0):
            
            eigenvals = eigenval_dict['jacobian'][freq_idx]
            all_iterations.extend([iteration] * len(eigenvals))
            all_eigenvals.extend(eigenvals)
    
    # Add L-BFGS phase data (offset iterations)
    adam_max_iter = NUM_EPOCHS_ADAM if eigenvalue_data['adam'] else 0
    for iteration, eigenval_dict in eigenvalue_data['lbfgs']:
        if ('jacobian' in eigenval_dict and 
            isinstance(eigenval_dict['jacobian'], dict) and 
            freq_idx in eigenval_dict['jacobian'] and
            len(eigenval_dict['jacobian'][freq_idx]) > 0):
            
            eigenvals = eigenval_dict['jacobian'][freq_idx]
            all_iterations.extend([adam_max_iter + iteration] * len(eigenvals))
            all_eigenvals.extend(eigenvals)
    
    if len(all_eigenvals) == 0:
        ax.text(0.5, 0.5, f'No Jacobian eigenvalue\ndata for freq {freq_idx}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'L1 reg = {l1_reg:.1e}\nFreq {freq_idx}: ω = {freq_value:.3f} rad/s')
        plt.tight_layout()
        
        # Save with L1-specific filename in L1 subdirectory
        l1_reg_str = f'{l1_reg:.1e}'.replace('.', 'p').replace('-', 'm').replace('+', 'p')
        freq_str = f'{freq_value:.3f}'.replace('.', 'p')
        save_figure_with_l1_reg(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_l1_reg_{l1_reg_str}_omega_{freq_str}', l1_reg)
        # plt.show()  # Commented out to prevent interactive display
        plt.close(fig)
        return
    
    # Convert to numpy arrays
    all_iterations = np.array(all_iterations)
    all_eigenvals = np.array(all_eigenvals)
    
    # Normalize iterations for color mapping
    if len(np.unique(all_iterations)) > 1:
        norm_iterations = (all_iterations - all_iterations.min()) / (all_iterations.max() - all_iterations.min())
    else:
        norm_iterations = np.zeros_like(all_iterations)
    
    # Plot eigenvalues with color representing training progress
    scatter = ax.scatter(
        all_eigenvals.real, 
        all_eigenvals.imag,
        c=norm_iterations,
        cmap=cmap,
        alpha=0.7,
        s=20
    )
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Jacobian Eigenvalue Evolution During Training\n'
                 f'L1 reg = {l1_reg:.1e} | Frequency {freq_idx}: ω = {freq_value:.3f} rad/s\n'
                 f'Static input = {static_input_value:.3f} | Final loss: {final_loss:.2e}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Progress')
    
    plt.tight_layout()
    
    # Save with L1-specific filename in L1 subdirectory
    l1_reg_str = f'{l1_reg:.1e}'.replace('.', 'p').replace('-', 'm').replace('+', 'p')
    freq_str = f'{freq_value:.3f}'.replace('.', 'p')
    save_figure_with_l1_reg(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_l1_reg_{l1_reg_str}_omega_{freq_str}', l1_reg)
    # plt.show()  # Commented out to prevent interactive display
    plt.close(fig)


def plot_eigenvalue_evolution(all_results, l1_reg_values):
    """
    Backward compatibility function - calls the new comparison function.
    """
    plot_eigenvalue_evolution_comparison(all_results, l1_reg_values)




# =============================================================================
# =============================================================================
# SUMMARY PLOTTING FUNCTIONS
# =============================================================================
# =============================================================================
def create_l1_reg_summary_plots(all_results, l1_reg_values):
    """
    Create summary plots comparing different metrics across L1 regularization levels.
    
    Arguments:
        all_results: list of results dictionaries
        l1_reg_values: list of L1 regularization values tested
    """
    # Extract summary metrics
    final_losses = []
    num_fixed_points = []
    num_slow_points = []
    spectral_radii = []
    
    for result in all_results:
        final_losses.append(result['final_loss'])
        
        # Count fixed points
        if result['all_fixed_points']:
            num_fps = sum(len(task_fps) for task_fps in result['all_fixed_points'])
            num_fixed_points.append(num_fps)
        else:
            num_fixed_points.append(0)
        
        # Count slow points
        if result['all_slow_points']:
            num_sps = sum(len(task_sps) for task_sps in result['all_slow_points'])
            num_slow_points.append(num_sps)
        else:
            num_slow_points.append(0)
        
        # Compute spectral radius of connectivity matrix
        J = result['trained_params']['J']
        eigenvals = np.linalg.eigvals(np.array(J))
        spectral_radius = np.max(np.abs(eigenvals))
        spectral_radii.append(spectral_radius)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Final loss vs L1 regularization
    axes[0, 0].semilogx(l1_reg_values, final_losses, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('L1 Regularization')
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Training Loss vs L1 Regularization')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of fixed points vs L1 regularization
    axes[0, 1].semilogx(l1_reg_values, num_fixed_points, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('L1 Regularization')
    axes[0, 1].set_ylabel('Number of Fixed Points')
    axes[0, 1].set_title('Fixed Points vs L1 Regularization')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral radius vs L1 regularization
    axes[1, 0].semilogx(l1_reg_values, spectral_radii, 'go-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Unit circle')
    axes[1, 0].set_xlabel('L1 Regularization')
    axes[1, 0].set_ylabel('Spectral Radius')
    axes[1, 0].set_title('Spectral Radius vs L1 Regularization')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Slow points vs L1 regularization (if applicable)
    if any(n > 0 for n in num_slow_points):
        axes[1, 1].semilogx(l1_reg_values, num_slow_points, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('L1 Regularization')
        axes[1, 1].set_ylabel('Number of Slow Points')
        axes[1, 1].set_title('Slow Points vs L1 Regularization')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No slow point analysis\nperformed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Slow Points vs L1 Regularization')
    
    plt.suptitle('L1 Regularization Experiment Summary', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"l1_reg_summary_plots_{timestamp}"
    save_figure(fig, filename)
    # plt.show()  # Commented out to prevent interactive display


def create_unstable_eigenvalue_table(all_results, l1_reg_values):
    """
    Create a table showing the distribution of unstable eigenvalues per fixed point across L1 regularization levels.
    
    Arguments:
        all_results: list of results dictionaries
        l1_reg_values: list of L1 regularization values tested
    """
    # Collect unstable eigenvalue counts for each L1 regularization level
    unstable_counts_per_l1_reg = {}
    max_unstable_count = 0
    
    for result in all_results:
        l1_reg = result['l1_reg_strength']
        unstable_counts = []
        
        # Extract unstable eigenvalue counts from fixed points
        if result['all_unstable_eig_freq'] and len(result['all_unstable_eig_freq']) > 0:
            for task_unstable_freqs in result['all_unstable_eig_freq']:
                if task_unstable_freqs:  # Check if not None/empty
                    for fp_unstable_freqs in task_unstable_freqs:
                        if fp_unstable_freqs is not None:
                            # Count unstable eigenvalues for this fixed point
                            num_unstable = len(fp_unstable_freqs)
                            unstable_counts.append(num_unstable)
                            max_unstable_count = max(max_unstable_count, num_unstable)
        
        unstable_counts_per_l1_reg[l1_reg] = unstable_counts
    
    # If no unstable eigenvalues found, create a simple message plot
    if max_unstable_count == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No unstable eigenvalues found\nacross any L1 regularization levels', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Unstable Eigenvalue Distribution Table')
        plt.tight_layout()
        
        l1_reg_str = '_'.join([f'{l1:.1e}'.replace('.', 'p').replace('-', 'm').replace('+', 'p') for l1 in l1_reg_values])
        save_figure(fig, f'unstable_eigenvalue_table_l1_reg_{l1_reg_str}')
        plt.close(fig)
        return
    
    # Create the table: rows = number of unstable eigenvalues, cols = L1 regularization levels
    table_data = np.zeros((max_unstable_count + 1, len(l1_reg_values)), dtype=int)
    
    for l1_reg_idx, l1_reg in enumerate(l1_reg_values):
        unstable_counts = unstable_counts_per_l1_reg[l1_reg]
        
        # Count how many fixed points have each number of unstable eigenvalues
        for num_unstable in range(max_unstable_count + 1):
            count = unstable_counts.count(num_unstable)
            table_data[num_unstable, l1_reg_idx] = count
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, len(l1_reg_values) * 1.5), max(6, max_unstable_count * 0.8)))
    
    # Create heatmap
    im = ax.imshow(table_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(l1_reg_values)))
    ax.set_yticks(np.arange(max_unstable_count + 1))
    ax.set_xticklabels([f'{l1:.1e}' for l1 in l1_reg_values], rotation=45)
    ax.set_yticklabels([f'{i}' for i in range(max_unstable_count + 1)])
    
    # Labels and title
    ax.set_xlabel('L1 Regularization Level')
    ax.set_ylabel('Number of Unstable Eigenvalues per Fixed Point')
    ax.set_title('Distribution of Unstable Eigenvalues per Fixed Point Across L1 Regularization Levels')
    
    # Add text annotations
    for i in range(max_unstable_count + 1):
        for j in range(len(l1_reg_values)):
            text = ax.text(j, i, f'{table_data[i, j]}',
                         ha="center", va="center", color="black" if table_data[i, j] < table_data.max()/2 else "white",
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Fixed Points')
    
    plt.tight_layout()
    
    # Save with L1-specific filename
    l1_reg_str = '_'.join([f'{l1:.1e}'.replace('.', 'p').replace('-', 'm').replace('+', 'p') for l1 in l1_reg_values])
    save_figure(fig, f'unstable_eigenvalue_table_l1_reg_{l1_reg_str}')
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("UNSTABLE EIGENVALUE DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for l1_reg_idx, l1_reg in enumerate(l1_reg_values):
        unstable_counts = unstable_counts_per_l1_reg[l1_reg]
        total_fps = len(unstable_counts)
        if total_fps > 0:
            avg_unstable = np.mean(unstable_counts)
            max_unstable_this_l1_reg = max(unstable_counts) if unstable_counts else 0
            print(f"L1 reg {l1_reg:.1e}: {total_fps} fixed points, "
                  f"avg {avg_unstable:.1f} unstable eigenvalues, max {max_unstable_this_l1_reg}")
        else:
            print(f"L1 reg {l1_reg:.1e}: No fixed points found")
    print(f"{'='*60}")

    


# =============================================================================
# =============================================================================
# MAIN EXECUTION
# =============================================================================
# =============================================================================
def main():
    """
    Main function to run comprehensive L1 regularization experiments with full analysis.
    """
    # Get L1 regularization values from user
    l1_reg_values = get_l1_reg_values()
    
    # Get eigenvalue computation preferences from user
    compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies = get_eigenvalue_options()
    
    # Setup custom output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"L1_Regularization_Experiments_{timestamp}"
    output_base_path = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs"
    full_output_path = os.path.join(output_base_path, output_dir_name)
    
    # Set custom output directory for all saves
    set_custom_output_dir(full_output_path)
    
    # Show experiment configuration
    eigenval_types = []
    if compute_jacobian_eigenvals:
        if jacobian_frequencies is not None:
            eigenval_types.append(f"Jacobian (frequencies: {jacobian_frequencies})")
        else:
            eigenval_types.append("Jacobian")
    if compute_connectivity_eigenvals:
        eigenval_types.append("Connectivity Matrix")
    eigenval_msg = f"Eigenvalue types: {', '.join(eigenval_types)}" if eigenval_types else "No eigenvalue tracking"
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE L1 REGULARIZATION EXPERIMENTS")
    print(f"Testing {len(l1_reg_values)} L1 regularization strengths")
    print(f"Fixed sparsity: s = 0.0 (no structural sparsity)")
    print(f"{eigenval_msg}")
    if eigenval_types:
        print(f"Eigenvalue sampling frequency: every {EIGENVALUE_SAMPLE_FREQUENCY} iterations")
    print(f"Output directory: {full_output_path}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(NUMPY_SEED)
    key = random.PRNGKey(JAX_SEED)
    
    # Generate a single set of base parameters that will be used for all L1 regularization strengths
    # This ensures all networks start from the same underlying structure, only varying in L1 regularization
    analysis_key, base_key = random.split(key)
    
    # Store results for all L1 regularization strengths
    all_results = []
    
    for i, l1_reg in enumerate(l1_reg_values):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING L1 REGULARIZATION STRENGTH {i+1}/{len(l1_reg_values)}: {l1_reg}")
        print(f"{'='*80}")
        
        # Use the same base key for all L1 regularization strengths to ensure consistent initial parameters
        mask, params = init_params(base_key, l1_reg)
        
        # Generate a unique analysis key for this L1 regularization strength
        analysis_key, subkey = random.split(analysis_key)
        
        # Run full training and analysis pipeline
        results = train_with_full_analysis(params, mask, subkey, 
                                          compute_jacobian_eigenvals, 
                                          compute_connectivity_eigenvals, 
                                          sparsity_value=0.0, 
                                          l1_reg_strength=l1_reg,
                                          jacobian_frequencies=jacobian_frequencies)
        
        # Store results
        all_results.append(results)
        
        print(f"\nCompleted L1 regularization {l1_reg}: final loss = {results['final_loss']:.6e}")
        
        # Save individual L1 regularization results
        save_variable_with_l1_reg(results, f"complete_results_l1_reg_{l1_reg}", l1_reg, s=0.0)
    

    # ========================================
    # CREATE COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION
    # ========================================
    if compute_jacobian_eigenvals or compute_connectivity_eigenvals:
        print(f"\n{'='*60}")
        print("CREATING COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION")
        print(f"{'='*60}")
        
        plot_eigenvalue_evolution_comparison(all_results, l1_reg_values)
    else:
        print(f"\n{'='*60}")
        print("SKIPPING EIGENVALUE VISUALIZATION (NO EIGENVALUE DATA)")
        print(f"{'='*60}")
    

    # ========================================
    # CREATE COMPREHENSIVE SUMMARY PLOTS
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE SUMMARY VISUALIZATIONS")
    print(f"{'='*60}")
    
    # plot_loss_evolution_comparison(all_results, l1_reg_values)  # Function not implemented
    create_l1_reg_summary_plots(all_results, l1_reg_values)
    

    # ========================================
    # CREATE UNSTABLE EIGENVALUE DISTRIBUTION TABLE
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING UNSTABLE EIGENVALUE DISTRIBUTION TABLE")
    print(f"{'='*60}")
    
    create_unstable_eigenvalue_table(all_results, l1_reg_values)
    

    # ========================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================
    print(f"\n{'='*60}")
    print("SAVING COMPLETE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    # Save all results together
    complete_experiment_results = {
        'l1_reg_values': l1_reg_values,
        'all_results': all_results,
        'experiment_config': {
            'eigenvalue_sample_frequency': EIGENVALUE_SAMPLE_FREQUENCY,
            'compute_jacobian_eigenvals': compute_jacobian_eigenvals,
            'compute_connectivity_eigenvals': compute_connectivity_eigenvals,
            'jacobian_frequencies': jacobian_frequencies,
            'num_epochs_adam': NUM_EPOCHS_ADAM,
            'num_epochs_lbfgs': NUM_EPOCHS_LBFGS,
            'network_size': N,
            'num_tasks': num_tasks,
            'slow_point_search': SLOW_POINT_SEARCH,
            'fixed_sparsity': 0.0,  # L1 experiments use fixed sparsity
            'timestamp': timestamp,
            'output_directory': full_output_path
        }
    }
    save_variable(complete_experiment_results, f"complete_l1_regularization_experiment_{timestamp}")
    

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*80}")
    print("COMPREHENSIVE L1 REGULARIZATION EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Experiment timestamp: {timestamp}")
    print(f"Output directory: {full_output_path}")
    print(f"Tested L1 regularization values: {l1_reg_values}")
    print(f"Fixed sparsity: s = 0.0")
    print(f"Eigenvalue tracking: {eigenval_msg}")
    print("Final losses by L1 regularization:")
    for result in all_results:
        print(f"  L1 reg = {result['l1_reg_strength']:.1e}: loss = {result['final_loss']:.6e}")
    
    print(f"\nNumber of fixed points found by L1 regularization:")
    for result in all_results:
        if result['all_fixed_points']:
            num_fps = sum(len(task_fps) for task_fps in result['all_fixed_points'])
            print(f"  L1 reg = {result['l1_reg_strength']:.1e}: {num_fps} fixed points")
    
    if SLOW_POINT_SEARCH:
        print(f"\nNumber of slow points found by L1 regularization:")
        for result in all_results:
            if result['all_slow_points']:
                num_sps = sum(len(task_sps) for task_sps in result['all_slow_points'])
                print(f"  L1 reg = {result['l1_reg_strength']:.1e}: {num_sps} slow points")
    
    print(f"\nAll results and visualizations saved to: {full_output_path}")


if __name__ == "__main__":
    main()