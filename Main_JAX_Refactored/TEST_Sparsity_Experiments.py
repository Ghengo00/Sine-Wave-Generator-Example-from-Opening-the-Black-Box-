"""
Sparsity experiments: Train RNN with different sparsity levels and track Jacobian eigenvalue evolution.

This script trains RNNs at different sparsity levels and visualizes how the eigenvalues
of the Jacobian matrix at fixed points evolve during the optimization process.
"""


import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from functools import partial
from tqdm import tqdm
import time
import os

# Import all modules
from _1_config import (
    N, I, num_tasks, s, dt, omegas, static_inputs,
    NUM_EPOCHS_ADAM, NUM_EPOCHS_LBFGS, LOSS_THRESHOLD,
    NUMPY_SEED, JAX_SEED, EIGENVALUE_SAMPLE_FREQUENCY,
    TOL, MAXITER, GAUSSIAN_STD, NUM_ATTEMPTS, SLOW_POINT_SEARCH
)
from _2_utils import Timer, save_variable, set_custom_output_dir, get_output_dir
from _3_data_generation import generate_all_inputs_and_targets
from _4_rnn_model import init_params, compute_driving_final_states, run_batch_diagnostics
from _5_training import setup_optimizers, create_training_functions_with_state_chaining
from _6_analysis import find_points, compute_jacobian, find_and_analyze_points, generate_point_summaries
from _7_visualization import (save_figure, plot_trajectories_vs_targets, plot_parameter_matrices_for_tasks, 
                          analyze_jacobians_visualization, analyze_unstable_frequencies_visualization)
from _8_pca_analysis import run_pca_analysis


def get_sparsity_values():
    """
    Get sparsity values from user input.
    
    Returns:
        list: List of sparsity values to test
    """
    print("=" * 60)
    print("SPARSITY EIGENVALUE EXPERIMENT SETUP")
    print("=" * 60)
    
    print("\nEnter sparsity values to test (between 0.0 and 1.0)")
    print("Examples: 0.0, 0.2, 0.5, 0.8")
    print("Enter values separated by commas (e.g., '0.0, 0.2, 0.5')")
    
    while True:
        try:
            user_input = input("\nSparsity values: ").strip()
            sparsity_values = [float(x.strip()) for x in user_input.split(',')]
            
            # Validate sparsity values
            for s_val in sparsity_values:
                if not (0.0 <= s_val <= 1.0):
                    raise ValueError(f"Sparsity value {s_val} must be between 0.0 and 1.0")
            
            print(f"\nWill test sparsity values: {sparsity_values}")
            confirm = input("Proceed? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return sparsity_values
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter valid comma-separated numbers between 0.0 and 1.0")


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


def modified_scan_with_eigenvalues(params, opt_state, step_fn, num_steps, tag, sample_frequency, 
                                   compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Modified training loop that periodically samples eigenvalues.
    
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
    else:
        print(f"Starting {tag} optimization (no eigenvalue sampling)...")
    
    # Initialize tracking variables
    best_loss = jnp.inf
    best_params = params
    eigenvalue_history = []
    
    # Sample eigenvalues at iteration 0 if requested
    if compute_any_eigenvals:
        print(f"[{tag}] Sampling eigenvalues at iteration 0...")
        initial_eigenval_data = compute_eigenvalues_during_training(params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
        eigenvalue_history.append((0, initial_eigenval_data))
    
    # Training loop
    current_params = params
    current_opt_state = opt_state
    
    for step in tqdm(range(num_steps), desc=f"{tag} Training"):
        # Perform optimization step
        current_params, current_opt_state, loss = step_fn(current_params, current_opt_state)
        
        # Update best parameters if improved
        if loss < best_loss:
            best_loss = loss
            best_params = current_params
        
        # Sample eigenvalues at specified frequency if requested
        if compute_any_eigenvals and (step + 1) % sample_frequency == 0:
            print(f"[{tag}] step {step + 1}: loss {loss:.6e} - Sampling eigenvalues...")
            eigenval_data = compute_eigenvalues_during_training(current_params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
            eigenvalue_history.append((step + 1, eigenval_data))
    
    return best_params, best_loss, current_params, current_opt_state, eigenvalue_history


def train_with_full_analysis(params, mask, sparsity_value, key, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Train model with full analysis pipeline (replicating _0_main_refactored.py functionality).
    
    Arguments:
        params: initial parameters
        mask: sparsity mask
        sparsity_value: current sparsity level being tested
        key: JAX random key for analysis
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues during training
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues during training
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        results_dict: comprehensive results dictionary containing all analysis outputs
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SPARSITY s = {sparsity_value}")
    print(f"{'='*60}")
    
    results = {}
    results['sparsity'] = sparsity_value
    
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Compute driving phase final states
    print("Computing driving phase final states...")
    driving_final_states = compute_driving_final_states(params)
    print(f"Driving phase completed. States shape: {driving_final_states.shape}")
    
    # Create training functions with state chaining
    adam_step, lbfgs_step = create_training_functions_with_state_chaining(adam_opt, lbfgs_opt, mask)
    
    # Track eigenvalues during training
    eigenvalue_data = {}
    
    # =============================================================================
    # TRAINING WITH EIGENVALUE TRACKING
    # =============================================================================
    print("\n2. TRAINING WITH EIGENVALUE TRACKING")
    print("-" * 40)
    
    with Timer(f"Training with sparsity {sparsity_value}"):
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
    
    # =============================================================================
    # POST-TRAINING DIAGNOSTICS
    # =============================================================================
    print("\n3. POST-TRAINING DIAGNOSTICS")
    print("-" * 40)
    
    # Collect final trajectories and fixed-point initializations
    final_loss_check, state_traj_states, state_fixed_point_inits = run_batch_diagnostics(trained_params, key)
    print(f"Final loss after training: {final_loss_check:.6e}")
    
    results['state_traj_states'] = state_traj_states
    results['state_fixed_point_inits'] = state_fixed_point_inits
    
    # =============================================================================
    # SAVE TRAINING RESULTS
    # =============================================================================
    print("\n4. SAVING TRAINING RESULTS")
    print("-" * 40)
    
    print("Saving training results...")
    save_variable(np.array(trained_params["J"]), f"J_param_sparsity_{sparsity_value}", s=sparsity_value)
    save_variable(np.array(trained_params["B"]), f"B_param_sparsity_{sparsity_value}", s=sparsity_value)
    save_variable(np.array(trained_params["b_x"]), f"b_x_param_sparsity_{sparsity_value}", s=sparsity_value)
    save_variable(np.array(trained_params["w"]), f"w_param_sparsity_{sparsity_value}", s=sparsity_value)
    save_variable(np.array(trained_params["b_z"]), f"b_z_param_sparsity_{sparsity_value}", s=sparsity_value)

    state_dict = {
        "traj_states": state_traj_states,
        "fixed_point_inits": state_fixed_point_inits
    }
    save_variable(state_dict, f"state_sparsity_{sparsity_value}", s=sparsity_value)
    
    # =============================================================================
    # VISUALIZATION OF BASIC RESULTS
    # =============================================================================
    print("\n5. VISUALIZATION OF BASIC RESULTS")
    print("-" * 40)
    
    # Plot trajectories vs targets
    print("Plotting produced trajectories vs. target signals...")
    plot_trajectories_vs_targets(trained_params)
    
    # Plot parameter matrices
    plot_parameter_matrices_for_tasks(trained_params)
    
    # =============================================================================
    # FIXED POINT ANALYSIS
    # =============================================================================
    print("\n6. FIXED POINT ANALYSIS")
    print("-" * 40)
    
    with Timer("Fixed Point Analysis"):
        # Find and analyze fixed points
        all_fixed_points, all_jacobians, all_unstable_eig_freq = find_and_analyze_points(
            state_traj_states, state_fixed_point_inits, trained_params, point_type="fixed"
        )
        
        # Save fixed point results
        save_variable(all_fixed_points, f"all_fixed_points_sparsity_{sparsity_value}", s=sparsity_value)
        save_variable(all_jacobians, f"all_fixed_jacobians_sparsity_{sparsity_value}", s=sparsity_value)
        save_variable(all_unstable_eig_freq, f"all_fixed_unstable_eig_freq_sparsity_{sparsity_value}", s=sparsity_value)
    
    results['all_fixed_points'] = all_fixed_points
    results['all_jacobians'] = all_jacobians
    results['all_unstable_eig_freq'] = all_unstable_eig_freq
    
    # Generate fixed point summaries
    generate_point_summaries(point_type="fixed", all_points=all_fixed_points, 
                            all_unstable_eig_freq=all_unstable_eig_freq)
    
    # Visualize fixed point analysis
    analyze_jacobians_visualization(point_type="fixed", all_jacobians=all_jacobians)
    analyze_unstable_frequencies_visualization(point_type="fixed", 
                                              all_unstable_eig_freq=all_unstable_eig_freq)
    
    # =============================================================================
    # SLOW POINT ANALYSIS (OPTIONAL)
    # =============================================================================
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
            save_variable(all_slow_points, f"all_slow_points_sparsity_{sparsity_value}", s=sparsity_value)
            save_variable(all_slow_jacobians, f"all_slow_jacobians_sparsity_{sparsity_value}", s=sparsity_value)
            save_variable(all_slow_unstable_eig_freq, f"all_slow_unstable_eig_freq_sparsity_{sparsity_value}", s=sparsity_value)
        
        # Generate slow point summaries
        generate_point_summaries(point_type="slow", all_points=all_slow_points, 
                                all_unstable_eig_freq=all_slow_unstable_eig_freq)
        
        # Visualize slow point analysis
        analyze_jacobians_visualization(point_type="slow", all_jacobians=all_slow_jacobians)
        analyze_unstable_frequencies_visualization(point_type="slow", 
                                                  all_unstable_eig_freq=all_slow_unstable_eig_freq)
    
    results['all_slow_points'] = all_slow_points
    results['all_slow_jacobians'] = all_slow_jacobians
    results['all_slow_unstable_eig_freq'] = all_slow_unstable_eig_freq
    
    # =============================================================================
    # PCA ANALYSIS
    # =============================================================================
    print("\n8. PCA ANALYSIS")
    print("-" * 40)
    
    with Timer("PCA Analysis"):
        # Run PCA analysis
        pca_results = run_pca_analysis(
            state_traj_states, 
            all_fixed_points=all_fixed_points,
            all_slow_points=all_slow_points,
            params=trained_params,
            slow_point_search=SLOW_POINT_SEARCH
        )
        
        # Save PCA results
        save_variable(pca_results, f"pca_results_sparsity_{sparsity_value}", s=sparsity_value)
    
    results['pca_results'] = pca_results
    
    print(f"\nCompleted full analysis for sparsity {sparsity_value}: final loss = {final_loss:.6e}")
    
    return results


def plot_eigenvalue_evolution_comparison(all_results, sparsity_values):
    """
    Create visualization showing eigenvalue evolution during training for all sparsity levels.
    Handles both Jacobian and connectivity eigenvalues, with separate plots for each Jacobian frequency.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
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
        plot_connectivity_eigenvalue_evolution(all_results, sparsity_values, cmap)
    
    # Plot Jacobian eigenvalues for each frequency if present
    if has_jacobian and jacobian_frequencies:
        for freq_idx in jacobian_frequencies:
            plot_jacobian_eigenvalue_evolution_for_frequency(all_results, sparsity_values, freq_idx, cmap)


def plot_connectivity_eigenvalue_evolution(all_results, sparsity_values, cmap):
    """
    Plot connectivity matrix eigenvalue evolution for all sparsity levels.
    """
    n_sparsity = len(sparsity_values)
    fig, axes = plt.subplots(1, n_sparsity, figsize=(5*n_sparsity, 5))
    if n_sparsity == 1:
        axes = [axes]
    
    for sparsity_idx, result in enumerate(all_results):
        sparsity = result['sparsity']
        eigenvalue_data = result['eigenvalue_data']
        final_loss = result['final_loss']
        
        ax = axes[sparsity_idx]
        
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
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
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
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Sparsity s = {sparsity:.2f}\nFinal loss: {final_loss:.2e}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar for first subplot
        if sparsity_idx == 0 and len(all_eigenvals) > 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Progress')
    
    plt.suptitle('Connectivity Matrix Eigenvalue Evolution During Training', fontsize=16)
    plt.tight_layout()
    save_figure('connectivity_eigenvalue_evolution_comparison')
    plt.show()


def plot_jacobian_eigenvalue_evolution_for_frequency(all_results, sparsity_values, freq_idx, cmap):
    """
    Plot Jacobian eigenvalue evolution for a specific frequency across all sparsity levels.
    """
    n_sparsity = len(sparsity_values)
    fig, axes = plt.subplots(1, n_sparsity, figsize=(5*n_sparsity, 5))
    if n_sparsity == 1:
        axes = [axes]
    
    # Get frequency information for plot title
    freq_value = float(omegas[freq_idx])
    static_input_value = float(static_inputs[freq_idx])
    
    for sparsity_idx, result in enumerate(all_results):
        sparsity = result['sparsity']
        eigenvalue_data = result['eigenvalue_data']
        final_loss = result['final_loss']
        
        ax = axes[sparsity_idx]
        
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
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
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
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Sparsity s = {sparsity:.2f}\nFinal loss: {final_loss:.2e}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar for first subplot
        if sparsity_idx == 0 and len(all_eigenvals) > 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Progress')
    
    plt.suptitle(f'Jacobian Eigenvalue Evolution During Training\n'
                 f'Frequency {freq_idx}: ω = {freq_value:.3f} rad/s, static_input = {static_input_value:.3f}', 
                 fontsize=16)
    plt.tight_layout()
    save_figure(f'jacobian_eigenvalue_evolution_freq_{freq_idx}_omega_{freq_value:.3f}')
    plt.show()


def create_sparsity_summary_plots(all_results, sparsity_values):
    """
    Create summary plots comparing different metrics across sparsity levels.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
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
    
    # Plot 1: Final loss vs sparsity
    axes[0, 0].plot(sparsity_values, final_losses, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Sparsity')
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Training Loss vs Sparsity')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of fixed points vs sparsity
    axes[0, 1].plot(sparsity_values, num_fixed_points, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sparsity')
    axes[0, 1].set_ylabel('Number of Fixed Points')
    axes[0, 1].set_title('Fixed Points vs Sparsity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral radius vs sparsity
    axes[1, 0].plot(sparsity_values, spectral_radii, 'go-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Unit circle')
    axes[1, 0].set_xlabel('Sparsity')
    axes[1, 0].set_ylabel('Spectral Radius')
    axes[1, 0].set_title('Spectral Radius vs Sparsity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Slow points vs sparsity (if applicable)
    if any(n > 0 for n in num_slow_points):
        axes[1, 1].plot(sparsity_values, num_slow_points, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Sparsity')
        axes[1, 1].set_ylabel('Number of Slow Points')
        axes[1, 1].set_title('Slow Points vs Sparsity')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No slow point analysis\nperformed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Slow Points vs Sparsity')
    
    plt.suptitle('Sparsity Experiment Summary', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"sparsity_summary_plots_{timestamp}"
    save_figure(fig, filename)
    plt.show()


def plot_eigenvalue_evolution(all_results, sparsity_values):
    """
    Backward compatibility function - calls the new comparison function.
    """
    plot_eigenvalue_evolution_comparison(all_results, sparsity_values)


def main():
    """
    Main function to run comprehensive sparsity experiments with full analysis.
    """
    # Get sparsity values from user
    sparsity_values = get_sparsity_values()
    
    # Get eigenvalue computation preferences from user
    compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies = get_eigenvalue_options()
    
    # Setup custom output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"Sparsity_Experiments_{timestamp}"
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
    print("COMPREHENSIVE SPARSITY EXPERIMENTS")
    print(f"Testing {len(sparsity_values)} sparsity levels")
    print(f"{eigenval_msg}")
    if eigenval_types:
        print(f"Eigenvalue sampling frequency: every {EIGENVALUE_SAMPLE_FREQUENCY} iterations")
    print(f"Output directory: {full_output_path}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(NUMPY_SEED)
    key = random.PRNGKey(JAX_SEED)
    
    # Store results for all sparsity levels
    all_results = []
    
    for i, sparsity in enumerate(sparsity_values):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING SPARSITY LEVEL {i+1}/{len(sparsity_values)}: s = {sparsity}")
        print(f"{'='*80}")
        
        # Generate new random key for this sparsity level
        key, subkey = random.split(key)
        
        # Initialize parameters with current sparsity level
        mask, params = init_params_with_sparsity(subkey, sparsity)
        
        # Run full training and analysis pipeline
        results = train_with_full_analysis(params, mask, sparsity, subkey, 
                                          compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
        
        # Store results
        all_results.append(results)
        
        print(f"\nCompleted sparsity {sparsity}: final loss = {results['final_loss']:.6e}")
        
        # Save individual sparsity results
        save_variable(results, f"complete_results_sparsity_{sparsity}", s=sparsity)
    
    # =============================================================================
    # CREATE COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION
    # =============================================================================
    if compute_jacobian_eigenvals or compute_connectivity_eigenvals:
        print(f"\n{'='*60}")
        print("CREATING COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION")
        print(f"{'='*60}")
        
        plot_eigenvalue_evolution_comparison(all_results, sparsity_values)
    else:
        print(f"\n{'='*60}")
        print("SKIPPING EIGENVALUE VISUALIZATION (NO EIGENVALUE DATA)")
        print(f"{'='*60}")
    
    # =============================================================================
    # CREATE COMPREHENSIVE SUMMARY PLOTS
    # =============================================================================
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE SUMMARY VISUALIZATIONS")
    print(f"{'='*60}")
    
    create_sparsity_summary_plots(all_results, sparsity_values)
    
    # =============================================================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # =============================================================================
    print(f"\n{'='*60}")
    print("SAVING COMPLETE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    # Save all results together
    complete_experiment_results = {
        'sparsity_values': sparsity_values,
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
            'timestamp': timestamp,
            'output_directory': full_output_path
        }
    }
    save_variable(complete_experiment_results, f"complete_sparsity_experiment_{timestamp}")
    
    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SPARSITY EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Experiment timestamp: {timestamp}")
    print(f"Output directory: {full_output_path}")
    print(f"Tested sparsity values: {sparsity_values}")
    print(f"Eigenvalue tracking: {eigenval_msg}")
    print("Final losses by sparsity:")
    for result in all_results:
        print(f"  s = {result['sparsity']:.3f}: loss = {result['final_loss']:.6e}")
    
    print(f"\nNumber of fixed points found by sparsity:")
    for result in all_results:
        if result['all_fixed_points']:
            num_fps = sum(len(task_fps) for task_fps in result['all_fixed_points'])
            print(f"  s = {result['sparsity']:.3f}: {num_fps} fixed points")
    
    if SLOW_POINT_SEARCH:
        print(f"\nNumber of slow points found by sparsity:")
        for result in all_results:
            if result['all_slow_points']:
                num_sps = sum(len(task_sps) for task_sps in result['all_slow_points'])
                print(f"  s = {result['sparsity']:.3f}: {num_sps} slow points")
    
    print(f"\nAll results and visualizations saved to: {full_output_path}")


def init_params_with_sparsity(key, sparsity_val):
    """Initialize RNN parameters with specific sparsity value."""
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    mask = random.bernoulli(k_mask, p=1.0 - sparsity_val, shape=(N, N)).astype(jnp.float32)
    J_unscaled = random.normal(k1, (N, N)) / jnp.sqrt(N) * mask
    J = J_unscaled / jnp.sqrt(1 - sparsity_val) if sparsity_val < 1.0 else J_unscaled

    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)
    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}


if __name__ == "__main__":
    main()
