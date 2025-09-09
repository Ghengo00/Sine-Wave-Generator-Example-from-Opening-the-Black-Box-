"""
Sparsity experiments: Train RNN with different sparsity levels OR apply post-hoc sparsification to pre-trained networks.

This script supports two modes:
1. 'train' mode: Train RNNs at different sparsity levels and track Jacobian eigenvalue evolution (original behavior)
2. 'import' mode: Load pre-trained dense parameters and apply post-hoc sparsification at different levels

In 'import' mode, the script visualizes eigenvalues of the connectivity matrix J before and after sparsification.


all_results: dict containing (one per sparsity level - keys are sparsity_value):
- results: dict containing:
    - 'sparsity'
    - 'trained_params'
    - 'final_loss'
    - 'eigenvalue_data': dict containing the following keys:
        - 'adam': list of (iteration, eigenvalue_data) tuples for Adam phase
        - 'lbfgs': list of (iteration, eigenvalue_data) tuples for L-BFGS phase
        - eigenvalue_data dict contains:
            - 'jacobian': dict mapping frequency indices to eigenvalue arrays (if computed)
            - 'connectivity': array of connectivity matrix eigenvalues (if computed)
    - 'training_loss_data': dict containing the following keys:
        - 'adam'
        - 'lbfgs'
        - 'final'
    - 'state_traj_states'
    - 'state_fixed_point_inits'
    - 'all_fixed_points'
    - 'all_jacobians'
    - 'all_unstable_eig_freq'
    - 'all_slow_points' (if SLOW_POINT_SEARCH is True)
    - 'all_slow_jacobians' (if SLOW_POINT_SEARCH is True)
    - 'all_slow_unstable_eig_freq' (if SLOW_POINT_SEARCH is True)
    - 'pca_results': dict containing PCA analysis results (one per combinations of skip and tanh options - keys are f"skip_{skip_steps}_tanh_{apply_tanh}"):
        - 'pca': PCA object
        - 'proj_trajs': projected trajectories
        - 'all_trajectories_combined': all trajectories combined (before PCA)
        - 'skip_initial_steps': number of initial steps skipped
        - 'apply_tanh': whether tanh transformation was applied
        - 'pca_components': PCA components
        - 'pca_explained_variance_ratio': explained variance ratio for each component
        - 'proj_fixed': projected fixed points
        - 'proj_slow': projected slow points (if SLOW_POINT_SEARCH is True)




OUTPUT VARIABLES SAVED (per sparsity level):
Eigenvalue evolution data during training:
- 'eigenvalue_data_sparsity_{sparsity_value}' (dict containing):
  - 'adam': list of (iteration, eigenvalue_data) tuples for Adam phase
  - 'lbfgs': list of (iteration, eigenvalue_data) tuples for L-BFGS phase
  - eigenvalue_data dict contains:
    - 'jacobian': dict mapping frequency indices to eigenvalue arrays (if computed)
    - 'connectivity': array of connectivity matrix eigenvalues (if computed)

Training loss evolution data during training:
- 'training_loss_over_iterations_sparsity_{sparsity_value}' (dict containing):
  - 'losses': array of loss values at each eigenvalue sample point
  - 'iterations': array of training iterations corresponding to loss values

Training results:
- 'J_param_sparsity_{sparsity_value}', shape (N, N)
- 'B_param_sparsity_{sparsity_value}', shape (N, I)
- 'b_x_param_sparsity_{sparsity_value}', shape (N,)
- 'w_param_sparsity_{sparsity_value}', shape (I, N)  
- 'b_z_param_sparsity_{sparsity_value}', shape (I,)
- 'state_sparsity_{sparsity_value}' (dict containing):
  - 'traj_states': array of shape (num_tasks, T_train, N)
  - 'fixed_point_inits': array of shape (num_tasks, N)

Fixed point and frequency analysis results (replicated for slow points if enabled):
- 'all_fixed_points_sparsity_{sparsity_value}': list of lists, outer list over tasks, inner list over fixed points for a given task
- 'all_fixed_jacobians_sparsity_{sparsity_value}': list of lists, outer list over tasks, inner list over fixed point Jacobians for a given task
- 'all_fixed_unstable_eig_freq_sparsity_{sparsity_value}': list of lists of lists, outer list over tasks, inner list over fixed points, inner list over unstable eigen frequencies for a given task

PCA analysis results:
- 'pca_results' (dict containing dictionaries, one for each combination of skip and tanh options):
    - proj_trajs: list of projected trajectories, shape (num_tasks, num_steps_train+1-skip_initial_steps, n_components)
    - skip_initial_steps: number of initial steps skipped
    - apply_tanh: whether tanh transformation was applied
    - pca_components: PCA components, shape (n_components, N)
    - pca_explained_variance_ratio: explained variance ratio for each component, shape (n_components,)
    - proj_fixed: list of fixed points for PCA analysis, same structure as all_fixed_points, so shape (num_total_fixed_ponts, N)
    - proj_slow if SLOW_POINT_SEARCH is True


    

CROSS-SPARSITY OUTPUT VARIABLES SAVED:
Experiment summary:
- 'complete_sparsity_experiment_{timestamp}' (dict containing):
  - 'sparsity_values': array of all tested sparsity values
  - 'experiment_config': dict with experiment configuration including:
    - 'eigenvalue_sample_frequency': frequency of eigenvalue sampling during training
    - 'compute_jacobian_eigenvals': boolean indicating if Jacobian eigenvalues were computed
    - 'compute_connectivity_eigenvals': boolean indicating if connectivity eigenvalues were computed
    - 'jacobian_frequencies': list of frequency indices used for Jacobian eigenvalue computation
    - 'num_epochs_adam': number of Adam optimization epochs
    - 'num_epochs_lbfgs': number of L-BFGS optimization epochs
    - 'network_size': network size (N)
    - 'num_tasks': number of tasks
    - 'slow_point_search': boolean indicating if slow point search was enabled
    - 'timestamp': experiment timestamp
    - 'output_directory': full path to experiment output directory

    
========================================
========================================


OUTPUT PLOTS SAVED (per sparsity level):
Eigenvalue evolution plots (single sparsity):
- 'jacobian_eigenvalue_evolution_freq_{freq_idx}_sparsity_{sparsity_str}_omega_{freq_str}' plots:
        Evolution of Jacobian eigenvalues during training for specific frequency and sparsity
        From 'plot_jacobian_eigenvalue_evolution_for_single_sparsity' function
        One for each frequency index and sparsity level combination
- 'connectivity_eigenvalue_evolution_sparsity_{sparsity_str}' plots:
        Evolution of connectivity matrix eigenvalues during training for specific sparsity
        From 'plot_connectivity_eigenvalue_evolution_for_single_sparsity' function
        One plot for each sparsity level

Training progress:
- 'training_loss_evolution_sparsity_{sparsity_str}' plots:
        Plot of training loss evolution during optimization
        From 'plot_training_loss_evolution' function

Visualization of basic results:
- 'trajectory_vs_target' plots:
        From 'plot_trajectories_vs_targets' function
        One for each of TEST_INDICES
- 'parameter_matrices' plots:
        Plot of J, B, and b_x matrices with colour scales
        From 'plot_parameter_matrices_for_tasks' function
        For test_indices=0

Fixed point and frequency plots (replicated for slow points if enabled):
- 'Jacobian_Matrices' plots:
        From 'analyze_jacobians_visualization' function
        One for each of TEST_INDICES
- 'Unstable_Eigenvalues' plots:
        Plot of the unstable eigenvalues (for given task) on the complex plane
        From 'analyze_unstable_frequencies_visualization' function, and within that, from 'plot_unstable_eigenvalues' function
        One for each of TEST_INDICES
- 'Frequency_Comparison' plots:
        Plot of the maximum norm of the imaginary component of the unstable eigenvalues vs the target frequency for each task
        From 'analyze_unstable_frequencies_visualization' function, and within that, from 'plot_frequency_comparison' function

PCA plots (multiple skip/tanh combinations):
- 'pca_explained_variance_ratio' plots:
        Plot of the explained variance ratio for each PCA component
        From 'run_pca_analysis' function, and within that, from 'plot_pca_explained_variance_ratio' function
- 'pca_plot_fixed' plots:
        3-D PCA plot, .png
        From 'run_pca_analysis' function, and within that, from 'plot_pca_trajectories_and_points' function
- 'subset_of_indices_pca_plot_fixed' plots:
        3-D PCA plot, .png, only for specified tasks
- 'interactive_pca_plot_fixed' plots:
        Interactive 3-D PCA plot, .html
        From 'run_pca_analysis' function, and within that, from 'plot_interactive_pca_trajectories_and_points' function
- 'subset_of_indices_interactive_pca_plot_fixed' plots:
        Interactive 3-D PCA plot, .html, only for specified tasks




CROSS-SPARSITY OUTPUT PLOTS SAVED:
Eigenvalue evolution comparison plots:
- 'connectivity_eigenvalue_evolution_sparsities_{sparsity_str}' plots:
        Evolution of connectivity matrix eigenvalues during training comparing all sparsity levels
        From 'plot_connectivity_eigenvalue_evolution' function within 'plot_eigenvalue_evolution_comparison'
        Only saved if connectivity eigenvalues were computed
        Sparsity_str contains all tested sparsity values in filename
        Shows eigenvalues on complex plane with color indicating training progress

Training loss comparison plots:
- 'training_loss_evolution_sparsities_{sparsity_str}' plots:
        Training loss evolution during optimization comparing all sparsity levels
        From 'plot_training_loss_evolution_comparison' function

Trajectory comparison plots:
- 'trajectory_vs_target_task_{j}_sparsities_{sparsity_str}' plots:
        Trajectory vs target comparison for specific task across all sparsity levels
        From 'plot_trajectories_vs_targets_comparison' function
        One plot for each task index in TEST_INDICES

Frequency comparison plots:
- 'frequency_comparison_sparsities_{sparsity_str}' plots:
        Frequency comparison (beteween target task frequency and maximum norm of imaginary part of unstable eigenvalues for that task) plots across all sparsity levels
        From 'plot_frequency_comparison_across_sparsity' function

PCA comparison plots:
- 'pca_explained_variance_ratio_skip_{skip}_tanh_{tanh}_sparsities_{sparsity_str}' plots:
        PCA explained variance ratio comparison across all sparsity levels
        From 'plot_pca_explained_variance_comparison' function
        One for each combination of skip steps and tanh options
- 'pca_plot_fixed_skip_{skip}_tanh_{tanh}_sparsities_{sparsity_str}' plots:
        3D PCA plots comparing all sparsity levels
        From 'plot_pca_3d_comparison' function
        One for each combination of skip steps and tanh options
- 'subset_of_indices_pca_plot_fixed_skip_{skip}_tanh_{tanh}_sparsities_{sparsity_str}' plots:
        3D PCA plots for subset of tasks comparing all sparsity levels
        From 'plot_pca_3d_comparison' function
        One for each combination of skip steps and tanh options

Summary comparison plots:
- 'sparsity_summary_plots_{timestamp}' plots:
        2x2 subplot comparing metrics across all sparsity levels including:
        - Final training loss vs sparsity (log scale)
        - Number of fixed points vs sparsity
        - Spectral radius vs sparsity (with unit circle reference)
        - Number of slow points vs sparsity (if slow point search enabled)
        From 'create_sparsity_summary_plots' function

Unstable eigenvalue distribution plots:
- 'unstable_eigenvalue_table_sparsities_{sparsity_str}' plots:
        Heatmap table showing distribution of unstable eigenvalues per fixed point across all sparsity levels
        Rows: number of unstable eigenvalues per fixed point (0, 1, 2, ...)
        Columns: sparsity levels
        Values: count of fixed points with that number of unstable eigenvalues
        From 'create_unstable_eigenvalue_table' function
        Sparsity_str contains all tested sparsity values in filename

Fixed points per task distribution plots:
- 'fixed_points_per_task_table_sparsities_{sparsity_str}' plots:
        Heatmap table showing distribution of fixed points per task across all sparsity levels
        Rows: number of fixed points per task (0, 1, 2, ...)
        Columns: sparsity levels
        Values: count of tasks with that number of fixed points
        From 'create_fixed_points_per_task_table' function
        Sparsity_str contains all tested sparsity values in filename
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
    TOL, MAXITER, GAUSSIAN_STD, NUM_ATTEMPTS, SLOW_POINT_SEARCH, TEST_INDICES,
    PCA_N_COMPONENTS, PCA_SKIP_OPTIONS, PCA_TANH_OPTIONS,
    time_drive, time_train, num_steps_train, TEST_INDICES_EXTREMES, COLORS
)
from _2_utils import (Timer, save_variable, set_custom_output_dir, get_output_dir, 
                     save_variable_with_sparsity, save_figure_with_sparsity, get_sparsity_output_dir, load_variable)
from _3_data_generation import generate_all_inputs_and_targets
from _4_rnn_model import init_params, run_batch_diagnostics, simulate_trajectory
from _5_training import setup_optimizers, create_training_functions, scan_with_history
from _6_analysis import find_points, compute_jacobian, find_and_analyze_points, generate_point_summaries
from _7_visualization import (save_figure, plot_single_trajectory_vs_target, plot_trajectories_vs_targets, plot_parameter_matrices_for_tasks, 
                          plot_frequency_comparison, analyze_jacobians_visualization, analyze_unstable_frequencies_visualization)
from _8_pca_analysis import plot_explained_variance_ratio, plot_pca_trajectories_and_points, run_pca_analysis


# =============================================================================
# =============================================================================
# USER INPUT FUNCTIONS
# =============================================================================
# =============================================================================
def get_mode_selection():
    """
    Get user selection for experiment mode.
    
    Returns:
        str: either "train" or "import"
    """
    print("=" * 60)
    print("SPARSITY EXPERIMENT MODE SELECTION")
    print("=" * 60)
    
    print("\nChoose experiment mode:")
    print("1. Train mode - Train networks with sparsity enforcement during training")
    print("2. Import mode - Load pre-trained dense network and apply post-training sparsification")
    
    while True:
        try:
            choice = input("\nEnter choice (1-2): ").strip()
            
            if choice == '1':
                return "train"
            elif choice == '2':
                return "import"
            else:
                print("Please enter 1 or 2")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(1)


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




# =============================================================================
# =============================================================================
# CONTEXT MANAGER
# =============================================================================
# =============================================================================
@contextmanager
def sparsity_output_context(sparsity_value):
    """
    Context manager to temporarily set output directory to sparsity-specific folder.
    """
    from _2_utils import _CUSTOM_OUTPUT_DIR, set_custom_output_dir
    
    # Save current custom output directory
    original_custom_dir = _CUSTOM_OUTPUT_DIR
    
    # Set to sparsity-specific directory  
    sparsity_dir = get_sparsity_output_dir(sparsity_value)
    set_custom_output_dir(sparsity_dir)
    
    try:
        yield sparsity_dir
    finally:
        # Restore original directory
        if original_custom_dir:
            set_custom_output_dir(original_custom_dir)
        else:
            set_custom_output_dir(None)




# =============================================================================
# =============================================================================
# PARAMETER INITIALIZATION AND IMPORT FUNCTIONS
# =============================================================================
# =============================================================================
def init_params_with_sparsity(key, sparsity_val):
    """
    Initialize RNN parameters with specific sparsity value.
    
    Note: All sparsity levels use the same base random key to ensure
    that the underlying parameter distributions are identical, with only
    the sparsity mask differing between experiments.
    """
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    mask = random.bernoulli(k_mask, p=1.0 - sparsity_val, shape=(N, N)).astype(jnp.float32)
    J_unscaled = random.normal(k1, (N, N)) / jnp.sqrt(N) * mask
    J = J_unscaled / (1 - sparsity_val) if sparsity_val < 1.0 else J_unscaled

    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)

    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}


def load_dense_parameters():
    """
    Load pre-trained dense parameters from specified file path.
    
    Returns:
        dict: parameters dictionary containing J, B, b_x, w, b_z
    """
    base_dir = "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250624_133230/Sparsity_0p00"

    try:
        # Load individual parameter files
        param_files = {
            'J': "J_param_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl",
            'B': "B_param_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl",
            'b_x': "b_x_param_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl",
            'w': "w_param_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl",
            'b_z': "b_z_param_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
        }
        
        params = {}
        for param_name, filename in param_files.items():
            filepath = os.path.join(base_dir, filename)
            params[param_name] = load_variable(filepath)
            print(f"✓ Loaded {param_name}")

        return params
        
    except Exception as e:
        print(f"Error loading dense parameters: {e}")
        raise


def apply_sparsity_to_params(params, sparsity_val, key):
    """
    Apply sparsity to dense parameters by creating a random Bernoulli mask.
    
    Arguments:
        params: dense parameters dictionary
        sparsity_val: sparsity level to apply
        key: JAX random key for mask generation
        
    Returns:
        mask: sparsity mask used
        sparsified_params: parameters with sparsity applied to J matrix
    """
    # Create random Bernoulli mask for sparsity
    mask = random.bernoulli(key, p=1.0 - sparsity_val, shape=(N, N)).astype(jnp.float32)
    
    # Apply mask to J matrix
    J_sparse = params["J"] * mask
    
    # Scale to maintain expected norm (same as in training)
    if sparsity_val < 1.0:
        J_sparse = J_sparse / jnp.sqrt(1 - sparsity_val)
    
    # Create new parameter dictionary with sparsified J
    sparsified_params = {
        "J": J_sparse,
        "B": params["B"],  # Keep other parameters unchanged
        "b_x": params["b_x"],
        "w": params["w"],
        "b_z": params["b_z"]
    }
    
    return mask, sparsified_params




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
def modified_scan_with_eigenvalues(params, opt_state, step_fn, loss_fn, num_steps, tag, sample_frequency, 
                                   compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Optimized training loop that uses lax.scan for chunks between eigenvalue sampling.
    
    This hybrid approach uses JAX's lax.scan for fast optimization steps while keeping
    eigenvalue sampling in Python for flexibility and debugging.
    
    Arguments:
        params: initial parameters
        opt_state: initial optimizer state
        step_fn: optimization step function
        loss_fn: loss function to evaluate training loss
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
        loss_history: list of (iteration, training_loss) tuples
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

        return best_params, best_loss, final_params, final_opt_state, [], []
    
    # EIGENVALUE SAMPLING CASE
    # Initialize tracking variables
    best_loss = float('inf')  # Ensure it's a proper Python float
    best_params = params
    eigenvalue_history = []
    loss_history = []
    current_params = params
    current_opt_state = opt_state
    
    # Sample eigenvalues and loss at iteration 0
    print(f"[{tag}] Sampling eigenvalues and loss at iteration 0...")
    initial_eigenval_data = compute_eigenvalues_during_training(
        params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
    )
    loss_result = loss_fn(params)
    initial_loss = float(loss_result[0] if isinstance(loss_result, tuple) else loss_result)
    eigenvalue_history.append((0, initial_eigenval_data))
    loss_history.append((0, initial_loss))
    
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
        # Convert to float to ensure proper comparison (handle JAX array scalars)
        chunk_best_loss_val = float(chunk_best_loss)
        best_loss_val = float(best_loss)
        if chunk_best_loss_val < best_loss_val:
            best_loss = chunk_best_loss_val
            best_params = chunk_best_params
        
        # Sample eigenvalues and loss at end of chunk
        step_num = (chunk_idx + 1) * chunk_size
        print(f"[{tag}] step {step_num}: best_loss {best_loss:.6e} - Sampling eigenvalues and loss...")
        eigenval_data = compute_eigenvalues_during_training(
            current_params, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
        )
        loss_result = loss_fn(current_params)
        current_loss = float(loss_result[0] if isinstance(loss_result, tuple) else loss_result)
        eigenvalue_history.append((step_num, eigenval_data))
        loss_history.append((step_num, current_loss))
    
    # Handle remaining steps if any
    if remaining_steps > 0:
        print(f"[{tag}] Processing final {remaining_steps} steps...")
        final_best_params, final_best_loss, current_params, current_opt_state = scan_with_history(
            current_params, current_opt_state, step_fn, remaining_steps, f"{tag}-Final"
        )
        
        # Update global best if final chunk improved
        # Convert to float to ensure proper comparison (handle JAX array scalars)
        final_best_loss_val = float(final_best_loss)
        best_loss_val = float(best_loss)
        if final_best_loss_val < best_loss_val:
            best_loss = final_best_loss_val
            best_params = final_best_params
    
    print(f"[{tag}] Optimization completed. Final best loss: {best_loss:.6e}")
    
    return best_params, best_loss, current_params, current_opt_state, eigenvalue_history, loss_history




# =============================================================================
# =============================================================================
# SPARSIFICATION AND ANALYSIS PIPELINE (IMPORT MODE)
# =============================================================================
# =============================================================================
def sparsify_and_analyze(dense_params, sparsity_value, key, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Apply sparsification to dense parameters and run full analysis pipeline.
    
    Arguments:
        dense_params: pre-trained dense parameters
        sparsity_value: sparsity level to apply
        key: JAX random key for analysis
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        results_dict: comprehensive results dictionary containing all analysis outputs
    """
    print(f"\n{'='*60}")
    print(f"SPARSIFYING AND ANALYZING WITH SPARSITY s = {sparsity_value}")
    print(f"{'='*60}")
    
    # ========================================
    # 1. SETUP AND SPARSIFICATION
    # ========================================
    results = {}
    results['sparsity'] = sparsity_value
    
    # Store original dense J matrix for before/after comparison
    dense_J = dense_params["J"]
    results['dense_J'] = dense_J  # Store for aggregate plotting
    
    # Create sparsity mask using the same approach as in training
    mask_key, analysis_key = random.split(key)
    mask = random.bernoulli(mask_key, p=1.0 - sparsity_value, shape=(N, N)).astype(jnp.float32)
    
    # Apply sparsification to J matrix
    J_sparse = dense_J * mask
    # Rescale to maintain spectral properties (same as in training)
    if sparsity_value < 1.0:
        J_sparse = J_sparse / (1 - sparsity_value)
    
    # Create sparsified parameters
    params = {**dense_params, "J": J_sparse}
    
    # Plot connectivity eigenvalues before and after sparsification
    plot_connectivity_eigenvalues_before_after(dense_J, J_sparse, sparsity_value)
    
    print(f"Applied sparsity {sparsity_value} to connectivity matrix")
    print(f"Dense J matrix spectral radius: {np.max(np.abs(np.linalg.eigvals(np.array(dense_J)))):.4f}")
    print(f"Sparse J matrix spectral radius: {np.max(np.abs(np.linalg.eigvals(np.array(J_sparse)))):.4f}")
    
    # ========================================
    # 2. SKIP TRAINING - PARAMETERS ARE FIXED
    # ========================================
    print("\n2. USING PRE-TRAINED PARAMETERS (SKIPPING TRAINING)")
    print("-" * 40)
    
    # No training needed - use sparsified parameters directly
    trained_params = params
    final_loss = None  # We don't have a loss for imported parameters
    
    # No eigenvalue evolution data for import mode
    eigenvalue_data = {}
    training_loss_data = {}
    
    results['trained_params'] = trained_params
    results['final_loss'] = final_loss
    results['eigenvalue_data'] = eigenvalue_data
    results['training_loss_data'] = training_loss_data
    
    # ========================================
    # 3. RUN COMPLETE ANALYSIS PIPELINE
    # ========================================
    print("\n3. RUNNING COMPLETE ANALYSIS PIPELINE")
    print("-" * 40)
    
    # Generate inputs and targets
    # all_inputs, all_targets = generate_all_inputs_and_targets()
    
    # Simulate trajectories with sparsified parameters
    with Timer(f"Trajectory simulation with sparsity {sparsity_value}"):
        _, states_trajectory, final_states = run_batch_diagnostics(trained_params, key)
    
    # Store trajectory results
    results['state_traj_states'] = states_trajectory
    results['state_fixed_point_inits'] = final_states
    
    # Save individual parameter matrices
    save_variable_with_sparsity(trained_params["J"], "J_param", sparsity_value)
    save_variable_with_sparsity(trained_params["B"], "B_param", sparsity_value)
    save_variable_with_sparsity(trained_params["b_x"], "b_x_param", sparsity_value)
    save_variable_with_sparsity(trained_params["w"], "w_param", sparsity_value)
    save_variable_with_sparsity(trained_params["b_z"], "b_z_param", sparsity_value)
    
    # Save state data
    state_data = {
        'traj_states': states_trajectory,
        'fixed_point_inits': final_states
    }
    save_variable_with_sparsity(state_data, "state", sparsity_value)
    
    # ========================================
    # 4. FIXED POINT AND FREQUENCY ANALYSIS
    # ========================================
    print("\n4. FIXED POINT AND FREQUENCY ANALYSIS")
    print("-" * 40)
    
    with Timer(f"Fixed point analysis with sparsity {sparsity_value}"):
        # Run fixed point and frequency analysis
        all_fixed_points, all_jacobians, all_unstable_eig_freqs = find_and_analyze_points(
            states_trajectory, final_states, trained_params, point_type='fixed'
        )
        
        results['all_fixed_points'] = all_fixed_points
        results['all_jacobians'] = all_jacobians  
        results['all_unstable_eig_freq'] = all_unstable_eig_freqs
        
        # Save fixed point analysis results
        save_variable_with_sparsity(all_fixed_points, "all_fixed_points", sparsity_value)
        save_variable_with_sparsity(all_jacobians, "all_fixed_jacobians", sparsity_value)
        save_variable_with_sparsity(all_unstable_eig_freqs, "all_fixed_unstable_eig_freq", sparsity_value)
    
    # ========================================
    # 5. SLOW POINT ANALYSIS (if enabled)
    # ========================================
    if SLOW_POINT_SEARCH:
        print("\n5. SLOW POINT ANALYSIS")
        print("-" * 40)
        
        with Timer(f"Slow point analysis with sparsity {sparsity_value}"):
            all_slow_points, all_slow_jacobians, all_slow_unstable_eig_freqs = find_and_analyze_points(
                states_trajectory, final_states, trained_params, point_type='slow'
            )
            
            results['all_slow_points'] = all_slow_points
            results['all_slow_jacobians'] = all_slow_jacobians
            results['all_slow_unstable_eig_freq'] = all_slow_unstable_eig_freqs
            
            # Save slow point analysis results
            save_variable_with_sparsity(all_slow_points, "all_slow_points", sparsity_value)
            save_variable_with_sparsity(all_slow_jacobians, "all_slow_jacobians", sparsity_value) 
            save_variable_with_sparsity(all_slow_unstable_eig_freqs, "all_slow_unstable_eig_freq", sparsity_value)
    else:
        # Set default values when slow point search is disabled
        all_slow_points = None
        all_slow_jacobians = None
        all_slow_unstable_eig_freqs = None
        results['all_slow_points'] = all_slow_points
        results['all_slow_jacobians'] = all_slow_jacobians
        results['all_slow_unstable_eig_freq'] = all_slow_unstable_eig_freqs
    
    # ========================================
    # 6. PCA ANALYSIS
    # ========================================
    print("\n6. PCA ANALYSIS")
    print("-" * 40)

    with Timer(f"PCA analysis with sparsity {sparsity_value}"):
        print(f"\nRunning PCA analysis for all skip/tanh combinations...")
        
        # Initialize PCA results dictionary
        pca_results = {}
        
        # Check trajectory length to determine feasible skip options
        traj_length = states_trajectory.shape[1] if len(states_trajectory.shape) >= 2 else 0
        print(f"Trajectory length: {traj_length} time steps")
        
        # Run PCA analysis for all combinations of skip and tanh options
        for skip_steps in PCA_SKIP_OPTIONS:
            for apply_tanh in PCA_TANH_OPTIONS:
                print(f"  Processing skip_steps={skip_steps}, apply_tanh={apply_tanh}")
                
                # Check if we have enough time steps after skipping
                if skip_steps >= traj_length:
                    print(f"    WARNING: Skip steps ({skip_steps}) >= trajectory length ({traj_length}). Skipping this combination.")
                    # Store None result to indicate this combination was skipped
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = None
                    continue
                
                # Run PCA analysis (save plots in sparsity-specific directory)
                try:
                    with sparsity_output_context(sparsity_value):
                        # Temporarily update the config sparsity for plot titles
                        import _1_config
                        original_s = _1_config.s
                        _1_config.s = sparsity_value
                        
                        try:
                            pca_result = run_pca_analysis(
                                states_trajectory, 
                                all_fixed_points=all_fixed_points,
                                all_slow_points=all_slow_points,
                                params=trained_params,
                                slow_point_search=SLOW_POINT_SEARCH,
                                skip_initial_steps=skip_steps,
                                apply_tanh=apply_tanh,
                                n_components=PCA_N_COMPONENTS
                            )
                        finally:
                            # Restore original sparsity value
                            _1_config.s = original_s
                    
                    # Store results with key indicating skip and tanh settings
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = pca_result
                    
                    # Debug: check what was stored
                    print(f"    Stored PCA result for key '{key}': {list(pca_result.keys()) if pca_result else 'None'}")
                    if pca_result and 'proj_trajs' in pca_result:
                        print(f"    - Number of projected trajectories: {len(pca_result['proj_trajs'])}")
                    if pca_result and 'pca' in pca_result:
                        print(f"    - PCA object available: {pca_result['pca'] is not None}")
                
                except Exception as e:
                    print(f"    ERROR in PCA analysis for skip={skip_steps}, tanh={apply_tanh}: {e}")
                    print(f"    Storing None result and continuing with other combinations...")
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = None
        
        print(f"Completed PCA analysis for all combinations. Total PCA keys stored: {list(pca_results.keys())}")
        successful_pca_keys = [k for k, v in pca_results.items() if v is not None]
        print(f"Successful PCA analyses: {len(successful_pca_keys)} out of {len(pca_results)}")
    
    results['pca_results'] = pca_results
    
    # ========================================
    # 7. VISUALIZATION
    # ========================================
    print("\n7. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    with Timer(f"Visualization with sparsity {sparsity_value}"):
        # Basic trajectory and parameter visualizations
        plot_trajectories_vs_targets(trained_params, test_indices=TEST_INDICES, sparsity_value=sparsity_value)
        plot_parameter_matrices_for_tasks(trained_params, test_indices=[0], sparsity_value=sparsity_value)
        
        # Fixed point analysis visualizations
        analyze_jacobians_visualization(point_type="fixed", all_jacobians=all_jacobians)
        analyze_unstable_frequencies_visualization(point_type="fixed", 
                                                      all_unstable_eig_freq=all_unstable_eig_freqs)
        
        # Slow point visualizations (if enabled)
        if SLOW_POINT_SEARCH and all_slow_points is not None:
            analyze_jacobians_visualization(point_type="slow", all_jacobians=all_slow_jacobians)
            analyze_unstable_frequencies_visualization(point_type="slow", all_unstable_eig_freq=all_slow_unstable_eig_freqs)
    
    print(f"\nCompleted sparsification and analysis for sparsity {sparsity_value}")
    return results




# =============================================================================
# =============================================================================
# FULL RUN FUNCTION (TRAINING MODE)
# =============================================================================
# =============================================================================
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
    
    
    # ========================================
    # 1. SETUP AND INITIALIZATION
    # ========================================
    results = {}
    results['sparsity'] = sparsity_value
    
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Create training functions 
    adam_step, lbfgs_step, loss_fn = create_training_functions(adam_opt, lbfgs_opt, mask)
    
    # Track eigenvalues during training
    eigenvalue_data = {}
    training_loss_data = {}
    

    # ========================================
    # 2. TRAINING WITH EIGENVALUE TRACKING
    # ========================================
    print("\n2. TRAINING WITH EIGENVALUE TRACKING")
    print("-" * 40)
    
    with Timer(f"Training with sparsity {sparsity_value}"):
        # Adam training phase with eigenvalue tracking
        best_params_adam, best_loss_adam, params_after_adam, adam_state_after, adam_eigenvals, adam_losses = \
            modified_scan_with_eigenvalues(
                params, adam_state, adam_step, loss_fn,
                NUM_EPOCHS_ADAM, "ADAM", EIGENVALUE_SAMPLE_FREQUENCY,
                compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
            )
        
        eigenvalue_data['adam'] = adam_eigenvals
        training_loss_data['adam'] = adam_losses
        print(f"Adam phase completed. Best loss: {best_loss_adam:.6e}")
        
        # Continue with best parameters from Adam
        trained_params = best_params_adam
        final_loss = best_loss_adam
        
        # L-BFGS training phase (if needed)
        if best_loss_adam > LOSS_THRESHOLD:
            print("Loss threshold not met, starting L-BFGS phase...")
            best_params_lbfgs, best_loss_lbfgs, params_after_lbfgs, lbfgs_state_after, lbfgs_eigenvals, lbfgs_losses = \
                modified_scan_with_eigenvalues(
                    trained_params, lbfgs_state, lbfgs_step, loss_fn,
                    NUM_EPOCHS_LBFGS, "L-BFGS", EIGENVALUE_SAMPLE_FREQUENCY,
                    compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies
                )
            
            eigenvalue_data['lbfgs'] = lbfgs_eigenvals
            training_loss_data['lbfgs'] = lbfgs_losses
            
            # Use L-BFGS results if better
            if best_loss_lbfgs < best_loss_adam:
                trained_params = best_params_lbfgs
                final_loss = best_loss_lbfgs
                print(f"L-BFGS improved results. Best loss: {best_loss_lbfgs:.6e}")
        else:
            eigenvalue_data['lbfgs'] = []
            training_loss_data['lbfgs'] = []
            print("Loss threshold met, skipping L-BFGS phase.")
    
    # Add final best loss to training loss data
    training_loss_data['final'] = final_loss
    
    results['trained_params'] = trained_params
    results['final_loss'] = final_loss
    results['eigenvalue_data'] = eigenvalue_data
    results['training_loss_data'] = training_loss_data
    

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
    save_variable_with_sparsity(np.array(trained_params["J"]), f"J_param_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    save_variable_with_sparsity(np.array(trained_params["B"]), f"B_param_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    save_variable_with_sparsity(np.array(trained_params["b_x"]), f"b_x_param_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    save_variable_with_sparsity(np.array(trained_params["w"]), f"w_param_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    save_variable_with_sparsity(np.array(trained_params["b_z"]), f"b_z_param_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)

    state_dict = {
        "traj_states": state_traj_states,
        "fixed_point_inits": state_fixed_point_inits
    }
    save_variable_with_sparsity(state_dict, f"state_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    
    # Save eigenvalue evolution data during training
    print("Saving eigenvalue evolution data...")
    save_variable_with_sparsity(eigenvalue_data, f"eigenvalue_data_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    
    # Save training loss evolution data
    print("Saving training loss evolution data...")
    save_variable_with_sparsity(training_loss_data, f"training_loss_over_iterations_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    
    # Plot training loss evolution
    print("Plotting training loss evolution...")
    plot_training_loss_evolution(training_loss_data, sparsity_value)
    

    # ========================================
    # 5. VISUALIZATION OF BASIC RESULTS
    # ========================================
    print("\n5. VISUALIZATION OF BASIC RESULTS")
    print("-" * 40)
    
    # Plot trajectories vs targets (save in sparsity-specific folder)
    print("Plotting produced trajectories vs. target signals...")
    plot_trajectories_vs_targets(trained_params, test_indices=TEST_INDICES, sparsity_value=sparsity_value)
    
    # Plot parameter matrices (save in sparsity-specific folder)
    plot_parameter_matrices_for_tasks(trained_params, test_indices=[0], sparsity_value=sparsity_value)


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
        save_variable_with_sparsity(all_fixed_points, f"all_fixed_points_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
        save_variable_with_sparsity(all_jacobians, f"all_fixed_jacobians_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
        save_variable_with_sparsity(all_unstable_eig_freq, f"all_fixed_unstable_eig_freq_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
    
    results['all_fixed_points'] = all_fixed_points
    results['all_jacobians'] = all_jacobians
    results['all_unstable_eig_freq'] = all_unstable_eig_freq
    
    # Generate fixed point summaries
    generate_point_summaries(point_type="fixed", all_points=all_fixed_points, 
                            all_unstable_eig_freq=all_unstable_eig_freq)
    
    # Visualize fixed point analysis (save in sparsity-specific directory)
    with sparsity_output_context(sparsity_value):
        # Temporarily update the config sparsity for plot titles
        import _1_config
        original_s = _1_config.s
        _1_config.s = sparsity_value
        
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
            save_variable_with_sparsity(all_slow_points, f"all_slow_points_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
            save_variable_with_sparsity(all_slow_jacobians, f"all_slow_jacobians_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
            save_variable_with_sparsity(all_slow_unstable_eig_freq, f"all_slow_unstable_eig_freq_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
        
        # Generate slow point summaries
        generate_point_summaries(point_type="slow", all_points=all_slow_points, 
                                all_unstable_eig_freq=all_slow_unstable_eig_freq)
        
        # Visualize slow point analysis (save in sparsity-specific directory)
        with sparsity_output_context(sparsity_value):
            # Temporarily update the config sparsity for plot titles
            import _1_config
            original_s = _1_config.s
            _1_config.s = sparsity_value
            
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
    # 8. PCA ANALYSIS
    # ========================================
    print("\n8. PCA ANALYSIS")
    print("-" * 40)
    
    with Timer("PCA Analysis"):
        print(f"\nRunning PCA analysis for all skip/tanh combinations...")
        
        # Initialize PCA results dictionary
        pca_results = {}
        
        # Check trajectory length to determine feasible skip options
        traj_length = state_traj_states.shape[1] if len(state_traj_states.shape) >= 2 else 0
        print(f"Trajectory length: {traj_length} time steps")
        
        # Run PCA analysis for all combinations of skip and tanh options
        for skip_steps in PCA_SKIP_OPTIONS:
            for apply_tanh in PCA_TANH_OPTIONS:
                print(f"  Processing skip_steps={skip_steps}, apply_tanh={apply_tanh}")
                
                # Check if we have enough time steps after skipping
                if skip_steps >= traj_length:
                    print(f"    WARNING: Skip steps ({skip_steps}) >= trajectory length ({traj_length}). Skipping this combination.")
                    # Store None result to indicate this combination was skipped
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = None
                    continue
                
                # Run PCA analysis (save plots in sparsity-specific directory)
                try:
                    with sparsity_output_context(sparsity_value):
                        # Temporarily update the config sparsity for plot titles
                        import _1_config
                        original_s = _1_config.s
                        _1_config.s = sparsity_value
                        
                        try:
                            pca_result = run_pca_analysis(
                                state_traj_states, 
                                all_fixed_points=all_fixed_points,
                                all_slow_points=all_slow_points,
                                params=trained_params,
                                slow_point_search=SLOW_POINT_SEARCH,
                                skip_initial_steps=skip_steps,
                                apply_tanh=apply_tanh,
                                n_components=PCA_N_COMPONENTS
                            )
                        finally:
                            # Restore original sparsity value
                            _1_config.s = original_s
                    
                    # Store results with key indicating skip and tanh settings
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = pca_result
                    
                    # Debug: check what was stored
                    print(f"    Stored PCA result for key '{key}': {list(pca_result.keys()) if pca_result else 'None'}")
                    if pca_result and 'proj_trajs' in pca_result:
                        print(f"    - Number of projected trajectories: {len(pca_result['proj_trajs'])}")
                    if pca_result and 'pca' in pca_result:
                        print(f"    - PCA object available: {pca_result['pca'] is not None}")
                
                except Exception as e:
                    print(f"    ERROR in PCA analysis for skip={skip_steps}, tanh={apply_tanh}: {e}")
                    print(f"    Storing None result and continuing with other combinations...")
                    key = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    pca_results[key] = None
        
        # Save PCA results
        # save_variable_with_sparsity(pca_results, f"pca_results_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
        
        print(f"Completed PCA analysis for all combinations. Total PCA keys stored: {list(pca_results.keys())}")
        successful_pca_keys = [k for k, v in pca_results.items() if v is not None]
        print(f"Successful PCA analyses: {len(successful_pca_keys)} out of {len(pca_results)}")
    
    results['pca_results'] = pca_results
    
    print(f"Completed full analysis for sparsity {sparsity_value}: final loss = {final_loss:.6e}")
    
    return results




# =============================================================================
# =============================================================================
# TRAINING LOSS PLOTTING FUNCTION
# =============================================================================
# =============================================================================
def plot_training_loss_evolution(training_loss_data, sparsity_value, ax=None, for_comparison=False, ylim=None):
    """
    Plot training loss evolution during optimization for a single sparsity level.
    
    Arguments:
        training_loss_data: dictionary with 'adam', 'lbfgs', and 'final' keys
        sparsity_value: sparsity level for this experiment
        ax: optional matplotlib axis to plot on. If None, creates new figure
        for_comparison: if True, adjusts styling for comparison plots (smaller markers, shorter labels)
        ylim: optional tuple (ymin, ymax) to set consistent y-axis limits across comparison plots
    """
    # Create new figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        standalone_plot = True
    else:
        fig = ax.get_figure()
        standalone_plot = False
    
    # Combine loss data from both training phases
    all_iterations = []
    all_losses = []
    phase_colors = []
    
    # Add Adam phase data
    for iteration, loss in training_loss_data['adam']:
        all_iterations.append(iteration)
        all_losses.append(loss)
        phase_colors.append('blue')
    
    # Add L-BFGS phase data (offset iterations)
    adam_max_iter = NUM_EPOCHS_ADAM if training_loss_data['adam'] else 0
    for iteration, loss in training_loss_data['lbfgs']:
        all_iterations.append(adam_max_iter + iteration)
        all_losses.append(loss)
        phase_colors.append('red')
    
    if len(all_losses) == 0:
        no_data_msg = 'No training loss data available' if standalone_plot else 'No training\nloss data'
        ax.text(0.5, 0.5, no_data_msg, 
               ha='center', va='center', transform=ax.transAxes)
        
        # Set title based on context
        if standalone_plot:
            ax.set_title(f'Training Loss Evolution - Sparsity s = {sparsity_value:.2f}')
            plt.tight_layout()
            # Save with sparsity-specific filename
            sparsity_str = f'{sparsity_value:.2f}'.replace('.', 'p')
            save_figure_with_sparsity(fig, f'training_loss_evolution_sparsity_{sparsity_str}', sparsity_value)
            plt.close(fig)
        else:
            ax.set_title(f'Sparsity s = {sparsity_value:.2f}')
        return
    
    # Plot the loss evolution
    adam_iterations = [iter for iter, color in zip(all_iterations, phase_colors) if color == 'blue']
    adam_losses = [loss for loss, color in zip(all_losses, phase_colors) if color == 'blue']
    lbfgs_iterations = [iter for iter, color in zip(all_iterations, phase_colors) if color == 'red']
    lbfgs_losses = [loss for loss, color in zip(all_losses, phase_colors) if color == 'red']
    
    # Adjust styling based on context
    markersize = 6 if standalone_plot else 4
    linewidth = 2
    
    if adam_iterations:
        ax.plot(adam_iterations, adam_losses, 'b-o', linewidth=linewidth, markersize=markersize, label='Adam', alpha=0.8)
    if lbfgs_iterations:
        ax.plot(lbfgs_iterations, lbfgs_losses, 'r-o', linewidth=linewidth, markersize=markersize, label='L-BFGS', alpha=0.8)
    
    # Mark the final best loss
    final_loss = training_loss_data['final']
    if all_iterations:
        final_label = f'Final best loss: {final_loss:.2e}' if standalone_plot else f'Final: {final_loss:.2e}'
        ax.axhline(y=final_loss, color='green', linestyle='--', alpha=0.7, label=final_label)
    
    # Formatting
    ax.set_xlabel('Training Iteration', fontsize=18)
    ax.set_ylabel('Training Loss', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Set title based on context - remove final loss for publication readiness
    if standalone_plot:
        ax.set_title(f'Training Loss Evolution - Sparsity s = {sparsity_value:.2f}', fontsize=20)
    else:
        ax.set_title(f'Sparsity s = {sparsity_value:.2f}', fontsize=20)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='upper right')

    # Set consistent y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Save only for standalone plots
    if standalone_plot:
        plt.tight_layout()
        sparsity_str = f'{sparsity_value:.2f}'.replace('.', 'p')
        save_figure_with_sparsity(fig, f'training_loss_evolution_sparsity_{sparsity_str}', sparsity_value)
        plt.close(fig)




# =============================================================================
# =============================================================================
# EIGENVALUE EVOLUTION PLOTTING FUNCTIONS
# =============================================================================
# =============================================================================
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
            # Create separate plot for each sparsity level
            for sparsity_idx, result in enumerate(all_results):
                plot_jacobian_eigenvalue_evolution_for_single_sparsity(result, freq_idx, cmap)


def plot_connectivity_eigenvalue_evolution(all_results, sparsity_values, cmap):
    """
    Plot connectivity matrix eigenvalue evolution for all sparsity levels.
    """
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts (same as other aggregate plots)
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_sparsity == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Store scatter plot for colorbar
    scatter_plot = None
    
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
            s=1
        )
        
        # Store the first valid scatter plot for colorbar
        if scatter_plot is None and len(all_eigenvals) > 0:
            scatter_plot = scatter
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Real', fontsize=18)
        ax.set_ylabel('Imaginary', fontsize=18)
        ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Hide unused subplots if any
    for idx in range(n_sparsity, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar to the right of all subplots
    if scatter_plot is not None:
        plt.tight_layout()
        # Create space for colorbar
        fig.subplots_adjust(right=0.85)
        # Add colorbar
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Training Progress', fontsize=18)
        cbar.ax.tick_params(labelsize=15)
    
    # Remove super title for publication readiness
    # plt.suptitle('Connectivity Matrix Eigenvalue Evolution During Training', fontsize=16)
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'connectivity_eigenvalue_evolution_sparsities_{sparsity_str}')
    plt.close(fig)  # Close figure to free memory
    # plt.show()  # Commented out to prevent interactive display


def plot_jacobian_eigenvalue_evolution_for_single_sparsity(result, freq_idx, cmap):
    """
    Plot Jacobian eigenvalue evolution for a specific frequency and single sparsity level.
    """
    sparsity = result['sparsity']
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
        ax.set_title(f'Sparsity s = {sparsity:.2f}\nFreq {freq_idx}: ω = {freq_value:.3f} rad/s')
        plt.tight_layout()
        
        # Save with sparsity-specific filename in sparsity subdirectory
        sparsity_str = f'{sparsity:.2f}'.replace('.', 'p')
        freq_str = f'{freq_value:.3f}'.replace('.', 'p')
        save_figure_with_sparsity(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_sparsity_{sparsity_str}_omega_{freq_str}', sparsity)
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
        s=1
    )
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Jacobian Eigenvalue Evolution During Training\n'
                 f'Sparsity s = {sparsity:.2f} | Frequency {freq_idx}: ω = {freq_value:.3f} rad/s\n'
                 f'Static input = {static_input_value:.3f} | Final loss: {final_loss:.2e}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Progress')
    
    plt.tight_layout()
    
    # Save with sparsity-specific filename in sparsity subdirectory
    sparsity_str = f'{sparsity:.2f}'.replace('.', 'p')
    freq_str = f'{freq_value:.3f}'.replace('.', 'p')
    save_figure_with_sparsity(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_sparsity_{sparsity_str}_omega_{freq_str}', sparsity)
    # plt.show()  # Commented out to prevent interactive display
    plt.close(fig)




# =============================================================================
# =============================================================================
# CONNECTIVITY EIGENVALUE BEFORE/AFTER SPARSIFICATION PLOTTING FUNCTIONS
# =============================================================================
# =============================================================================
def plot_connectivity_eigenvalues_before_after(dense_J, sparse_J, sparsity_value):
    """
    Plot connectivity matrix eigenvalues before and after sparsification for a single sparsity level.
    
    Arguments:
        dense_J: dense connectivity matrix
        sparse_J: sparsified connectivity matrix  
        sparsity_value: sparsity level applied
    """
    # Compute eigenvalues
    dense_eigenvals = np.linalg.eigvals(np.array(dense_J))
    sparse_eigenvals = np.linalg.eigvals(np.array(sparse_J))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot eigenvalues
    ax.scatter(dense_eigenvals.real, dense_eigenvals.imag, 
               c='blue', alpha=0.7, s=20, label='Before sparsification (dense)')
    ax.scatter(sparse_eigenvals.real, sparse_eigenvals.imag, 
               c='red', alpha=0.7, s=20, label=f'After sparsification (s={sparsity_value})')
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('Real Part', fontsize=18)
    ax.set_ylabel('Imaginary Part', fontsize=18)
    ax.set_title(f'Connectivity Matrix Eigenvalues\nBefore vs After Sparsification (s={sparsity_value})', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set axis tick fontsize
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Set axis limits to show unit circle clearly
    max_val = max(np.max(np.abs(dense_eigenvals)), np.max(np.abs(sparse_eigenvals)), 1.2)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    # Save figure
    sparsity_str = f"{sparsity_value:.3f}".replace('.', 'p')
    save_figure_with_sparsity(fig, f"connectivity_eigenvalues_before_after_sparsity_{sparsity_str}", sparsity_value)
    plt.close(fig)


def plot_connectivity_eigenvalues_before_after_aggregate(all_results, sparsity_values):
    """
    Plot connectivity matrix eigenvalues before and after sparsification for all sparsity levels.
    
    Arguments:
        all_results: list of results dictionaries for each sparsity level
        sparsity_values: list of sparsity values tested
    """
    # Determine subplot layout
    n_sparsity = len(sparsity_values)
    n_cols = min(3, n_sparsity)
    n_rows = (n_sparsity + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_sparsity == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get dense J matrix from first result (should be same for all)
    dense_J = None
    for result in all_results:
        if 'dense_J' in result:
            dense_J = result['dense_J']
            break
    
    if dense_J is None:
        print("Warning: No dense J matrix found in results")
        return
    
    dense_eigenvals = np.linalg.eigvals(np.array(dense_J))
    
    for i, (result, sparsity) in enumerate(zip(all_results, sparsity_values)):
        ax = axes[i]
        
        # Get sparsified J matrix
        sparse_J = result['trained_params']['J']
        sparse_eigenvals = np.linalg.eigvals(np.array(sparse_J))
        
        # Plot eigenvalues
        ax.scatter(dense_eigenvals.real, dense_eigenvals.imag, 
                   c='blue', alpha=0.7, s=20, label='Before (dense)')
        ax.scatter(sparse_eigenvals.real, sparse_eigenvals.imag, 
                   c='red', alpha=0.7, s=20, label=f'After (s={sparsity})')
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('Real Part', fontsize=18)
        ax.set_ylabel('Imaginary Part', fontsize=18)
        ax.set_title(f'Sparsity = {sparsity}', fontsize=20)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        
        # Set axis tick fontsize
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        # Set axis limits
        max_val = max(np.max(np.abs(dense_eigenvals)), np.max(np.abs(sparse_eigenvals)), 1.2)
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
    
    # Hide unused subplots
    for i in range(n_sparsity, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    sparsity_str = "_".join([f"{s:.3f}".replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f"connectivity_eigenvalues_before_after_aggregate_sparsities_{sparsity_str}")
    plt.close(fig)




# =============================================================================
# =============================================================================
# SUMMARY PLOTTING FUNCTIONS
# =============================================================================
# =============================================================================
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
    
    # Check if this is import mode (final_loss is None)
    is_import_mode = all_results and all_results[0]['final_loss'] is None
    
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
    
    # Plot 1: Final loss vs sparsity (only for training mode)
    if not is_import_mode and any(loss is not None for loss in final_losses):
        axes[0, 0].plot(sparsity_values, final_losses, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Sparsity')
        axes[0, 0].set_ylabel('Final Loss')
        axes[0, 0].set_title('Training Loss vs Sparsity')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Training loss not available\n(Import mode)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Training Loss vs Sparsity')
    
    # Plot 2: Number of fixed points vs sparsity
    axes[0, 1].plot(sparsity_values, num_fixed_points, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sparsity')
    axes[0, 1].set_ylabel('Number of Fixed Points')
    axes[0, 1].set_title('Fixed Points vs Sparsity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral radius vs sparsity
    axes[1, 0].plot(sparsity_values, spectral_radii, 'go-', linewidth=2, markersize=8)
    # Add horizontal reference line at y=1.0 (unit circle)
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Unit circle')
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
    plt.close(fig)  # Close figure to free memory
    # plt.show()  # Commented out to prevent interactive display


def create_unstable_eigenvalue_table(all_results, sparsity_values):
    """
    Create a table showing the distribution of unstable eigenvalues per fixed point across sparsity levels.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
    """
    # Collect unstable eigenvalue counts for each sparsity level
    unstable_counts_per_sparsity = {}
    max_unstable_count = 0
    
    for result in all_results:
        sparsity = result['sparsity']
        unstable_counts = []
        
        # Extract unstable eigenvalue counts from fixed points
        # Add error handling for missing keys
        if 'all_unstable_eig_freq' not in result:
            print(f"Warning: Missing 'all_unstable_eig_freq' key for sparsity {sparsity}")
            unstable_counts_per_sparsity[sparsity] = []
            continue
            
        if result['all_unstable_eig_freq'] and len(result['all_unstable_eig_freq']) > 0:
            for task_unstable_freqs in result['all_unstable_eig_freq']:
                if task_unstable_freqs:  # Check if not None/empty
                    for fp_unstable_freqs in task_unstable_freqs:
                        if fp_unstable_freqs is not None:
                            # Count unstable eigenvalues for this fixed point
                            num_unstable = len(fp_unstable_freqs)
                            unstable_counts.append(num_unstable)
                            max_unstable_count = max(max_unstable_count, num_unstable)
        
        unstable_counts_per_sparsity[sparsity] = unstable_counts
    
    # If no unstable eigenvalues found, create a simple message plot
    if max_unstable_count == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No unstable eigenvalues found\nacross any sparsity levels', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Unstable Eigenvalue Distribution Table')
        plt.tight_layout()
        
        sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'unstable_eigenvalue_table_sparsities_{sparsity_str}')
        plt.close(fig)
        return
    
    # Create the table: rows = number of unstable eigenvalues, cols = sparsity levels
    table_data = np.zeros((max_unstable_count + 1, len(sparsity_values)), dtype=int)
    
    for sparsity_idx, sparsity in enumerate(sparsity_values):
        unstable_counts = unstable_counts_per_sparsity[sparsity]
        
        # Count how many fixed points have each number of unstable eigenvalues
        for num_unstable in range(max_unstable_count + 1):
            count = unstable_counts.count(num_unstable)
            table_data[num_unstable, sparsity_idx] = count
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, len(sparsity_values) * 1.5), max(6, max_unstable_count * 0.8)))
    
    # Create heatmap
    im = ax.imshow(table_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(sparsity_values)))
    ax.set_yticks(np.arange(max_unstable_count + 1))
    ax.set_xticklabels([f'{s:.2f}' for s in sparsity_values])
    ax.set_yticklabels([f'{i}' for i in range(max_unstable_count + 1)])
    
    # Labels and title
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel('Number of Unstable Eigenvalues per Fixed Point')
    ax.set_title('Distribution of Unstable Eigenvalues per Fixed Point Across Sparsity Levels')
    
    # Add text annotations
    for i in range(max_unstable_count + 1):
        for j in range(len(sparsity_values)):
            text = ax.text(j, i, f'{table_data[i, j]}',
                         ha="center", va="center", color="black" if table_data[i, j] < table_data.max()/2 else "white",
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Fixed Points')
    
    plt.tight_layout()
    
    # Save with sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'unstable_eigenvalue_table_sparsities_{sparsity_str}')
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("UNSTABLE EIGENVALUE DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for sparsity_idx, sparsity in enumerate(sparsity_values):
        unstable_counts = unstable_counts_per_sparsity[sparsity]
        total_fps = len(unstable_counts)
        if total_fps > 0:
            avg_unstable = np.mean(unstable_counts)
            max_unstable_this_sparsity = max(unstable_counts) if unstable_counts else 0
            print(f"Sparsity {sparsity:.2f}: {total_fps} fixed points, "
                  f"avg {avg_unstable:.1f} unstable eigenvalues, max {max_unstable_this_sparsity}")
        else:
            print(f"Sparsity {sparsity:.2f}: No fixed points found")
    print(f"{'='*60}")


def create_fixed_points_per_task_table(all_results, sparsity_values):
    """
    Create a table showing the distribution of fixed points per task across sparsity levels.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
    """
    # Collect fixed point counts per task for each sparsity level
    fp_counts_per_sparsity = {}
    max_fp_count = 0
    
    for result in all_results:
        sparsity = result['sparsity']
        fp_counts = []
        
        # Extract fixed point counts per task
        if result['all_fixed_points'] and len(result['all_fixed_points']) > 0:
            for task_fixed_points in result['all_fixed_points']:
                num_fps = len(task_fixed_points) if task_fixed_points else 0
                fp_counts.append(num_fps)
                max_fp_count = max(max_fp_count, num_fps)
        else:
            # If no fixed points data, assume 0 fixed points for all tasks
            fp_counts = [0] * num_tasks
        
        fp_counts_per_sparsity[sparsity] = fp_counts
    
    # If no fixed points found, create a simple message plot
    if max_fp_count == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No fixed points found\nacross any sparsity levels', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Fixed Points per Task Distribution Table')
        plt.tight_layout()
        
        sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'fixed_points_per_task_table_sparsities_{sparsity_str}')
        plt.close(fig)
        return
    
    # Create the table: rows = number of fixed points per task, cols = sparsity levels
    table_data = np.zeros((max_fp_count + 1, len(sparsity_values)), dtype=int)
    
    for sparsity_idx, sparsity in enumerate(sparsity_values):
        fp_counts = fp_counts_per_sparsity[sparsity]
        
        # Count how many tasks have each number of fixed points
        for num_fps in range(max_fp_count + 1):
            count = fp_counts.count(num_fps)
            table_data[num_fps, sparsity_idx] = count
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, len(sparsity_values) * 1.5), max(6, max_fp_count * 0.8)))
    
    # Create heatmap
    im = ax.imshow(table_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(sparsity_values)))
    ax.set_yticks(np.arange(max_fp_count + 1))
    ax.set_xticklabels([f'{s:.2f}' for s in sparsity_values])
    ax.set_yticklabels([f'{i}' for i in range(max_fp_count + 1)])
    
    # Labels and title
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel('Number of Fixed Points per Task')
    ax.set_title('Distribution of Fixed Points per Task Across Sparsity Levels')
    
    # Add text annotations
    for i in range(max_fp_count + 1):
        for j in range(len(sparsity_values)):
            text = ax.text(j, i, f'{table_data[i, j]}',
                         ha="center", va="center", color="black" if table_data[i, j] < table_data.max()/2 else "white",
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Tasks')
    
    plt.tight_layout()
    
    # Save with sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'fixed_points_per_task_table_sparsities_{sparsity_str}')
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("FIXED POINTS PER TASK DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for sparsity_idx, sparsity in enumerate(sparsity_values):
        fp_counts = fp_counts_per_sparsity[sparsity]
        total_tasks = len(fp_counts)
        if total_tasks > 0:
            avg_fps = np.mean(fp_counts)
            std_fps = np.std(fp_counts)
            min_fps = np.min(fp_counts)
            max_fps = np.max(fp_counts)
            print(f"Sparsity {sparsity:.2f}: {total_tasks} tasks, "
                  f"avg {avg_fps:.1f}±{std_fps:.1f} FPs/task, "
                  f"range [{min_fps}-{max_fps}] FPs/task")
        else:
            print(f"Sparsity {sparsity:.2f}: No task data available")
    print(f"{'='*60}")




# =============================================================================
# =============================================================================
# AGGREGATE PLOTTING FUNCTIONS (CROSS-SPARSITY)
# =============================================================================
# =============================================================================
def plot_training_loss_evolution_comparison(all_results, sparsity_values):
    """
    Create visualization showing training loss evolution for all sparsity levels.
    This function now reuses the individual plotting logic to avoid code duplication.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
    """
    # Check if this is import mode (training_loss_data is empty)
    if all_results and not all_results[0]['training_loss_data']:
        print("Skipping training loss evolution comparison - not applicable for import mode")
        return
    
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_sparsity == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Calculate global y-axis limits for consistent scaling
    all_loss_values = []
    for result in all_results:
        training_loss_data = result['training_loss_data']
        
        # Collect all loss values from both training phases
        for iteration, loss in training_loss_data['adam']:
            all_loss_values.append(loss)
        for iteration, loss in training_loss_data['lbfgs']:
            all_loss_values.append(loss)
        
        # Also include final loss
        if 'final' in training_loss_data:
            all_loss_values.append(training_loss_data['final'])
    
    # Determine y-axis limits with some padding
    if all_loss_values:
        min_loss = min(all_loss_values)
        max_loss = max(all_loss_values)
        
        # Add padding in log space
        log_min = np.log10(min_loss)
        log_max = np.log10(max_loss)
        log_range = log_max - log_min
        padding = 0.1 * log_range if log_range > 0 else 0.5
        
        ylim = (10**(log_min - padding), 10**(log_max + padding))
    else:
        ylim = None
    
    # Use the individual plotting function for each sparsity level
    for sparsity_idx, result in enumerate(all_results):
        ax = axes[sparsity_idx]
        
        # Call the individual plotting function with the provided axis and consistent y-limits
        plot_training_loss_evolution(
            training_loss_data=result['training_loss_data'],
            sparsity_value=result['sparsity'],
            ax=ax,
            for_comparison=True,
            ylim=ylim
        )
    
    # Hide unused subplots if any
    for idx in range(n_sparsity, len(axes)):
        axes[idx].set_visible(False)
    
    # Remove super title for publication readiness
    # plt.suptitle('Training Loss Evolution Across Sparsity Levels', fontsize=16)
    plt.tight_layout()
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'training_loss_evolution_sparsities_{sparsity_str}')
    plt.close(fig)  # Close figure to free memory


def plot_trajectories_vs_targets_comparison(all_results, sparsity_values, test_index):
    """
    Create visualization showing trajectory vs target for specific test index across all sparsity levels.
    This function now reuses the individual plotting logic to avoid code duplication.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
        test_index: specific test index to plot
    """
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_sparsity == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Get task information for title
    j = test_index
    omega = omegas[j]
    u_off = static_inputs[j]
    
    # Common initial state for all sparsity levels
    x0_test = jnp.zeros((N,))
    
    # Use the individual plotting function for each sparsity level
    for sparsity_idx, result in enumerate(all_results):
        ax = axes[sparsity_idx]
        sparsity = result['sparsity']
        trained_params = result['trained_params']
        final_loss = result['final_loss']
        
        try:
            # Call the individual plotting function with the provided axis
            error = plot_single_trajectory_vs_target(
                params=trained_params,
                task_index=j,
                x0_test=x0_test,
                ax=ax,
                for_comparison=True
            )
            
            # Add sparsity information to the title for comparison context (remove omega and MSE for publication)
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
            
            # Enhance axis formatting for publication readiness
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(18)

        except Exception as e:
            # ax.text(0.5, 0.5, f'Simulation failed\n{str(e)[:30]}...', 
            #        ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
    
    # Hide unused subplots if any
    for idx in range(n_sparsity, len(axes)):
        axes[idx].set_visible(False)
    
    # Remove super title for publication readiness
    # plt.suptitle(f'Trajectory vs Target Across Sparsity Levels\nTask {j}: ω = {omega:.3f} rad/s, u_offset = {u_off:.3f}', fontsize=16)
    plt.tight_layout()
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'trajectory_vs_target_task_{j}_sparsities_{sparsity_str}')
    plt.close(fig)


def plot_frequency_comparison_across_sparsity(all_results, sparsity_values):
    """
    Create visualization showing frequency comparison plots across all sparsity levels.
    This function now reuses the individual plotting logic to avoid code duplication.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
    """
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_sparsity == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Use the individual plotting function for each sparsity level
    for sparsity_idx, result in enumerate(all_results):
        ax = axes[sparsity_idx]
        sparsity = result['sparsity']
        all_unstable_eig_freq = result['all_unstable_eig_freq']
        final_loss = result['final_loss']
        
        # Check if this is import mode (final_loss is None)
        is_import_mode = final_loss is None
        
        if not all_unstable_eig_freq or len(all_unstable_eig_freq) == 0:
            ax.text(0.5, 0.5, 'No unstable\neigenvalue data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
            continue
        
        try:
            # Call the individual plotting function with the provided axis
            plot_frequency_comparison(
                all_unstable_eig_freq=all_unstable_eig_freq,
                omegas=omegas,
                ax=ax,
                sparsity_value=sparsity,
                for_comparison=True
            )
            
            # Add final loss information to the title only if available (training mode)
            current_title = ax.get_title()
            if not is_import_mode and final_loss is not None:
                if current_title and not current_title.startswith('Sparsity'):
                    ax.set_title(f'Sparsity s = {sparsity:.2f}\nFinal loss: {final_loss:.2e}')
                elif 'Sparsity' in current_title and 'Final loss' not in current_title:
                    ax.set_title(f'{current_title}\nFinal loss: {final_loss:.2e}')
            else:
                # Import mode - just show sparsity
                if current_title and not current_title.startswith('Sparsity'):
                    ax.set_title(f'Sparsity s = {sparsity:.2f}')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot failed\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
    
    # Hide unused subplots if any
    for idx in range(n_sparsity, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title and formatting
    # plt.suptitle('Frequency Comparison Across Sparsity Levels', fontsize=16)
    plt.tight_layout()
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'frequency_comparison_sparsities_{sparsity_str}')
    plt.close(fig)


def plot_pca_explained_variance_comparison(all_results, sparsity_values, skip_steps, apply_tanh):
    """
    Create visualization showing PCA explained variance ratio across all sparsity levels.
    This function now reuses the individual plotting logic to avoid code duplication.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
        skip_steps: skip steps parameter for PCA
        apply_tanh: tanh application parameter for PCA
    """
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_sparsity == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    pca_key = f"skip_{skip_steps}_tanh_{apply_tanh}"
    
    # Use the individual plotting function for each sparsity level
    for sparsity_idx, result in enumerate(all_results):
        ax = axes[sparsity_idx]
        sparsity = result['sparsity']
        pca_results = result['pca_results']
        final_loss = result['final_loss']
        
        if pca_key not in pca_results:
            ax.text(0.5, 0.5, f'No PCA data for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
            continue
        
        pca_data = pca_results[pca_key]
        pca_object = pca_data['pca']
        
        try:
            # Call the individual plotting function with the provided axis
            plot_explained_variance_ratio(
                pca=pca_object,
                skip_initial_steps=skip_steps,
                apply_tanh=apply_tanh,
                ax=ax,
                sparsity_value=sparsity,
                for_comparison=True
            )
            
            # Format for publication readiness: enhance title and font sizes, enlarge grey boxes
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Make grey boxes larger by adjusting bar width in future calls
            # (This would need to be done in the individual plotting function if accessible)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot failed\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
    
    # Hide unused subplots if any
    for idx in range(n_sparsity, len(axes)):
        axes[idx].set_visible(False)
    
    # Remove super title for publication readiness
    # plt.suptitle(f'PCA Explained Variance Ratio Across Sparsity Levels\n(skip_steps={skip_steps}, apply_tanh={apply_tanh})', fontsize=16)
    plt.tight_layout()
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'pca_explained_variance_ratio_skip_{skip_steps}_tanh_{apply_tanh}_sparsities_{sparsity_str}')
    plt.close(fig)


def plot_pca_3d_comparison(all_results, sparsity_values, skip_steps, apply_tanh, plot_type='fixed'):
    """
    Create 3D PCA plots across all sparsity levels.
    This function now reuses the individual plotting logic to avoid code duplication.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
        skip_steps: skip steps parameter for PCA
        apply_tanh: tanh application parameter for PCA
        plot_type: 'fixed' for regular plot, 'subset' for subset plot
    """
    n_sparsity = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_sparsity <= 3:
        nrows, ncols = 1, n_sparsity
    elif n_sparsity <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_sparsity + 2) // 3
    
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    
    pca_key = f"skip_{skip_steps}_tanh_{apply_tanh}"
    
    # Use the individual plotting function for each sparsity level
    for sparsity_idx, result in enumerate(all_results):
        sparsity = result['sparsity']
        pca_results = result['pca_results']
        final_loss = result['final_loss']
        
        print(f"\nProcessing sparsity {sparsity:.2f} for PCA 3D comparison:")
        print(f"  Available PCA result keys: {list(pca_results.keys()) if pca_results else 'None'}")
        
        ax = fig.add_subplot(nrows, ncols, sparsity_idx + 1, projection='3d')
        
        if pca_key not in pca_results:
            ax.text(0.5, 0.5, 0.5, f'No PCA data for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
            print(f"WARNING: No PCA data found for key '{pca_key}' in sparsity {sparsity:.2f}")
            print(f"Available PCA keys: {list(pca_results.keys())}")
            continue
        
        pca_data = pca_results[pca_key]
        pca_object = pca_data.get('pca', None)
        proj_trajs = pca_data.get('proj_trajs', [])
        
        # Debug information
        print(f"DEBUG: Sparsity {sparsity:.2f}, PCA key '{pca_key}':")
        print(f"  - PCA object available: {pca_object is not None}")
        print(f"  - Number of projected trajectories: {len(proj_trajs) if proj_trajs else 0}")
        print(f"  - PCA data keys: {list(pca_data.keys())}")
        
        if pca_object is None:
            ax.text(0.5, 0.5, 0.5, f'No PCA object for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
            continue
        
        # Get fixed points from main results, not from PCA data
        all_fixed_points = result.get('all_fixed_points', [])
        proj_fixed_points = pca_data.get('proj_fixed', np.array([]))
        
        print(f"  - Fixed points from result: {len(all_fixed_points) if all_fixed_points else 0} tasks")
        print(f"  - Projected fixed points shape: {proj_fixed_points.shape if proj_fixed_points.size > 0 else 'Empty'}")
        if all_fixed_points:
            total_fps = sum(len(task_fps) for task_fps in all_fixed_points if task_fps is not None)
            print(f"  - Total fixed points across all tasks: {total_fps}")
        
        if not proj_trajs or len(proj_trajs) == 0:
            ax.text(0.5, 0.5, 0.5, 'No projection\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}')
            print(f"WARNING: No projected trajectories found for sparsity {sparsity:.2f}")
            continue
        
        print(f"  - First trajectory shape: {proj_trajs[0].shape if len(proj_trajs) > 0 else 'N/A'}")
        print(f"  - All trajectory shapes: {[traj.shape for traj in proj_trajs[:3]]}...")  # Show first 3
        
        # Define which trajectories to plot based on plot_type
        if plot_type == 'subset':
            # Use TEST_INDICES_EXTREMES if available, otherwise first few tasks
            try:
                task_indices = TEST_INDICES_EXTREMES[:min(len(TEST_INDICES_EXTREMES), len(proj_trajs))]
                # Filter trajectories to plot only the subset
                subset_proj_trajs = [proj_trajs[i] for i in task_indices if i < len(proj_trajs)]
            except (NameError, AttributeError):
                task_indices = list(range(min(3, len(proj_trajs))))
                subset_proj_trajs = proj_trajs[:min(3, len(proj_trajs))]
            
            # Also filter fixed points if they exist per task
            if all_fixed_points and len(all_fixed_points) > 0:
                subset_all_fixed_points = [all_fixed_points[i] if i < len(all_fixed_points) else [] for i in task_indices]
            else:
                subset_all_fixed_points = all_fixed_points
        else:
            subset_proj_trajs = proj_trajs
            subset_all_fixed_points = all_fixed_points
        
        try:
            # Call the individual plotting function with the provided axis
            plot_pca_trajectories_and_points(
                pca=pca_object,
                proj_trajs=subset_proj_trajs,
                point_type="fixed",
                all_points=subset_all_fixed_points,
                proj_points=proj_fixed_points,
                params=result.get('trained_params', {}),
                skip_initial_steps=skip_steps,
                apply_tanh=apply_tanh,
                filename_prefix="",
                ax=ax,
                sparsity_value=sparsity,
                for_comparison=True
            )
            
            # Format for publication readiness: remove legend, clean title, enhance font sizes
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=24)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()  # Remove legend for cleaner publication appearance
            
            # Enhance axis label font sizes with proper padding for 3D plots
            ax.set_xlabel(ax.get_xlabel(), fontsize=18, labelpad=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18, labelpad=10)
            ax.set_zlabel(ax.get_zlabel() if hasattr(ax, 'get_zlabel') else '', fontsize=18, labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=15)
            
            # Adjust 3D plot view and spacing
            ax.view_init(elev=20, azim=45)  # Set a good viewing angle
            
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f'Plot failed\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sparsity s = {sparsity:.2f}', fontsize=20)
    
    plot_name = 'pca_plot_fixed' if plot_type == 'fixed' else 'subset_of_indices_pca_plot_fixed'
    
    # Remove super title for publication readiness
    # plt.suptitle(f'3D PCA Plots Across Sparsity Levels ({plot_name})\n(skip_steps={skip_steps}, apply_tanh={apply_tanh})', fontsize=16)
    
    # Use subplots_adjust for better control over 3D plot spacing
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.2, hspace=0.3)
    
    # Create sparsity-specific filename
    sparsity_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
    save_figure(fig, f'{plot_name}_skip_{skip_steps}_tanh_{apply_tanh}_sparsities_{sparsity_str}')
    plt.close(fig)


def create_all_aggregate_plots(all_results, sparsity_values):
    """
    Create all aggregate plots across sparsity levels.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values tested
    """
    print("\n" + "="*60)
    print("CREATING AGGREGATE PLOTS ACROSS SPARSITY LEVELS")
    print("="*60)
    
    # 1. Training loss evolution comparison
    print("Creating training loss evolution comparison...")
    plot_training_loss_evolution_comparison(all_results, sparsity_values)
    
    # 2. Trajectory vs target comparisons for each test index
    print("Creating trajectory vs target comparisons...")
    for test_idx in TEST_INDICES:
        print(f"  Processing test index {test_idx}...")
        plot_trajectories_vs_targets_comparison(all_results, sparsity_values, test_idx)
    
    # 3. Frequency comparison across sparsity
    print("Creating frequency comparison across sparsity...")
    plot_frequency_comparison_across_sparsity(all_results, sparsity_values)
    
    # 4. PCA plots for each combination of skip steps and tanh activations
    print("Creating PCA comparison plots...")
    for skip_steps in PCA_SKIP_OPTIONS:
        for apply_tanh in PCA_TANH_OPTIONS:
            print(f"  Processing skip_steps={skip_steps}, apply_tanh={apply_tanh}...")
            
            # PCA explained variance ratio
            plot_pca_explained_variance_comparison(all_results, sparsity_values, skip_steps, apply_tanh)
            
            # PCA 3D plots
            plot_pca_3d_comparison(all_results, sparsity_values, skip_steps, apply_tanh, 'fixed')
            plot_pca_3d_comparison(all_results, sparsity_values, skip_steps, apply_tanh, 'subset')
    
    print("Completed all aggregate plots!")




# =============================================================================
# =============================================================================
# MAIN LOOP FUNCTION
# =============================================================================
# =============================================================================
def main():
    """
    Main function to run comprehensive sparsity experiments with full analysis.
    Supports both training mode and import mode.
    """
    # Get mode selection from user
    mode = get_mode_selection()
    
    # Get sparsity values from user
    sparsity_values = get_sparsity_values()
    
    # Get eigenvalue computation preferences from user (for training mode)
    if mode == "train":
        compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies = get_eigenvalue_options()
    else:
        # For import mode, we don't need eigenvalue evolution tracking
        compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies = False, False, None
    
    # Load dense parameters if in import mode
    if mode == "import":
        dense_params = load_dense_parameters()
    
    # Setup custom output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if mode == "train":
        output_dir_name = f"Sparsity_Experiments_{timestamp}"
    else:
        output_dir_name = f"Sparsity_Experiments_from_Trained_{timestamp}"
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Move to parent directory, then to Outputs
    parent_dir = os.path.dirname(script_dir)
    output_base_path = os.path.join(parent_dir, 'Outputs')
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
    print(f"COMPREHENSIVE SPARSITY EXPERIMENTS - {mode.upper()} MODE")
    print(f"Testing {len(sparsity_values)} sparsity levels")
    if mode == "train":
        print(f"{eigenval_msg}")
        if eigenval_types:
            print(f"Eigenvalue sampling frequency: every {EIGENVALUE_SAMPLE_FREQUENCY} iterations")
    else:
        print("Using pre-trained dense parameters for sparsification")
    print(f"Output directory: {full_output_path}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(NUMPY_SEED)
    key = random.PRNGKey(JAX_SEED)
    
    # Setup base parameters based on mode
    if mode == "train":
        # Generate a single set of base parameters that will be used for all sparsity levels
        # This ensures all networks start from the same underlying structure, only varying in sparsity
        analysis_key, base_key = random.split(key)
    else:
        # Use pre-loaded dense parameters
        analysis_key = key
    
    # Store results for all sparsity levels
    all_results = []
    
    for i, sparsity in enumerate(sparsity_values):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING SPARSITY LEVEL {i+1}/{len(sparsity_values)}: s = {sparsity}")
        print(f"{'='*80}")
        
        # Initialize results structure with basic info
        results = {
            'sparsity': sparsity,
            'success': False,
            'error_message': None
        }
        
        try:
            # Generate a unique analysis key for this sparsity level
            analysis_key, subkey = random.split(analysis_key)
            
            if mode == "train":
                # Use the same base key for all sparsity levels to ensure consistent initial parameters
                # Only the sparsity mask will differ between runs
                mask, params = init_params_with_sparsity(base_key, sparsity)
                
                # Run full training and analysis pipeline
                results = train_with_full_analysis(params, mask, sparsity, subkey, 
                                                  compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
            else:
                # Run sparsification and analysis pipeline
                results = sparsify_and_analyze(dense_params, sparsity, subkey, 
                                             compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
            
            # Mark as successful if we get here
            results['success'] = True
            
            # Store results
            all_results.append(results)
            if mode == "train":
                print(f"\nCompleted sparsity {sparsity}: final loss = {results['final_loss']:.6e}")
            else:
                print(f"\nCompleted sparsity {sparsity}: sparsification and analysis")
                
        except Exception as e:
            error_msg = str(e)
            print(f"\nERROR processing sparsity {sparsity}: {error_msg}")
            print(f"Skipping this sparsity level and continuing with remaining levels...")
            
            # Store the error information for debugging
            results['error_message'] = error_msg
            results['success'] = False
            all_results.append(results)  # Still append so we know what failed
            
            # Continue with the next sparsity level instead of crashing
    
    # Check if we have any successful results before proceeding
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    if not successful_results:
        print(f"\nERROR: No successful results obtained for any sparsity level!")
        print("Cannot proceed with aggregate analysis. Exiting...")
        if failed_results:
            print("\nFailure summary:")
            for result in failed_results:
                sparsity = result.get('sparsity', 'unknown')
                error = result.get('error_message', 'unknown error')
                print(f"  s = {sparsity}: {error}")
        return
    
    print(f"\nSuccessfully processed {len(successful_results)} out of {len(sparsity_values)} sparsity levels.")
    if failed_results:
        print(f"Failed to process {len(failed_results)} sparsity levels:")
        for result in failed_results:
            sparsity = result.get('sparsity', 'unknown')
            error = result.get('error_message', 'unknown error')
            print(f"  s = {sparsity}: {error}")
    
    # Use only successful results for subsequent analysis
    working_results = successful_results
    working_sparsity_values = [r['sparsity'] for r in successful_results]
    

    # ========================================
    # CREATE MODE-SPECIFIC VISUALIZATIONS
    # ========================================
    if mode == "train":
        # CREATE COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION (TRAINING MODE)
        if compute_jacobian_eigenvals or compute_connectivity_eigenvals:
            print(f"\n{'='*60}")
            print("CREATING COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION")
            print(f"{'='*60}")
            
            try:
                plot_eigenvalue_evolution_comparison(working_results, working_sparsity_values)
                print("Successfully created eigenvalue evolution comparison.")
            except Exception as e:
                print(f"Error creating eigenvalue evolution comparison: {e}")
                print("Continuing with remaining visualizations...")
        else:
            print(f"\n{'='*60}")
            print("SKIPPING EIGENVALUE VISUALIZATION (NO EIGENVALUE DATA)")
            print(f"{'='*60}")
    else:
        # CREATE BEFORE/AFTER EIGENVALUE VISUALIZATION (IMPORT MODE)
        print(f"\n{'='*60}")
        print("CREATING BEFORE/AFTER EIGENVALUE VISUALIZATION")
        print(f"{'='*60}")
        
        try:
            plot_connectivity_eigenvalues_before_after_aggregate(working_results, working_sparsity_values)
            print("Successfully created before/after eigenvalue visualization.")
        except Exception as e:
            print(f"Error creating before/after eigenvalue visualization: {e}")
            print("Continuing with remaining visualizations...")
    

    # ========================================
    # CREATE COMPREHENSIVE SUMMARY PLOTS
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE SUMMARY VISUALIZATIONS")
    print(f"{'='*60}")
    
    try:
        create_sparsity_summary_plots(working_results, working_sparsity_values)
        print("Successfully created summary plots.")
    except Exception as e:
        print(f"Error creating summary plots: {e}")
        print("Continuing with remaining visualizations...")
    

    # ========================================
    # CREATE UNSTABLE EIGENVALUE DISTRIBUTION TABLE
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING UNSTABLE EIGENVALUE DISTRIBUTION TABLE")
    print(f"{'='*60}")
    
    try:
        create_unstable_eigenvalue_table(working_results, working_sparsity_values)
        print("Successfully created unstable eigenvalue distribution table.")
    except Exception as e:
        print(f"Error creating unstable eigenvalue table: {e}")
        print("Continuing with remaining visualizations...")
    

    # ========================================
    # CREATE FIXED POINTS PER TASK DISTRIBUTION TABLE
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING FIXED POINTS PER TASK DISTRIBUTION TABLE")
    print(f"{'='*60}")
    
    try:
        create_fixed_points_per_task_table(working_results, working_sparsity_values)
        print("Successfully created fixed points per task distribution table.")
    except Exception as e:
        print(f"Error creating fixed points per task table: {e}")
        print("Continuing with remaining visualizations...")
    

    # ========================================
    # CREATE ALL AGGREGATE PLOTS ACROSS SPARSITY LEVELS
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING ALL AGGREGATE PLOTS ACROSS SPARSITY LEVELS")
    print(f"{'='*60}")
    
    try:
        create_all_aggregate_plots(working_results, working_sparsity_values)
        print("Successfully created all aggregate plots.")
    except Exception as e:
        print(f"Error creating aggregate plots: {e}")
        print("Continuing with saving experiment results...")
    
    

    # ========================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================
    print(f"\n{'='*60}")
    print("SAVING COMPLETE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    try:
        # Save all results together
        complete_experiment_results = {
            'sparsity_values': working_sparsity_values,
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
                'output_directory': full_output_path,
                'mode': mode,  # Add mode information
                'successful_sparsity_levels': len(working_results),
                'failed_sparsity_levels': len(failed_results),
                'total_sparsity_levels': len(sparsity_values)
            }
        }
        save_variable(complete_experiment_results, f"complete_sparsity_experiment_{timestamp}")
        print("Successfully saved complete experiment results.")
    except Exception as e:
        print(f"Error saving experiment results: {e}")
    
    

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SPARSITY EXPERIMENT COMPLETE - {mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Experiment timestamp: {timestamp}")
    print(f"Output directory: {full_output_path}")
    print(f"Tested sparsity values: {sparsity_values}")
    print(f"Successfully processed: {working_sparsity_values}")
    if failed_results:
        failed_sparsities = [r.get('sparsity', 'unknown') for r in failed_results]
        print(f"Failed to process: {failed_sparsities}")
    if mode == "train":
        print(f"Eigenvalue tracking: {eigenval_msg}")
        print("Final losses by sparsity:")
        for result in working_results:
            print(f"  s = {result['sparsity']:.3f}: loss = {result['final_loss']:.6e}")
    else:
        print("Mode: Post-training sparsification of pre-trained dense network")
    
    print(f"\nNumber of fixed points found by sparsity:")
    for result in working_results:
        if result['all_fixed_points']:
            num_fps = sum(len(task_fps) for task_fps in result['all_fixed_points'])
            print(f"  s = {result['sparsity']:.3f}: {num_fps} fixed points")
    
    if SLOW_POINT_SEARCH:
        print(f"\nNumber of slow points found by sparsity:")
        for result in working_results:
            if result['all_slow_points']:
                num_sps = sum(len(task_sps) for task_sps in result['all_slow_points'])
                print(f"  s = {result['sparsity']:.3f}: {num_sps} slow points")
    
    print(f"\nAll results and visualizations saved to: {full_output_path}")




if __name__ == "__main__":
    main()