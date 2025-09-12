"""
Sparsity experiments: Train RNN with different sparsity levels and track Jacobian eigenvalue evolution.

This script trains RNNs at different sparsity levels and visualizes how the eigenvalues
of the Jacobian matrix at fixed points evolve during the optimization process.


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
    - 'all_unstable_eig_freqs'
    - 'all_slow_points' (if SLOW_POINT_SEARCH is True)
    - 'all_slow_jacobians' (if SLOW_POINT_SEARCH is True)
    - 'all_slow_unstable_eig_freqs' (if SLOW_POINT_SEARCH is True)
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
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import scipy.stats
    import optax
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import JAX packages: {e}. Ensure jax & jaxlib are installed in the active environment.")

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import matplotlib: {e}. Install via 'pip install matplotlib'.")

# Quick runtime diagnostics
def _print_startup_diagnostics():
    import sys
    print("[Startup] Python:", sys.version.split()[0])
    try:
        import jax
        devs = jax.devices()
        print(f"[Startup] JAX devices ({len(devs)}):", devs)
        if not any(d.platform == 'gpu' for d in devs):
            print("[Startup][WARN] No GPU detected by JAX; computation will run on CPU.")
    except Exception as e:
        print(f"[Startup][ERROR] Could not query JAX devices: {e}")
_print_startup_diagnostics()

from functools import partial
from tqdm import tqdm
import time
import os
import gc
from contextlib import contextmanager

# Memory monitoring function
def print_memory_usage(label=""):
    """Print current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"[Memory] {label}: {memory_mb:.1f} MB")
        return memory_mb
    except ImportError:
        print(f"[Memory] {label}: psutil not available")
        return 0

def cleanup_memory():
    """Force garbage collection and JAX memory cleanup"""
    gc.collect()
    try:
        import jax
        # Clear JAX compilation cache periodically
        jax.clear_backends()
    except:
        pass

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
                     save_variable_with_sparsity, save_figure_with_sparsity, get_sparsity_output_dir,
                     save_variable_with_rank, save_figure_with_rank, get_rank_output_dir, rank_output_context)
from _3_data_generation import generate_all_inputs_and_targets
from _4_rnn_model import run_batch_diagnostics, run_batch_diagnostics_low_rank, simulate_trajectory, rnn_step_low_rank, simulate_trajectory_low_rank, solve_task_low_rank, vmap_solve_task_low_rank, batched_loss_low_rank
from _5_training import setup_optimizers, scan_with_history, apply_column_balancing, create_training_functions_low_rank, train_model_low_rank
from _6_analysis import find_points, compute_jacobian, find_and_analyze_points, generate_point_summaries
from _7_visualization import (save_figure, plot_single_trajectory_vs_target, plot_trajectories_vs_targets, plot_parameter_matrices_for_tasks, 
                          plot_frequency_comparison, analyze_jacobians_visualization, analyze_unstable_frequencies_visualization)
from _8_pca_analysis import plot_explained_variance_ratio, plot_pca_trajectories_and_points, run_pca_analysis


# OVERWRITE
# NUM_EPOCHS_LBFGS = 1000




# =============================================================================
# =============================================================================
# USER INPUT FUNCTIONS
# =============================================================================
# =============================================================================
def get_rank_values():
    """
    Get rank values from user input.
    
    Returns:
        list: List of rank values to test
    """
    print("=" * 60)
    print("LOW-RANK CONNECTIVITY EXPERIMENT SETUP")
    print("=" * 60)
    
    print(f"\nEnter rank values to test (between 1 and 200)")
    print("Examples: 1, 5, 10, 20")
    print("Enter values separated by commas (e.g., '1, 5, 10')")
    
    while True:
        try:
            user_input = input("Rank values: ").strip()
            if not user_input:
                print("Please enter at least one rank value.")
                continue
                
            rank_values = [int(x.strip()) for x in user_input.split(',')]
            
            # Validate rank values
            invalid_ranks = [r for r in rank_values if r < 1 or r > 200]
            if invalid_ranks:
                print(f"Invalid rank values: {invalid_ranks}. Rank must be between 1 and 200.")
                continue
                
            print(f"Selected rank values: {rank_values}")
            return rank_values
            
        except ValueError:
            print("Please enter valid integers separated by commas.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()


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
                if '-' not in range_input:
                    raise ValueError("Range must contain a dash (e.g., '0-10')")
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
# CONTEXT MANAGER AND PARAMETER DICTIONARY HELPER
# =============================================================================
# =============================================================================
@contextmanager
def connectivity_context(params):
    """
    Context manager to temporarily add the full connectivity matrix J = U @ V.T to params.
    This allows compatibility with visualization functions that expect J.
    """
    params_with_J = add_connectivity_matrix(params)
    try:
        yield params_with_J
    finally:
        pass  # No cleanup needed

@contextmanager
def rank_output_context(rank_value):
    """
    Context manager for setting up output directory for specific rank.
    """
    # Save current output directory
    original_output_dir = get_output_dir()
    
    # Set up rank-specific output directory
    rank_output_dir = get_sparsity_output_dir().replace('sparsity', 'rank')  # Reuse the structure
    rank_output_dir = rank_output_dir.replace(f'sparsity_{0.0}', f'rank_{rank_value}')
    set_custom_output_dir(rank_output_dir)
    
    try:
        yield rank_output_dir
    finally:
        # Restore original output directory
        set_custom_output_dir(original_output_dir)


def add_connectivity_matrix(params: dict) -> dict:
    """
    Add the reconstructed connectivity matrix J = U @ V.T to params dict.
    
    Args:
        params: parameter dictionary containing U and V
        
    Returns:
        params: parameter dictionary with J added
    """
    if "U" in params and "V" in params:
        return {**params, "J": params["U"] @ params["V"].T}
    else:
        return params  # Already has J or different structure




# =============================================================================
# =============================================================================
# PARAMETER INITIALIZATION FUNCTIONS
# =============================================================================
# =============================================================================
def make_general_covariance_from_orthogonal(key, R: int) -> jnp.ndarray:
    """
    Create a general covariance matrix for generating low-rank matrices using an orthogonal basis.
    
    The covariance is constructed as:
    1. Generate a random orthogonal matrix
    2. Generate a diagonal matrix with random positive entries
    3. Construct the covariance matrix as the product of the orthogonal matrix, diagonal matrix, and its transpose
    
    Args:
        key  : JAX PRNGKey for random generation
        R    : target rank
        
    Returns:
        General covariance matrix of shape (2R, 2R)
    """
    # Convert JAX PRNGKey to integer seed for SciPy
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    
    # Generate a random orthogonal matrix sampled uniformly from the orthogonal group
    Q = scipy.stats.ortho_group.rvs(dim=2*R, random_state=seed).astype(jnp.float64)  # shape (2R, 2R)
    
    # Generate a diagonal matrix with random entries sampled uniformly from (0.01, 1.0)
    D = jnp.diag(jax.random.uniform(key, (2*R,), minval=0.01, maxval=1.0, dtype=jnp.float64))
    
    # Construct the covariance matrix
    cov_general_from_orthogonal = Q @ D @ Q.T
    
    return cov_general_from_orthogonal


def generate_low_rank_matrix_factors(key, R: int, N: int, cov: jnp.ndarray):
    """
    Generate U and V factors for an N x N matrix of rank R as a sum of R outer products.

    For each i in 0, ..., N-1, draw the 2R-dimensional vector [m_1[i], ..., m_R[i], n_1[i], ..., n_R[i]]
    from a joint Gaussian with zero mean and covariance `cov` (shape 2R x 2R).

    Args:
      key   : JAX PRNGKey
      R     : target rank (R < N)
      N     : matrix size
      cov   : covariance matrix of shape (2R, 2R)

    Returns:
      U     : left factors of shape (N, R)
      V     : right factors of shape (N, R)
    """
    subkey, key = random.split(key)
    
    # # Sample N vectors, each of length 2R, from multivariate Gaussian
    # samples = random.multivariate_normal(subkey, jnp.zeros(2*R), cov, shape=(N,))
    
    # Sample N independent draws from N(0, cov)
    # using the Cholesky decomposition of cov
    Z       = random.normal(subkey, shape=(N, 2 * R), dtype=jnp.float64)        # shape (N, 2R), each element is inependently N(0, 1)
    L       = jnp.linalg.cholesky(cov)                       # L is a lower triangular matrix such that cov = L L^T (the Cholesky decomposition)
    samples = Z @ L.T                                        # shape (N, 2R), where each row (length 2R) is a sample from N(0, cov)

    # Split samples into matrices of the left and right vectors
    R_int = int(R)  # Ensure R is a Python integer for array slicing
    U  = samples[:, :R_int]        # shape (N, R), each column is a vector u_r
    V  = samples[:, R_int:]        # shape (N, R), each column is a vector v_r
    
    return U, V


def init_params_low_rank(key, rank):
    """
    Initialize RNN parameters with low-rank connectivity matrix.
    
    Arguments:
        key: JAX random key
        rank: desired rank of connectivity matrix
        
    Returns:
        params: dictionary containing U, V, B, b_x, w, b_z
    """
    k_cov, k_factors, k2, k3, k4, k5 = random.split(key, 6)

    # Generate covariance matrix for low-rank generation
    cov = make_general_covariance_from_orthogonal(k_cov, rank)
    
    # Generate U and V factors
    U, V = generate_low_rank_matrix_factors(k_factors, rank, N, cov)
    
    # Scale U and V each by 1/sqrt(1-s) (but s=0 for low-rank, so scale by 1)
    # Actually, we want to scale for spectral radius control, so scale by desired factor
    scaling_factor = 1.0  # No sparsity in low-rank case
    U = U * scaling_factor
    V = V * scaling_factor
    
    # Other parameters (same as original)
    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0)
    
    return {"U": U, "V": V, "B": B, "b_x": b_x, "w": w, "b_z": b_z}



# =============================================================================
# =============================================================================
# EIGENVALUE COMPUTATION FUNCTIONS
# =============================================================================
# =============================================================================
def compute_connectivity_eigenvalues(params):
    """
    Compute eigenvalues of the connectivity matrix J.
    For low-rank connectivity, reconstructs J = U @ V.T first.
    
    Arguments:
        params: RNN parameters (either with J or with U, V for low-rank)
        
    Returns:
        eigenvalues: complex array of eigenvalues of the connectivity matrix J
    """
    try:
        # Check if this is low-rank connectivity (U, V) or standard connectivity (J)
        if "U" in params and "V" in params:
            # Low-rank connectivity: reconstruct J = U @ V.T
            U = np.array(params["U"])
            V = np.array(params["V"])
            J = U @ V.T
        elif "J" in params:
            # Standard connectivity
            J = np.array(params["J"])
        else:
            raise ValueError("Parameters must contain either 'J' or both 'U' and 'V'")
            
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
    # Check if this is low-rank connectivity (U, V) or standard connectivity (J)
    if "U" in params and "V" in params:
        # Low-rank connectivity: reconstruct J = U @ V.T
        U_trained = np.array(params["U"])
        V_trained = np.array(params["V"])
        J_trained = U_trained @ V_trained.T
    elif "J" in params:
        # Standard connectivity
        J_trained = np.array(params["J"])
    else:
        raise ValueError("Parameters must contain either 'J' or both 'U' and 'V'")
        
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
# FULL RUN FUNCTION
# =============================================================================
# =============================================================================
def train_with_full_analysis(params, rank_value, key, compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies=None):
    """
    Train model with full analysis pipeline using low-rank connectivity.
    
    Arguments:
        params: initial parameters containing U, V matrices
        rank_value: current rank level being tested
        key: JAX random key for analysis
        compute_jacobian_eigenvals: whether to compute Jacobian eigenvalues during training
        compute_connectivity_eigenvals: whether to compute connectivity eigenvalues during training
        jacobian_frequencies: list of frequency indices for Jacobian eigenvalue computation
        
    Returns:
        results_dict: comprehensive results dictionary containing all analysis outputs
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WITH RANK r = {rank_value}")
    print(f"{'='*60}")
    
    
    # ========================================
    # 1. SETUP AND INITIALIZATION
    # ========================================
    results = {}
    results['rank'] = rank_value
    
    # Setup optimizers
    adam_opt, lbfgs_opt = setup_optimizers()
    
    # Initialize optimizer states
    adam_state = adam_opt.init(params)
    lbfgs_state = lbfgs_opt.init(params)
    
    # Create training functions for low-rank connectivity
    adam_step, lbfgs_step, loss_fn = create_training_functions_low_rank(adam_opt, lbfgs_opt)
    
    # Track eigenvalues during training
    eigenvalue_data = {}
    training_loss_data = {}
    

    # ========================================
    # 2. TRAINING WITH EIGENVALUE TRACKING
    # ========================================
    print("\n2. TRAINING WITH EIGENVALUE TRACKING")
    print("-" * 40)
    
    with Timer(f"Training with rank {rank_value}"):
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
    final_loss_check, state_traj_states, state_fixed_point_inits = run_batch_diagnostics_low_rank(trained_params, key)
    print(f"Final loss after training: {final_loss_check:.6e}")
    
    results['state_traj_states'] = state_traj_states
    results['state_fixed_point_inits'] = state_fixed_point_inits
    

    # ========================================
    # 4. SAVE TRAINING RESULTS
    # ========================================
    print("\n4. SAVING TRAINING RESULTS")
    print("-" * 40)
    
    print("Saving training results...")
    # For low-rank connectivity, reconstruct J = U @ V.T for saving
    J_reconstructed = trained_params["U"] @ trained_params["V"].T
    save_variable_with_rank(np.array(J_reconstructed), f"J_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["U"]), f"U_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["V"]), f"V_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["B"]), f"B_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["b_x"]), f"b_x_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["w"]), f"w_param_rank_{rank_value}", rank_value, r=rank_value)
    save_variable_with_rank(np.array(trained_params["b_z"]), f"b_z_param_rank_{rank_value}", rank_value, r=rank_value)

    state_dict = {
        "traj_states": state_traj_states,
        "fixed_point_inits": state_fixed_point_inits
    }
    save_variable_with_rank(state_dict, f"state_rank_{rank_value}", rank_value, r=rank_value)
    
    # Save eigenvalue evolution data during training
    print("Saving eigenvalue evolution data...")
    save_variable_with_rank(eigenvalue_data, f"eigenvalue_data_rank_{rank_value}", rank_value, r=rank_value)
    
    # Save training loss evolution data
    print("Saving training loss evolution data...")
    save_variable_with_rank(training_loss_data, f"training_loss_over_iterations_rank_{rank_value}", rank_value, r=rank_value)
    
    # Plot training loss evolution
    print("Plotting training loss evolution...")
    plot_training_loss_evolution(training_loss_data, rank_value)
    

    # ========================================
    # 5. VISUALIZATION OF BASIC RESULTS
    # ========================================
    print("\n5. VISUALIZATION OF BASIC RESULTS")
    print("-" * 40)
    
    # Plot trajectories vs targets (save in rank-specific folder)
    print("Plotting produced trajectories vs. target signals...")
    with connectivity_context(trained_params) as params_with_J:
        plot_trajectories_vs_targets(params_with_J, test_indices=TEST_INDICES, rank_value=rank_value)
    
    # Plot parameter matrices (save in rank-specific folder)
    with connectivity_context(trained_params) as params_with_J:
        plot_parameter_matrices_for_tasks(params_with_J, test_indices=[0], rank_value=rank_value)


    # ========================================
    # 6. FIXED POINT ANALYSIS
    # ========================================
    print("\n6. FIXED POINT ANALYSIS")
    print("-" * 40)
    
    with Timer("Fixed Point Analysis"):
        # Find and analyze fixed points
        with connectivity_context(trained_params) as params_with_J:
            all_fixed_points, all_jacobians, all_unstable_eig_freq = find_and_analyze_points(
                state_traj_states, state_fixed_point_inits, params_with_J, point_type="fixed"
            )
        
        # Save fixed point results
        save_variable_with_rank(all_fixed_points, f"all_fixed_points_rank_{rank_value}", rank_value, r=rank_value)
        save_variable_with_rank(all_jacobians, f"all_fixed_jacobians_rank_{rank_value}", rank_value, r=rank_value)
        save_variable_with_rank(all_unstable_eig_freq, f"all_fixed_unstable_eig_freq_rank_{rank_value}", rank_value, r=rank_value)
    
    results['all_fixed_points'] = all_fixed_points
    results['all_jacobians'] = all_jacobians
    results['all_unstable_eig_freq'] = all_unstable_eig_freq
    
    # Generate fixed point summaries
    generate_point_summaries(point_type="fixed", all_points=all_fixed_points, 
                            all_unstable_eig_freq=all_unstable_eig_freq)
    
    # Visualize fixed point analysis (save in rank-specific directory)
    with rank_output_context(rank_value):
        # Temporarily update the config for plot titles
        import _1_config
        original_s = _1_config.s
        _1_config.s = 0  # Use 0 for sparsity since we're using rank
        
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
            with connectivity_context(trained_params) as params_with_J:
                all_slow_points, all_slow_jacobians, all_slow_unstable_eig_freq = find_and_analyze_points(
                    state_traj_states, state_fixed_point_inits, params_with_J, point_type="slow"
                )
            
            # Save slow point results
            save_variable_with_rank(all_slow_points, f"all_slow_points_rank_{rank_value}", rank_value, r=rank_value)
            save_variable_with_rank(all_slow_jacobians, f"all_slow_jacobians_rank_{rank_value}", rank_value, r=rank_value)
            save_variable_with_rank(all_slow_unstable_eig_freq, f"all_slow_unstable_eig_freq_rank_{rank_value}", rank_value, r=rank_value)
        
        # Generate slow point summaries
        generate_point_summaries(point_type="slow", all_points=all_slow_points, 
                                all_unstable_eig_freq=all_slow_unstable_eig_freq)
        
        # Visualize slow point analysis (save in rank-specific directory)
        with rank_output_context(rank_value):
            # Temporarily update the config for plot titles
            import _1_config
            original_s = _1_config.s
            _1_config.s = 0  # Use 0 for sparsity since we're using rank
            
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
        
        # Run PCA analysis for all combinations of skip and tanh options
        for skip_steps in PCA_SKIP_OPTIONS:
            for apply_tanh in PCA_TANH_OPTIONS:
                print(f"  Processing skip_steps={skip_steps}, apply_tanh={apply_tanh}")
                
                # Run PCA analysis (save plots in rank-specific directory)
                with rank_output_context(rank_value):
                    # Temporarily update the config for plot titles
                    import _1_config
                    original_s = _1_config.s
                    _1_config.s = 0  # Use 0 for sparsity since we're using rank
                    
                    try:
                        with connectivity_context(trained_params) as params_with_J:
                            pca_result = run_pca_analysis(
                                state_traj_states, 
                                all_fixed_points=all_fixed_points,
                                all_slow_points=all_slow_points,
                                params=params_with_J,
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
        
        # Save PCA results
        # save_variable_with_sparsity(pca_results, f"pca_results_sparsity_{sparsity_value}", sparsity_value, s=sparsity_value)
        
        print(f"Completed PCA analysis for all combinations. Total PCA keys stored: {list(pca_results.keys())}")
    
    results['pca_results'] = pca_results
    
    print(f"Completed full analysis for rank {rank_value}: final loss = {final_loss:.6e}")
    
    return results




# =============================================================================
# =============================================================================
# TRAINING LOSS PLOTTING FUNCTION
# =============================================================================
# =============================================================================
def plot_training_loss_evolution(training_loss_data, rank_value, ax=None, for_comparison=False, ylim=None):
    """
    Plot training loss evolution during optimization for a single rank level.
    
    Arguments:
        training_loss_data: dictionary with 'adam', 'lbfgs', and 'final' keys
        rank_value: rank level for this experiment
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
            ax.set_title(f'Training Loss Evolution - Rank r = {rank_value}')
            plt.tight_layout()
            # Save with rank-specific filename
            rank_str = f'{rank_value}'
            save_figure_with_rank(fig, f'training_loss_evolution_rank_{rank_str}', rank_value)
            plt.close(fig)
        else:
            ax.set_title(f'Rank r = {rank_value}')
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
        ax.set_title(f'Training Loss Evolution - Rank r = {rank_value}', fontsize=20)
    else:
        ax.set_title(f'Rank r = {rank_value}', fontsize=20)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='upper right')

    # Set consistent y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Save only for standalone plots
    if standalone_plot:
        plt.tight_layout()
        rank_str = f'{rank_value}'
        save_figure_with_rank(fig, f'training_loss_evolution_rank_{rank_str}', rank_value)
        plt.close(fig)




# =============================================================================
# =============================================================================
# EIGENVALUE EVOLUTION PLOTTING FUNCTIONS
# =============================================================================
# =============================================================================
def plot_eigenvalue_evolution_comparison(all_results, rank_values):
    """
    Create visualization showing eigenvalue evolution during training for all rank levels.
    Handles both Jacobian and connectivity eigenvalues, with separate plots for each Jacobian frequency.
    
    Arguments:
        all_results: list of results dictionaries
        rank_values: list of rank values tested
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
        plot_connectivity_eigenvalue_evolution(all_results, rank_values, cmap)
    
    # Plot Jacobian eigenvalues for each frequency if present
    if has_jacobian and jacobian_frequencies:
        for freq_idx in jacobian_frequencies:
            # Create separate plot for each rank level
            for rank_idx, result in enumerate(all_results):
                plot_jacobian_eigenvalue_evolution_for_single_sparsity(result, freq_idx, cmap)


def plot_connectivity_eigenvalue_evolution(all_results, rank_values, cmap):
    """
    Plot connectivity matrix eigenvalue evolution for all rank levels.
    """
    n_ranks = len(rank_values)
    
    # Calculate subplot layout - prefer wider layouts (same as other aggregate plots)
    if n_ranks <= 3:
        nrows, ncols = 1, n_ranks
    elif n_ranks <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_ranks + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_ranks == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Store scatter plot for colorbar
    scatter_plot = None
    
    for rank_idx, result in enumerate(all_results):
        rank = result['rank']
        eigenvalue_data = result['eigenvalue_data']
        final_loss = result['final_loss']
        
        ax = axes[rank_idx]
        
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
            ax.set_title(f'Rank r = {rank}')
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
        ax.set_title(f'Rank r = {rank}', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Hide unused subplots if any
    for idx in range(n_ranks, len(axes)):
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
    
    # Create rank-specific filename
    rank_str = '_'.join([f'{r}' for r in rank_values])
    save_figure(fig, f'connectivity_eigenvalue_evolution_ranks_{rank_str}')
    plt.close(fig)  # Close figure to free memory
    # plt.show()  # Commented out to prevent interactive display


def plot_jacobian_eigenvalue_evolution_for_single_sparsity(result, freq_idx, cmap):
    """
    Plot Jacobian eigenvalue evolution for a specific frequency and single rank level.
    """
    rank = result['rank']
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
        ax.set_title(f'Rank r = {rank}\nFreq {freq_idx}: ω = {freq_value:.3f} rad/s')
        plt.tight_layout()
        
        # Save with rank-specific filename in rank subdirectory
        rank_str = f'{rank}'
        freq_str = f'{freq_value:.3f}'.replace('.', 'p')
        save_figure_with_rank(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_rank_{rank_str}_omega_{freq_str}', rank)
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
                 f'Rank r = {rank} | Frequency {freq_idx}: ω = {freq_value:.3f} rad/s\n'
                 f'Static input = {static_input_value:.3f} | Final loss: {final_loss:.2e}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Progress')
    
    plt.tight_layout()
    
    # Save with rank-specific filename in rank subdirectory
    rank_str = f'{rank}'
    freq_str = f'{freq_value:.3f}'.replace('.', 'p')
    save_figure_with_rank(fig, f'jacobian_eigenvalue_evolution_freq_{freq_idx}_rank_{rank_str}_omega_{freq_str}', rank)
    # plt.show()  # Commented out to prevent interactive display
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
        trained_params = result['trained_params']
        if "U" in trained_params and "V" in trained_params:
            U = np.array(trained_params['U'])
            V = np.array(trained_params['V'])
            J = U @ V.T
        else:
            # Fallback to direct J if available
            J = trained_params['J']
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


def create_rank_summary_plots(all_results, rank_values):
    """
    Create summary plots comparing different metrics across rank levels.
    
    Arguments:
        all_results: list of results dictionaries
        rank_values: list of rank values tested
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
        # For low-rank connectivity, reconstruct J = U @ V.T
        trained_params = result['trained_params']
        if "U" in trained_params and "V" in trained_params:
            U = np.array(trained_params['U'])
            V = np.array(trained_params['V'])
            J = U @ V.T
        else:
            # Fallback to direct J if available
            J = trained_params['J']
            
        eigenvals = np.linalg.eigvals(np.array(J))
        spectral_radius = np.max(np.abs(eigenvals))
        spectral_radii.append(spectral_radius)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Final loss vs rank
    axes[0, 0].plot(rank_values, final_losses, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Rank')
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Training Loss vs Rank')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of fixed points vs rank
    axes[0, 1].plot(rank_values, num_fixed_points, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Rank')
    axes[0, 1].set_ylabel('Number of Fixed Points')
    axes[0, 1].set_title('Fixed Points vs Rank')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral radius vs rank
    axes[1, 0].plot(rank_values, spectral_radii, 'go-', linewidth=2, markersize=8)
    # Add horizontal reference line at y=1.0 (unit circle)
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Unit circle')
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Spectral Radius')
    axes[1, 0].set_title('Spectral Radius vs Rank')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Slow points vs rank (if applicable)
    if any(n > 0 for n in num_slow_points):
        axes[1, 1].plot(rank_values, num_slow_points, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Number of Slow Points')
        axes[1, 1].set_title('Slow Points vs Rank')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No slow point analysis\nperformed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Slow Points vs Rank')
    
    plt.suptitle('Low-Rank Connectivity Experiment Summary', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"rank_summary_plots_{timestamp}"
    save_figure(fig, filename)
    plt.close(fig)  # Close figure to free memory


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


def create_rank_unstable_eigenvalue_table(all_results, rank_values):
    """
    Create a table showing the distribution of unstable eigenvalues per fixed point across rank levels.
    
    Arguments:
        all_results: list of results dictionaries
        rank_values: list of rank values tested
    """
    # Collect unstable eigenvalue counts for each rank level
    unstable_counts_per_rank = {}
    max_unstable_count = 0
    
    for result in all_results:
        rank = result['rank']
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
        
        unstable_counts_per_rank[rank] = unstable_counts
    
    # If no unstable eigenvalues found, create a simple message plot
    if max_unstable_count == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No unstable eigenvalues found\nacross any rank levels', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Unstable Eigenvalue Distribution Table')
        plt.tight_layout()
        
        rank_str = '_'.join([f'{r}' for r in rank_values])
        save_figure(fig, f'unstable_eigenvalue_table_ranks_{rank_str}')
        plt.close(fig)
        return
    
    # Create the table: rows = number of unstable eigenvalues, cols = rank levels
    table_data = np.zeros((max_unstable_count + 1, len(rank_values)), dtype=int)
    
    for rank_idx, rank in enumerate(rank_values):
        unstable_counts = unstable_counts_per_rank[rank]
        
        # Count how many fixed points have each number of unstable eigenvalues
        for num_unstable in range(max_unstable_count + 1):
            count = unstable_counts.count(num_unstable)
            table_data[num_unstable, rank_idx] = count
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, len(rank_values) * 1.5), max(6, max_unstable_count * 0.8)))
    
    # Create heatmap
    im = ax.imshow(table_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(rank_values)))
    ax.set_yticks(np.arange(max_unstable_count + 1))
    ax.set_xticklabels([f'{r}' for r in rank_values])
    ax.set_yticklabels([f'{i}' for i in range(max_unstable_count + 1)])
    
    # Labels and title
    ax.set_xlabel('Rank Level')
    ax.set_ylabel('Number of Unstable Eigenvalues per Fixed Point')
    ax.set_title('Distribution of Unstable Eigenvalues per Fixed Point Across Rank Levels')
    
    # Add text annotations
    for i in range(max_unstable_count + 1):
        for j in range(len(rank_values)):
            text = ax.text(j, i, f'{table_data[i, j]}',
                         ha="center", va="center", color="black" if table_data[i, j] < table_data.max()/2 else "white",
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Fixed Points')
    
    plt.tight_layout()
    
    # Save with rank-specific filename
    rank_str = '_'.join([f'{r}' for r in rank_values])
    save_figure(fig, f'unstable_eigenvalue_table_ranks_{rank_str}')
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("UNSTABLE EIGENVALUE DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for rank_idx, rank in enumerate(rank_values):
        unstable_counts = unstable_counts_per_rank[rank]
        total_fps = len(unstable_counts)
        if total_fps > 0:
            avg_unstable = np.mean(unstable_counts)
            max_unstable_this_rank = max(unstable_counts) if unstable_counts else 0
            print(f"Rank {rank}: {total_fps} fixed points, "
                  f"avg {avg_unstable:.1f} unstable eigenvalues, max {max_unstable_this_rank}")
        else:
            print(f"Rank {rank}: No fixed points found")
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


def create_rank_fixed_points_per_task_table(all_results, rank_values):
    """
    Create a table showing the distribution of fixed points per task across rank levels.
    
    Arguments:
        all_results: list of results dictionaries
        rank_values: list of rank values tested
    """
    # Collect fixed point counts per task for each rank level
    fp_counts_per_rank = {}
    max_fp_count = 0
    
    for result in all_results:
        rank = result['rank']
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
        
        fp_counts_per_rank[rank] = fp_counts
    
    # If no fixed points found, create a simple message plot
    if max_fp_count == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No fixed points found\nacross any rank levels', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Fixed Points per Task Distribution Table')
        plt.tight_layout()
        
        rank_str = '_'.join([f'{r}' for r in rank_values])
        save_figure(fig, f'fixed_points_per_task_table_ranks_{rank_str}')
        plt.close(fig)
        return
    
    # Create the table: rows = number of fixed points per task, cols = rank levels
    table_data = np.zeros((max_fp_count + 1, len(rank_values)), dtype=int)
    
    for rank_idx, rank in enumerate(rank_values):
        fp_counts = fp_counts_per_rank[rank]
        
        # Count how many tasks have each number of fixed points
        for num_fps in range(max_fp_count + 1):
            count = fp_counts.count(num_fps)
            table_data[num_fps, rank_idx] = count
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, len(rank_values) * 1.5), max(6, max_fp_count * 0.8)))
    
    # Create heatmap
    im = ax.imshow(table_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(rank_values)))
    ax.set_yticks(np.arange(max_fp_count + 1))
    ax.set_xticklabels([f'{r}' for r in rank_values])
    ax.set_yticklabels([f'{i}' for i in range(max_fp_count + 1)])
    
    # Labels and title
    ax.set_xlabel('Rank Level')
    ax.set_ylabel('Number of Fixed Points per Task')
    ax.set_title('Distribution of Fixed Points per Task Across Rank Levels')
    
    # Add text annotations
    for i in range(max_fp_count + 1):
        for j in range(len(rank_values)):
            text = ax.text(j, i, f'{table_data[i, j]}',
                         ha="center", va="center", color="black" if table_data[i, j] < table_data.max()/2 else "white",
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Tasks')
    
    plt.tight_layout()
    
    # Save with rank-specific filename
    rank_str = '_'.join([f'{r}' for r in rank_values])
    save_figure(fig, f'fixed_points_per_task_table_ranks_{rank_str}')
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("FIXED POINTS PER TASK DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    for rank_idx, rank in enumerate(rank_values):
        fp_counts = fp_counts_per_rank[rank]
        total_tasks = len(fp_counts)
        if total_tasks > 0:
            avg_fps = np.mean(fp_counts)
            std_fps = np.std(fp_counts)
            min_fps = np.min(fp_counts)
            max_fps = np.max(fp_counts)
            print(f"Rank {rank}: {total_tasks} tasks, "
                  f"avg {avg_fps:.1f}±{std_fps:.1f} FPs/task, "
                  f"range [{min_fps}-{max_fps}] FPs/task")
        else:
            print(f"Rank {rank}: No task data available")
    print(f"{'='*60}")






def plot_training_loss_evolution_flexible(training_loss_data, param_value, param_name, param_label, param_prefix, ax=None, for_comparison=False, ylim=None):
    """
    Plot training loss evolution during optimization for a single parameter level.
    Flexible version that works with both sparsity and rank parameters.
    
    Arguments:
        training_loss_data: dictionary with 'adam', 'lbfgs', and 'final' keys
        param_value: parameter level for this experiment (sparsity or rank)
        param_name: name of the parameter ('sparsity' or 'rank')
        param_label: label for the parameter ('Sparsity' or 'Rank')
        param_prefix: prefix for titles ('s' or 'r')
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
            if param_name == 'rank':
                ax.set_title(f'Training Loss Evolution - {param_label} r = {param_value}')
            else:
                ax.set_title(f'Training Loss Evolution - {param_label} s = {param_value:.2f}')
            plt.tight_layout()
            
            # Save with parameter-specific filename
            if param_name == 'rank':
                save_figure(fig, f'training_loss_evolution_rank_{param_value}')
            else:
                param_str = f'{param_value:.2f}'.replace('.', 'p')
                save_figure_with_sparsity(fig, f'training_loss_evolution_sparsity_{param_str}', param_value)
            plt.close(fig)
        else:
            if param_name == 'rank':
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
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
    
    # Set title based on context
    if standalone_plot:
        if param_name == 'rank':
            ax.set_title(f'Training Loss Evolution - {param_label} r = {param_value}', fontsize=20)
        else:
            ax.set_title(f'Training Loss Evolution - {param_label} s = {param_value:.2f}', fontsize=20)
    else:
        if param_name == 'rank':
            ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
        else:
            ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='upper right')
    
    # Apply consistent y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Save individual plots if standalone
    if standalone_plot:
        plt.tight_layout()
        if param_name == 'rank':
            save_figure(fig, f'training_loss_evolution_rank_{param_value}')
        else:
            param_str = f'{param_value:.2f}'.replace('.', 'p')
            save_figure_with_sparsity(fig, f'training_loss_evolution_sparsity_{param_str}', param_value)
        plt.close(fig)


# =============================================================================
# =============================================================================
# AGGREGATE PLOTTING FUNCTIONS (CROSS-SPARSITY)
# =============================================================================
# =============================================================================
def plot_training_loss_evolution_comparison(all_results, sparsity_values):
    """
    Create visualization showing training loss evolution for all sparsity/rank levels.
    This function works with both sparsity-based and rank-based results.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values or rank values tested
    """
    # Detect if we have rank-based or sparsity-based results
    is_rank_based = any('rank' in result for result in all_results)
    param_name = 'rank' if is_rank_based else 'sparsity'
    param_label = 'Rank' if is_rank_based else 'Sparsity'
    param_prefix = 'r' if is_rank_based else 's'
    
    n_values = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_values <= 3:
        nrows, ncols = 1, n_values
    elif n_values <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_values + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_values == 1:
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
    
    # Use the individual plotting function for each parameter value
    for param_idx, result in enumerate(all_results):
        ax = axes[param_idx]
        
        # Get the parameter value for this result
        param_value = result[param_name]
        
        # Call the individual plotting function with the provided axis and consistent y-limits
        plot_training_loss_evolution_flexible(
            training_loss_data=result['training_loss_data'],
            param_value=param_value,
            param_name=param_name,
            param_label=param_label,
            param_prefix=param_prefix,
            ax=ax,
            for_comparison=True,
            ylim=ylim
        )
    
    # Hide unused subplots if any
    for idx in range(n_values, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Create parameter-specific filename
    if is_rank_based:
        param_str = '_'.join([f'{r}' for r in sparsity_values])
        save_figure(fig, f'training_loss_evolution_ranks_{param_str}')
    else:
        param_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'training_loss_evolution_sparsities_{param_str}')
    plt.close(fig)  # Close figure to free memory


def plot_trajectories_vs_targets_comparison(all_results, sparsity_values, test_index):
    """
    Create visualization showing trajectory vs target for specific test index across all sparsity/rank levels.
    This function works with both sparsity-based and rank-based results.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values or rank values tested
        test_index: specific test index to plot
    """
    # Detect if we have rank-based or sparsity-based results
    is_rank_based = any('rank' in result for result in all_results)
    param_name = 'rank' if is_rank_based else 'sparsity'
    param_label = 'Rank' if is_rank_based else 'Sparsity'
    
    n_values = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_values <= 3:
        nrows, ncols = 1, n_values
    elif n_values <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_values + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_values == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Get task information for title
    j = test_index
    omega = omegas[j]
    u_off = static_inputs[j]
    
    # Common initial state for all parameter levels
    x0_test = jnp.zeros((N,))
    
    # Use the individual plotting function for each parameter value
    for param_idx, result in enumerate(all_results):
        ax = axes[param_idx]
        param_value = result[param_name]
        trained_params = result['trained_params']
        final_loss = result['final_loss']
        
        try:
            # Call the individual plotting function with the provided axis
            with connectivity_context(trained_params) as params_with_J:
                error = plot_single_trajectory_vs_target(
                    params=params_with_J,
                    task_index=j,
                    x0_test=x0_test,
                    ax=ax,
                    for_comparison=True
                )
            
            # Add parameter information to the title for comparison context
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
            
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
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
    
    # Hide unused subplots if any
    for idx in range(n_values, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Create parameter-specific filename
    if is_rank_based:
        param_str = '_'.join([f'{r}' for r in sparsity_values])
        save_figure(fig, f'trajectory_vs_target_task_{j}_ranks_{param_str}')
    else:
        param_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'trajectory_vs_target_task_{j}_sparsities_{param_str}')
    plt.close(fig)


def plot_frequency_comparison_across_sparsity(all_results, sparsity_values):
    """
    Create visualization showing frequency comparison plots across all sparsity/rank levels.
    This function works with both sparsity-based and rank-based results.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values or rank values tested
    """
    # Detect if we have rank-based or sparsity-based results
    is_rank_based = any('rank' in result for result in all_results)
    param_name = 'rank' if is_rank_based else 'sparsity'
    param_label = 'Rank' if is_rank_based else 'Sparsity'
    
    n_values = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_values <= 3:
        nrows, ncols = 1, n_values
    elif n_values <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_values + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_values == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    # Use the individual plotting function for each parameter value
    for param_idx, result in enumerate(all_results):
        ax = axes[param_idx]
        param_value = result[param_name]
        all_unstable_eig_freq = result['all_unstable_eig_freq']
        final_loss = result['final_loss']
        
        if not all_unstable_eig_freq or len(all_unstable_eig_freq) == 0:
            ax.text(0.5, 0.5, 'No unstable\neigenvalue data', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
            continue
        
        try:
            # Call the individual plotting function with the provided axis
            plot_frequency_comparison(
                all_unstable_eig_freq=all_unstable_eig_freq,
                omegas=omegas,
                ax=ax,
                sparsity_value=param_value,
                for_comparison=True
            )
            
            # Add final loss information to the title
            current_title = ax.get_title()
            if is_rank_based:
                if current_title and not current_title.startswith('Rank'):
                    ax.set_title(f'{param_label} r = {param_value}\nFinal loss: {final_loss:.2e}')
                elif 'Rank' in current_title and 'Final loss' not in current_title:
                    ax.set_title(f'{current_title}\nFinal loss: {final_loss:.2e}')
            else:
                if current_title and not current_title.startswith('Sparsity'):
                    ax.set_title(f'{param_label} s = {param_value:.2f}\nFinal loss: {final_loss:.2e}')
                elif 'Sparsity' in current_title and 'Final loss' not in current_title:
                    ax.set_title(f'{current_title}\nFinal loss: {final_loss:.2e}')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot failed\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
    
    # Hide unused subplots if any
    for idx in range(n_values, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Create parameter-specific filename
    if is_rank_based:
        param_str = '_'.join([f'{r}' for r in sparsity_values])
        save_figure(fig, f'frequency_comparison_ranks_{param_str}')
    else:
        param_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'frequency_comparison_sparsities_{param_str}')
    plt.close(fig)


def plot_pca_explained_variance_comparison(all_results, sparsity_values, skip_steps, apply_tanh):
    """
    Create visualization showing PCA explained variance ratio across all sparsity/rank levels.
    This function works with both sparsity-based and rank-based results.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values or rank values tested
        skip_steps: skip steps parameter for PCA
        apply_tanh: tanh application parameter for PCA
    """
    # Detect if we have rank-based or sparsity-based results
    is_rank_based = any('rank' in result for result in all_results)
    param_name = 'rank' if is_rank_based else 'sparsity'
    param_label = 'Rank' if is_rank_based else 'Sparsity'
    param_prefix = 'r' if is_rank_based else 's'
    
    n_values = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_values <= 3:
        nrows, ncols = 1, n_values
    elif n_values <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_values + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_values == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array for single row
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    pca_key = f"skip_{skip_steps}_tanh_{apply_tanh}"
    
    # Use the individual plotting function for each parameter value
    for param_idx, result in enumerate(all_results):
        ax = axes[param_idx]
        param_value = result[param_name]
        pca_results = result['pca_results']
        final_loss = result['final_loss']
        
        if pca_key not in pca_results:
            ax.text(0.5, 0.5, f'No PCA data for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
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
                sparsity_value=param_value,
                for_comparison=True
            )
            
            # Format for publication readiness: enhance title and font sizes
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot failed\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
    
    # Hide unused subplots if any
    for idx in range(n_values, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Create parameter-specific filename
    if is_rank_based:
        param_str = '_'.join([f'{r}' for r in sparsity_values])
        save_figure(fig, f'pca_explained_variance_ratio_skip_{skip_steps}_tanh_{apply_tanh}_ranks_{param_str}')
    else:
        param_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'pca_explained_variance_ratio_skip_{skip_steps}_tanh_{apply_tanh}_sparsities_{param_str}')
    plt.close(fig)


def plot_pca_3d_comparison(all_results, sparsity_values, skip_steps, apply_tanh, plot_type='fixed'):
    """
    Create 3D PCA plots across all sparsity/rank levels.
    This function works with both sparsity-based and rank-based results.
    
    Arguments:
        all_results: list of results dictionaries
        sparsity_values: list of sparsity values or rank values tested
        skip_steps: skip steps parameter for PCA
        apply_tanh: tanh application parameter for PCA
        plot_type: 'fixed' for regular plot, 'subset' for subset plot
    """
    # Detect if we have rank-based or sparsity-based results
    is_rank_based = any('rank' in result for result in all_results)
    param_name = 'rank' if is_rank_based else 'sparsity'
    param_label = 'Rank' if is_rank_based else 'Sparsity'
    
    n_values = len(sparsity_values)
    
    # Calculate subplot layout - prefer wider layouts
    if n_values <= 3:
        nrows, ncols = 1, n_values
    elif n_values <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, (n_values + 2) // 3
    
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    
    pca_key = f"skip_{skip_steps}_tanh_{apply_tanh}"
    
    # Use the individual plotting function for each parameter value
    for param_idx, result in enumerate(all_results):
        param_value = result[param_name]
        pca_results = result['pca_results']
        final_loss = result['final_loss']
        
        print(f"\nProcessing {param_name} {param_value} for PCA 3D comparison:")
        print(f"  Available PCA result keys: {list(pca_results.keys()) if pca_results else 'None'}")
        
        ax = fig.add_subplot(nrows, ncols, param_idx + 1, projection='3d')
        
        if pca_key not in pca_results:
            ax.text(0.5, 0.5, 0.5, f'No PCA data for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
            print(f"WARNING: No PCA data found for key '{pca_key}' in {param_name} {param_value}")
            print(f"Available PCA keys: {list(pca_results.keys())}")
            continue
        
        pca_data = pca_results[pca_key]
        pca_object = pca_data.get('pca', None)
        proj_trajs = pca_data.get('proj_trajs', [])
        
        # Debug information
        print(f"DEBUG: {param_label} {param_value}, PCA key '{pca_key}':")
        print(f"  - PCA object available: {pca_object is not None}")
        print(f"  - Number of projected trajectories: {len(proj_trajs) if proj_trajs else 0}")
        print(f"  - PCA data keys: {list(pca_data.keys())}")
        
        if pca_object is None:
            ax.text(0.5, 0.5, 0.5, f'No PCA object for\nskip={skip_steps}, tanh={apply_tanh}', 
                   ha='center', va='center', transform=ax.transAxes)
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
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
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}')
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}')
            print(f"WARNING: No projected trajectories found for {param_name} {param_value}")
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
                sparsity_value=param_value,
                for_comparison=True
            )
            
            # Format for publication readiness: remove legend, clean title, enhance font sizes
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=24)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=24)
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
            if is_rank_based:
                ax.set_title(f'{param_label} r = {param_value}', fontsize=20)
            else:
                ax.set_title(f'{param_label} s = {param_value:.2f}', fontsize=20)
    
    plot_name = 'pca_plot_fixed' if plot_type == 'fixed' else 'subset_of_indices_pca_plot_fixed'
    
    # Use subplots_adjust for better control over 3D plot spacing
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.2, hspace=0.3)
    
    # Create parameter-specific filename
    if is_rank_based:
        param_str = '_'.join([f'{r}' for r in sparsity_values])
        save_figure(fig, f'{plot_name}_skip_{skip_steps}_tanh_{apply_tanh}_ranks_{param_str}')
    else:
        param_str = '_'.join([f'{s:.2f}'.replace('.', 'p') for s in sparsity_values])
        save_figure(fig, f'{plot_name}_skip_{skip_steps}_tanh_{apply_tanh}_sparsities_{param_str}')
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


def create_rank_aggregate_plots(all_results, rank_values):
    """
    Create all aggregate plots across rank levels.
    
    Arguments:
        all_results: list of results dictionaries
        rank_values: list of rank values tested
    """
    print("\n" + "="*60)
    print("CREATING AGGREGATE PLOTS ACROSS RANK LEVELS")
    print("="*60)
    
    # 1. Training loss evolution comparison
    print("Creating training loss evolution comparison...")
    plot_training_loss_evolution_comparison(all_results, rank_values)
    
    # 2. Trajectory vs target comparisons for each test index
    print("Creating trajectory vs target comparisons...")
    for test_idx in TEST_INDICES:
        print(f"  Processing test index {test_idx}...")
        plot_trajectories_vs_targets_comparison(all_results, rank_values, test_idx)
    
    # 3. Frequency comparison across rank levels
    print("Creating frequency comparison across rank levels...")
    plot_frequency_comparison_across_sparsity(all_results, rank_values)
    
    # 4. PCA plots for each combination of skip steps and tanh activations
    print("Creating PCA comparison plots...")
    for skip_steps in PCA_SKIP_OPTIONS:
        for apply_tanh in PCA_TANH_OPTIONS:
            print(f"  Processing skip_steps={skip_steps}, apply_tanh={apply_tanh}...")
            
            # PCA explained variance ratio
            plot_pca_explained_variance_comparison(all_results, rank_values, skip_steps, apply_tanh)
            
            # PCA 3D plots
            plot_pca_3d_comparison(all_results, rank_values, skip_steps, apply_tanh, 'fixed')
            plot_pca_3d_comparison(all_results, rank_values, skip_steps, apply_tanh, 'subset')
    
    print("Completed all aggregate plots!")




# =============================================================================
# =============================================================================
# MAIN LOOP FUNCTION
# =============================================================================
# =============================================================================
def main():
    """
    Main function to run comprehensive sparsity experiments with full analysis.
    """
    # Get rank values from user
    rank_values = get_rank_values()
    
    # Get eigenvalue computation preferences from user
    compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies = get_eigenvalue_options()
    
    # Setup custom output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"Sparsity_Experiments_for_Low_Rank_{timestamp}"
    
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
    print("COMPREHENSIVE LOW-RANK CONNECTIVITY EXPERIMENTS")
    print(f"Testing {len(rank_values)} rank levels")
    print(f"{eigenval_msg}")
    if eigenval_types:
        print(f"Eigenvalue sampling frequency: every {EIGENVALUE_SAMPLE_FREQUENCY} iterations")
    print(f"Output directory: {full_output_path}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(NUMPY_SEED)
    key = random.PRNGKey(JAX_SEED)
    
    # Generate a single set of base parameters that will be used for all rank levels
    # This ensures all networks start from the same underlying structure, only varying in rank
    analysis_key, base_key = random.split(key)
    
    # Store results for all rank levels
    all_results = []
    
    for i, rank in enumerate(rank_values):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING RANK LEVEL {i+1}/{len(rank_values)}: r = {rank}")
        print(f"{'='*80}")
        
        # Memory monitoring
        print_memory_usage(f"Before rank {rank}")
        
        # Validate rank is reasonable for low-rank approximation
        if rank > N:
            print(f"WARNING: Rank {rank} > N={N}. This is not low-rank and may cause memory issues.")
            print("Skipping this rank value.")
            continue
            
        try:
            # Use the same base key for all rank levels to ensure consistent initial parameters
            # Only the rank will differ between runs
            params = init_params_low_rank(base_key, rank)
            
            # Generate a unique analysis key for this rank level
            analysis_key, subkey = random.split(analysis_key)
            
            # Run full training and analysis pipeline
            results = train_with_full_analysis(params, rank, subkey, 
                                              compute_jacobian_eigenvals, compute_connectivity_eigenvals, jacobian_frequencies)
            
            # Store results
            all_results.append(results)
            
            print(f"\nCompleted rank {rank}: final loss = {results['final_loss']:.6e}")
            
        except Exception as e:
            print(f"\nERROR processing rank {rank}: {e}")
            print(f"Error type: {type(e).__name__}")
            print("Continuing with next rank...")
            cleanup_memory()
            continue
        
        # Memory cleanup after each rank
        print_memory_usage(f"After rank {rank}")
        cleanup_memory()
        print_memory_usage(f"After cleanup for rank {rank}")
    

    # ========================================
    # CREATE COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION
    # ========================================
    if compute_jacobian_eigenvals or compute_connectivity_eigenvals:
        print(f"\n{'='*60}")
        print("CREATING COMPARATIVE EIGENVALUE EVOLUTION VISUALIZATION")
        print(f"{'='*60}")
        
        plot_eigenvalue_evolution_comparison(all_results, rank_values)
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
    
    # create_rank_summary_plots(all_results, rank_values)  # TODO: Implement rank-specific plots
    create_rank_summary_plots(all_results, rank_values)
    

    # ========================================
    # CREATE UNSTABLE EIGENVALUE DISTRIBUTION TABLE
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING UNSTABLE EIGENVALUE DISTRIBUTION TABLE")
    print(f"{'='*60}")
    
    # create_unstable_eigenvalue_table(all_results, rank_values)  # TODO: Implement rank-specific tables
    create_rank_unstable_eigenvalue_table(all_results, rank_values)
    

    # ========================================
    # CREATE FIXED POINTS PER TASK DISTRIBUTION TABLE
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING FIXED POINTS PER TASK DISTRIBUTION TABLE")
    print(f"{'='*60}")
    
    # create_fixed_points_per_task_table(all_results, rank_values)  # TODO: Implement rank-specific tables
    create_rank_fixed_points_per_task_table(all_results, rank_values)
    

    # ========================================
    # CREATE ALL AGGREGATE PLOTS ACROSS SPARSITY LEVELS
    # ========================================
    print(f"\n{'='*60}")
    print("CREATING ALL AGGREGATE PLOTS ACROSS SPARSITY LEVELS")
    print(f"{'='*60}")
    
    # create_all_aggregate_plots(all_results, rank_values)  # TODO: Implement rank-specific plots
    create_rank_aggregate_plots(all_results, rank_values)
    

    # ========================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================
    print(f"\n{'='*60}")
    print("SAVING COMPLETE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    # Save all results together
    complete_experiment_results = {
        'rank_values': rank_values,
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
    save_variable(complete_experiment_results, f"complete_rank_experiment_{timestamp}")
    

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*80}")
    print("COMPREHENSIVE LOW-RANK CONNECTIVITY EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Experiment timestamp: {timestamp}")
    print(f"Output directory: {full_output_path}")
    print(f"Tested rank values: {rank_values}")
    print(f"Eigenvalue tracking: {eigenval_msg}")
    print("Final losses by rank:")
    for result in all_results:
        print(f"  r = {result['rank']}: loss = {result['final_loss']:.6e}")
    
    print(f"\nNumber of fixed points found by rank:")
    for result in all_results:
        if result['all_fixed_points']:
            num_fps = sum(len(task_fps) for task_fps in result['all_fixed_points'])
            print(f"  r = {result['rank']}: {num_fps} fixed points")
    
    if SLOW_POINT_SEARCH:
        print(f"\nNumber of slow points found by rank:")
        for result in all_results:
            if result['all_slow_points']:
                num_sps = sum(len(task_sps) for task_sps in result['all_slow_points'])
                print(f"  r = {result['rank']}: {num_sps} slow points")
    
    print(f"\nAll results and visualizations saved to: {full_output_path}")




if __name__ == "__main__":
    main()