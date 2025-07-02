#!/usr/bin/env python3
"""
TEST_Frequency_Generalization.py

This script provides two main functionalities:
1. GAP TRAINING MODE: Train RNN on frequencies 0.1-0.25 and 0.45-0.6 (excluding the gap 0.25-0.45)
2. TESTING MODE: Test a trained RNN's ability to generalize to new frequencies (0.01-1.2 range)

Features:
- Train with frequency gaps to test generalization
- Variable testing time based on frequency
- Custom test frequency ranges
- Comprehensive analysis and visualization

Configuration:
All parameters are configured directly in the script in the CONFIGURATION SETTINGS section.
Set EXECUTION_MODE to "train" or "test" to choose the operation mode.
No command line arguments or user input required.

Usage:
    # Simply run the script after configuring the parameters
    python TEST_Frequency_Generalization.py


OUTPUTS:
    FILES:
        - trained_params: dict of trained RNN parameters after gap training
            - Saved with name "Gap_Trained_Params_{EXECUTION_MODE}_{timestamp}.pkl"
            - Contains the following keys:
                - 'J'
                - 'B'
                - 'w'
                - 'b_x'
                - 'b_z'
        - all_fixed_points: list of list of points, one list per comprehensive test frequency
            - Saved with name "Fixed_Points_{EXECUTION_MODE}_{timestamp}.pkl"
        - all_jacobians: list of list of Jacobians, one list per comprehensive test frequency
            - Saved with name "Jacobian_Matrices_{EXECUTION_MODE}_{timestamp}.pkl"
        - all_unstable_eig_freq: list of list of list of unstable frequencies, one middle list per comprehensive test frequency, one inner list per fixed point for that frequency
            - Saved with name "Unstable_Eigen_Frequencies_{EXECUTION_MODE}_{timestamp}.pkl"
        - frequency_labels: list of frequencies used for fixed point and PCA analysis
            - Saved with name "Analysis_Frequencies_{EXECUTION_MODE}_{timestamp}.pkl"
        - trajs_and_inits: dict
            - Saved with name "Trajectory_States_and_Fixed_Point_Inits_{EXECUTION_MODE}_{timestamp}.pkl"
            - Contains the following keys:
                - 'trajectory_states': list of trajectory states from training, one for each frequency
                - 'fixed_point_inits': list of initial states for fixed point search, one for each frequency
                'frequency_labels': list of frequencies corresponding to trajectory states
        # - **CURRENTLY NOT BEING SAVED** pca_results: dict containing PCA analysis results
        #     - Saved with name "PCA_Results_{EXECUTION_MODE}_{timestamp}.pkl"
        #     - Contains the following keys:
        #         - 'pca'
        #         - 'proj_trajs'
        #         - 'proj_trajs_by_region': trajectories categorized by frequency region
        #             - 'lower_extrapolation': trajectories from frequencies below training range
        #             - 'lower_training': trajectories from lower training range (0.1-0.25)
        #             - 'gap': trajectories from gap region (0.25-0.45)
        #             - 'upper_training': trajectories from upper training range (0.45-0.6)
        #             - 'higher_extrapolation': trajectories from frequencies above training range
        #         - 'frequency_labels': list of frequencies corresponding to trajectories
        #         - 'all_trajectories_combined'
        #         - 'skip_initial_steps'
        #         - 'apply_tanh'
        #         - 'pca_components'
        #         - 'pca_explained_variance_ratio'
        #         - 'proj_fixed'
        - summary_data: dict summarizing the results of the gap training and testing
            - Saved with name "Frequency_Generalization_Summary_{EXECUTION_MODE}_{timestamp}.pkl"
            - Contains the following keys:
                - 'title'
                - 'training_configuration': dict containing the following keys:
                    - 'execution_mode'
                    - 'run_testing_after_training'
                    - 'training_frequency_ranges'
                    - 'excluded_gap'
                    - 'test_frequency_range'
                    - 'variable_test_time'
                    - 'number_of_samples_from_gap'
                    - 'number_of_samples_from_each_training_range'
                    - 'number_of_samples_from_extrapolation_higher'
                    - 'number_of_samples_from_extrapolation_lower'
                - 'results': dictionary with a key for each frequency tested; results[freq] contains:
                    - 'mse'
                    - 'static_input'
                    - 'x_drive_final': shape (N,)
                    - 'xs_train': shape (num_steps_train_or_test, N)
                    - 'test_time'
                    - 'rnn_output'
                    - 'targets'
                - 'performance_by_region': dict with a key for each region ('training_regions', 'gap_region', 'extrapolation_regions'); performance_by_region[region] contains:
                    - 'frequency_count'
                    - 'mean_mse'
                    - 'max_mse'
    PLOTS:
        - Plot of output trajectories vs target trajectories for each frequency
            - Saved with name
            - From function plot_frequency_comparison_gap()
        - Plot of explained variance ratio for PCA analysis
            - Saved with name
            - From function plot_explained_variance_ratio()
        - Plot of PCA trajectories with fixed points (for all points and then just a subset)
            - Saved with name
            - From function plot_pca_trajectories_and_points()
        - MSE vs Frequency plot with shaded training ranges and highlighted gap region
            - Saved with name "MSE_vs_Frequency_with_Gaps_{EXECUTION_MODE}_{timestamp}.png"
            - From function create_error_vs_frequency_plot_with_gaps()
        - Comprehensive subplot visualization showing multiple frequencies
            - Saved with name "Comprehensive_Trajectory_Subplots_{n_frequencies}_frequencies_{EXECUTION_MODE}_{timestamp}.png"
            - From function create_comprehensive_trajectory_subplots()
"""
# ============================================================
# ============================================================
# IMPORTS
# ============================================================
# ============================================================
import os
import sys
import pickle
import time
from tqdm import tqdm
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit
from scipy.optimize import root, least_squares
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save (not display) figures
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _1_config import *
from _2_utils import Timer, get_output_dir, set_custom_output_dir
from _3_data_generation import get_drive_input, get_train_input
from _4_rnn_model import simulate_trajectory, create_loss_function
from _5_training import train_model
from _6_analysis import compute_F, analyze_points
from _7_visualization import plot_trajectories_vs_targets
from _8_pca_analysis import perform_pca_analysis, project_points_to_pca, plot_explained_variance_ratio, plot_pca_trajectories_and_points




# ============================================================
# ============================================================
# CONFIGURATION SETTINGS
# ============================================================
# ============================================================

# ============================================================
# EXECUTION MODE SETTINGS

# Set the execution mode
EXECUTION_MODE               = "train"   # Options: "train" or "test"
RUN_TESTING_AFTER_TRAINING   = True      # If True, will run testing after gap training completes

# Current timestamp for output folder naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Random key
key                          = random.PRNGKey(JAX_SEED)


# ============================================================
# SETTINGS TO BE CONTROLLED BY TEST_FREQUENCY_GENERALIZATION_WITH_SPARSITY_AND_RANK.py

# Set the sparsity
S                       = 0.0                           # Sparsity level for the J matrix (0.0 to 1.0)

# Choose whether or not to explicitly control the rank of the connectivity matrix J
CONTROL_RANK            = False                         # If True, will control the rank of J during initialization
# Set the rank
R                       = 50                            # Rank of the J matrix (1 to N, where N is the number of neurons)
# Enusre the rank does not exceed the dimension
if R > N:
    R = N
# Covariance matrix for controlled rank initialization
cov_general             = (jax.random.normal(key, (2*R, 2*R))) \
    @ (jax.random.normal(key, (2*R, 2*R))).T \
    + 1e-6 * jnp.eye(2*R)                               # Symmetric positive-definite covariance matrix for generating low-rank matrices, shape (2R, 2R)


# ============================================================
# TRAINING MODE SETTINGS

# Gap training configuration
TRAINING_FREQ_RANGES    = [(0.1, 0.25), (0.45, 0.6)]    # Training frequency ranges (excluding 0.25-0.45)
TESTING_FREQ_RANGE      = (0.01, 1.2)                   # Full testing range
VARIABLE_TEST_TIME      = True                          # Enable frequency-dependent testing time

# Create training frequency mask for gap training
def create_training_mask():
    """Create a boolean mask for training frequencies based on gap ranges."""
    mask = jnp.zeros(len(omegas), dtype=bool)
    for freq_min, freq_max in TRAINING_FREQ_RANGES:
        freq_mask = (omegas >= freq_min) & (omegas <= freq_max)
        mask = mask | freq_mask
    return mask

# Training mask and filtered parameters for gap training
training_mask            = create_training_mask()
training_omegas          = omegas[training_mask]            # Frequencies you actually train on
training_static_inputs   = static_inputs[training_mask]     # Static inputs you actually use for training
num_training_tasks       = len(training_omegas)


# ============================================================
# TESTING MODE SETTINGS - GENERAL

# Minimum cycles for frequencies at test time (only used if VARIABLE_TEST_TIME = True)
MIN_CYCLES = 2

# Number of frequencies to test at in each region
NUM_SAMPLES_MIDDLE                 = 20
NUM_SAMPLES_TRAINING               = 10       # Test NUM_SAMPLES_TRAINING frequencies in each training range
NUM_SAMPLES_EXTRAPOLATION_HIGHER   = 30
NUM_SAMPLES_EXTRAPOLATION_LOWER    = 15


# ============================================================
# TESTING MODE SETTINGS - IF EXECUTION_MODE = "test" (so no gap training)

# Default parameter path
DEFAULT_PARAM_PATH = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Frequency_Generalization_train_20250627_152729/Gap_Trained_Params_20250627_152729.pkl"

# Function to import trajectory data from pickle files
def read_trajectory_data(file_path):
    """Read trajectory data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

# RNN parameters from parameter paths
RNN_PARAMS = {
    'J': read_trajectory_data(DEFAULT_PARAM_PATH)["J"],
    'B': read_trajectory_data(DEFAULT_PARAM_PATH)["B"],
    'w': read_trajectory_data(DEFAULT_PARAM_PATH)["w"],
    'b_x': read_trajectory_data(DEFAULT_PARAM_PATH)["b_x"],
    'b_z': read_trajectory_data(DEFAULT_PARAM_PATH)["b_z"]
}


# ============================================================
# PLOTTING SETTINGS

# Settings for create_comprehensive_trajectory_subplots
MAX_PLOTS = 20
NUM_PLOTS_MIDDLE                 = 6
NUM_PLOTS_TRAINING               = 2       # Plot NUM_PLOT_TRAINING frequencies in each training range
NUM_PLOTS_EXTRAPOLATION_HIGHER   = 5
NUM_PLOTS_EXTRAPOLATION_LOWER    = 5


# ============================================================
# PCA SETTINGS

PCA_SKIP_INITIAL_STEPS  = 400
PCA_APPLY_TANH          = False
PCA_N_COMPONENTS        = 10



# ============================================================
# ============================================================
# READING AND WRITING FUNCTIONS
# ============================================================
# ============================================================
def save_file_to_pkl(data, base_name):
    """
    Save a dictionary to a pickle file in the Outputs/Frequency_Generalization_{timestamp} folder.
    
    Parameters
    ----------
    data : dict
        The dictionary to save.
    base_name : str
        The name of the file.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Move to parent directory, then to Outputs/Frequency_Generalization_{timestamp}
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{EXECUTION_MODE}_{timestamp}')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename using the base name
    filename = f"{base_name}_{timestamp}.pkl"

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the dictionary
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Dictionary saved to: {file_path}")


def save_plot_to_png(plot_figure, base_name):
    """
    Save a plot figure to a PNG file in the Outputs/Frequency_Generalization_{timestamp} folder.
    
    Parameters
    ----------
    plot_figure : matplotlib.figure.Figure
        The matplotlib figure to save.
    base_name : str
        The base name of the file.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Move to parent directory, then to Outputs/Frequency_Generalization_{timestamp}
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{EXECUTION_MODE}_{timestamp}')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename using the base name and including all the sparsity levels
    filename = f"{base_name}_{timestamp}.png"

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the plot
    plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {file_path}")




# ============================================================
# ============================================================
# INPUT AND TARGET GENERATION FUNCTIONS
# ============================================================
# ============================================================
def calculate_static_input(frequency):
    """
    Calculate static input based on frequency following the training pattern.
    Map the input frequency to the same pattern used in training.
    For frequencies outside the training range, we extrapolate the linear relationship.
    """
    # Get the training frequency range
    freq_min = float(omegas.min())  # 0.1
    freq_max = float(omegas.max())  # 0.6

    # In training, frequencies are:   omegas        = jnp.linspace(0.1, 0.6, num_tasks)
    # And static inputs are:          static_inputs = jnp.linspace(0, num_tasks - 1, num_tasks) / num_tasks + 0.25
    # This means:                     static_input  = task_index / num_tasks + 0.25

    # Calculate the equivalent task index for the current frequency (potentially by exptrapolating)
    # Over the training range:   frequency  = freq_min + task_index * (freq_max - freq_min) / (num_tasks - 1)
    # Solving for task_index:    task_index = (frequency - freq_min) * (num_tasks - 1) / (freq_max - freq_min)
    task_index_continuous = (frequency - freq_min) * (num_tasks - 1) / (freq_max - freq_min)
    
    # Calculate static input using the same formula as training: task_index / num_tasks + 0.25
    static_input = task_index_continuous / num_tasks + 0.25
    
    return float(static_input)


def generate_sine_wave_data(frequency, static_input, time_drive, time_train_or_test):
    """Generate input and target data for a given frequency and static input."""
    # Generate driving phase input (sine wave with static offset)
    drive_input  = (jnp.sin(frequency * time_drive) + static_input)[:, None]                 # Shape: (num_steps_drive, 1)
    # Generate training/testing phase input (static input only)
    train_input  = jnp.full((len(time_train_or_test), 1), static_input, dtype=jnp.float32)   # Shape: (num_steps_train, 1)
    
    # Combine inputs
    inputs       = jnp.concatenate([drive_input, train_input], axis=0)
    
    # Generate target for driving phase (zeros)
    drive_target = jnp.zeros((len(time_drive), 1), dtype=jnp.float32)                        # Shape: (num_steps_drive, 1)
    # Generate target (for training/testing phase only)
    train_target = jnp.sin(frequency * time_train_or_test)[:, None]                          # Shape: (num_steps_train, 1)
    
    # Combine targets
    targets = jnp.concatenate([drive_target, train_target], axis=0)
    
    return {
        'inputs': inputs,
        'targets': targets
    }


def get_gap_drive_input(gap_task_index):
    """
    Get the driving phase input for a given gap training task.
    
    Arguments:
        gap_task_index: index of the gap training task (0 to num_training_tasks-1)
    
    Returns:
        drive_input: driving phase input, shape (num_steps_drive, I)
    """
    return (jnp.sin(training_omegas[gap_task_index] * time_drive) + training_static_inputs[gap_task_index])[:, None]


def get_gap_train_input(gap_task_index):
    """
    Get the training phase input for a given gap training task.
    
    Arguments:
        gap_task_index: index of the gap training task (0 to num_training_tasks-1)
    
    Returns:
        train_input: training phase input, shape (num_steps_train, I)
    """
    return jnp.full((num_steps_train, I), training_static_inputs[gap_task_index], dtype=jnp.float32)


def get_gap_train_target(gap_task_index):
    """
    Get the training phase target for a given gap training task.
    
    Arguments:
        gap_task_index: index of the gap training task (0 to num_training_tasks-1)
    
    Returns:
        train_target: training phase target, shape (num_steps_train,)
    """
    return jnp.sin(training_omegas[gap_task_index] * time_train)


# Note that this is the original version, which is kept for reference
# def generate_gap_training_data():
#     """Generate training data for gap training (excluding frequencies in the gap)."""
#     # Use the filtered training frequencies and static inputs
#     drive_inputs = (
#         jnp.sin(training_omegas[:, None] * time_drive[None, :]) + 
#         training_static_inputs[:, None]
#     )[..., None]            # shape (num_training_tasks, num_steps_drive, 1)
    
#     train_inputs = jnp.broadcast_to(
#         training_static_inputs[:, None, None],
#         (len(training_omegas), num_steps_train, 1)
#     ).astype(jnp.float32)   # shape (num_training_tasks, num_steps_train, 1)
    
#     train_targets = jnp.sin(training_omegas[:, None] * time_train[None, :]) # shape (num_training_tasks, num_steps_train)
    
#     return drive_inputs, train_inputs, train_targets




# ============================================================
# ============================================================
# GAP TRAINING FUNCTIONS
# ============================================================
# ============================================================
def init_params_gap(key, sparsity):
    """
    Initialize RNN parameters with proper scaling.
    
    Arguments:
        key: JAX random key
        sparsity: sparsity level for the J matrix (0.0 to 1.0)
        
    Returns:
        mask: sparsity mask for connections
        params: dictionary containing J, B, b_x, w, b_z
    """
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    mask = random.bernoulli(k_mask, p=1.0 - sparsity, shape=(N, N)).astype(jnp.float32)
    J_unscaled = random.normal(k1, (N, N)) / jnp.sqrt(N) * mask
    # By Girko's law, we know that, in the limit N -> infinity, the eigenvalues of J will converge to a uniform distribution
    # within a disk centred at the origin of radius sqrt(Var * N) = sqrt((1 / N) * N) = 1.
    # After masking, the elements remain independently distributed with mean zero,
    # but their variance becomes Var_m = (1 - s) / N, meaning the spectral radius becomes sqrt(1 - s).
    # Thus, to maintain the same spectral radius as the full matrix, we scale J by 1 / sqrt(1 - s).
    J = J_unscaled / jnp.sqrt(1 - sparsity)

    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)
    
    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}


def init_params_gap_control_rank(key, sparsity, R, N, cov):
    """
    Initialize RNN parameters with controlled rank.
    
    Arguments:
        key: JAX random key
        sparsity: sparsity level for the J matrix (0.0 to 1.0)
        rank: desired rank of the J matrix (1 to N)
        
    Returns:
        mask: sparsity mask for connections
        params: dictionary containing J, B, b_x, w, b_z
    """
    k_mask, k1, k2, k3, k4, k5 = random.split(key, 6)

    # ----------------------------------------
    # CREATE MASK
    mask = random.bernoulli(k_mask, p=1.0 - sparsity, shape=(N, N)).astype(jnp.float32)

    # ----------------------------------------
    # CREATE UNSPARSIFIED J, WITH RANK R
    subkey, key = random.split(key)

    # Check that cov is a valid covariance matrix
    if cov.shape != (2 * R, 2 * R):
        raise ValueError(f"cov should be of shape (2R, 2R), but got {cov.shape}.")
    if not jnp.allclose(cov, cov.T):
        raise ValueError("cov should be symmetric, but it is not.")
    if not jnp.all(jnp.linalg.eigvalsh(cov) > 0):
        raise ValueError("cov should be positive definite, but it is not.")
    
    # Sample N independent draws from N(0, cov)
    # using the Cholesky decomposition of cov
    Z       = random.normal(subkey, shape=(N, 2 * R))        # shape (N, 2R), each element is inependently N(0, 1)
    L       = jnp.linalg.cholesky(cov)                       # L is a lower triangular matrix such that cov = L L^T (the Cholesky decomposition)
    samples = Z @ L.T                                        # shape (N, 2R), where each row (length 2R) is a sample from N(0, cov)

    # Split samples into matrices of the left and right vectors
    R_int = int(R)  # Ensure R is a Python integer for array slicing
    U  = samples[:, :R_int].T        # shape (R, N), each column is a vector u_r
    V  = samples[:, R_int:].T        # shape (R, N), each column is a vector v_r

    # Recover P as sum of outer products: P[ij] = sum_r u_r[i] * v_r[j]
    J_unscaled_unmasked = jnp.sum(U[:, :, None] * V[:, None, :], axis=0)

    # ----------------------------------------
    # SPARSIFY J AND SCALE IT
    J_unscaled = J_unscaled_unmasked * mask
    # By Girko's law, we know that, in the limit N -> infinity, the eigenvalues of J will converge to a uniform distribution
    # within a disk centred at the origin of radius sqrt(Var * N) = sqrt((1 / N) * N) = 1.
    # After masking, the elements remain independently distributed with mean zero,
    # but their variance becomes Var_m = (1 - s) / N, meaning the spectral radius becomes sqrt(1 - s).
    # Thus, to maintain the same spectral radius as the full matrix, we scale J by 1 / sqrt(1 - s).
    J = J_unscaled / jnp.sqrt(1 - sparsity)

    # ----------------------------------------
    # CREATE OTHER PARAMETERS
    B = random.normal(k2, (N, I)) / jnp.sqrt(N)
    b_x = jnp.zeros((N,))
    w = random.normal(k4, (N,)) / jnp.sqrt(N)
    b_z = jnp.array(0.0, dtype=jnp.float32)
    
    return mask, {"J": J, "B": B, "b_x": b_x, "w": w, "b_z": b_z}


@jit
def solve_gap_task(params, gap_task_index):
    """
    Run the drive and train phases all the way through for a single gap training task.
    Follows the exact same pattern as solve_task from _4_rnn_model.py.
    
    Arguments:
        params: RNN parameters
        gap_task_index: index of the gap training task (0 to num_training_tasks-1)
    
    Returns:
        zs_tr: training phase outputs, shape (num_steps_train,)
    """
    x0 = jnp.zeros((N,), dtype=jnp.float32)                     # Initial state for the task
    
    # Drive phase
    d_u       = get_gap_drive_input(gap_task_index)             # shape (num_steps_drive, I)
    xs_d, _   = simulate_trajectory(x0, d_u, params)
    x_final   = xs_d[-1]
    
    # Train phase
    t_u       = get_gap_train_input(gap_task_index)             # shape (num_steps_train, I)
    _, zs_tr  = simulate_trajectory(x_final, t_u, params)

    return zs_tr                                                # shape (num_steps_train,)


# Vectorize solve_gap_task over all gap task values (same pattern as in _4_rnn_model.py)
vmap_solve_gap_task = vmap(solve_gap_task, in_axes=(None, 0))   # vmap_solve_gap_task gives an array of shape (num_training_tasks, num_steps_train)


@jit
def gap_batched_loss(params):
    """
    Compute the loss for gap training, averaged over all gap training tasks and all time steps.
    Follows the exact same pattern as batched_loss from _4_rnn_model.py.

    Arguments:
        params: dictionary of RNN weights
    
    Returns:
        loss: scalar loss value averaged over all gap training tasks
    """
    gap_tasks      = jnp.arange(num_training_tasks)

    # Vectorized simulation over all gap training tasks
    zs_all         = vmap_solve_gap_task(params, gap_tasks)      # shape: (num_training_tasks, num_steps_train)

    # Get the training targets for all gap training tasks
    targets_all    = vmap(get_gap_train_target)(gap_tasks)       # shape: (num_training_tasks, num_steps_train)

    # Compute the MSE over tasks and time
    return jnp.mean((zs_all - targets_all) ** 2)


def run_gap_training():
    """Run gap training with the specified frequency ranges."""
    # Initialize model parameters
    key             = random.PRNGKey(JAX_SEED)
    _, subkey       = random.split(key)
    if CONTROL_RANK == False:
        mask, params    = init_params_gap(subkey, S)
    else:
        mask, params    = init_params_gap_control_rank(subkey, S, R, N, cov_general)
    
    # Create custom loss function for gap training
    # Using the optimized version that follows _4_rnn_model.py patterns exactly
    gap_loss_fn     = gap_batched_loss
    
    # Train the model
    print("Starting gap training...")
    with Timer("Gap training"):
        trained_params, final_loss = train_model(params, mask, loss_fn=gap_loss_fn)
    print(f"Gap training completed! Final loss: {final_loss:.6e}")
    
    # Save trained parameters
    save_file_to_pkl(trained_params, "Gap_Trained_Params")
    
    return trained_params, final_loss


# Note that this is the original version, which is kept for reference
# def create_gap_training_loss_function_original():
#     """Create a loss function that only uses the gap training frequencies (original version for reference)."""
#     # Create a custom loss function using the filtered training data
#     drive_inputs, train_inputs, train_targets = generate_gap_training_data()
    
#     def gap_training_loss(params):
#         """Custom loss function for gap training (original sequential implementation)."""
#         total_loss = 0.0
#         for i in range(len(training_omegas)):
#             # Initialize state
#             x0 = jnp.zeros((N,), dtype=jnp.float32)
            
#             # Drive phase
#             xs_drive, _ = simulate_trajectory(x0, drive_inputs[i], params)
#             x_drive_final = xs_drive[-1]
            
#             # Training phase
#             _, zs_train = simulate_trajectory(x_drive_final, train_inputs[i], params)
            
#             # Compute loss for this task
#             task_loss = jnp.mean((zs_train - train_targets[i]) ** 2)
#             total_loss += task_loss
        
#         return total_loss / len(training_omegas)
    
#     return gap_training_loss




# ============================================================
# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================
# ============================================================
def find_points_gap(
    x0_guess, u_const, J_trained, B_trained, b_x_trained,
    mode: str,
    num_attempts=None, tol=None, maxiter=None, alt_inits=None, gaussian_std=None):
    """
    Unified finder for fixed points and slow points.

    Notable arguments:
        mode        : 'fixed' for fixed point search or 'slow' for slow point search
        num_attempts: number of attempts to find a point
        tol         : tolerance for two fixed/slow points to be considered distinct
        maxiter     : maximum iterations for the solver
        alt_inits   : alternative initial conditions from training trajectories
        gaussian_std: standard deviation for Gaussian noise in perturbed initial conditions

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
        
        # Define multiple solver strategies for fixed points
        def make_solver(method, maxiter_val, xtol=1e-6):
            if method == 'lm':
                return lambda x: root(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                      x, method=method, options={'xtol': xtol})
            elif method == 'hybr':
                return lambda x: root(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                      x, method=method, options={'maxfev': maxiter_val, 'xtol': xtol})
            else:  # broyden1 and others
                return lambda x: root(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                      x, method=method, options={'maxfev': maxiter_val})
        
        # Try multiple solver configurations
        solvers = [
            make_solver('hybr', maxiter),         # Powell's hybrid method
        ]
        
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
        raise ValueError("Argument 'mode' must be either 'fixed' or 'slow'.")

    points = []

    # Debug: Check the initial guess quality
    F_initial = compute_F(x0_guess, u_const, J_trained, B_trained, b_x_trained)
    F_initial_norm = np.linalg.norm(F_initial)
    if F_initial_norm > 50:  # Only warn if very far
        print(f"DEBUG: Initial guess is very far from a {mode} point (||F|| = {F_initial_norm:.2e})")

    def _attempt(x_init):
        # For fixed points, try multiple solvers
        if mode == 'fixed':
            for i, solver in enumerate(solvers):
                try:
                    sol = solver(x_init)
                    if success_cond(sol):
                        return extract(sol)
                except Exception as e:
                    continue  # Try next solver
        else:
            # For slow points, use the original single solver approach
            try:
                sol = solver(x_init)
                if success_cond(sol):
                    return extract(sol)
            except Exception as e:
                pass  # Silently fail for now
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

    # # (III) Alternative initial conditions from training trajectories
    # if alt_inits is not None:
    #     for cand in tqdm(alt_inits, desc=f"Finding {mode} points (initial conditions from training)"):
    #         pt = _attempt(cand)
    #         if pt is not None and all(np.linalg.norm(pt - p) >= tol for p in points):
    #             points.append(pt)

    return points


def find_and_analyze_points_gap(state_traj_states, state_fixed_point_inits, params, frequencies, point_type="fixed"):
    """
    Unified function to find and analyze either fixed points or slow points for all frequencies.
    
    Arguments:
        state_traj_states: list of trajectory states from training, one for each frequency
        state_fixed_point_inits: list of initial states for fixed point search, one for each frequency
        params: trained model parameters
        frequencies: list of frequencies to analyze (from generate_comprehensive_test_frequencies)
        point_type: str, either "fixed" or "slow" to specify which type of points to find
    
    Returns:
        all_points              : list of lists of points, one list per frequency
        all_jacobians           : list of lists of Jacobians, one list per frequency  
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies
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
    all_points = []              # List of lists of points, one list per frequency
    all_jacobians = []           # List of lists of Jacobians, one list per frequency
    all_unstable_eig_freq = []   # List of lists of lists of unstable frequencies
    
    # Track search time
    start_time = time.time()
    print(f"\nStarting {search_name} search...")
    
    # Get number of frequencies to analyze
    num_frequencies = len(frequencies)
    
    # Iterate over each frequency to find points
    for j in range(num_frequencies):
        if j % 5 == 0:
            print(f"\n Processing Frequency {j+1} of {num_frequencies} (ω={frequencies[j]:.3f} rad/s)...")

        # Calculate the static input for this frequency using the same pattern as training
        u_const = calculate_static_input(frequencies[j])
        x0_guess = state_fixed_point_inits[j]
        
        # Debug: Print key values for the first few frequencies
        if j < 3:
            print(f"   DEBUG Frequency {j+1}: ω={frequencies[j]:.4f}, u_const={u_const:.4f}, x0_guess_norm={np.linalg.norm(x0_guess):.4f}")
        
        # Build fallback initializations from training trajectories
        # Use all available trajectory states for fallback initializations
        max_fallback_attempts = min(len(state_traj_states), num_attempts)
        if max_fallback_attempts > 0:
            idxs = np.random.choice(len(state_traj_states),
                                    size=max_fallback_attempts, replace=False)
            fallback_inits = []
            for idx in idxs:
                traj = state_traj_states[idx]                   # shape (num_steps_train+1, N)
                row = np.random.randint(0, traj.shape[0])
                fallback_inits.append(traj[row])                # shape (N,)
        else:
            fallback_inits = []
        
        # Find the points
        task_points = find_points_gap(
            x0_guess, u_const, J_trained, B_trained, b_x_trained,
            mode=point_type,
            num_attempts=num_attempts, 
            tol=tol, 
            maxiter=maxiter, 
            alt_inits=fallback_inits, 
            gaussian_std=gaussian_std
        )
        all_points.append(task_points)
        
        # Debug: Print how many points were found for the first few frequencies
        if j < 3:
            print(f"   DEBUG Frequency {j+1}: Found {len(task_points)} fixed points")
        
        # Analyze the points to compute Jacobians and unstable eigenvalues
        task_jacobians, task_unstable_freqs = analyze_points(task_points, J_trained)
        all_jacobians.append(task_jacobians)
        all_unstable_eig_freq.append(task_unstable_freqs)
    
    # Calculate search time
    search_time = time.time() - start_time
    print(f"{search_name.capitalize()} search completed in {search_time:.2f} seconds")
    
    return all_points, all_jacobians, all_unstable_eig_freq




# ============================================================
# ============================================================
# TESTING FUNCTIONS
# ============================================================
# ============================================================
def calculate_variable_test_time(frequency, min_cycles=MIN_CYCLES):
    """Calculate variable test time based on frequency to ensure adequate coverage."""
    if not VARIABLE_TEST_TIME:
        return T_train
    
    # Safety check for very low frequencies
    if frequency <= 1e-6:
        print(f"Warning: Very low frequency {frequency}, using default training time")
        return T_train
    
    # Ensure at least 2 full cycles for lower frequencies
    cycle_time   = 2 * jnp.pi / frequency
    min_time     = min_cycles * cycle_time
    
    # Use the maximum of default time and minimum required time
    return float(max(T_train, min_time))


def generate_comprehensive_test_frequencies(training_freq_ranges, testing_freq_range, training_static_inputs,
                                            num_samples_middle=NUM_SAMPLES_MIDDLE, num_samples_training=NUM_SAMPLES_TRAINING,
                                            num_samples_extrapolation_higher=NUM_SAMPLES_EXTRAPOLATION_HIGHER, num_samples_extrapolation_lower=NUM_SAMPLES_EXTRAPOLATION_LOWER):
    """Generate a comprehensive set of test frequencies covering the full range."""
    # Dense sampling around the gap
    gap_freqs      = np.linspace(training_freq_ranges[0][1], training_freq_ranges[1][0], num_samples_middle)
    
    # Sparse sampling in training ranges
    lower_train    = np.linspace(training_freq_ranges[0][0], training_freq_ranges[0][1], num_samples_training)
    upper_train    = np.linspace(training_freq_ranges[1][0], training_freq_ranges[1][1], num_samples_training)
    
    # Extrapolation ranges
    low_extrap     = np.linspace(testing_freq_range[0], training_freq_ranges[0][0], num_samples_extrapolation_lower)
    high_extrap    = np.linspace(training_freq_ranges[1][1], testing_freq_range[1], num_samples_extrapolation_higher)
    
    # Combine all frequencies - make sure to also test all training frequencies
    all_freqs      = np.concatenate([low_extrap, lower_train, gap_freqs, upper_train, high_extrap, training_omegas])
    
    # Remove duplicates and sort
    return sorted(list(set(all_freqs)))


def generate_trajectory_states_for_frequencies(frequencies, params):
    """
    Generate trajectory states and fixed point initializations for a list of frequencies.
    
    Arguments:
        frequencies: list of frequencies to analyze
        params: trained RNN parameters
        
    Returns:
        fixed_point_inits: list of initial states after drive phase for each frequency
        traj_states: list of trajectory states during training phase for each frequency
        frequency_labels: list of frequencies corresponding to each trajectory
    """
    fixed_point_inits = []
    traj_states = []
    frequency_labels = []
    
    for freq in frequencies:
        # Calculate static input for this frequency
        static_input = calculate_static_input(freq)
        
        # Generate drive phase input
        drive_input = (jnp.sin(freq * time_drive) + static_input)[:, None]
        
        # Generate training phase input (static input only)
        train_input = jnp.full((num_steps_train, 1), static_input, dtype=jnp.float32)
        
        # Initialize state and run drive phase
        x0 = jnp.zeros((N,), dtype=jnp.float32)
        xs_drive, _ = simulate_trajectory(x0, drive_input, params)
        x_drive_final = xs_drive[-1]
        
        # Run training phase to get trajectory states
        xs_train, _ = simulate_trajectory(x_drive_final, train_input, params)
        
        # Store results
        fixed_point_inits.append(x_drive_final)
        traj_states.append(xs_train)
        frequency_labels.append(freq)
    
    return fixed_point_inits, traj_states, frequency_labels


def test_frequency_generalization_with_params(params, test_frequencies):
    """Test frequency generalization using provided parameters."""
    # Test each frequency
    results = {}
    
    for freq in test_frequencies:
        print(f"Testing frequency: {freq:.3f} rad/s")
        
        # Calculate static input
        static_input   = calculate_static_input(freq)
        # Calculate variable test time
        test_time      = calculate_variable_test_time(freq)
        # Generate time arrays for this frequency
        time_test      = jnp.arange(0, test_time, dt, dtype=jnp.float32)
        # Generate test data with variable time
        test_data      = generate_sine_wave_data(frequency=freq,
                                                 static_input=static_input,
                                                 time_drive=time_drive,
                                                 time_train_or_test=time_test)
        
        try:
            # Initialize with zeros
            x0        = jnp.zeros(N)
            
            # Run RNN simulation
            xs, zs     = simulate_trajectory(x0, test_data['inputs'], params)
            # xs has shape (num_steps_drive + num_steps_test, N)
            # zs has shape (num_steps_drive + num_steps_test,)
            
            # Extract outputs (only from test phase) - use actual drive phase length
            actual_drive_steps = len(time_drive)
            x_drive_final      = xs[actual_drive_steps - 1]                   # Final state after drive phase, shape (N,)
            xs_train           = xs[actual_drive_steps:]                      # shape (num_steps_train_or_test, N)
            rnn_output         = zs[actual_drive_steps:, None]  
            test_targets       = test_data['targets'][actual_drive_steps:]
            
            # Calculate MSE
            mse          = np.mean((rnn_output - test_targets) ** 2)
            print(f"MSE: {mse:.6f}")
            
            # Store results
            results[freq] = {'mse': float(mse),
                             'static_input':    static_input,
                             'test_time':       test_time,
                             'x_drive_final':   x_drive_final,
                             'xs_train':        xs_train,
                             'rnn_output':      rnn_output,
                             'targets':         test_targets}
            
        except Exception as e:
            print(f"Error: {e}")
            results[freq] = {'error': str(e)}
    
    # Create plot
    successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
    if successful_results:
        
        # Create plot of MSE vs frequency with gaps
        test_freqs = sorted(successful_results.keys())
        mse_values = [successful_results[freq]['mse'] for freq in test_freqs]
        create_error_vs_frequency_plot_with_gaps(test_freqs, mse_values)
        
        # Create comprehensive subplot visualization of output trajectories vs targets
        create_comprehensive_trajectory_subplots(results)
    
    # Create comprehensive summary of results
    save_comprehensive_summary(results)
    
    return results




# ============================================================
# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
# ============================================================
def create_error_vs_frequency_plot_with_gaps(test_frequencies, mse_values):
    """
    Create MSE vs frequency plot with shaded training ranges and highlighted training gaps.
    
    Arguments:
        test_frequencies:   Frequencies used for testing (list or array)
        mse_values:         Mean squared error values for each frequency
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the error vs frequency
    plt.plot(test_frequencies, mse_values, 'bo-', linewidth=2, markersize=6, label='Test Performance')
    
    # Define region colors (same as create_comprehensive_trajectory_subplots)
    region_colors = {
        'lower_extrap': None,           # No shading for lower extrapolation
        'lower_training': '#E6F7E6',   # Very light green  
        'gap': '#FFE6CC',              # Very light orange
        'upper_training': '#E6F7E6',   # Very light green (same as lower training)
        'higher_extrap': None          # No shading for higher extrapolation
    }
    
    # Find the frequency range for plotting
    freq_min = min(test_frequencies)
    freq_max = max(test_frequencies)
    
    # Shade the lower training region
    lower_train_start = max(TRAINING_FREQ_RANGES[0][0], freq_min)
    lower_train_end = min(TRAINING_FREQ_RANGES[0][1], freq_max)
    if lower_train_start < lower_train_end and region_colors['lower_training']:
        plt.axvspan(lower_train_start, lower_train_end, alpha=0.4, 
                   color=region_colors['lower_training'], label='Lower Training Range')
    
    # Shade the gap region
    gap_start = max(TRAINING_FREQ_RANGES[0][1], freq_min)  # 0.25
    gap_end = min(TRAINING_FREQ_RANGES[1][0], freq_max)    # 0.45
    if gap_start < gap_end and region_colors['gap']:
        plt.axvspan(gap_start, gap_end, alpha=0.5, color=region_colors['gap'], 
                   label=f'Gap ({TRAINING_FREQ_RANGES[0][1]:.2f}-{TRAINING_FREQ_RANGES[1][0]:.2f} rad/s)')
    
    # Shade the upper training region
    upper_train_start = max(TRAINING_FREQ_RANGES[1][0], freq_min)
    upper_train_end = min(TRAINING_FREQ_RANGES[1][1], freq_max)
    if upper_train_start < upper_train_end and region_colors['upper_training']:
        plt.axvspan(upper_train_start, upper_train_end, alpha=0.4, 
                   color=region_colors['upper_training'], label='Upper Training Range')
    
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('RNN Performance vs Test Frequency (Gap Training)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Save the plot using simplified saving function
    save_plot_to_png(plt, "MSE_vs_Frequency_with_Gaps")
    plt.close()


def create_comprehensive_trajectory_subplots(results, max_plots=MAX_PLOTS,
                                             num_plots_middle=NUM_PLOTS_MIDDLE,
                                             num_plots_training=NUM_PLOTS_TRAINING,
                                             num_plots_extrapolation_higher=NUM_PLOTS_EXTRAPOLATION_HIGHER,
                                             num_plots_extrapolation_lower=NUM_PLOTS_EXTRAPOLATION_LOWER):
    """
    Create a comprehensive subplot visualization showing RNN outputs vs targets for multiple frequencies.
    
    Arguments:
        results:    Dictionary of test results from test_frequency_generalization_with_params()
        max_plots:  Maximum number of subplots to create (to avoid overcrowding)
    """
    # ----------------------------------------
    # SELECT FREQUENCIES TO PLOT
    # Filter successful results and sort by frequency
    successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
    if not successful_results:
        print("No successful results to plot")
        return
    
    all_freqs = sorted(successful_results.keys())
    
    # If we have fewer frequencies than max_plots, use all of them
    if len(all_freqs) <= max_plots:
        test_freqs = all_freqs
    else:
        # Categorize frequencies by region
        lower_extrap = []
        lower_training = []
        gap = []
        upper_training = []
        higher_extrap = []
        
        for freq in all_freqs:
            if freq < TRAINING_FREQ_RANGES[0][0]:  # Lower extrapolation
                lower_extrap.append(freq)
            elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:  # Lower training range
                lower_training.append(freq)
            elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:  # Gap region
                gap.append(freq)
            elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:  # Upper training range
                upper_training.append(freq)
            else:  # Higher extrapolation
                higher_extrap.append(freq)
        
        # Select frequencies with equal distribution within each region
        def select_equally_distributed(freq_list, n_select):
            """Select n_select frequencies equally distributed from freq_list"""
            if len(freq_list) == 0:
                return []
            if len(freq_list) <= n_select:
                return freq_list
            
            # Select indices that are equally distributed
            indices = np.linspace(0, len(freq_list) - 1, n_select, dtype=int)
            return [freq_list[i] for i in indices]
        
        # Select frequencies according to the specified priorities
        selected_freqs = []
        selected_freqs.extend(select_equally_distributed(lower_extrap, num_plots_extrapolation_lower))
        selected_freqs.extend(select_equally_distributed(lower_training, num_plots_training))
        selected_freqs.extend(select_equally_distributed(gap, num_plots_middle))
        selected_freqs.extend(select_equally_distributed(upper_training, num_plots_training))
        selected_freqs.extend(select_equally_distributed(higher_extrap, num_plots_extrapolation_higher))
        
        # Sort the selected frequencies and limit to max_plots
        test_freqs = sorted(selected_freqs)[:max_plots]
    

    # ----------------------------------------
    # SET UP SUBPLOTS
    n_plots = len(test_freqs)
    
    # Calculate subplot grid dimensions (roughly square)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig.suptitle('RNN Output vs Target Trajectories for Different Frequencies', fontsize=16)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors for each region
    region_colors = {
        'lower_extrap': '#FFE6E6',      # Very light red
        'lower_training': '#E6F7E6',    # Very light green
        'gap': '#FFE6CC',               # Very light orange
        'upper_training': '#E6F7E6',    # Very light green (same as lower training)
        'higher_extrap': '#E6E6FF'      # Very light blue
    }
    
    # Function to determine region for a frequency
    def get_frequency_region(freq):
        if freq < TRAINING_FREQ_RANGES[0][0]:
            return 'lower_extrap'
        elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:
            return 'lower_training'
        elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:
            return 'gap'
        elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:
            return 'upper_training'
        else:
            return 'higher_extrap'
    

    # ----------------------------------------
    # PLOT SUBPLOTS
    # Plot each frequency in its own subplot
    for i, freq in enumerate(test_freqs):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        result = successful_results[freq]
        rnn_output = result['rnn_output']
        targets = result['targets']
        mse = result['mse']
        
        # Create time axis
        time_axis = np.arange(len(rnn_output)) * dt
        
        # Determine region and set background color
        region = get_frequency_region(freq)
        ax.set_facecolor(region_colors[region])
        
        # Plot trajectories
        ax.plot(time_axis, rnn_output.flatten(), 'b-', label='RNN Output', linewidth=1.5)
        ax.plot(time_axis, targets.flatten(), 'r--', label='Target', linewidth=1.5, alpha=0.7)
        
        # Formatting
        ax.set_title(f'ω={freq:.2f} rad/s\nMSE={mse:.2e}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=8)
    

    # ----------------------------------------
    # STYLE AND LAYOUT
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for i in range(n_plots, total_subplots):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    # Create region legend in the top right corner of the figure
    legend_elements = [
        Patch(facecolor=region_colors['lower_extrap'], label='Lower Extrapolation'),
        Patch(facecolor=region_colors['lower_training'], label='Lower Training'),
        Patch(facecolor=region_colors['gap'], label='Gap Region'),
        Patch(facecolor=region_colors['upper_training'], label='Upper Training'),
        Patch(facecolor=region_colors['higher_extrap'], label='Higher Extrapolation')
    ]
    
    # Adjust layout first to calculate proper spacing
    plt.tight_layout()
    
    # Add the legend to the figure outside the plot area to avoid overlap
    # Place it on the right side of the figure, outside the subplot area
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
               fontsize=10, title='Frequency Regions', title_fontsize=11)
    
    # Adjust the subplot layout to make room for the legend
    plt.subplots_adjust(right=0.85)
    save_plot_to_png(fig, f"Comprehensive_Trajectory_Subplots_{n_plots}_frequencies")
    plt.close()


def save_comprehensive_summary(results):
    """Save comprehensive summary of gap training test results."""
    # Analyze results by region
    successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
    
    summary_data = {"title": "Gap Training Frequency Generalization Test Results",
                    "training_configuration": {"execution_mode": EXECUTION_MODE,
                                               "run_testing_after_training": RUN_TESTING_AFTER_TRAINING,
                                               "training_frequency_ranges": TRAINING_FREQ_RANGES,
                                               "excluded_gap": [TRAINING_FREQ_RANGES[0][1], TRAINING_FREQ_RANGES[1][0]],
                                               "test_frequency_range": TESTING_FREQ_RANGE,
                                               "variable_test_time": VARIABLE_TEST_TIME,
                                               "number_of_samples_from_gap": NUM_SAMPLES_MIDDLE,
                                               "number_of_samples_from_each_training_range": NUM_SAMPLES_TRAINING,
                                               "number_of_samples_from_extrapolation_higher": NUM_SAMPLES_EXTRAPOLATION_HIGHER,
                                               "number_of_samples_from_extrapolation_lower": NUM_SAMPLES_EXTRAPOLATION_LOWER},
                    "results": results}
    
    if successful_results:
        # Categorize frequencies
        in_gap = []
        in_training = []
        extrapolation = []
        
        for freq in successful_results.keys():
            if TRAINING_FREQ_RANGES[0][1] <= freq <= TRAINING_FREQ_RANGES[1][0]:
                in_gap.append(freq)
            elif any(r[0] <= freq <= r[1] for r in TRAINING_FREQ_RANGES):
                in_training.append(freq)
            else:
                extrapolation.append(freq)
        
        # Add performance by region
        performance_by_region = {}
        
        if in_training:
            mse_training = [successful_results[freq]['mse'] for freq in in_training]
            performance_by_region["training_regions"] = {
                "frequency_count": len(in_training),
                "mean_mse": float(np.mean(mse_training)),
                "max_mse": float(np.max(mse_training))
            }
        
        if in_gap:
            mse_gap = [successful_results[freq]['mse'] for freq in in_gap]
            performance_by_region["gap_region"] = {
                "frequency_count": len(in_gap),
                "mean_mse": float(np.mean(mse_gap)),
                "max_mse": float(np.max(mse_gap))
            }
        
        if extrapolation:
            mse_extrap = [successful_results[freq]['mse'] for freq in extrapolation]
            performance_by_region["extrapolation_regions"] = {
                "frequency_count": len(extrapolation),
                "mean_mse": float(np.mean(mse_extrap)),
                "max_mse": float(np.max(mse_extrap))
            }
        
        summary_data["performance_by_region"] = performance_by_region
    
    save_file_to_pkl(summary_data, "Frequency_Generalization_Summary")


def save_fixed_point_analysis_summary(all_fixed_points, all_unstable_eig_freq, frequency_labels, base_filename="Fixed_Point_Analysis_Summary"):
    """
    Generate and save a comprehensive text summary of fixed point analysis.
    
    Arguments:
        all_fixed_points: list of lists of fixed points, one list per frequency
        all_unstable_eig_freq: list of lists of lists of unstable frequencies
        frequency_labels: list of frequencies corresponding to each analysis
        base_filename: base name for the output text file
    """
    # Helper function to determine frequency region
    def get_frequency_region(freq):
        if freq < TRAINING_FREQ_RANGES[0][0]:
            return "Lower Extrapolation"
        elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:
            return "Lower Training"
        elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:
            return "Gap Region"
        elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:
            return "Upper Training"
        else:
            return "Higher Extrapolation"
    
    # Create summary content
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("FIXED POINT ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Count frequencies by number of fixed points
    fp_count_distribution = {}
    for i, freq_fixed_points in enumerate(all_fixed_points):
        num_fps = len(freq_fixed_points)
        if num_fps not in fp_count_distribution:
            fp_count_distribution[num_fps] = 0
        fp_count_distribution[num_fps] += 1
    
    summary_lines.append("1. DISTRIBUTION OF FREQUENCIES BY NUMBER OF FIXED POINTS:")
    summary_lines.append("-" * 60)
    for num_fps in sorted(fp_count_distribution.keys()):
        summary_lines.append(f"   {fp_count_distribution[num_fps]} frequencies have {num_fps} fixed point(s)")
    summary_lines.append("")
    
    # NEW SECTION: Count frequencies by number of fixed points BY REGION
    fp_count_by_region = {}
    for i, freq_fixed_points in enumerate(all_fixed_points):
        freq = frequency_labels[i]
        region = get_frequency_region(freq)
        num_fps = len(freq_fixed_points)
        
        if region not in fp_count_by_region:
            fp_count_by_region[region] = {}
        if num_fps not in fp_count_by_region[region]:
            fp_count_by_region[region][num_fps] = 0
        fp_count_by_region[region][num_fps] += 1
    
    summary_lines.append("2. DISTRIBUTION OF FREQUENCIES BY NUMBER OF FIXED POINTS BY REGION:")
    summary_lines.append("-" * 60)
    for region in ["Lower Extrapolation", "Lower Training", "Gap Region", "Upper Training", "Higher Extrapolation"]:
        if region in fp_count_by_region:
            summary_lines.append(f"   {region}:")
            for num_fps in sorted(fp_count_by_region[region].keys()):
                summary_lines.append(f"      {fp_count_by_region[region][num_fps]} frequencies have {num_fps} fixed point(s)")
        else:
            summary_lines.append(f"   {region}: No frequencies in this region")
    summary_lines.append("")
    
    # Count fixed points by number of unstable eigenvalues
    unstable_eig_distribution = {}
    total_fixed_points = 0
    
    for i, freq_unstable_eigs in enumerate(all_unstable_eig_freq):
        for fp_unstable_eigs in freq_unstable_eigs:
            total_fixed_points += 1
            num_unstable = len(fp_unstable_eigs)
            if num_unstable not in unstable_eig_distribution:
                unstable_eig_distribution[num_unstable] = 0
            unstable_eig_distribution[num_unstable] += 1
    
    summary_lines.append("3. DISTRIBUTION OF FIXED POINTS BY NUMBER OF UNSTABLE EIGENVALUES:")
    summary_lines.append("-" * 60)
    for num_unstable in sorted(unstable_eig_distribution.keys()):
        summary_lines.append(f"   {unstable_eig_distribution[num_unstable]} fixed point(s) have {num_unstable} unstable eigenvalue(s)")
    summary_lines.append("")
    summary_lines.append(f"   Total fixed points found: {total_fixed_points}")
    summary_lines.append("")
    
    # NEW SECTION: Count fixed points by number of unstable eigenvalues BY REGION
    unstable_eig_by_region = {}
    
    for i, freq_unstable_eigs in enumerate(all_unstable_eig_freq):
        freq = frequency_labels[i]
        region = get_frequency_region(freq)
        
        if region not in unstable_eig_by_region:
            unstable_eig_by_region[region] = {}
            
        for fp_unstable_eigs in freq_unstable_eigs:
            num_unstable = len(fp_unstable_eigs)
            if num_unstable not in unstable_eig_by_region[region]:
                unstable_eig_by_region[region][num_unstable] = 0
            unstable_eig_by_region[region][num_unstable] += 1
    
    summary_lines.append("4. DISTRIBUTION OF FIXED POINTS BY NUMBER OF UNSTABLE EIGENVALUES BY REGION:")
    summary_lines.append("-" * 60)
    for region in ["Lower Extrapolation", "Lower Training", "Gap Region", "Upper Training", "Higher Extrapolation"]:
        if region in unstable_eig_by_region:
            total_fps_in_region = sum(unstable_eig_by_region[region].values())
            summary_lines.append(f"   {region} (Total: {total_fps_in_region} fixed points):")
            for num_unstable in sorted(unstable_eig_by_region[region].keys()):
                summary_lines.append(f"      {unstable_eig_by_region[region][num_unstable]} fixed point(s) have {num_unstable} unstable eigenvalue(s)")
        else:
            summary_lines.append(f"   {region}: No fixed points in this region")
    summary_lines.append("")
    
    # NEW SECTION: Distribution of frequencies by number of unstable eigenvalues
    freq_unstable_distribution = {}
    for i, freq_unstable_eigs in enumerate(all_unstable_eig_freq):
        # Count total unstable eigenvalues for this frequency across all its fixed points
        total_unstable_for_freq = sum(len(fp_unstable_eigs) for fp_unstable_eigs in freq_unstable_eigs)
        
        if total_unstable_for_freq not in freq_unstable_distribution:
            freq_unstable_distribution[total_unstable_for_freq] = 0
        freq_unstable_distribution[total_unstable_for_freq] += 1
    
    summary_lines.append("5. DISTRIBUTION OF FREQUENCIES BY NUMBER OF UNSTABLE EIGENVALUES:")
    summary_lines.append("-" * 60)
    for num_unstable in sorted(freq_unstable_distribution.keys()):
        summary_lines.append(f"   {freq_unstable_distribution[num_unstable]} frequency(ies) have {num_unstable} total unstable eigenvalue(s)")
    summary_lines.append("")
    
    # NEW SECTION: Distribution of frequencies by number of unstable eigenvalues BY REGION
    freq_unstable_by_region = {}
    for i, freq_unstable_eigs in enumerate(all_unstable_eig_freq):
        freq = frequency_labels[i]
        region = get_frequency_region(freq)
        
        # Count total unstable eigenvalues for this frequency across all its fixed points
        total_unstable_for_freq = sum(len(fp_unstable_eigs) for fp_unstable_eigs in freq_unstable_eigs)
        
        if region not in freq_unstable_by_region:
            freq_unstable_by_region[region] = {}
        if total_unstable_for_freq not in freq_unstable_by_region[region]:
            freq_unstable_by_region[region][total_unstable_for_freq] = 0
        freq_unstable_by_region[region][total_unstable_for_freq] += 1
    
    summary_lines.append("6. DISTRIBUTION OF FREQUENCIES BY NUMBER OF UNSTABLE EIGENVALUES BY REGION:")
    summary_lines.append("-" * 60)
    for region in ["Lower Extrapolation", "Lower Training", "Gap Region", "Upper Training", "Higher Extrapolation"]:
        if region in freq_unstable_by_region:
            total_freqs_in_region = sum(freq_unstable_by_region[region].values())
            summary_lines.append(f"   {region} (Total: {total_freqs_in_region} frequencies):")
            for num_unstable in sorted(freq_unstable_by_region[region].keys()):
                summary_lines.append(f"      {freq_unstable_by_region[region][num_unstable]} frequency(ies) have {num_unstable} total unstable eigenvalue(s)")
        else:
            summary_lines.append(f"   {region}: No frequencies in this region")
    summary_lines.append("")
    
    # Frequency-by-frequency detailed summary
    summary_lines.append("7. FREQUENCY-BY-FREQUENCY DETAILED SUMMARY:")
    summary_lines.append("-" * 60)
    
    for i, freq in enumerate(frequency_labels):
        freq_fixed_points = all_fixed_points[i]
        freq_unstable_eigs = all_unstable_eig_freq[i]
        
        # Determine frequency region
        region = get_frequency_region(freq)
        
        summary_lines.append(f"Frequency {freq:.4f} rad/s ({region}):")
        summary_lines.append(f"   Number of fixed points: {len(freq_fixed_points)}")
        
        if len(freq_fixed_points) > 0:
            for fp_idx, fp_unstable_eigs in enumerate(freq_unstable_eigs):
                summary_lines.append(f"   Fixed point {fp_idx + 1}: {len(fp_unstable_eigs)} unstable eigenvalue(s)")
        else:
            summary_lines.append("   No fixed points found")
        summary_lines.append("")
    
    # Save to text file
    summary_text = "\n".join(summary_lines)
    
    # Create output directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{EXECUTION_MODE}_{timestamp}')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save file
    filename = f"{base_filename}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(summary_text)



def plot_frequency_comparison_gap(all_unstable_eig_freq, test_frequencies):
    """
    Plot a comparison of target frequencies and maximum unstable mode frequencies (imaginary components) 
    for all frequencies in the frequency generalization experiment.

    Arguments:
        all_unstable_eig_freq: list of lists of lists, where the outer list is over frequencies,
            the middle list is over fixed points,
            and the inner list contains unstable eigenvalues for each fixed point
        test_frequencies: array of test frequencies from the frequency generalization experiment
    """
    # ----------------------------------------
    # SET-UP
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Define region colors (same as create_comprehensive_trajectory_subplots)
    region_colors = {
        'lower_extrap': '#FFE6E6',      # Very light red
        'lower_training': '#E6F7E6',    # Very light green
        'gap': '#FFE6CC',               # Very light orange
        'upper_training': '#E6F7E6',    # Very light green (same as lower training)
        'higher_extrap': '#E6E6FF'      # Very light blue
    }
    
    # Function to determine region for a frequency
    def get_frequency_region(freq):
        if freq < TRAINING_FREQ_RANGES[0][0]:
            return 'lower_extrap'
        elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:
            return 'lower_training'
        elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:
            return 'gap'
        elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:
            return 'upper_training'
        else:
            return 'higher_extrap'

    # Function to determine index range for frequency ranges
    def freq_to_index_range(freq_start, freq_end):
        # Find approximate index ranges for frequency ranges
        start_idx = None
        end_idx = None
        for i, freq in enumerate(test_frequencies):
            if start_idx is None and freq >= freq_start:
                start_idx = i
            if freq <= freq_end:
                end_idx = i
        return start_idx, end_idx

    # Initialize lists to store data for plotting
    freq_indices = []
    target_frequencies = []
    unstable_frequencies = []

    # ----------------------------------------
    # ORGANISE DATA
    # Collect data for all frequencies
    for j, task_freqs in enumerate(all_unstable_eig_freq):
        target_freq = test_frequencies[j]
        # Find the maximum imaginary component across all fixed points in this frequency
        max_imag = 0
        for fp_freqs in task_freqs:
            if fp_freqs:
                current_max = max(abs(np.imag(ev)) for ev in fp_freqs)
                max_imag = max(max_imag, current_max)
        
        if max_imag > 0:    # only consider frequencies with unstable frequencies
            freq_indices.append(j)
            target_frequencies.append(target_freq)
            unstable_frequencies.append(max_imag)
    
    if len(target_frequencies) == 0:
        # Handle case with no data
        ax.text(0.5, 0.5, 'No unstable eigenvalue data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Comparison of Target Frequencies and Maximum Unstable Mode Frequencies')
        plt.tight_layout()
        save_plot_to_png(fig, "Frequency_Comparison_Gap")
        plt.close()
        return
    
    # ----------------------------------------
    # CREATE PLOT
    # Create scatter plot with same colors for all dots
    scatter_size = 100
    
    # Plot target frequencies (black points)
    ax.scatter(freq_indices, target_frequencies, color='black', label='Target Frequency', s=scatter_size, zorder=3)
    
    # Plot unstable frequencies (red points - keep original color)
    ax.scatter(freq_indices, unstable_frequencies, color='red', label='Max Unstable Mode Frequency', 
              s=scatter_size, alpha=1.0, zorder=3)

    # Add background shading for different regions
    freq_min = min(test_frequencies)
    freq_max = max(test_frequencies)
    
    # Shade the lower training region
    if TRAINING_FREQ_RANGES[0][0] <= freq_max and TRAINING_FREQ_RANGES[0][1] >= freq_min:
        start_idx, end_idx = freq_to_index_range(TRAINING_FREQ_RANGES[0][0], TRAINING_FREQ_RANGES[0][1])
        if start_idx is not None and end_idx is not None:
            ax.axvspan(start_idx, end_idx, alpha=0.4, color=region_colors['lower_training'], zorder=1)
    
    # Shade the gap region
    if TRAINING_FREQ_RANGES[0][1] <= freq_max and TRAINING_FREQ_RANGES[1][0] >= freq_min:
        start_idx, end_idx = freq_to_index_range(TRAINING_FREQ_RANGES[0][1], TRAINING_FREQ_RANGES[1][0])
        if start_idx is not None and end_idx is not None:
            ax.axvspan(start_idx, end_idx, alpha=0.5, color=region_colors['gap'], zorder=1)
    
    # Shade the upper training region
    if TRAINING_FREQ_RANGES[1][0] <= freq_max and TRAINING_FREQ_RANGES[1][1] >= freq_min:
        start_idx, end_idx = freq_to_index_range(TRAINING_FREQ_RANGES[1][0], TRAINING_FREQ_RANGES[1][1])
        if start_idx is not None and end_idx is not None:
            ax.axvspan(start_idx, end_idx, alpha=0.4, color=region_colors['upper_training'], zorder=1)

    # Add decorations
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Frequency (rad/s)')
    ax.set_title('Comparison of Target Frequencies and Maximum Unstable Mode Frequencies\n(Frequency Generalization Experiment)')
    ax.legend()
    ax.grid(True, alpha=0.3, zorder=2)
    
    # Save the plot
    save_plot_to_png(fig, "Frequency_Comparison_Gap")
    plt.close()




# ============================================================
# ============================================================
# MAIN EXECUTION
# ============================================================
# ============================================================
def main():
    print("=" * 60)
    print("FREQUENCY GENERALIZATION TESTING SCRIPT")
    print("Execution Mode: ", EXECUTION_MODE)
    print("=" * 60)
    
    # Check if we're in training mode
    training_mode = (EXECUTION_MODE == 'train')
    
    
    # ----------------------------------------
    # A) TRAINING MODE
    if training_mode:
        try:
            # ----------------------------------------
            # TRAINING
            print(f"\nRunning gap training...")
            trained_params, train_final_loss = run_gap_training()
            print(f"\nGap training completed successfully!")

            # Ensure that J has the expected sparsity (the proportion of zero elements is the same as S)
            if trained_params['J'].size > 0:
                actual_sparsity = (trained_params['J'].size - jnp.count_nonzero(trained_params['J'])) / trained_params['J'].size
                if not jnp.isclose(actual_sparsity, S, rtol=1e-2):
                    print(f"Warning: J sparsity ratio {actual_sparsity:.2f} does not match expected {S:.2f}")
            else:
                print("Warning: J is empty, cannot check sparsity ratio")


            # ----------------------------------------
            # FIXED POINT ANALYSIS
            print(f"\nRunning fixed point analysis...")
            # Generate comprehensive test frequencies for analysis
            analysis_freqs = generate_comprehensive_test_frequencies(TRAINING_FREQ_RANGES, TESTING_FREQ_RANGE, training_static_inputs,
                                                                    NUM_SAMPLES_MIDDLE, NUM_SAMPLES_TRAINING,
                                                                    NUM_SAMPLES_EXTRAPOLATION_HIGHER, NUM_SAMPLES_EXTRAPOLATION_LOWER)
            
            # Generate trajectory states for comprehensive frequency set
            fixed_point_inits, traj_states, frequency_labels = generate_trajectory_states_for_frequencies(analysis_freqs, trained_params)
            # DEBUG - Check that the trajectories are being generated correctly
            for i, traj in enumerate(traj_states):
                print(f"Shape of trajectory {i+1}: {traj.shape} (steps={traj.shape[0]}, N={traj.shape[1]})")
            # Save traj_states and fixed_point_inits
            trajs_and_inits = {"trajectory_states": traj_states,
                                "fixed_point_inits": fixed_point_inits,
                                "frequency_labels": frequency_labels}
            save_file_to_pkl(trajs_and_inits, "Trajectory_States_and_Fixed_Point_Inits")

            # Run fixed point analysis on the trained model
            all_fixed_points, all_jacobians, all_unstable_eig_freq \
                = find_and_analyze_points_gap(state_traj_states=traj_states, state_fixed_point_inits=fixed_point_inits,
                                          params=trained_params, frequencies=analysis_freqs, point_type="fixed")
            
            # Save results
            save_file_to_pkl(all_fixed_points, "Fixed_Points")
            save_file_to_pkl(all_jacobians, "Jacobian_Matrices")
            save_file_to_pkl(all_unstable_eig_freq, "Unstable_Eigen_Frequencies")
            save_file_to_pkl(frequency_labels, "Analysis_Frequencies")  # Save the frequencies used
            
            # Generate and save fixed point analysis summary
            save_fixed_point_analysis_summary(all_fixed_points, all_unstable_eig_freq, frequency_labels)
            
            print(f"Fixed point analysis completed successfully!")


            # ----------------------------------------
            # FREQUENCY COMPARISON PLOT
            # Set custom output directory to match this script's output folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            custom_output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{EXECUTION_MODE}_{timestamp}')
            set_custom_output_dir(custom_output_dir)
            
            plot_frequency_comparison_gap(all_unstable_eig_freq=all_unstable_eig_freq,
                                      test_frequencies=jnp.array(frequency_labels))
            

            # ----------------------------------------
            # PCA ANALYSIS
            # Convert list of trajectory states to 3D array for PCA analysis
            traj_states_array = jnp.stack(traj_states, axis=0)  # Convert list to 3D array
            
            # Perform PCA on trajectories
            pca, proj_trajs, all_trajectories_combined \
                = perform_pca_analysis(state_traj_states=traj_states_array,
                skip_initial_steps=PCA_SKIP_INITIAL_STEPS, apply_tanh=PCA_APPLY_TANH, n_components=PCA_N_COMPONENTS)
            
            # --------------------
            # EXPLAINED VARIANCE RATIO PLOT
            # Set custom output directory to match this script's output folder
            set_custom_output_dir(custom_output_dir)
            # Plot explained variance ratio bar chart
            plot_explained_variance_ratio(pca,
                                          skip_initial_steps=PCA_SKIP_INITIAL_STEPS, apply_tanh=PCA_APPLY_TANH)
            
            # --------------------
            # PCA RESULTS DICTIONARY
            # Categorize trajectories by frequency region for comprehensive frequency set
            proj_trajs_by_region = {
                "lower_extrapolation": [],
                "lower_training": [], 
                "gap": [],
                "upper_training": [],
                "higher_extrapolation": []
            }
            
            for i, freq in enumerate(frequency_labels):
                if freq < TRAINING_FREQ_RANGES[0][0]:  # Lower extrapolation
                    proj_trajs_by_region["lower_extrapolation"].append(proj_trajs[i])
                elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:  # Lower training
                    proj_trajs_by_region["lower_training"].append(proj_trajs[i])
                elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:  # Gap region
                    proj_trajs_by_region["gap"].append(proj_trajs[i])
                elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:  # Upper training
                    proj_trajs_by_region["upper_training"].append(proj_trajs[i])
                else:  # Higher extrapolation
                    proj_trajs_by_region["higher_extrapolation"].append(proj_trajs[i])
            
            # pca_results = {"pca":                  pca,                        # Add the PCA object itself
            #     "proj_trajs":                      proj_trajs,
            #     "proj_trajs_by_region":            proj_trajs_by_region,       # Trajectories categorized by region
            #     "frequency_labels":                frequency_labels,           # Frequencies corresponding to trajectories
            #     "all_trajectories_combined":       all_trajectories_combined,  # Add combined trajectory matrix
            #     "skip_initial_steps":              PCA_SKIP_INITIAL_STEPS,
            #     "apply_tanh":                      PCA_APPLY_TANH,
            #     "pca_components":                  pca.components_,
            #     "pca_explained_variance_ratio":    pca.explained_variance_ratio_}
            
            # --------------------
            # PCA PLOTS (FULL)
            # Project and plot fixed points if available
            if all_fixed_points is not None:
                proj_fixed = project_points_to_pca(pca, all_fixed_points)
                # pca_results["proj_fixed"] = proj_fixed
                
                # Set custom output directory to match this script's output folder
                set_custom_output_dir(custom_output_dir)
                # Plot PCA with all fixed points and trajectories (static matplotlib)
                plot_pca_trajectories_and_points(pca, proj_trajs,
                                                 point_type="fixed",
                                                 all_points=all_fixed_points,
                                                 proj_points=proj_fixed,
                                                 params=trained_params,
                                                 skip_initial_steps=PCA_SKIP_INITIAL_STEPS,
                                                 apply_tanh=PCA_APPLY_TANH,
                                                 proj_trajs_by_region=proj_trajs_by_region,
                                                 show_points=False,
                                                 show_unstable_modes=False)
            
            # --------------------
            # PCA PLOTS (SUBSET)
            # Plot subset of trajectories with better distribution across regions
            total_freqs = len(frequency_labels)
            if total_freqs >= 6:
                # Select indices to get good representation: start, end, and some in between
                subset_indices = [0, total_freqs//4, total_freqs//2, 3*total_freqs//4, total_freqs-1]
            elif total_freqs >= 3:
                subset_indices = [0, total_freqs//2, total_freqs-1]
            else:
                subset_indices = list(range(total_freqs))
            
            subset_proj_trajs = [proj_trajs[i] for i in subset_indices if i < len(proj_trajs)]
            subset_fixed_points = [all_fixed_points[i] for i in subset_indices if i < len(all_fixed_points)]
            
            # Create subset trajectories by region using comprehensive frequency set
            subset_proj_trajs_by_region = {
                "lower_extrapolation": [],
                "lower_training": [], 
                "gap": [],
                "upper_training": [],
                "higher_extrapolation": []
            }
            
            for idx in subset_indices:
                if idx < len(frequency_labels):
                    freq = frequency_labels[idx]
                    if freq < TRAINING_FREQ_RANGES[0][0]:  # Lower extrapolation
                        subset_proj_trajs_by_region["lower_extrapolation"].append(proj_trajs[idx])
                    elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:  # Lower training
                        subset_proj_trajs_by_region["lower_training"].append(proj_trajs[idx])
                    elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:  # Gap region
                        subset_proj_trajs_by_region["gap"].append(proj_trajs[idx])
                    elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:  # Upper training
                        subset_proj_trajs_by_region["upper_training"].append(proj_trajs[idx])
                    else:  # Higher extrapolation
                        subset_proj_trajs_by_region["higher_extrapolation"].append(proj_trajs[idx])
            
            if subset_proj_trajs and subset_fixed_points:
                subset_proj_fixed = project_points_to_pca(pca, subset_fixed_points)
                
                # Set custom output directory to match this script's output folder
                set_custom_output_dir(custom_output_dir)
                # Plot PCA with subset with fixed points and trajectories (static matplotlib)
                plot_pca_trajectories_and_points(pca, subset_proj_trajs,
                                                 point_type="fixed",
                                                 all_points=subset_fixed_points,
                                                 proj_points=subset_proj_fixed,
                                                 params=trained_params,
                                                 skip_initial_steps=PCA_SKIP_INITIAL_STEPS,
                                                 apply_tanh=PCA_APPLY_TANH,
                                                 filename_prefix="subset_of_indices_",
                                                 proj_trajs_by_region=subset_proj_trajs_by_region,
                                                 show_points=False,
                                                 show_unstable_modes=False)
            
            # --------------------
            # SAVE PCA RESULTS DICTIONARY
            # save_file_to_pkl(pca_results, "PCA_Results")


            # ----------------------------------------
            # TESTING PHASE (IN TRAINING MODE)
            # Automatically run testing on the trained model if configured to do so
            if RUN_TESTING_AFTER_TRAINING:
                # Reuse the comprehensive test frequencies that were already generated for analysis
                test_freqs = analysis_freqs

                # Run the test
                try:
                    print("\nRunning frequency generalization test on trained model...")
                    test_results = test_frequency_generalization_with_params(params=trained_params,
                                                                            test_frequencies=test_freqs)
                    print(f"Frequency generalization test on trained model completed successfully!")
                
                except Exception as e:
                    print(f"Error during frequency generalization test: {e}")
                    sys.exit(1)
            
            else:
                print("Skipping testing phase (RUN_TESTING_AFTER_TRAINING = False)")
                
        except Exception as e:
            print(f"Error during gap training and testing: {e}")
            sys.exit(1)
    
    
    # ----------------------------------------
    # B) TESTING-ONLY MODE
    else:
        # Validate that parameters are loaded
        if RNN_PARAMS is None:
            print("Error: Cannot run testing mode without valid RNN parameters")
            print("Please check the parameter file paths in the configuration section")
            sys.exit(1)
        else:
            imported_params = RNN_PARAMS


        # ----------------------------------------
        # FIXED POINT ANALYSIS
        print(f"\nRunning fixed point analysis...")
        # Generate comprehensive test frequencies for analysis
        analysis_freqs = generate_comprehensive_test_frequencies(TRAINING_FREQ_RANGES, TESTING_FREQ_RANGE, training_static_inputs,
                                                                NUM_SAMPLES_MIDDLE, NUM_SAMPLES_TRAINING,
                                                                NUM_SAMPLES_EXTRAPOLATION_HIGHER, NUM_SAMPLES_EXTRAPOLATION_LOWER)
        
        # Generate trajectory states for comprehensive frequency set
        fixed_point_inits, traj_states, frequency_labels = generate_trajectory_states_for_frequencies(analysis_freqs, imported_params)
        # DEBUG - Check that the trajectories are being generated correctly
        for i, traj in enumerate(traj_states):
            print(f"Shape of trajectory {i+1}: {traj.shape} (steps={traj.shape[0]}, N={traj.shape[1]})")
        # Save traj_states and fixed_point_inits
        trajs_and_inits = {"trajectory_states": traj_states,
                            "fixed_point_inits": fixed_point_inits,
                            "frequency_labels": frequency_labels}
        save_file_to_pkl(trajs_and_inits, "Trajectory_States_and_Fixed_Point_Inits")

        # Run fixed point analysis on the trained model
        all_fixed_points, all_jacobians, all_unstable_eig_freq \
            = find_and_analyze_points_gap(state_traj_states=traj_states, state_fixed_point_inits=fixed_point_inits,
                                        params=imported_params, frequencies=analysis_freqs, point_type="fixed")
        
        # Save results
        save_file_to_pkl(all_fixed_points, "Fixed_Points")
        save_file_to_pkl(all_jacobians, "Jacobian_Matrices")
        save_file_to_pkl(all_unstable_eig_freq, "Unstable_Eigen_Frequencies")
        save_file_to_pkl(frequency_labels, "Analysis_Frequencies")  # Save the frequencies used
        
        # Generate and save fixed point analysis summary
        save_fixed_point_analysis_summary(all_fixed_points, all_unstable_eig_freq, frequency_labels)
        
        print(f"Fixed point analysis completed successfully!")


        # ----------------------------------------
        # FREQUENCY COMPARISON PLOT
        # Set custom output directory to match this script's output folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        custom_output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{EXECUTION_MODE}_{timestamp}')
        set_custom_output_dir(custom_output_dir)
        
        plot_frequency_comparison_gap(all_unstable_eig_freq=all_unstable_eig_freq,
                                    test_frequencies=jnp.array(frequency_labels))
        

        # ----------------------------------------
        # PCA ANALYSIS
        # Convert list of trajectory states to 3D array for PCA analysis
        traj_states_array = jnp.stack(traj_states, axis=0)  # Convert list to 3D array
        
        # Perform PCA on trajectories
        pca, proj_trajs, all_trajectories_combined \
            = perform_pca_analysis(state_traj_states=traj_states_array,
            skip_initial_steps=PCA_SKIP_INITIAL_STEPS, apply_tanh=PCA_APPLY_TANH, n_components=PCA_N_COMPONENTS)
        
        # --------------------
        # EXPLAINED VARIANCE RATIO PLOT
        # Set custom output directory to match this script's output folder
        set_custom_output_dir(custom_output_dir)
        # Plot explained variance ratio bar chart
        plot_explained_variance_ratio(pca,
                                        skip_initial_steps=PCA_SKIP_INITIAL_STEPS, apply_tanh=PCA_APPLY_TANH)
        
        # --------------------
        # PCA RESULTS DICTIONARY
        # Categorize trajectories by frequency region for comprehensive frequency set
        proj_trajs_by_region = {
            "lower_extrapolation": [],
            "lower_training": [], 
            "gap": [],
            "upper_training": [],
            "higher_extrapolation": []
        }
        
        for i, freq in enumerate(frequency_labels):
            if freq < TRAINING_FREQ_RANGES[0][0]:  # Lower extrapolation
                proj_trajs_by_region["lower_extrapolation"].append(proj_trajs[i])
            elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:  # Lower training
                proj_trajs_by_region["lower_training"].append(proj_trajs[i])
            elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:  # Gap region
                proj_trajs_by_region["gap"].append(proj_trajs[i])
            elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:  # Upper training
                proj_trajs_by_region["upper_training"].append(proj_trajs[i])
            else:  # Higher extrapolation
                proj_trajs_by_region["higher_extrapolation"].append(proj_trajs[i])
        
        # pca_results = {"pca":                  pca,                        # Add the PCA object itself
        #     "proj_trajs":                      proj_trajs,
        #     "proj_trajs_by_region":            proj_trajs_by_region,       # Trajectories categorized by region
        #     "frequency_labels":                frequency_labels,           # Frequencies corresponding to trajectories
        #     "all_trajectories_combined":       all_trajectories_combined,  # Add combined trajectory matrix
        #     "skip_initial_steps":              PCA_SKIP_INITIAL_STEPS,
        #     "apply_tanh":                      PCA_APPLY_TANH,
        #     "pca_components":                  pca.components_,
        #     "pca_explained_variance_ratio":    pca.explained_variance_ratio_}
        
        # --------------------
        # PCA PLOTS (FULL)
        # Project and plot fixed points if available
        if all_fixed_points is not None:
            proj_fixed = project_points_to_pca(pca, all_fixed_points)
            # pca_results["proj_fixed"] = proj_fixed
            
            # Set custom output directory to match this script's output folder
            set_custom_output_dir(custom_output_dir)
            # Plot PCA with all fixed points and trajectories (static matplotlib)
            plot_pca_trajectories_and_points(pca, proj_trajs,
                                                point_type="fixed",
                                                all_points=all_fixed_points,
                                                proj_points=proj_fixed,
                                                params=imported_params,
                                                skip_initial_steps=PCA_SKIP_INITIAL_STEPS,
                                                apply_tanh=PCA_APPLY_TANH,
                                                proj_trajs_by_region=proj_trajs_by_region,
                                                show_points=False,
                                                show_unstable_modes=False)
        
        # --------------------
        # PCA PLOTS (SUBSET)
        # Plot subset of trajectories with better distribution across regions
        total_freqs = len(frequency_labels)
        if total_freqs >= 6:
            # Select indices to get good representation: start, end, and some in between
            subset_indices = [0, total_freqs//4, total_freqs//2, 3*total_freqs//4, total_freqs-1]
        elif total_freqs >= 3:
            subset_indices = [0, total_freqs//2, total_freqs-1]
        else:
            subset_indices = list(range(total_freqs))
        
        subset_proj_trajs = [proj_trajs[i] for i in subset_indices if i < len(proj_trajs)]
        subset_fixed_points = [all_fixed_points[i] for i in subset_indices if i < len(all_fixed_points)]
        
        # Create subset trajectories by region using comprehensive frequency set
        subset_proj_trajs_by_region = {
            "lower_extrapolation": [],
            "lower_training": [], 
            "gap": [],
            "upper_training": [],
            "higher_extrapolation": []
        }
        
        for idx in subset_indices:
            if idx < len(frequency_labels):
                freq = frequency_labels[idx]
                if freq < TRAINING_FREQ_RANGES[0][0]:  # Lower extrapolation
                    subset_proj_trajs_by_region["lower_extrapolation"].append(proj_trajs[idx])
                elif TRAINING_FREQ_RANGES[0][0] <= freq <= TRAINING_FREQ_RANGES[0][1]:  # Lower training
                    subset_proj_trajs_by_region["lower_training"].append(proj_trajs[idx])
                elif TRAINING_FREQ_RANGES[0][1] < freq < TRAINING_FREQ_RANGES[1][0]:  # Gap region
                    subset_proj_trajs_by_region["gap"].append(proj_trajs[idx])
                elif TRAINING_FREQ_RANGES[1][0] <= freq <= TRAINING_FREQ_RANGES[1][1]:  # Upper training
                    subset_proj_trajs_by_region["upper_training"].append(proj_trajs[idx])
                else:  # Higher extrapolation
                    subset_proj_trajs_by_region["higher_extrapolation"].append(proj_trajs[idx])
        
        if subset_proj_trajs and subset_fixed_points:
            subset_proj_fixed = project_points_to_pca(pca, subset_fixed_points)
            
            # Set custom output directory to match this script's output folder
            set_custom_output_dir(custom_output_dir)
            # Plot PCA with subset with fixed points and trajectories (static matplotlib)
            plot_pca_trajectories_and_points(pca, subset_proj_trajs,
                                                point_type="fixed",
                                                all_points=subset_fixed_points,
                                                proj_points=subset_proj_fixed,
                                                params=imported_params,
                                                skip_initial_steps=PCA_SKIP_INITIAL_STEPS,
                                                apply_tanh=PCA_APPLY_TANH,
                                                filename_prefix="subset_of_indices_",
                                                proj_trajs_by_region=subset_proj_trajs_by_region,
                                                show_points=False,
                                                show_unstable_modes=False)
        
        # --------------------
        # SAVE PCA RESULTS DICTIONARY
        # save_file_to_pkl(pca_results, "PCA_Results")

        # ----------------------------------------
        # TESTING PHASE (IN TESTING MODE)
        test_freqs = analysis_freqs
        
        # Run the test
        try:
            print("\nRunning frequency generalization test on imported model...")
            results = test_frequency_generalization_with_params(params=imported_params,
                                                                test_frequencies=test_freqs)
            print(f"Frequency generalization test on imported model completed successfully!")
            
        except Exception as e:
            print(f"Error during frequency generalization test: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()