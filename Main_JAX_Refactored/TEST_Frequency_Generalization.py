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

Usage:
    # Gap training mode (set GAP_TRAINING_MODE = True in config section)
    python TEST_Frequency_Generalization.py --mode train
    
    # Testing mode with existing model
    python TEST_Frequency_Generalization.py --mode test --run_name your_run_name --test_frequencies 0.3 0.35 0.4
    
    # Interactive mode (will prompt for inputs)
    python TEST_Frequency_Generalization.py
"""
# ========================================
# IMPORTS
# ========================================
import os
import sys
import argparse
import pickle
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save (not display) figures
import matplotlib.pyplot as plt

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _1_config import (N, dt, time_drive, time_train, num_steps_drive, num_steps_train, 
                      T_drive, T_train, omegas, num_tasks, static_inputs,
                      TRAINING_FREQ_RANGES, TESTING_FREQ_RANGE, VARIABLE_TEST_TIME,
                      training_mask, training_omegas, training_static_inputs, num_training_tasks)
from _4_rnn_model import simulate_trajectory, init_params, create_loss_function
from _5_training import train_model
from _3_data_generation import get_drive_input, get_train_input
from _7_visualization import plot_trajectories_vs_targets
from _2_utils import Timer, get_output_dir, set_custom_output_dir




# ========================================
# CONFIGURATION SETTINGS
# ========================================
# Operation modes
GAP_TRAINING_MODE = True  # Set to True to train with frequency gaps, False to test existing model
DEFAULT_TEST_FREQUENCIES = [0.5, 1.5, 2.5]  # Default frequencies for testing mode

# Gap training parameters
GAP_TRAINING_OUTPUT_PREFIX = "gap_training"

# Default values for testing mode (when not training)
DEFAULT_RUN_PATH = "dummy_run_20240101_000000"
DEFAULT_J_PARAM_PATH = "/path/to/dummy/params.npz"
DEFAULT_B_PARAM_PATH = "/path/to/dummy/b_params.npz"
DEFAULT_W_PARAM_PATH = "/path/to/dummy/w_params.npz"
DEFAULT_B_X_PARAM_PATH = "/path/to/dummy/b_x_params.npz"
DEFAULT_B_Z_PARAM_PATH = "/path/to/dummy/b_z_params.npz"


def read_trajectory_data(file_path):
    """Read trajectory data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


# RNN parameters from parameter paths (used in testing mode with dummy data)
DUMMY_RNN_PARAMS = {
    'J': read_trajectory_data(DEFAULT_J_PARAM_PATH),
    'B': read_trajectory_data(DEFAULT_B_PARAM_PATH),
    'w': read_trajectory_data(DEFAULT_W_PARAM_PATH),
    'b_x': read_trajectory_data(DEFAULT_B_X_PARAM_PATH),
    'b_z': read_trajectory_data(DEFAULT_B_Z_PARAM_PATH)
}


# Current timestamp for output folder naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




# ========================================
# READING AND WRITING FUNCTIONS
# ========================================
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
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{timestamp}')
    
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
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_{timestamp}')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename using the base name and including all the sparsity levels
    filename = f"{base_name}_{timestamp}.png"

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the plot
    plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {file_path}")




# ========================================
# INPUT AND TARGET GENERATION FUNCTIONS
# ========================================
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


def generate_sine_wave_data(frequency, static_input, time_drive, time_train):
    """Generate sine wave data for a given frequency and static input."""
    # Generate driving phase input (sine wave with static offset)
    drive_input = (jnp.sin(frequency * time_drive) + static_input)[:, None]         # Shape: (num_steps_drive, 1)
    # Generate training phase input (static input only)
    train_input = jnp.full((len(time_train), 1), static_input, dtype=jnp.float32)   # Shape: (num_steps_train, 1)
    
    # Combine inputs
    inputs = jnp.concatenate([drive_input, train_input], axis=0)
    
    # Generate target for driving phase (zeros)
    drive_target = jnp.zeros((len(time_drive), 1), dtype=jnp.float32)               # Shape: (num_steps_drive, 1)
    # Generate target (for training phase only)
    train_target = jnp.sin(frequency * time_train)[:, None]                         # Shape: (num_steps_train, 1)
    
    # Combine targets
    targets = jnp.concatenate([drive_target, train_target], axis=0)
    
    return {
        'inputs': inputs,
        'targets': targets
    }


def calculate_variable_test_time(frequency):
    """Calculate variable test time based on frequency to ensure adequate coverage."""
    if not VARIABLE_TEST_TIME:
        return T_train
    
    # Ensure at least 2-3 full cycles for lower frequencies
    min_cycles = 3
    cycle_time = 2 * jnp.pi / frequency
    min_time = min_cycles * cycle_time
    
    # Use the maximum of default time and minimum required time
    return float(max(T_train, min_time))


# ========================================
# GAP TRAINING FUNCTIONS
# ========================================
def generate_gap_training_data():
    """Generate training data for gap training (excluding frequencies in the gap)."""
    # Use the filtered training frequencies and static inputs
    drive_inputs = (
        jnp.sin(training_omegas[:, None] * time_drive[None, :]) + 
        training_static_inputs[:, None]
    )[..., None]
    
    train_inputs = jnp.broadcast_to(
        training_static_inputs[:, None, None],
        (len(training_omegas), num_steps_train, 1)
    ).astype(jnp.float32)
    
    train_targets = jnp.sin(training_omegas[:, None] * time_train[None, :])
    
    return drive_inputs, train_inputs, train_targets


def create_gap_training_loss_function():
    """Create a loss function that only uses the gap training frequencies."""
    # Create a custom loss function using the filtered training data
    drive_inputs, train_inputs, train_targets = generate_gap_training_data()
    
    def gap_training_loss(params):
        """Custom loss function for gap training."""
        from _4_rnn_model import simulate_trajectory
        
        total_loss = 0.0
        for i in range(len(training_omegas)):
            # Initialize state
            x0 = jnp.zeros((N,), dtype=jnp.float32)
            
            # Drive phase
            xs_drive, _ = simulate_trajectory(x0, drive_inputs[i], params)
            x_drive_final = xs_drive[-1]
            
            # Training phase
            _, zs_train = simulate_trajectory(x_drive_final, train_inputs[i], params)
            
            # Compute loss for this task
            task_loss = jnp.mean((zs_train - train_targets[i]) ** 2)
            total_loss += task_loss
        
        return total_loss / len(training_omegas)
    
    return gap_training_loss


def run_gap_training():
    """Run gap training with the specified frequency ranges."""
    print("=" * 60)
    print("STARTING GAP TRAINING")
    print("=" * 60)
    print(f"Training frequency ranges: {TRAINING_FREQ_RANGES}")
    print(f"Number of training tasks: {num_training_tasks} (out of {num_tasks} total)")
    print(f"Training frequencies: {[f'{freq:.3f}' for freq in training_omegas]}")
    print()
    
    # Set up output directory
    set_custom_output_dir(GAP_TRAINING_OUTPUT_PREFIX)
    
    # Initialize model parameters
    key = jax.random.PRNGKey(42)  # Use fixed seed for reproducibility
    _, subkey = jax.random.split(key)
    mask, params = init_params(subkey)
    
    # Create custom loss function for gap training
    gap_loss_fn = create_gap_training_loss_function()
    
    # Train the model
    print("Starting gap training...")
    with Timer("Gap training"):
        trained_params, final_loss = train_model(params, mask, loss_fn=gap_loss_fn)
    
    print(f"Gap training completed! Final loss: {final_loss:.6e}")
    
    # Save trained parameters
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save parameters in .npz format
    param_file = os.path.join(output_dir, f"gap_trained_params_{timestamp}.npz")
    np.savez(param_file, params=trained_params)
    print(f"Parameters saved to: {param_file}")
    
    # Save training configuration
    config_file = os.path.join(output_dir, f"gap_training_config_{timestamp}.txt")
    with open(config_file, 'w') as f:
        f.write("Gap Training Configuration\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Training frequency ranges: {TRAINING_FREQ_RANGES}\n")
        f.write(f"Number of training tasks: {num_training_tasks}\n")
        f.write(f"Total available tasks: {num_tasks}\n")
        f.write(f"Network size (N): {N}\n")
        f.write(f"Training frequencies: {[f'{freq:.3f}' for freq in training_omegas]}\n")
        f.write(f"Final training loss: {final_loss:.6e}\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Configuration saved to: {config_file}")
    
    return trained_params, output_dir


# ========================================
# PLOTTING FUNCTIONS
# ========================================
def create_trajectory_plot(rnn_output, targets, frequency):
    """
    Create a simple trajectory vs target plot (amplitude vs time).

    Arguments:
        rnn_output:     RNN output array (shape: (num_steps, 1))
        targets:        Target array (shape: (num_steps, 1))
        frequency:      Frequency of the sine wave (for title)
    """
    plt.figure(figsize=(12, 6))
    
    # Create time axis (just use indices)
    time_axis = np.arange(len(rnn_output)) * dt
    
    plt.plot(time_axis, rnn_output.flatten(), 'b-', label='RNN Output', linewidth=2)
    plt.plot(time_axis, targets.flatten(), 'r--', label='Target', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'RNN Output vs Target (Frequency: {frequency:.2f} rad/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    save_plot_to_png(plt, f"Trajectory_vs_Target_for_{frequency:.2f}_rad_s")
    plt.close()  # Close to free memory


def create_error_vs_frequency_plot(test_frequencies, mse_values):
    """
    Create MSE vs frequency plot with shaded training range.
    
    Arguments:
        test_frequencies:   Frequencies used for testing (list or array)
        mse_values:         Mean squared error values for each frequency
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the error vs frequency
    plt.plot(test_frequencies, mse_values, 'bo-', linewidth=2, markersize=8, label='Test Performance')
    
    # Shade the training frequency range
    training_freq_min = float(omegas.min())
    training_freq_max = float(omegas.max())
    plt.axvspan(training_freq_min, training_freq_max, alpha=0.2, color='green', 
                label=f'Training Range ({training_freq_min:.1f}-{training_freq_max:.1f} rad/s)')
    
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('RNN Performance vs Test Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some padding to the y-axis
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    
    # Save the plot
    save_plot_to_png(plt, "MSE_vs_Frequency")
    plt.close()  # Close to free memory




# ========================================
# TESTING FUNCTION
# ========================================
def test_frequency_generalization(run_name, test_frequencies, output_dir=None):
    """Test RNN generalization to new frequencies."""
    
    # Set up paths
    outputs_base = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs"
    run_dir = os.path.join(outputs_base, run_name)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Use hard-coded RNN parameters
    params = DUMMY_RNN_PARAMS
    param_filename = "dummy_params.npz"
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"frequency_generalization_test_{timestamp}"
    
    output_path = os.path.join(outputs_base, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Testing RNN from run: {run_name}")
    print(f"RNN size: {N}")
    print(f"Test frequencies: {test_frequencies}")
    print(f"Output directory: {output_path}")
    
    # Test each frequency
    results = {}
    
    for freq in test_frequencies:
        print(f"\nTesting frequency: {freq}")
        
        # Calculate static input for this frequency
        static_input = calculate_static_input(freq)
        
        # Generate test data
        test_data = generate_sine_wave_data(
            frequency=freq,
            static_input=static_input,
            time_drive=time_drive,
            time_train=time_train
        )
        
        try:
            # Initialize with zeros
            x0 = jnp.zeros(N)
            
            # Run RNN simulation
            xs, zs = simulate_trajectory(x0, test_data['inputs'], params)
            
            # Extract outputs (only from training phase)
            rnn_output = zs[num_steps_drive:, None]  # Shape: (num_steps_train, 1)
            train_targets = test_data['targets'][num_steps_drive:]  # Shape: (num_steps_train, 1)
            
            # Create the plot
            create_trajectory_plot(
                rnn_output=rnn_output,
                targets=train_targets,
                frequency=freq
            )
            
            # Calculate and print error metrics
            mse = np.mean((rnn_output - train_targets) ** 2)
            print(f"  MSE: {mse:.6f}")
            
            # Store results
            results[freq] = {
                'mse': float(mse),
                'static_input': static_input,
                'rnn_output': rnn_output,
                'targets': train_targets
            }
            
        except Exception as e:
            print(f"  Error testing frequency {freq}: {e}")
            results[freq] = {'error': str(e)}
    
    # Create error vs frequency plot
    successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
    if successful_results:
        test_freqs = sorted(successful_results.keys())
        mse_values = [successful_results[freq]['mse'] for freq in test_freqs]
        
        create_error_vs_frequency_plot(test_freqs, mse_values)
    
    # Save summary results with detailed information
    summary_file = os.path.join(output_path, "frequency_generalization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Frequency Generalization Test Results\n")
        f.write(f"=====================================\n\n")
        
        # Detailed configuration information
        f.write(f"Configuration:\n")
        f.write(f"  Source run: {run_name}\n")
        f.write(f"  RNN size (N): {N} neurons\n")
        f.write(f"  Time step (dt): {dt} seconds\n")
        f.write(f"  Driving time (T_drive): {T_drive} seconds\n")
        f.write(f"  Training time (T_train): {T_train} seconds\n")
        f.write(f"  Total time steps: {num_steps_drive + num_steps_train}\n")
        f.write(f"  Training frequency range: {float(omegas.min()):.1f} - {float(omegas.max()):.1f} rad/s\n")
        f.write(f"  Test frequencies: {test_frequencies}\n\n")
        
        # Results for each frequency
        f.write(f"Results:\n")
        for freq, result in results.items():
            f.write(f"  Frequency {freq:.2f} rad/s:\n")
            if 'error' in result:
                f.write(f"    Error: {result['error']}\n")
            else:
                f.write(f"    Static input: {result['static_input']}\n")
                f.write(f"    MSE: {result['mse']:.6f}\n")
                
                # Indicate if this frequency is within training range
                is_in_training_range = float(omegas.min()) <= freq <= float(omegas.max())
                f.write(f"    Within training range: {'Yes' if is_in_training_range else 'No'}\n")
            f.write("\n")
        
        # Summary statistics
        if successful_results:
            f.write(f"Summary Statistics:\n")
            mse_values = [result['mse'] for result in successful_results.values()]
            f.write(f"  Mean MSE: {np.mean(mse_values):.6f}\n")
            f.write(f"  Std MSE: {np.std(mse_values):.6f}\n")
            f.write(f"  Min MSE: {np.min(mse_values):.6f}\n")
            f.write(f"  Max MSE: {np.max(mse_values):.6f}\n")
    
    print(f"\nFrequency generalization test completed!")
    
    return results




# ========================================
# MAIN EXECUTION
# ========================================
def main():
    parser = argparse.ArgumentParser(description="Gap training and frequency generalization testing")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                       help='Mode: "train" for gap training, "test" for testing existing model')
    parser.add_argument('--run_name', type=str, help='Name of the run to load parameters from (test mode)')
    parser.add_argument('--test_frequencies', type=float, nargs='+', help='Test frequencies (test mode)')
    parser.add_argument('--output_dir', type=str, help='Custom output directory name')
    
    args = parser.parse_args()
    
    # Check if we're in gap training mode based on config or command line
    training_mode = GAP_TRAINING_MODE or (args.mode == 'train')
    
    if training_mode:
        print("=" * 60)
        print("RUNNING IN GAP TRAINING MODE")
        print("=" * 60)
        try:
            trained_params, output_dir = run_gap_training()
            print(f"\nGap training completed successfully!")
            print(f"Results saved to: {output_dir}")
            
            # Optionally run testing on the trained model
            user_input = input("\nWould you like to test the trained model? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print("\nRunning frequency generalization test on trained model...")
                
                # Generate comprehensive test frequencies
                test_freqs = generate_comprehensive_test_frequencies()
                print(f"Testing {len(test_freqs)} frequencies from {min(test_freqs):.2f} to {max(test_freqs):.2f} rad/s")
                
                # Load the trained parameters for testing
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
                test_run_name = f"gap_training_test_{timestamp}"
                
                # Test with comprehensive frequency range
                test_results = test_frequency_generalization_with_params(
                    params=trained_params,
                    test_frequencies=test_freqs,
                    output_dir=f"gap_training_test_{timestamp}"
                )
                print(f"Testing completed! Results saved.")
                
        except Exception as e:
            print(f"Error during gap training: {e}")
            sys.exit(1)
    
    else:
        print("=" * 60)
        print("RUNNING IN TESTING MODE")
        print("=" * 60)
        
        # Determine the outputs directory
        outputs_dir = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs"
        
        # Get run name
        if args.run_name:
            run_name = args.run_name
            # Verify the run exists
            if not os.path.exists(os.path.join(outputs_dir, run_name)):
                print(f"Error: Run '{run_name}' not found in {outputs_dir}")
                sys.exit(1)
        else:
            run_name = DEFAULT_RUN_PATH
            print(f"Using dummy run name: {run_name}")
        
        # Get test frequencies
        if args.test_frequencies:
            test_frequencies = args.test_frequencies
        else:
            test_frequencies = DEFAULT_TEST_FREQUENCIES
            print(f"Using dummy test frequencies: {test_frequencies}")
        
        # Run the test
        try:
            results = test_frequency_generalization(
                run_name=run_name,
                test_frequencies=test_frequencies,
                output_dir=args.output_dir
            )
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Error during frequency generalization test: {e}")
            sys.exit(1)


def generate_comprehensive_test_frequencies():
    """Generate a comprehensive set of test frequencies covering the full range."""
    # Dense sampling around the gap
    gap_freqs = np.linspace(0.25, 0.45, 20)
    
    # Sparse sampling in training ranges
    lower_train = np.linspace(0.1, 0.25, 5)
    upper_train = np.linspace(0.45, 0.6, 5)
    
    # Extrapolation ranges
    low_extrap = np.linspace(0.01, 0.1, 10)
    high_extrap = np.linspace(0.6, 1.2, 15)
    
    # Combine all frequencies
    all_freqs = np.concatenate([low_extrap, lower_train, gap_freqs, upper_train, high_extrap])
    
    # Remove duplicates and sort
    return sorted(list(set(all_freqs)))


def test_frequency_generalization_with_params(params, test_frequencies, output_dir=None):
    """Test frequency generalization using provided parameters."""
    outputs_base = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs"
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"frequency_generalization_test_{timestamp}"
    
    output_path = os.path.join(outputs_base, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Testing {len(test_frequencies)} frequencies...")
    print(f"Output directory: {output_path}")
    
    # Test each frequency
    results = {}
    
    for freq in test_frequencies:
        print(f"Testing frequency: {freq:.3f} rad/s", end=" ... ")
        
        # Calculate static input and variable test time
        static_input = calculate_static_input(freq)
        test_time = calculate_variable_test_time(freq)
        
        # Generate time arrays for this frequency
        num_steps_test = int(test_time / dt)
        time_test = jnp.arange(0, test_time, dt, dtype=jnp.float32)
        
        # Generate test data with variable time
        test_data = generate_sine_wave_data_variable_time(
            frequency=freq,
            static_input=static_input,
            time_drive=time_drive,
            time_test=time_test
        )
        
        try:
            # Initialize with zeros
            x0 = jnp.zeros(N)
            
            # Run RNN simulation
            xs, zs = simulate_trajectory(x0, test_data['inputs'], params)
            
            # Extract outputs (only from test phase)
            rnn_output = zs[num_steps_drive:, None]  
            test_targets = test_data['targets'][num_steps_drive:]  
            
            # Calculate error metrics
            mse = np.mean((rnn_output - test_targets) ** 2)
            print(f"MSE: {mse:.6f}")
            
            # Store results
            results[freq] = {
                'mse': float(mse),
                'static_input': static_input,
                'test_time': test_time,
                'rnn_output': rnn_output,
                'targets': test_targets
            }
            
        except Exception as e:
            print(f"Error: {e}")
            results[freq] = {'error': str(e)}
    
    # Create plots and save results
    successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
    if successful_results:
        test_freqs = sorted(successful_results.keys())
        mse_values = [successful_results[freq]['mse'] for freq in test_freqs]
        
        create_error_vs_frequency_plot_with_gaps(test_freqs, mse_values, output_path)
    
    # Save comprehensive summary
    save_comprehensive_summary(results, output_path)
    
    return results


def generate_sine_wave_data_variable_time(frequency, static_input, time_drive, time_test):
    """Generate sine wave data with variable test time."""
    # Generate driving phase input (sine wave with static offset)
    drive_input = (jnp.sin(frequency * time_drive) + static_input)[:, None]
    # Generate test phase input (static input only)
    test_input = jnp.full((len(time_test), 1), static_input, dtype=jnp.float32)
    
    # Combine inputs
    inputs = jnp.concatenate([drive_input, test_input], axis=0)
    
    # Generate target for driving phase (zeros)
    drive_target = jnp.zeros((len(time_drive), 1), dtype=jnp.float32)
    # Generate target for test phase
    test_target = jnp.sin(frequency * time_test)[:, None]
    
    # Combine targets
    targets = jnp.concatenate([drive_target, test_target], axis=0)
    
    return {
        'inputs': inputs,
        'targets': targets
    }


def create_error_vs_frequency_plot_with_gaps(test_frequencies, mse_values, output_path):
    """Create error vs frequency plot highlighting gap training ranges."""
    plt.figure(figsize=(12, 8))
    
    # Plot the error vs frequency
    plt.plot(test_frequencies, mse_values, 'bo-', linewidth=2, markersize=6, label='Test Performance')
    
    # Shade the training frequency ranges (non-gap regions)
    for freq_min, freq_max in TRAINING_FREQ_RANGES:
        plt.axvspan(freq_min, freq_max, alpha=0.2, color='green', 
                   label=f'Training Range' if freq_min == TRAINING_FREQ_RANGES[0][0] else "")
    
    # Shade the gap region
    gap_start = TRAINING_FREQ_RANGES[0][1]  # 0.25
    gap_end = TRAINING_FREQ_RANGES[1][0]    # 0.45
    plt.axvspan(gap_start, gap_end, alpha=0.3, color='red', 
               label=f'Gap ({gap_start:.2f}-{gap_end:.2f} rad/s)')
    
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('RNN Performance vs Test Frequency (Gap Training)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Save the plot
    plot_path = os.path.join(output_path, "error_vs_frequency_gap_training.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gap training analysis plot saved: {plot_path}")


def save_comprehensive_summary(results, output_path):
    """Save comprehensive summary of gap training test results."""
    summary_file = os.path.join(output_path, "gap_training_test_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Gap Training Frequency Generalization Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  Gap training ranges: {TRAINING_FREQ_RANGES}\n")
        f.write(f"  Excluded gap: {TRAINING_FREQ_RANGES[0][1]:.2f} - {TRAINING_FREQ_RANGES[1][0]:.2f} rad/s\n")
        f.write(f"  Variable test time: {VARIABLE_TEST_TIME}\n")
        f.write(f"  Test frequency range: {TESTING_FREQ_RANGE}\n\n")
        
        # Analyze results by region
        successful_results = {freq: result for freq, result in results.items() if 'error' not in result}
        
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
            
            # Report statistics by region
            f.write("Performance by Region:\n")
            
            if in_training:
                mse_training = [successful_results[freq]['mse'] for freq in in_training]
                f.write(f"  Training regions: {len(in_training)} frequencies\n")
                f.write(f"    Mean MSE: {np.mean(mse_training):.6f}\n")
                f.write(f"    Max MSE: {np.max(mse_training):.6f}\n\n")
            
            if in_gap:
                mse_gap = [successful_results[freq]['mse'] for freq in in_gap]
                f.write(f"  Gap region: {len(in_gap)} frequencies\n")
                f.write(f"    Mean MSE: {np.mean(mse_gap):.6f}\n")
                f.write(f"    Max MSE: {np.max(mse_gap):.6f}\n\n")
            
            if extrapolation:
                mse_extrap = [successful_results[freq]['mse'] for freq in extrapolation]
                f.write(f"  Extrapolation regions: {len(extrapolation)} frequencies\n")
                f.write(f"    Mean MSE: {np.mean(mse_extrap):.6f}\n")
                f.write(f"    Max MSE: {np.max(mse_extrap):.6f}\n\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        for freq in sorted(results.keys()):
            result = results[freq]
            if 'error' in result:
                f.write(f"  {freq:.3f} rad/s: ERROR - {result['error']}\n")
            else:
                region = "GAP" if TRAINING_FREQ_RANGES[0][1] <= freq <= TRAINING_FREQ_RANGES[1][0] else \
                        "TRAIN" if any(r[0] <= freq <= r[1] for r in TRAINING_FREQ_RANGES) else "EXTRAP"
                f.write(f"  {freq:.3f} rad/s: MSE={result['mse']:.6f}, Time={result['test_time']:.1f}s [{region}]\n")
    
    print(f"Comprehensive summary saved: {summary_file}")


if __name__ == "__main__":
    main()