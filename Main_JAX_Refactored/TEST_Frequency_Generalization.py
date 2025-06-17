#!/usr/bin/env python3
"""
TEST_Frequency_Generalization.py

This script tests a trained RNN's ability to generalize to new frequencies
outside the training range. It allows users to:

1. Select trained RNN parameters from a chosen run in the Outputs folder
2. Specify multiple test frequencies (outside the training range)
3. Test the RNN's ability to reproduce the frequency signal
4. Save all outputs in the Outputs folder

Usage:
    # Interactive mode (will prompt for inputs)
    python TEST_Frequency_Generalization.py
    
    # Command line mode
    python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5
    
    # With custom output directory
    python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5 --output_dir custom_test_name
"""

import os
import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save (not display) figures
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _1_config import N, dt, time_drive, time_train, num_steps_drive, num_steps_train, T_drive, T_train, omegas, num_tasks
from _4_rnn_model import simulate_trajectory
from _3_data_generation import get_drive_input, get_train_input
from _7_visualization import plot_trajectories_vs_targets


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def create_trajectory_plot(rnn_output, targets, frequency, save_path):
    """Create a simple trajectory vs target plot."""
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory


def create_error_vs_frequency_plot(test_frequencies, mse_values, save_path):
    """Create error vs frequency plot with shaded training range."""
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory


class SimpleConfig:
    """Simple config class to hold parameters."""
    def __init__(self):
        self.N = N
        self.dt = dt
        self.time_drive = time_drive
        self.time_train = time_train 
        self.num_steps_drive = num_steps_drive
        self.num_steps_train = num_steps_train
        self.trial_length = num_steps_drive + num_steps_train
        self.learning_rate = 0.001  # Default value


def get_available_runs(outputs_dir):
    """Get list of available runs from the Outputs directory."""
    if not os.path.exists(outputs_dir):
        return []
    
    runs = []
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains RNN parameter files (either .npz or .pkl)
            param_files = []
            param_files.extend([f for f in os.listdir(item_path) if f.startswith('rnn_params_') and f.endswith('.npz')])
            param_files.extend([f for f in os.listdir(item_path) if 'param' in f and f.endswith('.pkl')])
            if param_files:
                runs.append(item)
    
    return sorted(runs)


def select_run_interactive(outputs_dir):
    """Interactively select a run from available runs."""
    runs = get_available_runs(outputs_dir)
    
    if not runs:
        print(f"No runs found in {outputs_dir}")
        sys.exit(1)
    
    print("Available runs:")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run}")
    
    while True:
        try:
            choice = int(input(f"Select a run (1-{len(runs)}): ")) - 1
            if 0 <= choice < len(runs):
                return runs[choice]
            else:
                print(f"Please enter a number between 1 and {len(runs)}")
        except ValueError:
            print("Please enter a valid number")


def get_frequencies_interactive():
    """Interactively get test frequencies from user."""
    print("\nEnter test frequencies (space-separated):")
    print("Example: 0.5 1.5 2.5")
    
    while True:
        try:
            freq_input = input("Test frequencies: ").strip()
            frequencies = [float(f) for f in freq_input.split()]
            if frequencies:
                return frequencies
            else:
                print("Please enter at least one frequency")
        except ValueError:
            print("Please enter valid numbers separated by spaces")


def load_rnn_parameters(run_dir):
    """Load RNN parameters from the run directory."""
    # Look for parameter files (both .npz and .pkl formats)
    npz_files = [f for f in os.listdir(run_dir) if f.startswith('rnn_params_') and f.endswith('.npz')]
    pkl_files = [f for f in os.listdir(run_dir) if 'param' in f and f.endswith('.pkl')]
    
    if npz_files:
        # Use .npz format (newer style)
        param_file = sorted(npz_files)[-1]
        param_path = os.path.join(run_dir, param_file)
        print(f"Loading parameters from: {param_file}")
        
        # Load the parameters
        params_data = np.load(param_path, allow_pickle=True)
        params = params_data['params'].item()
        
        return params, param_file
        
    elif pkl_files:
        # Use .pkl format (older style, load individual parameter files)
        print("Loading parameters from individual .pkl files")
        
        # Find all parameter files
        J_files = [f for f in pkl_files if f.startswith('J_param_')]
        B_files = [f for f in pkl_files if f.startswith('B_param_')]
        w_files = [f for f in pkl_files if f.startswith('w_param_')]
        b_x_files = [f for f in pkl_files if f.startswith('b_x_param_')]
        b_z_files = [f for f in pkl_files if f.startswith('b_z_param_')]
        
        if not all([J_files, B_files, w_files, b_x_files, b_z_files]):
            raise FileNotFoundError(f"Missing parameter files in {run_dir}")
        
        # Load individual parameters
        import pickle
        
        with open(os.path.join(run_dir, J_files[0]), 'rb') as f:
            J = pickle.load(f)
        with open(os.path.join(run_dir, B_files[0]), 'rb') as f:
            B = pickle.load(f)
        with open(os.path.join(run_dir, w_files[0]), 'rb') as f:
            w = pickle.load(f)
        with open(os.path.join(run_dir, b_x_files[0]), 'rb') as f:
            b_x = pickle.load(f)
        with open(os.path.join(run_dir, b_z_files[0]), 'rb') as f:
            b_z = pickle.load(f)
        
        # Convert to JAX arrays
        params = {
            'J': jnp.array(J),
            'B': jnp.array(B),
            'w': jnp.array(w),
            'b_x': jnp.array(b_x),
            'b_z': jnp.array(b_z)
        }
        
        return params, J_files[0]  # Return first file as reference
    
    else:
        raise FileNotFoundError(f"No RNN parameter files found in {run_dir}")


def extract_config_from_filename(param_filename):
    """Extract configuration parameters from the parameter filename."""
    try:
        if param_filename.startswith('rnn_params_'):
            # Expected format: rnn_params_N{N}_lr{lr}_freq{freq}.npz
            # Example: rnn_params_N100_lr0.001_freq1.0.npz
            config_part = param_filename.replace('rnn_params_', '').replace('.npz', '')
            
            # Parse components
            components = config_part.split('_')
            config_dict = {}
            
            for comp in components:
                if comp.startswith('N'):
                    config_dict['N'] = int(comp[1:])
                elif comp.startswith('lr'):
                    config_dict['lr'] = float(comp[2:])
                elif comp.startswith('freq'):
                    config_dict['freq'] = float(comp[4:])
            
            return config_dict
            
        elif 'param_' in param_filename and '_Neuron_Number_' in param_filename:
            # Expected format for .pkl files: J_param_YYYYMMDD_HHMMSS_Neuron_Number_200_...
            # Extract neuron number
            parts = param_filename.split('_')
            config_dict = {}
            
            for i, part in enumerate(parts):
                if part == 'Neuron' and i + 2 < len(parts) and parts[i + 1] == 'Number':
                    config_dict['N'] = int(parts[i + 2])
                # Add other parameters as they appear in the filename
            
            # Set default values for missing parameters
            config_dict.setdefault('lr', 0.001)
            config_dict.setdefault('freq', 1.0)
            
            return config_dict
            
        else:
            raise ValueError("Unknown filename format")
            
    except Exception as e:
        print(f"Warning: Could not parse config from filename {param_filename}: {e}")
        # Return default values if parsing fails
        return {'N': 200, 'lr': 0.001, 'freq': 1.0}  # Default to 200 neurons


def calculate_static_input(frequency):
    """Calculate static input based on frequency following the training pattern."""
    # In training, frequencies are: omegas = jnp.linspace(0.1, 0.6, num_tasks)
    # And static inputs are: static_inputs = jnp.linspace(0, num_tasks - 1, num_tasks) / num_tasks + 0.25
    # This means: static_input = task_index / num_tasks + 0.25
    
    # Get the training frequency range
    freq_min = float(omegas.min())  # 0.1
    freq_max = float(omegas.max())  # 0.6
    
    # Map the input frequency to the same pattern used in training
    # For frequencies outside the training range, we extrapolate the linear relationship
    
    # Calculate the equivalent task index for this frequency
    # frequency = freq_min + task_index * (freq_max - freq_min) / (num_tasks - 1)
    # Solving for task_index: task_index = (frequency - freq_min) * (num_tasks - 1) / (freq_max - freq_min)
    
    task_index_continuous = (frequency - freq_min) * (num_tasks - 1) / (freq_max - freq_min)
    
    # Calculate static input using the same formula as training: task_index / num_tasks + 0.25
    static_input = task_index_continuous / num_tasks + 0.25
    
    return float(static_input)


def generate_sine_wave_data(frequency, static_input, time_drive, time_train):
    """Generate sine wave data for a given frequency and static input."""
    # Generate driving phase
    drive_input = (jnp.sin(frequency * time_drive) + static_input)[:, None]
    
    # Generate training phase input (static input only)
    train_input = jnp.full((len(time_train), 1), static_input, dtype=jnp.float32)
    
    # Generate target (training phase only)
    train_target = jnp.sin(frequency * time_train)[:, None]
    
    # Combine inputs
    inputs = jnp.concatenate([drive_input, train_input], axis=0)
    
    # Targets are zeros during drive phase, sine wave during train phase
    targets = jnp.concatenate([
        jnp.zeros((len(time_drive), 1), dtype=jnp.float32),
        train_target
    ], axis=0)
    
    return {
        'inputs': inputs,
        'targets': targets
    }


def test_frequency_generalization(run_name, test_frequencies, output_dir=None):
    """Test RNN generalization to new frequencies."""
    
    # Set up paths
    outputs_base = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs"
    run_dir = os.path.join(outputs_base, run_name)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load RNN parameters
    params, param_filename = load_rnn_parameters(run_dir)
    
    # Extract configuration from filename
    config_dict = extract_config_from_filename(param_filename)
    
    # Create configuration object
    config = SimpleConfig()
    config.N = config_dict.get('N', 100)
    config.learning_rate = config_dict.get('lr', 0.001)
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"frequency_generalization_test_{timestamp}"
    
    output_path = os.path.join(outputs_base, output_dir)
    ensure_directory(output_path)
    
    print(f"Testing RNN from run: {run_name}")
    print(f"RNN size: {config.N}")
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
            time_drive=config.time_drive,
            time_train=config.time_train
        )
        
        try:
            # Initialize with zeros
            x0 = jnp.zeros(config.N)
            
            # Run RNN simulation
            xs, zs = simulate_trajectory(x0, test_data['inputs'], params)
            
            # Extract outputs (only from training phase)
            rnn_output = zs[config.num_steps_drive:, None]  # Shape: (num_steps_train, 1)
            train_targets = test_data['targets'][config.num_steps_drive:]  # Shape: (num_steps_train, 1)
            
            # Create trajectory vs target plot for this frequency - save directly in main output directory
            freq_fig_path = os.path.join(output_path, f"trajectory_vs_target_freq_{freq:.2f}.png")
            
            # Create the plot
            create_trajectory_plot(
                rnn_output=rnn_output,
                targets=train_targets,
                frequency=freq,
                save_path=freq_fig_path
            )
            
            print(f"  Saved trajectory plot: {freq_fig_path}")
            
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
        
        error_plot_path = os.path.join(output_path, "error_vs_frequency.png")
        create_error_vs_frequency_plot(test_freqs, mse_values, error_plot_path)
        print(f"Saved error vs frequency plot: {error_plot_path}")
    
    # Save summary results with detailed information
    summary_file = os.path.join(output_path, "frequency_generalization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Frequency Generalization Test Results\n")
        f.write(f"=====================================\n\n")
        
        # Detailed configuration information
        f.write(f"Configuration:\n")
        f.write(f"  Source run: {run_name}\n")
        f.write(f"  RNN size (N): {config.N} neurons\n")
        f.write(f"  Time step (dt): {config.dt} seconds\n")
        f.write(f"  Driving time (T_drive): {T_drive} seconds\n")
        f.write(f"  Training time (T_train): {T_train} seconds\n")
        f.write(f"  Total time steps: {config.trial_length}\n")
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
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test RNN frequency generalization")
    parser.add_argument('--run_name', type=str, help='Name of the run to load parameters from')
    parser.add_argument('--test_frequencies', type=float, nargs='+', help='Test frequencies')
    parser.add_argument('--output_dir', type=str, help='Custom output directory name')
    
    args = parser.parse_args()
    
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
        run_name = select_run_interactive(outputs_dir)
    
    # Get test frequencies
    if args.test_frequencies:
        test_frequencies = args.test_frequencies
    else:
        test_frequencies = get_frequencies_interactive()
    
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


if __name__ == "__main__":
    main()