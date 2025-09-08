"""
Utility functions for file I/O, saving, and timing operations.
"""


import os
import time
import pickle
import glob
from datetime import datetime
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# Import configuration parameters
try:
    from _1_config import (
        N, num_tasks, dt, T_drive, T_train, s, 
        NUM_EPOCHS_ADAM, NUM_EPOCHS_LBFGS
    )
except ImportError:
    # Default values if config is not available
    N = 200
    num_tasks = 51
    dt = 0.02
    T_drive = 8.0
    T_train = 64.0
    s = 0.0
    NUM_EPOCHS_ADAM = 1000
    NUM_EPOCHS_LBFGS = 2000




# =============================================================================
# CUSTOM OUTPUT DIRECTORY
# =============================================================================
# Global variable to override output directory
_CUSTOM_OUTPUT_DIR = None


def set_custom_output_dir(output_dir):
    """
    Set a custom output directory for saving files and figures.
    
    Arguments:
        output_dir: custom output directory path (None to use default)
    """
    global _CUSTOM_OUTPUT_DIR
    if output_dir:
        # If it's a relative path or just a folder name, put it in the main Outputs directory
        if not os.path.isabs(output_dir):
            # Get the script directory and go up one level to find Outputs
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_outputs_dir = os.path.join(os.path.dirname(script_dir), 'Outputs')
            # Add timestamp to the folder name
            timestamped_name = f"{output_dir}_{RUN_TIMESTAMP}"
            output_dir = os.path.join(base_outputs_dir, timestamped_name)
        
        _CUSTOM_OUTPUT_DIR = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Custom output directory set to: {output_dir}")
    else:
        _CUSTOM_OUTPUT_DIR = None
        print("Using default output directory")


def get_output_dir():
    """Get the current output directory (custom or default)."""
    if _CUSTOM_OUTPUT_DIR:
        return _CUSTOM_OUTPUT_DIR
    else:
        # Default to main Outputs directory, not subdirectory
        # Get the script directory and go up one level to find Outputs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_outputs_dir = os.path.join(os.path.dirname(script_dir), 'Outputs')
        return os.path.join(base_outputs_dir, f'JAX_Refactored_Outputs_{RUN_TIMESTAMP}')


# =============================================================================
# For TEST_Sparsity_Experiments.py
def get_sparsity_output_dir(sparsity_value=None):
    """
    Get the output directory for a specific sparsity level, creating a subdirectory if needed.
    
    Arguments:
        sparsity_value: sparsity level (if None, returns main output directory)
        
    Returns:
        output_dir: path to the sparsity-specific output directory
    """
    base_output_dir = get_output_dir()
    
    if sparsity_value is not None:
        # Create sparsity-specific subdirectory
        sparsity_str = f"{sparsity_value:.2f}".replace('.', 'p')
        sparsity_dir = os.path.join(base_output_dir, f"Sparsity_{sparsity_str}")
        os.makedirs(sparsity_dir, exist_ok=True)
        return sparsity_dir
    else:
        return base_output_dir


# =============================================================================
# For TEST_L_1_Experiments.py
def get_l1_output_dir(l1_reg_strength=None):
    """
    Get the output directory for a specific L1 regularization strength, creating a subdirectory if needed.
    
    Arguments:
        l1_reg_strength: L1 regularization strength (if None, returns main output directory)
        
    Returns:
        output_dir: path to the L1-specific output directory
    """
    base_output_dir = get_output_dir()
    
    if l1_reg_strength is not None:
        # Create L1-specific subdirectory
        if l1_reg_strength == 0.0:
            l1_str = "0p00"
        else:
            # Format to handle both small (e.g., 1e-5) and regular (e.g., 0.01) numbers
            if l1_reg_strength < 1e-3:
                l1_str = f"{l1_reg_strength:.2e}".replace('.', 'p').replace('e-', 'em').replace('e+', 'ep')
            else:
                l1_str = f"{l1_reg_strength:.4f}".replace('.', 'p').rstrip('0').rstrip('p')
                if l1_str == '':
                    l1_str = "0p00"
        l1_dir = os.path.join(base_output_dir, f"L1_Reg_{l1_str}")
        os.makedirs(l1_dir, exist_ok=True)
        return l1_dir
    else:
        return base_output_dir




# =============================================================================
# SAVING UTLITIES
# =============================================================================
# Create timestamp for output filenames - this ensures all files from a run will have the same timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_filename(
    variable_name, N=N, num_tasks=num_tasks,
    dt=dt, T_drive=T_drive, T_train=T_train,
    s=s,
    num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS
    ):
    """
    Generate a filename with timestamp and parameters.
    """
    return (f"{variable_name}_{RUN_TIMESTAMP}_Neuron_Number_{N}_Task_Number_{num_tasks}_"
            f"Time_Steps_{dt}_Driving_Time_{T_drive}_Training_Time_{T_train}_"
            f"Sparsity_{s}_Adam_Epochs_{num_epochs_adam}_LBFGS_Epochs_{num_epochs_lbfgs}")


def save_variable(
    variable, variable_name, N=N, num_tasks=num_tasks,
    dt=dt, T_drive=T_drive, T_train=T_train,
    s=s,
    num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS
    ):
    """
    Save a variable to a pickle file with a descriptive filename in the Outputs folder.
    """
    # Define the output directory
    output_dir = get_output_dir()

    # Create the output directory if it doesn't exist (exist_ok avoids error if it does)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train) + '.pkl'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")


def save_figure(fig, figure_name, N=N, num_tasks=num_tasks,
                dt=dt, T_drive=T_drive, T_train=T_train, s=s,
                num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS):
    """
    Save a matplotlib figure with timestamp and parameters.
    
    Arguments:
        fig: matplotlib figure object or None (will use current figure)
        figure_name: base name for the figure file
        Other parameters: configuration parameters for filename generation
    """
    # Define the output directory
    output_dir = get_output_dir()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(figure_name, N, num_tasks, dt, T_drive, T_train, s, num_epochs_adam, num_epochs_lbfgs) + '.pdf'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        if fig is None:
            # Save current figure if no figure object provided
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            # Save the specified figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filepath}")
    except Exception as e:
        print(f"Error saving figure to {filepath}: {e}")


# =============================================================================
# For TEST_Sparsity_Experiments.py
def save_variable_with_sparsity(
    variable, variable_name, sparsity_value=None, N=N, num_tasks=num_tasks,
    dt=dt, T_drive=T_drive, T_train=T_train, s=s,
    num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS
    ):
    """
    Save a variable to a pickle file in a sparsity-specific subdirectory.
    
    Arguments:
        variable: variable to save
        variable_name: base name for the variable
        sparsity_value: sparsity level (if None, saves to main directory)
        Other parameters: configuration parameters for filename generation
    """
    # Define the output directory (sparsity-specific if provided)
    output_dir = get_sparsity_output_dir(sparsity_value)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train, s, num_epochs_adam, num_epochs_lbfgs) + '.pkl'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")


def save_figure_with_sparsity(fig, figure_name, sparsity_value=None, N=N, num_tasks=num_tasks,
                               dt=dt, T_drive=T_drive, T_train=T_train, s=s,
                               num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS):
    """
    Save a matplotlib figure in a sparsity-specific subdirectory.
    
    Arguments:
        fig: matplotlib figure object or None (will use current figure)
        figure_name: base name for the figure file
        sparsity_value: sparsity level (if None, saves to main directory)
        Other parameters: configuration parameters for filename generation
    """
    # Define the output directory (sparsity-specific if provided)
    output_dir = get_sparsity_output_dir(sparsity_value)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(figure_name, N, num_tasks, dt, T_drive, T_train, s, num_epochs_adam, num_epochs_lbfgs) + '.pdf'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        if fig is None:
            # Save current figure if no figure object provided
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            # Save the specified figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filepath}")
    except Exception as e:
        print(f"Error saving figure to {filepath}: {e}")


# =============================================================================
# For TEST_L_1_Experiments.py
def save_variable_with_l1_reg(
    variable, variable_name, l1_reg_strength=None, N=N, num_tasks=num_tasks,
    dt=dt, T_drive=T_drive, T_train=T_train, s=s,
    num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS
    ):
    """
    Save a variable to a pickle file in a L1 regularization-specific subdirectory.
    
    Arguments:
        variable: variable to save
        variable_name: base name for the variable
        l1_reg_strength: L1 regularization strength (if None, saves to main directory)
        Other parameters: configuration parameters for filename generation
    """
    # Define the output directory (L1-specific if provided)
    output_dir = get_l1_output_dir(l1_reg_strength)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train, s, num_epochs_adam, num_epochs_lbfgs) + '.pkl'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")


def save_figure_with_l1_reg(fig, figure_name, l1_reg_strength=None, N=N, num_tasks=num_tasks,
                            dt=dt, T_drive=T_drive, T_train=T_train, s=s,
                            num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS):
    """
    Save a matplotlib figure in a L1 regularization-specific subdirectory.
    
    Arguments:
        fig: matplotlib figure object or None (will use current figure)
        figure_name: base name for the figure file
        l1_reg_strength: L1 regularization strength (if None, saves to main directory)
        Other parameters: configuration parameters for filename generation
    """
    # Define the output directory (L1-specific if provided)
    output_dir = get_l1_output_dir(l1_reg_strength)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(figure_name, N, num_tasks, dt, T_drive, T_train, s, num_epochs_adam, num_epochs_lbfgs) + '.pdf'
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure
    save_figure(fig, filepath)




# =============================================================================
# LOADING UTILITIES  
# =============================================================================
def load_variable(filepath):
    """
    Load a variable from a pickle file.
    
    Arguments:
        filepath: absolute path to the pickle file
        
    Returns:
        variable: the loaded variable
    """
    try:
        with open(filepath, 'rb') as f:
            variable = pickle.load(f)
        print(f"Successfully loaded data from {os.path.basename(filepath)}")
        return variable
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise


def find_files_by_pattern(directory, pattern):
    """
    Find files matching a pattern in a directory.
    
    Arguments:
        directory: directory to search in
        pattern: pattern to match (e.g., 'state_*.pkl')
        
    Returns:
        matching_files: list of full paths to matching files
    """
    search_pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(search_pattern)
    return sorted(matching_files)  # Sort for consistent ordering


def find_output_directories(base_path=None):
    """
    Find all output directories that contain JAX results.
    
    Arguments:
        base_path: base directory to search in (default: ../Outputs)
        
    Returns:
        output_dirs: list of output directory paths, sorted by creation time (newest first)
    """
    if base_path is None:
        # Get the script directory and go up one level to find Outputs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(os.path.dirname(script_dir), 'Outputs')
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Outputs directory not found at {base_path}")
    
    # Find all JAX output directories
    output_dirs = []
    for item in os.listdir(base_path):
        if item.startswith('JAX_') and 'Outputs_' in item:
            full_path = os.path.join(base_path, item)
            if os.path.isdir(full_path):
                output_dirs.append(full_path)
    
    if not output_dirs:
        raise FileNotFoundError("No JAX output directories found")
    
    # Sort by creation time (newest first)
    output_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return output_dirs


def list_available_runs(base_path=None, show_details=True):
    """
    List all available training runs with their timestamps and basic info.
    
    Arguments:
        base_path: base directory to search in (default: ../Outputs)
        show_details: whether to show detailed information about each run
        
    Returns:
        runs_info: list of dictionaries containing run information
    """
    try:
        output_dirs = find_output_directories(base_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    
    runs_info = []
    
    print(f"Found {len(output_dirs)} training run(s):")
    print("-" * 60)
    
    for i, output_dir in enumerate(output_dirs):
        dir_name = os.path.basename(output_dir)
        creation_time = os.path.getctime(output_dir)
        creation_time_str = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        
        run_info = {
            'index': i,
            'path': output_dir,
            'name': dir_name,
            'creation_time': creation_time_str
        }
        
        print(f"[{i}] {dir_name}")
        print(f"    Created: {creation_time_str}")
        print(f"    Path: {output_dir}")
        
        if show_details:
            # Check what files are available
            available_files = []
            file_patterns = {
                'Parameters': ['J_param_*.pkl', 'B_param_*.pkl', 'w_param_*.pkl'],
                'States': ['state_*.pkl'],
                'Fixed Points': ['all_fixed_points_*.pkl'],
                'Slow Points': ['all_slow_points_*.pkl'],
                'PCA Results': ['pca_results_*.pkl']
            }
            
            for category, patterns in file_patterns.items():
                found_any = False
                for pattern in patterns:
                    if find_files_by_pattern(output_dir, pattern):
                        found_any = True
                        break
                if found_any:
                    available_files.append(category)
            
            run_info['available_data'] = available_files
            print(f"    Available: {', '.join(available_files) if available_files else 'No data files found'}")
        
        print()
        runs_info.append(run_info)
    
    return runs_info




# =============================================================================
# TIMING UTILITY
# =============================================================================
class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.description} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.print_elapsed_time()
    
    def print_elapsed_time(self):
        """Print elapsed time in a readable format."""
        hours = int(self.total_time // 3600)
        minutes = int((self.total_time % 3600) // 60)
        seconds = self.total_time % 60
        
        print("\n" + "=" * 60)
        print(f"{self.description} completed!")
        print(f"Total execution time: {self.total_time:.2f} seconds")
        
        if hours > 0:
            print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f} (HH:MM:SS.mmm)")
        elif minutes > 0:
            print(f"Total execution time: {minutes:02d}:{seconds:06.3f} (MM:SS.mmm)")
        else:
            print(f"Total execution time: {seconds:.3f} seconds")
        print("=" * 60)