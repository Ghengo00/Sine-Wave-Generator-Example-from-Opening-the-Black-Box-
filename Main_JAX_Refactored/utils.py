"""
Utility functions for file I/O, saving, and timing operations.
"""


import os
import time
import pickle
from datetime import datetime


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# Import configuration parameters
try:
    from config import (
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
    output_dir = os.path.join(os.path.dirname(os.getcwd()), 'Outputs', f'JAX_Refactored_Outputs_{RUN_TIMESTAMP}')

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


def save_figure(
    fig, figure_name, N=N, num_tasks=num_tasks,
    dt=dt, T_drive=T_drive, T_train=T_train,
    s=s,
    num_epochs_adam=NUM_EPOCHS_ADAM, num_epochs_lbfgs=NUM_EPOCHS_LBFGS
    ):
    """
    Save a matplotlib figure with timestamp and parameters.
    """
    # Define the output directory
    output_dir = os.path.join(os.path.dirname(os.getcwd()), 'Outputs', f'JAX_Refactored_Outputs_{RUN_TIMESTAMP}')

    # Create the output directory if it doesn't exist (exist_ok avoids error if it does)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(figure_name, N, num_tasks, dt, T_drive, T_train) + '.png'
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        fig.savefig(filepath)
        print(f"Saved figure {figure_name} to {filepath}")
    except Exception as e:
        print(f"Error saving figure {figure_name} to {filepath}: {e}")


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