#!/usr/bin/env python3
"""
TEST_Frequency_Generalization_with_Sparsity.py

This script runs the frequency generalization test (gap training) for multiple sparsity values.
It systematically varies the sparsity parameter and collects results for aggregate analysis.

Features:
- Runs gap training (train mode) for different sparsity values
- Automatically tests generalization after each training
- Organizes outputs in structured folder hierarchy
- Saves individual results for each sparsity value in separate folders
- Minimal code overhead - leverages existing TEST_Frequency_Generalization.py

Folder Structure:
- Main folder: Frequency_Generalization_with_Sparsity_{timestamp}
- Individual sparsity folders: Sparsity_{sparsity_value:.3f}
- Each sparsity folder contains all outputs from that specific experiment

Configuration:
Set SPARSITY_VALUES to specify which sparsity levels to test.
All other parameters are inherited from TEST_Frequency_Generalization.py.

Usage:
    python TEST_Frequency_Generalization_with_Sparsity.py

Outputs:
    - Individual results for each sparsity value in separate folders
    - Aggregate results dictionary combining all sparsity experiments
    - Summary text file with experiment overview
"""
# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import pickle
import subprocess
import tempfile
import shutil
from datetime import datetime
from tqdm import tqdm

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))




# ============================================================
# CONFIGURATION
# ============================================================
# Sparsity values to test
SPARSITY_VALUES = [0.0, 0.1, 0.2]

# Current timestamp for output folder naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def modify_main_script(sparsity_value, temp_script_path, custom_output_dir):
    """
    Create a temporary version of the main script with modified sparsity value, train mode and testing enabled.
    
    Args:
        sparsity_value (float): The sparsity value to set
        temp_script_path (str): Path where to save the temporary script
        custom_output_dir (str): Custom output directory for this specific experiment
    """
    # Read the original script
    original_script_path = "TEST_Frequency_Generalization.py"
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace execution mode, testing settings, sparsity value, and set custom output directory
    lines = script_content.split('\n')
    modified_lines = []
    
    # Track if we've added the custom output directory setting
    custom_dir_added = False
    
    for line in lines:
        if line.strip().startswith('EXECUTION_MODE'):
            modified_lines.append('EXECUTION_MODE               = "train"   # Modified by sparsity experiment script')
        elif line.strip().startswith('RUN_TESTING_AFTER_TRAINING'):
            modified_lines.append('RUN_TESTING_AFTER_TRAINING   = True      # Modified by sparsity experiment script')
        elif line.strip().startswith('S ') and '=' in line and 'Sparsity level' in line:
            modified_lines.append(f'S                       = {sparsity_value}                           # Sparsity level for the J matrix (0.0 to 1.0) - Modified by sparsity experiment script')
        # Replace all hardcoded output directory constructions with get_output_dir()
        elif 'output_dir = os.path.join(parent_dir, \'Outputs\', f\'Frequency_Generalization_' in line:
            indent = len(line) - len(line.lstrip())
            modified_lines.append(' ' * indent + 'output_dir = get_output_dir()  # Modified by sparsity experiment script')
        elif 'custom_output_dir = os.path.join(parent_dir, \'Outputs\', f\'Frequency_Generalization_' in line:
            indent = len(line) - len(line.lstrip())
            modified_lines.append(' ' * indent + 'custom_output_dir = get_output_dir()  # Modified by sparsity experiment script')
        else:
            modified_lines.append(line)
            
        # Add custom output directory setting after the utils import
        if 'from _2_utils import Timer, get_output_dir, set_custom_output_dir' in line and not custom_dir_added:
            modified_lines.append('')
            modified_lines.append('# Set custom output directory for this sparsity experiment')
            modified_lines.append(f'set_custom_output_dir(r"{custom_output_dir}")')
            modified_lines.append('')
            custom_dir_added = True
    
    # Write the modified script
    with open(temp_script_path, 'w') as f:
        f.write('\n'.join(modified_lines))


# ------------------------------------------------------------


def run_single_sparsity_experiment(sparsity_value):
    """
    Run the frequency generalization experiment for a single sparsity value.
    
    Args:
        sparsity_value (float): The sparsity value to test
        
    Returns:
        dict: Results from the experiment, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT FOR SPARSITY = {sparsity_value}")
    print(f"{'='*60}")
    
    # Create the main experiment output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    main_output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_with_Sparsity_{timestamp}')
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create specific folder for this sparsity value
    sparsity_folder = f"Sparsity_{sparsity_value:.3f}"
    sparsity_output_dir = os.path.join(main_output_dir, sparsity_folder)
    os.makedirs(sparsity_output_dir, exist_ok=True)
    
    # Create temporary script in the current working directory instead of temp directory
    # This ensures the script can import local modules
    current_dir = os.getcwd()
    temp_script_path = os.path.join(current_dir, f"TEST_Frequency_Generalization_temp_sparsity_{sparsity_value}.py")
    
    try:
        # Create modified script file
        modify_main_script(sparsity_value, temp_script_path, sparsity_output_dir)
        
        # Run the experiment in the current directory
        result = subprocess.run([
            sys.executable, temp_script_path
        ], capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode != 0:
            print(f"Error running experiment:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        return {
            "sparsity": sparsity_value, 
            "success": True, 
            "output": result.stdout,
            "output_dir": sparsity_output_dir,
            "folder_name": sparsity_folder
        }
        
    except Exception as e:
        print(f"Exception during experiment: {e}")
        return None
    
    finally:
        # Clean up temporary script file
        try:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary script file: {e}")


# ------------------------------------------------------------


def collect_experiment_results(sparsity_values, experiment_results):
    """
    Collect and organize results from all sparsity experiments.
    
    Args:
        sparsity_values (list): List of sparsity values tested
        experiment_results (list): List of results from each experiment
        
    Returns:
        dict: Organized results dictionary
    """
    aggregate_results = {
        'sparsity_values': sparsity_values,
        'timestamp': timestamp,
        'successful_experiments': [],
        'failed_experiments': [],
        'experiment_folders': {},
        'summary_data': {},
        'output_directories': {}
    }
    
    # Process results from each experiment
    for i, sparsity in enumerate(sparsity_values):
        if i < len(experiment_results) and experiment_results[i] is not None:
            result = experiment_results[i]
            if result.get('success', False):
                aggregate_results['successful_experiments'].append(sparsity)
                aggregate_results['experiment_folders'][sparsity] = result.get('folder_name', f'Sparsity_{sparsity:.3f}')
                aggregate_results['output_directories'][sparsity] = result.get('output_dir', '')
                
                # Try to load summary data from this experiment's output directory
                output_dir = result.get('output_dir', '')
                if output_dir and os.path.exists(output_dir):
                    # Look for pickle files in the output directory
                    for file in os.listdir(output_dir):
                        if file.endswith('.pkl') and 'Summary' in file:
                            summary_file = os.path.join(output_dir, file)
                            try:
                                with open(summary_file, 'rb') as f:
                                    summary_data = pickle.load(f)
                                aggregate_results['summary_data'][sparsity] = summary_data
                                break
                            except Exception as e:
                                print(f"Could not load summary data for sparsity {sparsity}: {e}")
            else:
                aggregate_results['failed_experiments'].append(sparsity)
        else:
            aggregate_results['failed_experiments'].append(sparsity)
    
    return aggregate_results


# ------------------------------------------------------------


def save_aggregate_results(aggregate_results):
    """Save the aggregate results to a pickle file."""
    # Create output directory (main experiment folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_with_Sparsity_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results in the main folder
    results_file = os.path.join(output_dir, f'Aggregate_Sparsity_Results_{timestamp}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregate_results, f)




# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main execution function."""
    experiment_results = []
    
    try:
        # Run experiments for each sparsity value
        for i, sparsity in enumerate(tqdm(SPARSITY_VALUES, desc="Running sparsity experiments")):
            print(f"\nProgress: {i+1}/{len(SPARSITY_VALUES)}")
            result = run_single_sparsity_experiment(sparsity)
            experiment_results.append(result)
            
            # Small delay between experiments
            import time
            time.sleep(1)
        
        # Collect and organize results
        aggregate_results = collect_experiment_results(SPARSITY_VALUES, experiment_results)
        
        # Save aggregate results
        save_aggregate_results(aggregate_results)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()