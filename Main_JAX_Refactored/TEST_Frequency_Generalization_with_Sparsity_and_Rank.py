#!/usr/bin/env python3
"""
TEST_Frequency_Generalization_with_Sparsity_and_Rank.py

This script runs the frequency generalization test (gap training) for multiple combinations of sparsity and rank values.
It systematically varies both the sparsity and rank parameters and collects results for aggregate analysis.

Outputs:
    - Individual results for each sparsity-rank combination in separate folders
    - Aggregate results dictionary combining all experiments
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

# Import N from config to validate rank values
from _1_config import N




# ============================================================
# CONFIGURATION
# ============================================================
# Sparsity values to test
SPARSITY_VALUES = [0.0, 0.1, 0.2]

# Whether to control rank (if False, rank parameter will be ignored)
CONTROL_RANK = True
# Rank values to test
RANK_VALUES = [10, 25, 50]
# Validate rank values don't exceed network dimension
if CONTROL_RANK:
    invalid_ranks = [r for r in RANK_VALUES if r > N]
    if invalid_ranks:
        print(f"ERROR: The following rank values exceed the network dimension N={N}: {invalid_ranks}")
        sys.exit(1)

# Current timestamp for output folder naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def modify_main_script(sparsity_value, rank_value, temp_script_path, custom_output_dir):
    """
    Create a temporary version of the main script with modified sparsity value, rank value, train mode and testing enabled.
    
    Args:
        sparsity_value (float): The sparsity value to set
        rank_value (int): The rank value to set
        temp_script_path (str): Path where to save the temporary script
        custom_output_dir (str): Custom output directory for this specific experiment
    """
    # Read the original script
    original_script_path = "TEST_Frequency_Generalization.py"
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace execution mode, testing settings, sparsity value, rank value, and set custom output directory
    lines = script_content.split('\n')
    modified_lines = []
    
    # Track if we've added the custom output directory setting
    custom_dir_added = False
    
    for line in lines:
        if line.strip().startswith('EXECUTION_MODE'):
            modified_lines.append('EXECUTION_MODE               = "train"   # Modified by sparsity-rank experiment script')
        elif line.strip().startswith('RUN_TESTING_AFTER_TRAINING'):
            modified_lines.append('RUN_TESTING_AFTER_TRAINING   = True      # Modified by sparsity-rank experiment script')
        elif line.strip().startswith('S ') and '=' in line and 'Sparsity level' in line:
            modified_lines.append(f'S                       = {sparsity_value}                           # Sparsity level for the J matrix (0.0 to 1.0) - Modified by sparsity-rank experiment script')
        elif line.strip().startswith('CONTROL_RANK'):
            modified_lines.append(f'CONTROL_RANK            = {CONTROL_RANK}                         # If True, will control the rank of J during initialization - Modified by sparsity-rank experiment script')
        elif line.strip().startswith('R ') and '=' in line and 'Rank of the J matrix' in line:
            modified_lines.append(f'R                       = {rank_value}                            # Rank of the J matrix (1 to N, where N is the number of neurons) - Modified by sparsity-rank experiment script')
        # Replace all hardcoded output directory constructions with get_output_dir()
        elif 'output_dir = os.path.join(parent_dir, \'Outputs\', f\'Frequency_Generalization_' in line:
            indent = len(line) - len(line.lstrip())
            modified_lines.append(' ' * indent + 'output_dir = get_output_dir()  # Modified by sparsity-rank experiment script')
        elif 'custom_output_dir = os.path.join(parent_dir, \'Outputs\', f\'Frequency_Generalization_' in line:
            indent = len(line) - len(line.lstrip())
            modified_lines.append(' ' * indent + 'custom_output_dir = get_output_dir()  # Modified by sparsity-rank experiment script')
        else:
            modified_lines.append(line)
            
        # Add custom output directory setting after the utils import
        if 'from _2_utils import Timer, get_output_dir, set_custom_output_dir' in line and not custom_dir_added:
            modified_lines.append('')
            modified_lines.append('# Set custom output directory for this sparsity-rank experiment')
            modified_lines.append(f'set_custom_output_dir(r"{custom_output_dir}")')
            modified_lines.append('')
            custom_dir_added = True
    
    # Write the modified script
    with open(temp_script_path, 'w') as f:
        f.write('\n'.join(modified_lines))


# ------------------------------------------------------------


def run_single_experiment(sparsity_value, rank_value):
    """
    Run the frequency generalization experiment for a single sparsity-rank combination.
    
    Args:
        sparsity_value (float): The sparsity value to test
        rank_value (int): The rank value to test
        
    Returns:
        dict: Results from the experiment, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT FOR SPARSITY = {sparsity_value}, RANK = {rank_value}")
    print(f"{'='*60}")
    
    # Create the main experiment output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    main_output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_with_Sparsity_and_Rank_{timestamp}')
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create specific folder for this sparsity-rank combination
    if CONTROL_RANK:
        experiment_folder = f"Sparsity_{sparsity_value:.3f}_Rank_{rank_value:04d}"
    else:
        experiment_folder = f"Sparsity_{sparsity_value:.3f}_RankNotControlled"
    experiment_output_dir = os.path.join(main_output_dir, experiment_folder)
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Create temporary script in the current working directory instead of temp directory
    # This ensures the script can import local modules
    current_dir = os.getcwd()
    temp_script_path = os.path.join(current_dir, f"TEST_Frequency_Generalization_temp_sparsity_{sparsity_value}_rank_{rank_value:04d}.py")
    
    try:
        # Create modified script file
        modify_main_script(sparsity_value, rank_value, temp_script_path, experiment_output_dir)
        
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
            "rank": rank_value, 
            "success": True, 
            "output": result.stdout,
            "output_dir": experiment_output_dir,
            "folder_name": experiment_folder
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


def collect_experiment_results(sparsity_values, rank_values, experiment_results):
    """
    Collect and organize results from all sparsity-rank combination experiments.
    
    Args:
        sparsity_values (list): List of sparsity values tested
        rank_values (list): List of rank values tested
        experiment_results (list): List of results from each experiment
        
    Returns:
        dict: Organized results dictionary
    """
    aggregate_results = {
        'sparsity_values': sparsity_values,
        'rank_values': rank_values,
        'control_rank': CONTROL_RANK,
        'timestamp': timestamp,
        'successful_experiments': [],
        'failed_experiments': [],
        'experiment_folders': {},
        'summary_data': {},
        'output_directories': {}
    };
    
    # Process results from each experiment
    experiment_idx = 0
    for sparsity in sparsity_values:
        for rank in rank_values:
            if experiment_idx < len(experiment_results) and experiment_results[experiment_idx] is not None:
                result = experiment_results[experiment_idx]
                if result.get('success', False):
                    experiment_key = (sparsity, rank)
                    aggregate_results['successful_experiments'].append(experiment_key)
                    aggregate_results['experiment_folders'][experiment_key] = result.get('folder_name', f'Sparsity_{sparsity:.3f}_Rank_{rank:04d}')
                    aggregate_results['output_directories'][experiment_key] = result.get('output_dir', '')
                    
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
                                    aggregate_results['summary_data'][experiment_key] = summary_data
                                    break
                                except Exception as e:
                                    print(f"Could not load summary data for sparsity {sparsity}, rank {rank}: {e}")
                else:
                    aggregate_results['failed_experiments'].append((sparsity, rank))
            else:
                aggregate_results['failed_experiments'].append((sparsity, rank))
            
            experiment_idx += 1
    
    return aggregate_results;


# ------------------------------------------------------------


def save_aggregate_results(aggregate_results):
    """Save the aggregate results to a pickle file and create a summary text file."""
    # Create output directory (main experiment folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Frequency_Generalization_with_Sparsity_and_Rank_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results in the main folder
    results_file = os.path.join(output_dir, f'Aggregate_Sparsity_and_Rank_Results_{timestamp}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregate_results, f)




# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main execution function."""
    experiment_results = []
    
    # Create all combinations of sparsity and rank values
    # If rank control is disabled, only use the first rank value (it will be ignored anyway)
    if not CONTROL_RANK:
        rank_values_to_use = [RANK_VALUES[0]]  # Only use one rank value since it will be ignored
        print(f"Rank control disabled. Running experiments only for sparsity values: {SPARSITY_VALUES}")
    else:
        rank_values_to_use = RANK_VALUES
        print(f"Running experiments for sparsity values: {SPARSITY_VALUES} and rank values: {rank_values_to_use}")
    
    total_experiments = len(SPARSITY_VALUES) * len(rank_values_to_use)
    
    try:
        # Run experiments for each sparsity-rank combination
        experiment_combinations = [(s, r) for s in SPARSITY_VALUES for r in rank_values_to_use]
        
        for i, (sparsity, rank) in enumerate(tqdm(experiment_combinations, desc="Running sparsity-rank experiments")):
            print(f"\nProgress: {i+1}/{total_experiments}")
            
            result = run_single_experiment(sparsity, rank)
            experiment_results.append(result)
            
            # Small delay between experiments
            import time
            time.sleep(1)
        
        # Collect and organize results
        aggregate_results = collect_experiment_results(SPARSITY_VALUES, rank_values_to_use, experiment_results)
        
        # Save aggregate results
        save_aggregate_results(aggregate_results)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()