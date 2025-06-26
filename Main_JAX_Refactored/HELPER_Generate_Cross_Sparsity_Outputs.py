"""
Helper script to generate cross-sparsity and summary outputs from already-saved per-sparsity results.

This script allows you to recreate the aggregate visualizations and summary tables
from a sparsity experiment that was interrupted before generating cross-sparsity outputs.

Usage:
1. Set OUTPUT_DIR to point to your Sparsity_Experiments_{timestamp} directory
2. Set SPARSITY_VALUES to match the sparsity levels that were tested
3. Run this script

The script will:
- Load all per-sparsity saved results from .pkl files
- Reconstruct the all_results list as if the full experiment had completed
- Generate all cross-sparsity visualizations and summary tables
"""

import os
import glob
import pickle
import numpy as np

from _2_utils import load_variable, set_custom_output_dir
from TEST_Sparsity_Experiments import (
    plot_eigenvalue_evolution_comparison,
    create_sparsity_summary_plots,
    create_unstable_eigenvalue_table,
    create_fixed_points_per_task_table,
    create_all_aggregate_plots
)




# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR EXPERIMENT
# =============================================================================
# Set this to your specific experiment output directory
OUTPUT_DIR = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250624_133230"

# Set this to a list of the sparsity values that were tested in your experiment
SPARSITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]

# Set this to the extra info at the end of output file names (should include ".pkl")
EXTRA_INFO = {}
EXTRA_INFO["0.00"] = "20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
EXTRA_INFO["0.20"] = "20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.2_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
EXTRA_INFO["0.40"] = "20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.4_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
EXTRA_INFO["0.60"] = "20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.6_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
EXTRA_INFO["0.80"] = "20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.8_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"




# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_sparsity_results(output_dir, sparsity_value, extra_info, extra_info_zero):
    """
    Load all saved results for a specific sparsity level.

    Returns a results dictionary compatible with the original experiment structure.
    """
    # Set output directory for loading
    set_custom_output_dir(output_dir)

    # From the sparsity value, create a string representation
    sparsity_str = f"{sparsity_value:.2f}".replace('.', 'p')
    # Define the full filepath to the folder sparsity-specfic results
    sparsity_filepath = os.path.join(output_dir, f"Sparsity_{sparsity_str}")
    
    # Initialize results dictionary, and add sparsity value
    results = {}
    results['sparsity'] = sparsity_value
    
    try:
        # Load basic training results
        results['trained_params'] = {
            'J': load_variable(os.path.join(sparsity_filepath, f"J_param_sparsity_{sparsity_value}_{extra_info}")),
            'B': load_variable(os.path.join(sparsity_filepath, f"B_param_sparsity_{sparsity_value}_{extra_info}")),
            'b_x': load_variable(os.path.join(sparsity_filepath, f"b_x_param_sparsity_{sparsity_value}_{extra_info}")),
            'w': load_variable(os.path.join(sparsity_filepath, f"w_param_sparsity_{sparsity_value}_{extra_info}")),
            'b_z': load_variable(os.path.join(sparsity_filepath, f"b_z_param_sparsity_{sparsity_value}_{extra_info}"))
        }
        
        # Load state data
        state_data = load_variable(os.path.join(sparsity_filepath, f"state_sparsity_{sparsity_value}_{extra_info}"))
        results['state_traj_states'] = state_data['traj_states']
        results['state_fixed_point_inits'] = state_data['fixed_point_inits']
        
        # Load eigenvalue and training data
        results['eigenvalue_data'] = load_variable(os.path.join(sparsity_filepath, f"eigenvalue_data_sparsity_{sparsity_value}_{extra_info}"))
        results['training_loss_data'] = load_variable(os.path.join(sparsity_filepath, f"training_loss_over_iterations_sparsity_{sparsity_value}_{extra_info}"))
        results['final_loss'] = results['training_loss_data']['final']
        
        # Load fixed point analysis results
        results['all_fixed_points'] = load_variable(os.path.join(sparsity_filepath, f"all_fixed_points_sparsity_{sparsity_value}_{extra_info}"))
        results['all_jacobians'] = load_variable(os.path.join(sparsity_filepath, f"all_fixed_jacobians_sparsity_{sparsity_value}_{extra_info}"))
        results['all_unstable_eig_freq'] = load_variable(os.path.join(sparsity_filepath, f"all_fixed_unstable_eig_freq_sparsity_{sparsity_value}_{extra_info}"))
        
        # Load slow point analysis results (if available)
        try:
            results['all_slow_points'] = load_variable(os.path.join(sparsity_filepath, f"all_slow_points_sparsity_{sparsity_value}_{extra_info}"))
            results['all_slow_jacobians'] = load_variable(os.path.join(sparsity_filepath, f"all_slow_jacobians_sparsity_{sparsity_value}_{extra_info}"))
            results['all_slow_unstable_eig_freq'] = load_variable(os.path.join(sparsity_filepath, f"all_slow_unstable_eig_freq_sparsity_{sparsity_value}_{extra_info}"))
        except:
            # Slow point search may not have been enabled
            results['all_slow_points'] = None
            results['all_slow_jacobians'] = None
            results['all_slow_unstable_eig_freq'] = None
        
        # Load PCA results
        results['pca_results'] = {}
        results['pca_results'][f"skip_0_tanh_False"] = load_variable(os.path.join(sparsity_filepath, f"pca_results_skip_0_{extra_info_zero}"))
        results['pca_results'][f"skip_400_tanh_False"] = load_variable(os.path.join(sparsity_filepath, f"pca_results_skip_400_{extra_info_zero}"))
            
        return results
        
    except Exception as e:
        print(f"  ✗ Error loading data for sparsity {sparsity_value}: {e}")

        return None


def check_eigenvalue_data_availability(all_results):
    """
    Check if eigenvalue data is available for plotting eigenvalue evolution comparison.
    """
    has_jacobian = False
    has_connectivity = False
    
    for results in all_results:
        if results and 'eigenvalue_data' in results:
            eigenval_data = results['eigenvalue_data']
            for phase in ['adam', 'lbfgs']:
                if phase in eigenval_data:
                    for iteration, eig_data in eigenval_data[phase]:
                        if 'jacobian' in eig_data and eig_data['jacobian']:
                            has_jacobian = True
                        if 'connectivity' in eig_data and len(eig_data['connectivity']) > 0:
                            has_connectivity = True
    
    return has_jacobian, has_connectivity




# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main function to generate cross-sparsity outputs from saved per-sparsity results.
    """
    # ========================================
    # SET DIRECTORIES, LOAD AND CHECK DATA
    # Check the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return
    
    # Set output directory for saving new plots
    set_custom_output_dir(OUTPUT_DIR)
    
    # Load all sparsity results
    all_results = []
    successful_sparsity_values = []
    for sparsity in SPARSITY_VALUES:
        results = load_sparsity_results(OUTPUT_DIR, sparsity, EXTRA_INFO[f"{sparsity:.2f}"], EXTRA_INFO["0.00"])
        if results is not None:
            all_results.append(results)
            successful_sparsity_values.append(sparsity)
    
    # Check the all the results were loaded successfully
    if len(successful_sparsity_values) != len(SPARSITY_VALUES):
        print(f"Error: Only {len(successful_sparsity_values)} out of {len(SPARSITY_VALUES)} sparsity levels were successfully loaded.")
        for i, sparsity in enumerate(SPARSITY_VALUES):
            if sparsity not in successful_sparsity_values:
                print(f"  - Sparsity {sparsity:.2f} failed to load or was not found.")
        return
    
    # Check if eigenvalue data is available
    has_jacobian, has_connectivity = check_eigenvalue_data_availability(all_results)
    

    # ========================================
    # CREATE CROSS-SPARSITY OUTPUTS
    # 1. Create comparative eigenvalue evolution visualization
    if has_jacobian or has_connectivity:
        try:
            plot_eigenvalue_evolution_comparison(all_results, successful_sparsity_values)
        except Exception as e:
            print(f"   ✗ Error creating eigenvalue evolution comparison: {e}")
    else:
        print("\n1. Skipping eigenvalue evolution visualization (no eigenvalue data found)")
    
    # 2. Create comprehensive summary plots
    try:
        create_sparsity_summary_plots(all_results, successful_sparsity_values)
    except Exception as e:
        print(f"   ✗ Error creating summary plots: {e}")
    
    # 3. Create unstable eigenvalue distribution table
    try:
        create_unstable_eigenvalue_table(all_results, successful_sparsity_values)
    except Exception as e:
        print(f"   ✗ Error creating unstable eigenvalue table: {e}")
    
    # 4. Create fixed points per task distribution table
    try:
        create_fixed_points_per_task_table(all_results, successful_sparsity_values)
    except Exception as e:
        print(f"   ✗ Error creating fixed points per task table: {e}")
    
    # 5. Create all aggregate plots across sparsity levels
    try:
        create_all_aggregate_plots(all_results, successful_sparsity_values)
    except Exception as e:
        print(f"   ✗ Error creating aggregate plots: {e}")




if __name__ == "__main__":
    main()