"""
Helper script to generate cross-L1 and summary outputs from already-saved per-L1 results.

This script allows you to recreate the aggregate visualizations and summary tables
from a L1 experiment that was interrupted before generating cross-L1 outputs.

Usage:
1. Set OUTPUT_DIR to point to your L1_Regularization_Experiments_{timestamp} directory
2. Set L1_VALUES to match the L1 levels that were tested
3. Run this script

The script will:
- Load all per-L1 saved results from .pkl files
- Reconstruct the all_results list as if the full experiment had completed
- Generate all cross-L1 visualizations and summary tables
"""

import os
import glob
import pickle
import numpy as np

from _2_utils import load_variable, set_custom_output_dir
from TEST_L1_Experiments import (
    plot_eigenvalue_evolution_comparison,
    create_l1_reg_summary_plots,
    create_unstable_eigenvalue_table,
    create_fixed_points_per_task_table,
    create_all_aggregate_plots
)




# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR EXPERIMENT
# =============================================================================
# Set this to your specific experiment output directory
OUTPUT_DIR = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/L1_Regularization_Experiments_20250625_074229"

# Set this to a list of the L1 regularization values that were tested in your experiment
L1_REG_VALUES = [0.0, 1e-05, 1e-04, 0.001, 0.01, 0.1]

# Set this to the extra info at the end of output file names (should include ".pkl")
EXTRA_INFO = "20250625_074146_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"




# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_l1_results(output_dir, l1_reg_value, extra_info):
    """
    Load all saved results for a specific l1_reg level.

    Returns a results dictionary compatible with the original experiment structure.
    """
    # Set output directory for loading
    set_custom_output_dir(output_dir)

    # From the l1_reg value, create a string representation
    if l1_reg_value == 0.0:
        l1_reg_str = "0p00"
    elif l1_reg_value == 0.1:
        l1_reg_str = "0p1"
    elif l1_reg_value == 0.01:
        l1_reg_str = "0p01"
    elif l1_reg_value == 0.001:
        l1_reg_str = "0p001"
    elif l1_reg_value == 1e-04:
        l1_reg_str = "1p00em04"
    elif l1_reg_value == 1e-05:
        l1_reg_str = "1p00em05"
    # Define the full filepath to the folder l1_reg-specfic results
    l1_reg_filepath = os.path.join(output_dir, f"L1_Reg_{l1_reg_str}")
    
    # Initialize results dictionary, and add l1_reg value
    results = {}
    results['l1_reg_strength'] = l1_reg_value
    
    try:
        # Load basic training results
        results['trained_params'] = {
            'J': load_variable(os.path.join(l1_reg_filepath, f"J_param_l1_reg_{l1_reg_value}_{extra_info}")),
            'B': load_variable(os.path.join(l1_reg_filepath, f"B_param_l1_reg_{l1_reg_value}_{extra_info}")),
            'b_x': load_variable(os.path.join(l1_reg_filepath, f"b_x_param_l1_reg_{l1_reg_value}_{extra_info}")),
            'w': load_variable(os.path.join(l1_reg_filepath, f"w_param_l1_reg_{l1_reg_value}_{extra_info}")),
            'b_z': load_variable(os.path.join(l1_reg_filepath, f"b_z_param_l1_reg_{l1_reg_value}_{extra_info}"))
        }
        
        # Load state data
        state_data = load_variable(os.path.join(l1_reg_filepath, f"state_l1_reg_{l1_reg_value}_{extra_info}"))
        results['state_traj_states'] = state_data['traj_states']
        results['state_fixed_point_inits'] = state_data['fixed_point_inits']
        
        # Load eigenvalue and training data
        results['eigenvalue_data'] = load_variable(os.path.join(l1_reg_filepath, f"eigenvalue_data_l1_reg_{l1_reg_value}_{extra_info}"))
        results['training_loss_data'] = load_variable(os.path.join(l1_reg_filepath, f"training_loss_over_iterations_l1_reg_{l1_reg_value}_{extra_info}"))
        results['final_loss'] = results['training_loss_data']['final']
        
        # Load fixed point analysis results
        results['all_fixed_points'] = load_variable(os.path.join(l1_reg_filepath, f"all_fixed_points_l1_reg_{l1_reg_value}_{extra_info}"))
        results['all_jacobians'] = load_variable(os.path.join(l1_reg_filepath, f"all_fixed_jacobians_l1_reg_{l1_reg_value}_{extra_info}"))
        results['all_unstable_eig_freq'] = load_variable(os.path.join(l1_reg_filepath, f"all_fixed_unstable_eig_freq_l1_reg_{l1_reg_value}_{extra_info}"))
        
        # Load slow point analysis results (if available)
        try:
            results['all_slow_points'] = load_variable(os.path.join(l1_reg_filepath, f"all_slow_points_l1_reg_{l1_reg_value}_{extra_info}"))
            results['all_slow_jacobians'] = load_variable(os.path.join(l1_reg_filepath, f"all_slow_jacobians_l1_reg_{l1_reg_value}_{extra_info}"))
            results['all_slow_unstable_eig_freq'] = load_variable(os.path.join(l1_reg_filepath, f"all_slow_unstable_eig_freq_l1_reg_{l1_reg_value}_{extra_info}"))
        except:
            # Slow point search may not have been enabled
            results['all_slow_points'] = None
            results['all_slow_jacobians'] = None
            results['all_slow_unstable_eig_freq'] = None
        
        # Load PCA results
        results['pca_results'] = {}
        results['pca_results'][f"skip_0_tanh_False"] = load_variable(os.path.join(l1_reg_filepath, f"pca_results_skip_0_{extra_info}"))
        results['pca_results'][f"skip_400_tanh_False"] = load_variable(os.path.join(l1_reg_filepath, f"pca_results_skip_400_{extra_info}"))
            
        return results
        
    except Exception as e:
        print(f"  ✗ Error loading data for l1_reg {l1_reg_value}: {e}")

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
    Main function to generate cross-l1_reg outputs from saved per-L1 results.
    """
    # ========================================
    # SET DIRECTORIES, LOAD AND CHECK DATA
    # Check the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory does not exist: {OUTPUT_DIR}")
        return
    
    # Set output directory for saving new plots
    set_custom_output_dir(OUTPUT_DIR)
    
    # Load all l1_reg results
    all_results = []
    successful_l1_reg_values = []
    for l1_reg in L1_REG_VALUES:
        results = load_l1_results(OUTPUT_DIR, l1_reg, EXTRA_INFO)
        if results is not None:
            all_results.append(results)
            successful_l1_reg_values.append(l1_reg)
    
    # Check the all the results were loaded successfully
    if len(successful_l1_reg_values) != len(L1_REG_VALUES):
        print(f"Error: Only {len(successful_l1_reg_values)} out of {len(L1_REG_VALUES)} l1_reg levels were successfully loaded.")
        for i, l1_reg in enumerate(L1_REG_VALUES):
            if l1_reg not in successful_l1_reg_values:
                print(f"  - L1_Reg {l1_reg:.2f} failed to load or was not found.")
        return
    
    # Check if eigenvalue data is available
    has_jacobian, has_connectivity = check_eigenvalue_data_availability(all_results)
    

    # ========================================
    # CREATE CROSS-L1_REG OUTPUTS
    # 1. Create comparative eigenvalue evolution visualization
    if has_jacobian or has_connectivity:
        try:
            plot_eigenvalue_evolution_comparison(all_results, successful_l1_reg_values)
        except Exception as e:
            print(f"   ✗ Error creating eigenvalue evolution comparison: {e}")
    else:
        print("\n1. Skipping eigenvalue evolution visualization (no eigenvalue data found)")
    
    # 2. Create comprehensive summary plots
    try:
        create_l1_reg_summary_plots(all_results, successful_l1_reg_values)
    except Exception as e:
        print(f"   ✗ Error creating summary plots: {e}")
    
    # 3. Create unstable eigenvalue distribution table
    try:
        create_unstable_eigenvalue_table(all_results, successful_l1_reg_values)
    except Exception as e:
        print(f"   ✗ Error creating unstable eigenvalue table: {e}")
    
    # 4. Create fixed points per task distribution table
    try:
        create_fixed_points_per_task_table(all_results, successful_l1_reg_values)
    except Exception as e:
        print(f"   ✗ Error creating fixed points per task table: {e}")
    
    # 5. Create all aggregate plots across l1_reg levels
    try:
        create_all_aggregate_plots(all_results, successful_l1_reg_values)
    except Exception as e:
        print(f"   ✗ Error creating aggregate plots: {e}")




if __name__ == "__main__":
    main()