"""
Main execution script for RNN sine wave generation experiment.
Orchestrates the complete workflow from initialization to analysis.


OUTPUT VARIABLES SAVED:
Training results:
- 'J_param', shape (N, N)
- 'B_param', shape (N, I)
- 'b_x_param', shape (N,)
- 'w_param', shape (I, N)
- 'b_z_param', shape (I,)
- 'state' (dict containing):
  - 'traj_states': array of shape (num_tasks, T_train, N)
  - 'fixed_point_inits': array of shape (num_tasks, N)

Fixed point and frequency analysis results (replicated for slow points if enabled):
- 'all_fixed_points': list of lists, outer list over tasks, inner list over fixed points for a given task
- 'all_fixed_jacobians': list of lists, outer list over tasks, inner list over fixed point Jacobians for a given task
- 'all_fixed_unstable_eig_freq': list of lists of lists, outer list over tasks, inner list over fixed points, inner list over unstable eigen frequencies for a given task

PCA analysis results:
- 'pca_results' (dict containing dictionaries, one for each combination of skip and tanh options):
    - proj_trajs: list of projected trajectories, shape (num_tasks, num_steps_train+1-skip_initial_steps, n_components)
    - skip_initial_steps: number of initial steps skipped
    - apply_tanh: whether tanh transformation was applied
    - pca_components: PCA components, shape (n_components, N)
    - pca_explained_variance_ratio: explained variance ratio for each component, shape (n_components,)
    - all_fixed_points: list of fixed points for PCA analysis, same structure as all_fixed_points, so shape (num_total_fixed_ponts, N)
    - all_slow_points if SLOW_POINT_SEARCH is True


OUTPUT PLOTS SAVED:
Visualization of basic results:
- 'trajectory_vs_target' plots:
        From 'plot_trajectories_vs_targets' function
        One for each of TEST_INDICES
- 'parameter_matrices' plots:
        Plot of J, B, and b_x matrices with colour scales
        From 'plot_parameter_matrices_for_tasks' function
        One for each of TEST_INDICES
        
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

PCA plots (one for each combination of skip and tanh options):
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
"""


import numpy as np
import jax
from jax import random

# Import all modules
from _1_config import *
from _2_utils import Timer, save_variable
from _3_data_generation import generate_all_inputs_and_targets
from _4_rnn_model import init_params, run_batch_diagnostics
from _5_training import train_model
from _6_analysis import find_and_analyze_points, generate_point_summaries
from _7_visualization import (plot_trajectories_vs_targets, plot_parameter_matrices_for_tasks, 
                          analyze_jacobians_visualization, analyze_unstable_frequencies_visualization)
from _8_pca_analysis import run_pca_analysis

def main():
    """
    Main execution function that runs the complete RNN experiment.
    """
    
    with Timer("Full RNN Experiment"):
        print("=" * 60)
        print("RNN SINE WAVE GENERATION EXPERIMENT")
        print("=" * 60)
        
        # =============================================================================
        # 1. SETUP AND INITIALIZATION
        # =============================================================================
        print("\n1. SETUP AND INITIALIZATION")
        print("-" * 40)
        
        # Set random seeds
        np.random.seed(NUMPY_SEED)
        key = random.PRNGKey(JAX_SEED)
        key, subkey = random.split(key)
        
        # Initialize RNN parameters
        print(f"Initializing RNN with {N} neurons for {num_tasks} tasks...")
        mask, params = init_params(subkey)
        
        print(f"Network dimensions: {N} neurons, {I} input dimension")
        print(f"Task parameters: {num_tasks} tasks, frequencies from {omegas[0]:.3f} to {omegas[-1]:.3f}")
        print(f"Time parameters: dt={dt}, T_drive={T_drive}, T_train={T_train}")
        
        # =============================================================================
        # 2. TRAINING
        # =============================================================================
        print("\n2. TRAINING")
        print("-" * 40)
        
        with Timer("Training"):
            # Train the model
            trained_params, final_loss = train_model(params, mask)
        
        # =============================================================================
        # 3. POST-TRAINING DIAGNOSTICS
        # =============================================================================
        print("\n3. POST-TRAINING DIAGNOSTICS")
        print("-" * 40)
        
        # Collect final trajectories and fixed-point initializations
        final_loss_check, state_traj_states, state_fixed_point_inits = run_batch_diagnostics(trained_params, key)
        print(f"Final loss after training: {final_loss_check:.6e}")
        
        # =============================================================================
        # 4. SAVE TRAINING RESULTS
        # =============================================================================
        print("\n4. SAVING TRAINING RESULTS")
        print("-" * 40)
        
        print("Saving training results...")
        save_variable(np.array(trained_params["J"]), "J_param")
        save_variable(np.array(trained_params["B"]), "B_param")
        save_variable(np.array(trained_params["b_x"]), "b_x_param")
        save_variable(np.array(trained_params["w"]), "w_param")
        save_variable(np.array(trained_params["b_z"]), "b_z_param")

        state_dict = {
            "traj_states": state_traj_states,
            "fixed_point_inits": state_fixed_point_inits
        }
        save_variable(state_dict, "state")
        
        # =============================================================================
        # 5. VISUALIZATION OF BASIC RESULTS
        # =============================================================================
        print("\n5. VISUALIZATION OF BASIC RESULTS")
        print("-" * 40)
        
        # Plot trajectories vs targets
        print("Plotting produced trajectories vs. target signals...")
        plot_trajectories_vs_targets(trained_params)
        
        # Plot parameter matrices
        plot_parameter_matrices_for_tasks(trained_params)
        
        # =============================================================================
        # 6. FIXED POINT ANALYSIS
        # =============================================================================
        print("\n6. FIXED POINT ANALYSIS")
        print("-" * 40)
        
        with Timer("Fixed Point Analysis"):
            # Find and analyze fixed points
            all_fixed_points, all_jacobians, all_unstable_eig_freq = find_and_analyze_points(
                state_traj_states, state_fixed_point_inits, trained_params, point_type="fixed"
            )
            
            # Save fixed point results
            save_variable(all_fixed_points, "all_fixed_points")
            save_variable(all_jacobians, "all_fixed_jacobians")
            save_variable(all_unstable_eig_freq, "all_fixed_unstable_eig_freq")
        
        # Generate fixed point summaries
        generate_point_summaries(point_type="fixed", all_points=all_fixed_points, 
                                all_unstable_eig_freq=all_unstable_eig_freq)
        
        # Visualize fixed point analysis
        analyze_jacobians_visualization(point_type="fixed", all_jacobians=all_jacobians)
        analyze_unstable_frequencies_visualization(point_type="fixed", 
                                                  all_unstable_eig_freq=all_unstable_eig_freq)
        
        # =============================================================================
        # 7. SLOW POINT ANALYSIS (OPTIONAL)
        # =============================================================================
        all_slow_points = None
        all_slow_jacobians = None
        all_slow_unstable_eig_freq = None
        
        if SLOW_POINT_SEARCH:
            print("\n7. SLOW POINT ANALYSIS")
            print("-" * 40)
            
            with Timer("Slow Point Analysis"):
                # Find and analyze slow points
                all_slow_points, all_slow_jacobians, all_slow_unstable_eig_freq = find_and_analyze_points(
                    state_traj_states, state_fixed_point_inits, trained_params, point_type="slow"
                )
                
                # Save slow point results
                save_variable(all_slow_points, "all_slow_points")
                save_variable(all_slow_jacobians, "all_slow_jacobians")
                save_variable(all_slow_unstable_eig_freq, "all_slow_unstable_eig_freq")
            
            # Generate slow point summaries
            generate_point_summaries(point_type="slow", all_points=all_slow_points, 
                                    all_unstable_eig_freq=all_slow_unstable_eig_freq)
            
            # Visualize slow point analysis
            analyze_jacobians_visualization(point_type="slow", all_jacobians=all_slow_jacobians)
            analyze_unstable_frequencies_visualization(point_type="slow", 
                                                      all_unstable_eig_freq=all_slow_unstable_eig_freq)
        
        # =============================================================================
        # 8. PCA ANALYSIS
        # =============================================================================
        print("\n8. PCA ANALYSIS")
        print("-" * 40)
        
        with Timer("PCA Analysis"):
            print(f"Running PCA analysis with {len(skip_options)} skip options and {len(tanh_options)} transformation options...")
            print(f"Skip options: {skip_options}")
            print(f"Tanh transformation options: {tanh_options}")
            
            # Store all PCA results
            all_pca_results = {}
            
            for skip_steps in PCA_SKIP_OPTIONS:
                for apply_tanh in PCA_TANH_OPTIONS:
                    config_name = f"skip_{skip_steps}_tanh_{apply_tanh}"
                    print(f"\n--- Running PCA with skip={skip_steps}, tanh={apply_tanh} ---")
                    
                    # Run PCA analysis with current configuration
                    pca_results = run_pca_analysis(
                        state_traj_states=state_traj_states,
                        all_fixed_points=all_fixed_points,
                        all_slow_points=all_slow_points,
                        params=trained_params,
                        slow_point_search=SLOW_POINT_SEARCH,
                        skip_initial_steps=skip_steps,
                        apply_tanh=apply_tanh,
                        n_components=PCA_N_COMPONENTS
                    )
                    
                    # Store results with descriptive key
                    all_pca_results[config_name] = pca_results
                    
                    # Print summary for this configuration
                    explained_variance = pca_results["pca_explained_variance_ratio"]
                    total_variance = sum(explained_variance)
                    print(f"PCA explained variance ratio: {explained_variance}")
                    print(f"Total variance explained by first 3 components: {total_variance:.3f}")
            
            # Save all PCA results
            save_variable(all_pca_results, "all_pca_results")
            print(f"\nSaved results for {len(all_pca_results)} PCA configurations")
        
        # =============================================================================
        # 9. FINAL SUMMARY
        # =============================================================================
        print("\n9. EXPERIMENT COMPLETE")
        print("-" * 40)
        print(f"Final training loss: {final_loss:.6e}")
        print(f"Number of fixed points found: {sum(len(task_fps) for task_fps in all_fixed_points)}")
        if SLOW_POINT_SEARCH and all_slow_points:
            print(f"Number of slow points found: {sum(len(task_sps) for task_sps in all_slow_points)}")
        print(f"PCA analysis completed for {len(skip_options)} × {len(tanh_options)} = {len(skip_options) * len(tanh_options)} configurations")
        print("All results saved to the Outputs directory.")


if __name__ == "__main__":
    main()