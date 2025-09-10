import os
import pickle
from datetime import datetime

import numpy as np
from scipy.spatial import procrustes

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Random seeds and file paths to use as baselines
rand_seeds = [42, 15]
file_paths_for_baselines = ["/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250624_133230/Sparsity_0p00/state_sparsity_0.0_20250624_133209_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl",
                            "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_Baseline_for_Procrustes_20250626_114113/Sparsity_0p00/state_sparsity_0.0_20250626_114103_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"]

# Sparsity levels and file paths to use in experiments
sparsity_levels = ["0.0", "0.2", "0.4", "0.6", "0.8", "0.99"]
file_paths_to_trajs = ["/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p00/state_sparsity_0.0_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl",
                       "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p20/state_sparsity_0.2_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.2_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl",
                       "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p40/state_sparsity_0.4_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.4_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl",
                       "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p60/state_sparsity_0.6_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.6_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl",
                       "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p80/state_sparsity_0.8_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.8_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl",
                       "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250909_145322_15_Same_as_11_but_for_6_Values_of_Sparsity/Sparsity_0p99/state_sparsity_0.99_20250909_145319_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.99_Adam_Epochs_1000_LBFGS_Epochs_10000.pkl"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




# ========================================
# READING AND WRITING FUNCTIONS
# ========================================
def read_trajectory_data(file_path):
    """
    Read trajectory data from a pickle file.
    
    Parameters
    ----------
    file_path : str
        Path to the .pkl file containing trajectory data.
        
    Returns
    -------
    data : object
        The loaded trajectory data from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_file_to_pkl(data_dict, base_name):
    """
    Save a dictionary to a pickle file in the Outputs/Procrustes_Analysis_{timestamp} folder.
    
    Parameters
    ----------
    data_dict : dict
        The dictionary to save.
    base_name : str
        The name of the file.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Move to parent directory, then to Outputs/Procrustes_Analysis_{timestamp}
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Procrustes_Analysis_{timestamp}')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename using the base name and including all the sparsity levels
    filename = f"{base_name}_sparsity_levels_{'_'.join(sparsity_levels)}_{timestamp}.pkl"

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the dictionary
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dictionary saved to: {file_path}")


def save_plot_to_png(plot_figure, base_name):
    """
    Save a plot figure to a PNG file in the Outputs/Procrustes_Analysis_{timestamp} folder.
    
    Parameters
    ----------
    plot_figure : matplotlib.figure.Figure
        The matplotlib figure to save.
    base_name : str
        The base name of the file.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Move to parent directory, then to Outputs/Procrustes_Analysis_{timestamp}
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'Outputs', f'Procrustes_Analysis_{timestamp}')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename using the base name and including all the sparsity levels
    filename = f"{base_name}_sparsity_levels_{'_'.join(sparsity_levels)}_{timestamp}.pdf"

    # Full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the plot
    plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {file_path}")




# ========================================
# PROCRUSTES ANALYSIS FUNCTIONS
# ========================================
def perform_procrustes_with_tensor_flattening(X, Y):
    """
    Perform Procrustes analysis on two 3-D tensors by flattening the first two dimensions.

    Parameters
    ----------
    X : array_like, shape (K, T, N)
        Source tensor.
    Y : array_like, shape (K, T, N)
        Target tensor. Must have the same shape as X.

    Returns
    -------
    disparity : float
        Procrustes disparity between flattened, standardized X and Y.
    R2 : float
        Procrustes R-squared = (1 - disparity/2)**2.
    X_std : ndarray, shape (K, T, N)
        Centered & Frobenius-norm standardized X, reshaped back to (K, T, N).
    Y_transformed : ndarray, shape (K, T, N)
        Y after centering, normalization, optimal rotation+scale, reshaped back to (K, T, N).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    K, T, N = X.shape

    # 1) Flatten (K, T, N) -> (K*T, N)
    X_flat = X.reshape(K*T, N)
    Y_flat = Y.reshape(K*T, N)

    # 2) Run SciPy Procrustes (centers, scales both to unit norm, finds best rot+scale of Y)
    X_std_flat, Y_std_flat, disparity = procrustes(X_flat, Y_flat)

    # 3) Compute R and R²
    R = 1 - disparity/2
    R2 = R**2

    # 4) Reshape standardized and transformed arrays back to (K, T, N)
    X_std = X_std_flat.reshape(K, T, N)
    Y_transformed = Y_std_flat.reshape(K, T, N)

    return disparity, R2, X_std, Y_transformed


def perform_procrustes_with_averaging(X, Y):
    """
    Perform separate Procrustes fits on each slice of two 3-D tensors and sum the disparities.

    Parameters
    ----------
    X : array_like, shape (K, T, N)
        Source tensor.
    Y : array_like, shape (K, T, N)
        Target tensor. Must have the same shape as X.

    Returns
    -------
    total_disparity : float
        Sum of individual Procrustes disparities across the K slices.
    R2 : float
        Procrustes R-squared computed as (1 - total_disparity/2)**2.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    K, T, N = X.shape

    # 1) Run Procrustes on each (T, N) slice and accumulate disparities
    total_disparity = 0.0
    for k in range(K):
        _, _, disp_k = procrustes(X[k], Y[k])
        total_disparity += disp_k

    # 2) Compute R and R² in the same way as the flattened method
    R = 1 - total_disparity/2
    R2 = R**2

    return total_disparity, R2




# ========================================
# PLOTTING FUNCTIONS
# ========================================
def plot_table(results, sparsity_levels, value_key="disparity_flat"):
    """
    Create a heatmap table showing specified values for different sparsity level combinations.
    
    Parameters
    ----------
    results : dict
        Dictionary containing Procrustes analysis results with keys in format 
        "{sparsity_level_x}_vs_{sparsity_level_y}".
    sparsity_levels : list
        List of sparsity levels used for x-axis (columns) and y-axis (rows).
    value_key : str, optional
        The key to extract from each result dictionary (default: "disparity_flat").
        Options include: "disparity_flat", "R2_flat", "total_disparity_avg", "R2_avg".
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the heatmap table.
    """
    # Create a matrix to store the specified values
    n_levels = len(sparsity_levels)
    value_matrix = np.zeros((n_levels, n_levels))
    
    # Fill the matrix with the specified values
    for i, sparsity_y in enumerate(sparsity_levels):
        for j, sparsity_x in enumerate(sparsity_levels):
            key = f"{sparsity_x}_vs_{sparsity_y}"
            if key in results:
                value_matrix[i, j] = results[key][value_key]
            else:
                value_matrix[i, j] = np.nan
    
    # Create a DataFrame for better labeling
    df = pd.DataFrame(
        value_matrix,
        index=[f"Sparsity {level}" for level in sparsity_levels],
        columns=[f"Sparsity {level}" for level in sparsity_levels]
    )
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use Blues colormap, but flip it for disparity metrics (lower is better)
    if "disparity" in value_key:
        cmap = 'Blues_r'  # Reverse Blues for disparity (lower values = darker blue)
    else:
        cmap = 'Blues'    # Normal Blues for R2 (higher values = darker blue)
    cbar_label = f'Procrustes {value_key.replace("_", " ").title()}'
    
    # Create heatmap with annotations
    sns.heatmap(
        df,
        annot=True,
        fmt='.4f',
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Sparsity Level X', fontsize=12)
    ax.set_ylabel('Sparsity Level Y', fontsize=12)
    ax.set_title(f'Procrustes {value_key.replace("_", " ").title()} Matrix', 
                 fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot
    save_plot_to_png(fig, f"Procrustes_{value_key}_heatmap")
    
    return fig




# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    results = {}

    disparity_flat_base, R2_flat_base, _, _ = perform_procrustes_with_tensor_flattening(read_trajectory_data(file_paths_for_baselines[0])["traj_states"],
                                                                                        read_trajectory_data(file_paths_for_baselines[1])["traj_states"])
    total_disparity_avg_base, R2_avg_base = perform_procrustes_with_averaging(read_trajectory_data(file_paths_for_baselines[0])["traj_states"],
                                                                                        read_trajectory_data(file_paths_for_baselines[1])["traj_states"])

    if len(sparsity_levels) == len(file_paths_to_trajs):
        for idx_x, sparsity_level_x in enumerate(sparsity_levels):
            for idx_y, sparsity_level_y in enumerate(sparsity_levels):
                # Load trajectories for the current sparsity levels
                traj_x = read_trajectory_data(file_paths_to_trajs[idx_x])["traj_states"]
                traj_y = read_trajectory_data(file_paths_to_trajs[idx_y])["traj_states"]

                # Perform Procrustes analysis with tensor flattening
                disparity_flat, R2_flat, _, _ = perform_procrustes_with_tensor_flattening(traj_x, traj_y)
                # Normalize the results for tensor flattening by the base values
                disparity_flat_normalized, R2_flat_normalized = disparity_flat / disparity_flat_base, R2_flat / R2_flat_base

                # Perform Procrustes analysis with averaging
                total_disparity_avg, R2_avg = perform_procrustes_with_averaging(traj_x, traj_y)
                # Normalize the results for averaging by the base values
                total_disparity_avg_normalized, R2_avg_normalized = total_disparity_avg / total_disparity_avg_base, R2_avg / R2_avg_base

                # Save the results in the results dictionary with an appropriate key
                key = f"{sparsity_level_x}_vs_{sparsity_level_y}"
                results[key] = {
                    "disparity_flat": disparity_flat_normalized,
                    "R2_flat": R2_flat_normalized,
                    # "X_std_flat": X_std_flat,
                    # "Y_transformed_flat": Y_transformed_flat,
                    "total_disparity_avg": total_disparity_avg_normalized,
                    "R2_avg": R2_avg_normalized
                }

        # Save the results to a file
        save_file_to_pkl(results, "Procrustes_Analysis_Results")

        # Plot the results and save the table (for each of the 4 possible metrics)
        plot_table(results, sparsity_levels, value_key="disparity_flat")
        plot_table(results, sparsity_levels, value_key="R2_flat")
        plot_table(results, sparsity_levels, value_key="total_disparity_avg")
        plot_table(results, sparsity_levels, value_key="R2_avg")