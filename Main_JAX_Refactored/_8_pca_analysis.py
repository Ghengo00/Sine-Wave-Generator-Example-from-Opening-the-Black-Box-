"""
PCA analysis and visualization for trajectory and fixed point analysis.
"""


import numpy as np
from scipy.linalg import eig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo

import sys
import os
from tqdm import tqdm
import time

from _1_config import s
from _2_utils import get_output_dir, save_figure, save_variable, load_variable, set_custom_output_dir
from _6_analysis import compute_jacobian




# ================================================================================
# PCA FUNCTIONS
# ================================================================================
def perform_pca_analysis(state_traj_states, skip_initial_steps=0, apply_tanh=False, n_components=3):
    """
    Perform PCA on trajectory states and return PCA object and projections.
    
    Arguments:
        state_traj_states: trajectory states from training, shape (num_tasks, num_steps_train+1, N)
        skip_initial_steps: number of initial time steps to skip (default: 0)
        apply_tanh: whether to apply tanh transformation to states before PCA (default: False)
        n_components: number of principal components to compute (default: 3)
        
    Returns:
        pca: fitted PCA object
        proj_trajs: list of projected trajectories, shape (num_tasks, num_steps_train+1-skip, n_components)
        all_trajectories_combined: combined trajectory matrix used for PCA fitting, shape (num_tasks * (num_steps_train+1-skip), N)
    """
    # Track PCA computation time
    start_time = time.time()
    transform_info = " with tanh transformation" if apply_tanh else ""
    print(f"\nStarting PCA computation (skipping first {skip_initial_steps} time steps{transform_info})...")

    # Truncate trajectories if requested
    if skip_initial_steps > 0:
        state_traj_states_truncated = state_traj_states[:, skip_initial_steps:, :]
        print(f"Trajectory shape after truncation: {state_traj_states_truncated.shape}")
    else:
        state_traj_states_truncated = state_traj_states

    # Apply tanh transformation if requested
    if apply_tanh:
        state_traj_states_truncated = np.tanh(state_traj_states_truncated)
        print("Applied tanh transformation to trajectory states")

    # Concatenate all state trajectories from all tasks
    all_states = np.concatenate(state_traj_states_truncated, axis=0)  # shape is (num_tasks * (num_steps_train+1-skip), N)

    # Perform PCA with specified number of components
    pca = PCA(n_components=n_components)
    proj_all = pca.fit_transform(all_states)    # shape is (num_tasks * (num_steps_train+1-skip), n_components)

    # Project each trajectory into the PCA space
    proj_trajs = []
    start = 0
    for traj in tqdm(state_traj_states_truncated, desc="Projecting trajectories"):
        T = traj.shape[0]   # Number of time steps in the trajectory, T = num_steps_train + 1 - skip_initial_steps
        proj_traj = proj_all[start:start + T]   # Extracts the projected trajectory from the PCA space, resulting shape is (T, 3)
        proj_trajs.append(proj_traj)
        start += T

    # Calculate PCA computation time
    pca_time = time.time() - start_time
    print(f"\nPCA computation completed in {pca_time:.2f} seconds")
    
    return pca, proj_trajs, all_states


def project_points_to_pca(pca, all_points):
    """
    Project fixed points or slow points into PCA space.
    
    Arguments:
        pca: fitted PCA object
        all_points: list of lists of points
        
    Returns:
        proj_points: projected points in PCA space, shape (num_total_points, 3)
    """
    # Project all points from all tasks into the PCA space
    all_points_flat = []
    for task_points in all_points:
        all_points_flat.extend(task_points)
    
    if all_points_flat:   # Check if there are any points
        all_points_flat = np.array(all_points_flat)
        if all_points_flat.ndim == 1: # If it's a 1D array, reshape it to 2D
            all_points_flat = all_points_flat.reshape(1, -1)
        proj_points = pca.transform(all_points_flat)   # Projects all points from all tasks into the PCA space, resulting shape is (num_total_points, n_components)
    else:
        proj_points = np.empty((0, pca.n_components_))   # Empty array with correct shape for plotting
    
    return proj_points




# ================================================================================
# PLOTTING FUNCTIONS
# ================================================================================
def plot_explained_variance_ratio(pca, skip_initial_steps=0, apply_tanh=False, filename_prefix="", ax=None, sparsity_value=None, for_comparison=False, l1_reg_value=None):
    """
    Plot and save a bar chart of the explained variance ratio for each principal component.
    
    Arguments:
        pca: fitted PCA object
        skip_initial_steps: number of initial steps that were skipped (for title/filename)
        apply_tanh: whether tanh transformation was applied (for title/filename)
        filename_prefix: prefix to add to the filename (default: "")
        ax: optional matplotlib axis to plot on. If None, creates new figure
        sparsity_value: sparsity value for title and filename (used when ax=None or for_comparison=True)
        for_comparison: if True, adjusts styling for comparison plots
        l1_reg_value: if not None, use this L1 regularization value instead of sparsity for titles and filenames
    """
    # Create new figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone_plot = True
    else:
        fig = ax.get_figure()
        standalone_plot = False
    
    # Get explained variance ratios
    explained_variance_ratio = pca.explained_variance_ratio_
    n_components = len(explained_variance_ratio)
    
    # Create bar chart
    pc_labels = [f'PC{i+1}' for i in range(n_components)]
    bars = ax.bar(pc_labels, explained_variance_ratio, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, explained_variance_ratio)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    title_skip_info = f" (skipping first {skip_initial_steps} steps)" if skip_initial_steps > 0 else ""
    title_tanh_info = " (tanh transformed)" if apply_tanh else ""
    
    # Determine parameter values and labels for title and filename
    if l1_reg_value is not None:
        # Use L1 regularization value
        param_label = "L1 reg"
        param_format = f"{l1_reg_value:.1e}" if l1_reg_value != 0 else "0"
        param_value = l1_reg_value
    elif sparsity_value is not None:
        # Use sparsity value
        param_label = "Sparsity"
        param_format = f"{sparsity_value:.2f}"
        param_value = sparsity_value
    else:
        # Use global sparsity value
        param_label = "Sparsity"
        param_format = f"{s:.2f}"
        param_value = s
    
    # Set title
    ax.set_title(f'PCA Explained Variance Ratio{title_skip_info}{title_tanh_info} ({param_label}={param_format})', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Proportion of Total Variance', fontsize=12)
    
    # Set y-axis limits to show all variance clearly
    ax.set_ylim(0, max(explained_variance_ratio) * 1.1)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total variance explained as text
    total_variance = sum(explained_variance_ratio)
    ax.text(0.98, 0.98, f'Total variance explained: {total_variance:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    # Add variance explained by first 3 PCs as text (underneath the total variance box)
    first_3_variance = sum(explained_variance_ratio[:3]) if len(explained_variance_ratio) >= 3 else total_variance
    ax.text(0.98, 0.90, f'First 3 PCs variance: {first_3_variance:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    # Save only for standalone plots
    if standalone_plot:
        # Improve layout
        plt.tight_layout()
        
        # Save the figure
        param_str = param_format.replace('.', 'p').replace('+', 'p').replace('-', 'm')
        tanh_suffix = "_tanh" if apply_tanh else ""
        figure_name = f"{filename_prefix}pca_explained_variance_ratio_skip_{skip_initial_steps}{tanh_suffix}_{param_label.lower().replace(' ', '_')}_{param_str}"
        save_figure(fig, figure_name)
        
        plt.close()
        
        print(f"Explained variance ratio bar chart saved")


def plot_pca_trajectories_and_points(pca, proj_trajs, point_type="fixed", all_points=None, proj_points=None, params=None, skip_initial_steps=0, apply_tanh=False, filename_prefix="", ax=None, sparsity_value=None, for_comparison=False, l1_reg_value=None, proj_trajs_by_region=None):
    """
    Unified function to plot PCA trajectories with either fixed points or slow points.
    
    Arguments:
        pca          : fitted PCA object
        proj_trajs   : list of projected trajectories
        point_type   : str, either "fixed" or "slow" to specify which type of points to plot
        all_points   : list of lists of points, one list per task
        proj_points  : projected points in PCA space, shape (num_total_points, 3)
        params       : model parameters for computing Jacobians
        skip_initial_steps : number of initial steps that were skipped (for title/filename)
        apply_tanh   : whether tanh transformation was applied (for title/filename)
        filename_prefix : prefix to add to the filename (default: "")
        ax           : optional matplotlib 3D axis to plot on. If None, creates new figure
        sparsity_value : sparsity value for title and filename (used when ax=None or for_comparison=True)
        for_comparison : if True, adjusts styling for comparison plots
        l1_reg_value : if not None, use this L1 regularization value instead of sparsity for titles and filenames
        proj_trajs_by_region : dict with keys as region names and values as lists of trajectories for that region
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    J_trained = np.array(params["J"])

    # Create new figure only if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        standalone_plot = True
    else:
        fig = ax.get_figure()
        standalone_plot = False

    # Plot the projected trajectories with color gradient
    if proj_trajs_by_region is not None:
        # Plot trajectories by region with different base colors
        region_colors = {
            'lower_extrapolation': (0.5, 0.0, 0.5),   # Purple
            'lower_training': (0.0, 0.0, 1.0),        # Blue
            'gap': (1.0, 0.65, 0.0),                  # Orange
            'upper_training': (1.0, 0.0, 0.0),        # Red  
            'higher_extrapolation': (0.0, 0.5, 0.0)   # Green
        }
        
        # Track which regions actually have trajectories for legend
        plotted_regions = []
        
        for region, trajs in proj_trajs_by_region.items():
            if len(trajs) > 0:  # Only plot and add to legend if region has trajectories
                base_color = region_colors.get(region, (0.0, 0.0, 1.0))  # Default to blue
                plotted_regions.append((region, base_color))
                
                for traj_idx, traj in enumerate(trajs):
                    # Create a color gradient along the trajectory
                    num_points = len(traj)
                    if num_points > 1:
                        # Create segments for the line plot with varying colors
                        for i in range(num_points - 1):
                            # Calculate color intensity based on time step (0 = light, 1 = dark)
                            color_intensity = i / (num_points - 1)
                            # Create color with varying intensity (start with lighter and transition to darker)
                            region_color = (
                                base_color[0] * (0.3 + 0.7 * color_intensity),
                                base_color[1] * (0.3 + 0.7 * color_intensity),
                                base_color[2] * (0.3 + 0.7 * color_intensity)
                            )
                            
                            # Plot each segment with increased transparency
                            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2], 
                                   color=region_color, alpha=0.3, linewidth=1.5)
                    else:
                        # Fallback for single point trajectories
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=base_color, alpha=0.2)
        
        # Add region legend elements for plotting later
        region_legend_elements = []
        region_label_map = {
            'lower_extrapolation': 'Lower Extrapolation',
            'lower_training': 'Lower Training', 
            'gap': 'Gap Region',
            'upper_training': 'Upper Training',
            'higher_extrapolation': 'Higher Extrapolation'
        }
        
        for region, color in plotted_regions:
            # Create a line element for the legend with the region's base color
            from matplotlib.lines import Line2D
            region_legend_elements.append(
                Line2D([0], [0], color=color, lw=2, label=region_label_map.get(region, region))
            )
    else:
        # Initialize empty for when not using regions
        region_legend_elements = []
        
        # Original behavior: Plot all trajectories with blue gradient
        for traj_idx, traj in enumerate(proj_trajs):
            # Create a color gradient along the trajectory
            num_points = len(traj)
            if num_points > 1:
                # Create segments for the line plot with varying colors
                for i in range(num_points - 1):
                    # Calculate color intensity based on time step (0 = light, 1 = dark)
                    color_intensity = i / (num_points - 1)
                    # Create blue color with varying intensity (start with lighter blue and transition to darker blue)
                    blue_color = (0.7 - 0.5 * color_intensity, 0.7 - 0.5 * color_intensity, 1.0)
                    
                    # Plot each segment with increased transparency
                    ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2], 
                           color=blue_color, alpha=0.3, linewidth=1.5)
            else:
                # Fallback for single point trajectories
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.2)

    # Plot the projected points as green scatter points
    if proj_points.size > 0:
        # Adjust scatter point size for comparison plots
        point_size = 15 if not for_comparison else 15
        ax.scatter(proj_points[:, 0], proj_points[:, 1], proj_points[:, 2], 
                  color='green', s=point_size, alpha=0.8, label=point_name)

    # For each point, plot all unstable modes as red lines
    point_idx = 0
    unstable_mode_counts = {}  # Dictionary to track number of unstable modes per point
    
    for j, task_points in enumerate(all_points):
        for x_star in task_points:
            # Compute the Jacobian and find the eigenvalues at the point
            J_eff = np.array(compute_jacobian(x_star, J_trained))
            eigenvals, eigenvecs = eig(J_eff)
            # Find all the unstable eigenvalues for the given point
            unstable_idx = np.where(np.real(eigenvals) > 0)[0]
            num_unstable = len(unstable_idx)
            
            # Track unstable mode counts
            if num_unstable in unstable_mode_counts:
                unstable_mode_counts[num_unstable] += 1
            else:
                unstable_mode_counts[num_unstable] = 1
            
            if len(unstable_idx) > 0:
                # For each unstable eigenvalue, plot the corresponding eigenvector direction
                for idx in unstable_idx:
                    # Take the real part for the plotting direction
                    v = np.real(eigenvecs[:, idx])
                    scale = 1.0
                    # Project the unstable eigenvector direction into PCA space
                    v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_points[point_idx]
                    # Plot a line centred on the point in PCA space
                    line = np.vstack([
                        proj_points[point_idx] - v_proj,
                        proj_points[point_idx] + v_proj
                    ])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red', linewidth=2, alpha=0.5)
            point_idx += 1
    
    # Print summary of unstable modes (only for standalone plots)
    if standalone_plot:
        print(f"\nUnstable modes summary for {point_name}:")
        for num_modes in sorted(unstable_mode_counts.keys()):
            count = unstable_mode_counts[num_modes]
            print(f"Number of fixed points with {num_modes} unstable modes: {count}")

    # Calculate the overall data range for equal scaling
    all_data = np.concatenate([np.concatenate(proj_trajs, axis=0)])
    if proj_points.size > 0:
        all_data = np.concatenate([all_data, proj_points], axis=0)
    
    # Find the maximum absolute value across all dimensions
    max_range = np.max(np.abs(all_data))
    
    # Add decorations
    title_skip_info = f" (skipping first {skip_initial_steps} steps)" if skip_initial_steps > 0 else ""
    title_tanh_info = " (tanh transformed)" if apply_tanh else ""
    
    # Determine parameter values and labels for title and filename
    if l1_reg_value is not None:
        # Use L1 regularization value
        param_label = "L1 reg"
        param_format = f"{l1_reg_value:.1e}" if l1_reg_value != 0 else "0"
        param_value = l1_reg_value
    elif sparsity_value is not None:
        # Use sparsity value
        param_label = "Sparsity"
        param_format = f"{sparsity_value:.2f}"
        param_value = sparsity_value
    else:
        # Use global sparsity value
        param_label = "Sparsity"
        param_format = f"{s:.2f}"
        param_value = s
    
    ax.set_title(f'PCA of Network Trajectories and {point_name}{title_skip_info}{title_tanh_info} ({param_label}={param_format})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Set equal aspect ratio and equal scaling for all axes
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add legend - combine fixed points legend with region legend
    legend_elements = []
    
    # Add fixed points to legend if they exist
    if proj_points.size > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='green', 
                                         linestyle='None', markersize=8, 
                                         alpha=0.8, label=point_name))
    
    # Add region legend elements if they exist
    if region_legend_elements:
        legend_elements.extend(region_legend_elements)
    
    # Create the legend if there are elements to show
    if legend_elements:
        # Position legend outside the plot area to avoid overlap
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5), 
                 fontsize=9)

    # Save only for standalone plots
    if standalone_plot:
        # Save the figure
        param_str = param_format.replace('.', 'p').replace('+', 'p').replace('-', 'm')
        tanh_suffix = "_tanh" if apply_tanh else ""
        figure_name = f"{filename_prefix}pca_plot_{point_type}_skip_{skip_initial_steps}{tanh_suffix}_{param_label.lower().replace(' ', '_')}_{param_str}"
        save_figure(fig, figure_name)

        plt.close()


def plot_interactive_pca_trajectories_and_points(pca, proj_trajs, point_type="fixed", all_points=None, proj_points=None, params=None, skip_initial_steps=0, apply_tanh=False, filename_prefix="", sparsity_value=None, l1_reg_value=None):
    """
    Create an interactive 3D plot using Plotly that can be rotated and saved as HTML.
    
    Arguments:
        pca          : fitted PCA object
        proj_trajs   : list of projected trajectories
        point_type   : str, either "fixed" or "slow" to specify which type of points to plot
        all_points   : list of lists of points, one list per task
        proj_points  : projected points in PCA space, shape (num_total_points, 3)
        params       : model parameters for computing Jacobians
        skip_initial_steps : number of initial steps that were skipped (for title/filename)
        apply_tanh   : whether tanh transformation was applied (for title/filename)
        filename_prefix : prefix to add to the filename (default: "")
        sparsity_value : sparsity value for title and filename (optional)
        l1_reg_value : if not None, use this L1 regularization value instead of sparsity for titles and filenames
    """
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    # Create figure
    fig = go.Figure()
    
    # Add trajectories with color gradient from light to dark blue
    for i, traj in enumerate(proj_trajs):
        num_points = len(traj)
        if num_points > 1:
            # Create color values for gradient (0 = light blue, 1 = dark blue)
            color_values = np.linspace(0, 1, num_points)
            
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines+markers',
                line=dict(
                    color=color_values,
                    colorscale=[[0, 'lightblue'], [1, 'darkblue']],
                    width=4,
                    showscale=False
                ),
                marker=dict(
                    color=color_values,
                    colorscale=[[0, 'lightblue'], [1, 'darkblue']],
                    size=3,
                    showscale=False
                ),
                showlegend=False,
                opacity=0.3  # Reduced opacity for more transparency
            ))
        else:
            # Fallback for single point trajectories
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False,
                opacity=0.2  # Reduced opacity for more transparency
            ))
    
    # Add points
    if proj_points.size > 0:
        fig.add_trace(go.Scatter3d(
            x=proj_points[:, 0], y=proj_points[:, 1], z=proj_points[:, 2],
            mode='markers',
            marker=dict(color='green', size=3),  # Reduced size from 5 to 3
            name=point_name
        ))
    
    # Add unstable modes as lines
    if all_points is not None and params is not None:
        J_trained = np.array(params["J"])
        point_idx = 0
        unstable_lines_x, unstable_lines_y, unstable_lines_z = [], [], []
        unstable_mode_counts = {}  # Dictionary to track number of unstable modes per point
        
        for j, task_points in enumerate(all_points):
            for x_star in task_points:
                # Compute the Jacobian and find the eigenvalues at the point
                J_eff = np.array(compute_jacobian(x_star, J_trained))
                eigenvals, eigenvecs = eig(J_eff)
                # Find all the unstable eigenvalues for the given point
                unstable_idx = np.where(np.real(eigenvals) > 0)[0]
                num_unstable = len(unstable_idx)
                
                # Track unstable mode counts
                if num_unstable in unstable_mode_counts:
                    unstable_mode_counts[num_unstable] += 1
                else:
                    unstable_mode_counts[num_unstable] = 1
                
                if len(unstable_idx) > 0:
                    # For each unstable eigenvalue, plot the corresponding eigenvector direction
                    for idx in unstable_idx:
                        # Take the real part for the plotting direction
                        v = np.real(eigenvecs[:, idx])
                        scale = 1.0
                        # Project the unstable eigenvector direction into PCA space
                        v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_points[point_idx]
                        # Create line data
                        line_start = proj_points[point_idx] - v_proj
                        line_end = proj_points[point_idx] + v_proj
                        
                        # Add to line collections
                        unstable_lines_x.extend([line_start[0], line_end[0], None])
                        unstable_lines_y.extend([line_start[1], line_end[1], None])
                        unstable_lines_z.extend([line_start[2], line_end[2], None])
                
                point_idx += 1
        
        # Print summary of unstable modes
        print(f"\nUnstable modes summary for {point_name} (Interactive Plot):")
        for num_modes in sorted(unstable_mode_counts.keys()):
            count = unstable_mode_counts[num_modes]
            print(f"Number of fixed points with {num_modes} unstable modes: {count}")
        
        # Add unstable modes as a single trace
        if unstable_lines_x:
            fig.add_trace(go.Scatter3d(
                x=unstable_lines_x, y=unstable_lines_y, z=unstable_lines_z,
                mode='lines',
                line=dict(color='red', width=4),
                name='Unstable Modes',
                opacity=0.7
            ))
    
    # Calculate the overall data range for equal scaling
    all_data = np.concatenate([np.concatenate(proj_trajs, axis=0)])
    if proj_points.size > 0:
        all_data = np.concatenate([all_data, proj_points], axis=0)
    
    # Find the maximum absolute value across all dimensions
    max_range = np.max(np.abs(all_data))
    
    # Set layout and styling
    title_skip_info = f" (skipping first {skip_initial_steps} steps)" if skip_initial_steps > 0 else ""
    title_tanh_info = " (tanh transformed)" if apply_tanh else ""
    
    # Determine parameter values and labels for title and filename
    if l1_reg_value is not None:
        # Use L1 regularization value
        param_label = "L1 reg"
        param_format = f"{l1_reg_value:.1e}" if l1_reg_value != 0 else "0"
        param_value = l1_reg_value
    elif sparsity_value is not None:
        # Use sparsity value
        param_label = "Sparsity"
        param_format = f"{sparsity_value:.2f}"
        param_value = sparsity_value
    else:
        # Use global sparsity value
        param_label = "Sparsity"
        param_format = f"{s:.2f}"
        param_value = s
    
    fig.update_layout(
        title=f'Interactive PCA of Network Trajectories and {point_name}{title_skip_info}{title_tanh_info} ({param_label}={param_format})',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            aspectmode='cube',  # This ensures equal aspect ratios
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Default viewing angle
            )
        ),
        width=900,
        height=700
    )
    
    # Save as interactive HTML file
    param_str = param_format.replace('.', 'p').replace('+', 'p').replace('-', 'm')
    tanh_suffix = "_tanh" if apply_tanh else ""
    html_filename = f"{filename_prefix}interactive_pca_plot_{point_type}_skip_{skip_initial_steps}{tanh_suffix}_{param_label.lower().replace(' ', '_')}_{param_str}.html"
    
    # Use the same output directory as static plots
    output_dir = get_output_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    html_filepath = os.path.join(output_dir, html_filename)
    pyo.plot(fig, filename=html_filepath, auto_open=False)
    print(f"Interactive PCA plot saved as: {html_filepath}")
    
    return fig




# ================================================================================
# EXECUTION FUNCTION
# ================================================================================
def run_pca_analysis(state_traj_states, all_fixed_points=None, all_slow_points=None, params=None, slow_point_search=False, skip_initial_steps=0, apply_tanh=False, n_components=3):
    """
    Run complete PCA analysis including trajectory projection and point visualization.
    
    Arguments:
        state_traj_states: trajectory states from training
        all_fixed_points: list of lists of fixed points (optional)
        all_slow_points: list of lists of slow points (optional)
        params: model parameters
        slow_point_search: whether slow point search was performed
        skip_initial_steps: number of initial time steps to skip from trajectories (default: 0)
        apply_tanh: whether to apply tanh transformation to states before PCA (default: False)
        n_components: number of principal components to compute (default: 3)
        
    Returns:
        pca_results: dictionary containing PCA results
    """
    # Perform PCA on trajectories
    pca, proj_trajs, all_trajectories_combined = perform_pca_analysis(state_traj_states, skip_initial_steps=skip_initial_steps, apply_tanh=apply_tanh, n_components=n_components)
    
    # Plot explained variance ratio bar chart
    plot_explained_variance_ratio(pca, skip_initial_steps=skip_initial_steps, apply_tanh=apply_tanh)
    
    pca_results = {
        "pca": pca,  # Add the PCA object itself
        "proj_trajs": proj_trajs,
        "all_trajectories_combined": all_trajectories_combined,  # Add combined trajectory matrix
        "skip_initial_steps": skip_initial_steps,
        "apply_tanh": apply_tanh,
        "pca_components": pca.components_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    

    # ========================================
    # FIXED POINTS
    # Project and plot fixed points if available
    if all_fixed_points is not None:
        proj_fixed = project_points_to_pca(pca, all_fixed_points)
        pca_results["proj_fixed"] = proj_fixed
        
        # Plot PCA with fixed points - all trajectories (static matplotlib)
        plot_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="fixed",
            all_points=all_fixed_points,
            proj_points=proj_fixed,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
        
        # Create interactive plot - all trajectories (Plotly)
        plot_interactive_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="fixed",
            all_points=all_fixed_points,
            proj_points=proj_fixed,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
        
        # Plot subset of trajectories (first, middle, last: indices 0, 25, 50)
        subset_indices = [0, 25, 50]
        subset_proj_trajs = [proj_trajs[i] for i in subset_indices if i < len(proj_trajs)]
        subset_fixed_points = [all_fixed_points[i] for i in subset_indices if i < len(all_fixed_points)]
        
        if subset_proj_trajs and subset_fixed_points:
            subset_proj_fixed = project_points_to_pca(pca, subset_fixed_points)
            
            # Plot subset with fixed points (static matplotlib)
            plot_pca_trajectories_and_points(
                pca, subset_proj_trajs,
                point_type="fixed",
                all_points=subset_fixed_points,
                proj_points=subset_proj_fixed,
                params=params,
                skip_initial_steps=skip_initial_steps,
                apply_tanh=apply_tanh,
                filename_prefix="subset_of_indices_"
            )
            
            # Create interactive subset plot with modified filename
            subset_fig = plot_interactive_pca_trajectories_and_points(
                pca, subset_proj_trajs,
                point_type="fixed",
                all_points=subset_fixed_points,
                proj_points=subset_proj_fixed,
                params=params,
                skip_initial_steps=skip_initial_steps,
                apply_tanh=apply_tanh,
                filename_prefix="subset_of_indices_"
            )
    

    # ========================================
    # SLOW POINTS
    # Project and plot slow points if available
    if slow_point_search and all_slow_points is not None:
        proj_slow = project_points_to_pca(pca, all_slow_points)
        pca_results["proj_slow"] = proj_slow
        
        # Plot PCA with slow points - all trajectories (static matplotlib)
        plot_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="slow",
            all_points=all_slow_points,
            proj_points=proj_slow,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
        
        # Create interactive plot - all trajectories (Plotly)
        plot_interactive_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="slow",
            all_points=all_slow_points,
            proj_points=proj_slow,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
        
        # Plot subset of trajectories (first, middle, last: indices 0, 25, 50)
        subset_indices = [0, 25, 50]
        subset_proj_trajs = [proj_trajs[i] for i in subset_indices if i < len(proj_trajs)]
        subset_slow_points = [all_slow_points[i] for i in subset_indices if i < len(all_slow_points)]
        
        if subset_proj_trajs and subset_slow_points:
            subset_proj_slow = project_points_to_pca(pca, subset_slow_points)
            
            # Plot subset with slow points (static matplotlib)
            plot_pca_trajectories_and_points(
                pca, subset_proj_trajs,
                point_type="slow",
                all_points=subset_slow_points,
                proj_points=subset_proj_slow,
                params=params,
                skip_initial_steps=skip_initial_steps,
                apply_tanh=apply_tanh,
                filename_prefix="subset_of_indices_"
            )
            
            # Create interactive subset plot with modified filename
            subset_fig = plot_interactive_pca_trajectories_and_points(
                pca, subset_proj_trajs,
                point_type="slow",
                all_points=subset_slow_points,
                proj_points=subset_proj_slow,
                params=params,
                skip_initial_steps=skip_initial_steps,
                apply_tanh=apply_tanh,
                filename_prefix="subset_of_indices_"
            )
    

    # ========================================
    # SAVE PCA RESULTS
    tanh_suffix = "_tanh" if apply_tanh else ""
    results_filename = f"pca_results_skip_{skip_initial_steps}{tanh_suffix}"
    save_variable(pca_results, results_filename)
    
    return pca_results




# ================================================================================
# RUN THIS FILE INDEPENDENTLY
# ================================================================================
if __name__ == "__main__":
    # ========================================
    # MANUAL FILEPATH SPECIFICATION
    # ========================================
    # Base directory for the experiment
    base_dir = "/Users/gianlucacarrozzo/Documents/University and Education/UCL/Machine Learning/MSc Project/Palmigiano Lab/Code/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/JAX_Refactored_Outputs_20250616_103938"
    
    # REQUIRED FILES - Update these paths to your specific files
    # State trajectories file (required)
    state_file = f"{base_dir}/state_20250616_103938_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
    
    # Model parameter files (required)
    J_param_file = f"{base_dir}/J_param_20250616_103938_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"

    # OPTIONAL FILES - Set to None if not available or update paths
    # Fixed points file (optional)
    fixed_points_file = f"{base_dir}/all_fixed_points_20250616_103938_Neuron_Number_200_Task_Number_51_Time_Steps_0.02_Driving_Time_8.0_Training_Time_64.0_Sparsity_0.0_Adam_Epochs_1000_LBFGS_Epochs_2000.pkl"
    
    # Slow points file (optional)
    slow_points_file = None

    # Create a new subfolder 'Further_PCA' in the current directory
    further_pca_dir = os.path.join(base_dir, 'Further_PCA')
    # Create the directory if it doesn't exist
    if not os.path.exists(further_pca_dir):
        os.makedirs(further_pca_dir)
        print(f"Created new directory: {further_pca_dir}")
    else:
        print(f"Using existing directory: {further_pca_dir}")
    # Set the custom output directory for saving new results
    set_custom_output_dir(further_pca_dir)


    # ========================================
    # LOAD DATA FROM SPECIFIED FILES
    # ========================================    
    try:
        # Load trajectory states (required)
        print(f"Loading state trajectories from: {state_file}")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State trajectory file not found: {state_file}")
        state_states = load_variable(state_file)
        state_traj_states = state_states["traj_states"]
        
        # Load model parameters (required)
        params = {}
        param_files = {
            'J': J_param_file
        }
        
        for param_name, param_file in param_files.items():
            print(f"Loading {param_name} parameters from: {param_file}")
            if not os.path.exists(param_file):
                raise FileNotFoundError(f"{param_name} parameter file not found: {param_file}")
            params[param_name] = load_variable(param_file)
        
        # Load fixed points (optional)
        all_fixed_points = None
        if fixed_points_file is not None:
            if os.path.exists(fixed_points_file):
                print(f"Loading fixed points from: {fixed_points_file}")
                all_fixed_points = load_variable(fixed_points_file)
            else:
                print(f"Warning: Fixed points file specified but not found: {fixed_points_file}")
                print("PCA will be performed without fixed points")
        else:
            print("No fixed points file specified - PCA will be performed without fixed points")
        
        # Load slow points (optional)
        all_slow_points = None
        slow_point_search = False
        if slow_points_file is not None:
            if os.path.exists(slow_points_file):
                print(f"Loading slow points from: {slow_points_file}")
                all_slow_points = load_variable(slow_points_file)
                slow_point_search = True
            else:
                print(f"Warning: Slow points file specified but not found: {slow_points_file}")
                print("PCA will be performed without slow points")
        else:
            print("No slow points file specified - PCA will be performed without slow points")
        

        # ========================================
        # RUN PCA WITH SPECIFIED FILES
        # ========================================    
        # Run PCA analysis with different configurations
        print("\n" + "="*60)
        print("RUNNING PCA ANALYSIS")
        print("="*60)

        print(f"\nLoaded data shapes:")
        print(f"  State trajectories: {state_traj_states.shape}")
        if all_fixed_points is not None:
            print(f"  Fixed points: {len(all_fixed_points)} tasks")
        if all_slow_points is not None:
            print(f"  Slow points: {len(all_slow_points)} tasks")
        
        # Configuration options
        skip_options = [0, 200, 400, 600]  # Skip initial time steps - STANDARD IS [0, 200, 400, 600]
        tanh_options = [False, True]  # Apply tanh transformation - STANDARD IS [False, True]
        n_components = 10  # Number of principal components (3D plotting only supports 3 components)
        
        for skip_steps in skip_options:
            for apply_tanh in tanh_options:
                print(f"\n--- Running PCA with skip={skip_steps}, tanh={apply_tanh} ---")
                
                # Run PCA analysis
                pca_results = run_pca_analysis(
                    state_traj_states=state_traj_states,
                    all_fixed_points=all_fixed_points,
                    all_slow_points=all_slow_points,
                    params=params,
                    slow_point_search=slow_point_search,
                    skip_initial_steps=skip_steps,
                    apply_tanh=apply_tanh,
                    n_components=n_components
                )
                
                # Print summary
                explained_variance = pca_results["pca_explained_variance_ratio"]
                total_variance = sum(explained_variance)
                print(f"PCA explained variance ratio: {explained_variance}")
                print(f"Total variance explained by first 3 components: {total_variance:.3f}")
        
        print("\n" + "="*60)
        print("PCA ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Results and plots have been saved to the output directory.")
    

    # ========================================
    # HANDLE ERRORS
    # ========================================
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the specified directory exists and contains the necessary data files.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during PCA analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)