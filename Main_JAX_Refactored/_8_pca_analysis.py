"""
PCA analysis and visualization for trajectory and fixed point analysis.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
from scipy.linalg import eig
from _2_utils import save_figure, save_variable
from _6_analysis import compute_jacobian


def perform_pca_analysis(state_traj_states, skip_initial_steps=0, apply_tanh=False):
    """
    Perform PCA on trajectory states and return PCA object and projections.
    
    Arguments:
        state_traj_states: trajectory states from training, shape (num_tasks, num_steps_train+1, N)
        skip_initial_steps: number of initial time steps to skip (default: 0)
        apply_tanh: whether to apply tanh transformation to states before PCA (default: False)
        
    Returns:
        pca: fitted PCA object
        proj_trajs: list of projected trajectories
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

    # Perform PCA with 3 components
    pca = PCA(n_components=3)
    proj_all = pca.fit_transform(all_states)

    # Project each trajectory into the PCA space
    proj_trajs = []
    start = 0
    for traj in tqdm(state_traj_states_truncated, desc="Projecting trajectories"):
        T = traj.shape[0]   # Number of time steps in the trajectory
        proj_traj = proj_all[start:start + T]   # Extracts the projected trajectory from the PCA space, resulting shape is (T, 3)
        proj_trajs.append(proj_traj)
        start += T

    # Calculate PCA computation time
    pca_time = time.time() - start_time
    print(f"\nPCA computation completed in {pca_time:.2f} seconds")
    
    return pca, proj_trajs


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
        proj_points = pca.transform(all_points_flat)   # Projects all points from all tasks into the PCA space, resulting shape is (num_total_points, 3)
    else:
        proj_points = np.empty((0, 3))   # Empty array with correct shape for plotting
    
    return proj_points


def plot_pca_trajectories_and_points(pca, proj_trajs, point_type="fixed", all_points=None, proj_points=None, params=None, skip_initial_steps=0, apply_tanh=False):
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
    """
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    J_trained = np.array(params["J"])

    # Create the figure and axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the projected trajectories in blue
    for traj in proj_trajs:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.5)

    # Plot the projected points as green scatter points
    if proj_points.size > 0:
        ax.scatter(proj_points[:, 0], proj_points[:, 1], proj_points[:, 2], color='green', s=50, label=point_name)

    # For each point, plot all unstable modes as red lines
    point_idx = 0
    for j, task_points in enumerate(all_points):
        for x_star in task_points:
            # Compute the Jacobian and find the eigenvalues at the point
            J_eff = np.array(compute_jacobian(x_star, J_trained))
            eigenvals, eigenvecs = eig(J_eff)
            # Find all the unstable eigenvalues for the given point
            unstable_idx = np.where(np.real(eigenvals) > 0)[0]
            if len(unstable_idx) > 0:
                # For each unstable eigenvalue, plot the corresponding eigenvector direction
                for idx in unstable_idx:
                    # Take the real part for the plotting direction
                    v = np.real(eigenvecs[:, idx])
                    scale = 0.5
                    # Project the unstable eigenvector direction into PCA space
                    v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_points[point_idx]
                    # Plot a line centred on the point in PCA space
                    line = np.vstack([
                        proj_points[point_idx] - v_proj,
                        proj_points[point_idx] + v_proj
                    ])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='red', linewidth=2, alpha=0.5)
            point_idx += 1

    # Add decorations
    title_skip_info = f" (skipping first {skip_initial_steps} steps)" if skip_initial_steps > 0 else ""
    title_tanh_info = " (tanh transformed)" if apply_tanh else ""
    ax.set_title(f'PCA of Network Trajectories and {point_name}{title_skip_info}{title_tanh_info}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()

    # Save the figure
    tanh_suffix = "_tanh" if apply_tanh else ""
    figure_name = f"pca_plot_{point_type}_skip_{skip_initial_steps}{tanh_suffix}"
    save_figure(fig, figure_name)

    plt.close()


def run_pca_analysis(state_traj_states, all_fixed_points=None, all_slow_points=None, params=None, slow_point_search=False, skip_initial_steps=0, apply_tanh=False):
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
        
    Returns:
        pca_results: dictionary containing PCA results
    """
    # Perform PCA on trajectories
    pca, proj_trajs = perform_pca_analysis(state_traj_states, skip_initial_steps=skip_initial_steps, apply_tanh=apply_tanh)
    
    pca_results = {
        "proj_trajs": proj_trajs,
        "skip_initial_steps": skip_initial_steps,
        "apply_tanh": apply_tanh,
        "pca_components": pca.components_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    
    # Project and plot fixed points if available
    if all_fixed_points is not None:
        proj_fixed = project_points_to_pca(pca, all_fixed_points)
        pca_results["proj_fixed"] = proj_fixed
        
        # Plot PCA with fixed points
        plot_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="fixed",
            all_points=all_fixed_points,
            proj_points=proj_fixed,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
    
    # Project and plot slow points if available
    if slow_point_search and all_slow_points is not None:
        proj_slow = project_points_to_pca(pca, all_slow_points)
        pca_results["proj_slow"] = proj_slow
        
        # Plot PCA with slow points
        plot_pca_trajectories_and_points(
            pca, proj_trajs,
            point_type="slow",
            all_points=all_slow_points,
            proj_points=proj_slow,
            params=params,
            skip_initial_steps=skip_initial_steps,
            apply_tanh=apply_tanh
        )
    
    # Save PCA results
    tanh_suffix = "_tanh" if apply_tanh else ""
    results_filename = f"pca_results_skip_{skip_initial_steps}{tanh_suffix}"
    save_variable(pca_results, results_filename)
    
    return pca_results