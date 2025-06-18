"""
Visualization functions for plotting trajectories, parameters, Jacobians, and frequencies.
"""


import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from _2_utils import save_figure


# =============================================================================
# TRAJECTORY AND PARAMETER PLOTTING FUNCTIONS
# =============================================================================
def plot_trajectories_vs_targets(params, test_indices=None, sparsity_value=None):
    """
    Plot produced trajectories vs. target signals for selected tasks.
    
    Arguments:
        params: RNN parameters
        test_indices: indices of tasks to plot
        sparsity_value: sparsity level for saving plots in appropriate folder
    """
    from _1_config import omegas, static_inputs, time_train, N
    from _4_rnn_model import simulate_trajectory
    
    if test_indices is None:
        from _1_config import TEST_INDICES
        test_indices = TEST_INDICES
    
    x0_test = jnp.zeros((N,))

    for idx, j in enumerate(test_indices):
        omega = omegas[j]
        u_off = static_inputs[j]

        from _1_config import time_drive, num_steps_train
        # Build input sequences
        u_drive_np = (np.sin(omega * time_drive) + u_off).astype(np.float32).reshape(-1, 1)
        u_train_np = np.full((num_steps_train, 1), u_off, dtype=np.float32)
        # Build target signal
        target_train = np.sin(omega * time_train)

        u_drive = jnp.array(u_drive_np)
        u_train = jnp.array(u_train_np)

        xs_drive_test, _ = simulate_trajectory(x0_test, u_drive, params)
        x_drive_final_test = xs_drive_test[-1]

        _, zs_train_test = simulate_trajectory(x_drive_final_test, u_train, params)
        produced_traj = np.array(zs_train_test)

        fig = plt.figure(figsize=(10, 3))
        plt.plot(time_train, produced_traj, label="Produced Output", linewidth=2)
        plt.plot(time_train, target_train, 'k--', label="Target Signal", linewidth=1.5)
        plt.title(f"Task {j}: ω={omega:.3f}, u_offset={u_off:.3f}")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        # Save the figure in the appropriate directory
        if sparsity_value is not None:
            # For sparsity experiments, save in sparsity-specific folder
            from _2_utils import save_figure_with_sparsity
            save_figure_with_sparsity(fig, f"trajectory_vs_target_task_{j}", sparsity_value)
        else:
            # For regular experiments, save in main folder
            save_figure(fig, f"trajectory_vs_target_task_{j}")
        
        plt.close(fig)


def plot_parameter_matrices(J, B, b_x, j, omega, u_offset, sparsity_value=None):
    """
    Plot the parameter matrices J, B, and b_x for a given task.

    Arguments:
        J        : Jacobian matrix, shape (N, N)
        B        : Input matrix, shape (N, I)
        b_x      : Bias vector, shape (N,)
        j        : Task index
        omega    : Frequency for the task
        u_offset : Static input offset for the task
        sparsity_value : Sparsity level for folder organization and title
    """
    # Create the figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Set the title for the figure including sparsity information
    if sparsity_value is not None:
        fig.suptitle(f'Parameter Matrices (Task {j}, ω={omega:.3f}, u_offset={u_offset:.3f}, Sparsity={sparsity_value:.2f})', fontsize=16)
    else:
        fig.suptitle(f'Parameter Matrices (Task {j}, ω={omega:.3f}, u_offset={u_offset:.3f})', fontsize=16)

    # Plot J matrix
    im = axes[0].imshow(np.array(J), cmap='viridis')
    axes[0].set_title('J Matrix')
    plt.colorbar(im, ax=axes[0])

    # Plot B matrix
    im = axes[1].imshow(np.array(B), cmap='viridis')
    axes[1].set_title('B Matrix')
    plt.colorbar(im, ax=axes[1])

    # Plot b_x vector
    im = axes[2].imshow(np.array(b_x).reshape(-1, 1), cmap='viridis')
    axes[2].set_title('b_x Vector')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    
    # Save the figure in the appropriate directory
    if sparsity_value is not None:
        # For sparsity experiments, save in sparsity-specific folder
        from _2_utils import save_figure_with_sparsity
        save_figure_with_sparsity(fig, f"parameter_matrices_task_{j}", sparsity_value)
    else:
        # For regular experiments, save in main folder
        save_figure(fig, f"parameter_matrices_task_{j}")
    
    plt.close(fig)


def plot_parameter_matrices_for_tasks(params, test_indices=None, sparsity_value=None):
    """
    Plot parameter matrices for selected tasks.
    
    Arguments:
        params: RNN parameters
        test_indices: indices of tasks to plot
        sparsity_value: sparsity level for saving plots in appropriate folder
    """
    from _1_config import omegas, static_inputs
    
    if test_indices is None:
        from _1_config import TEST_INDICES_EXTREMES
        test_indices = TEST_INDICES_EXTREMES
    
    print("\nPlotting parameter matrices for selected tasks...")
    for j in test_indices:
        omega = omegas[j]
        u_off = static_inputs[j]
        plot_parameter_matrices(params["J"], params["B"], params["b_x"], j, omega, u_off, sparsity_value)


# =============================================================================
# JACOBIAN AND FREQUENCY PLOTTING FUNCTIONS
# =============================================================================
def plot_jacobian_matrices(jacobians, j):
    """
    Plot the Jacobian matrices for a given task.
    """
    from _1_config import omegas, s
    
    # Create the figure and subplots
    fig, axes = plt.subplots(len(jacobians), 1, figsize=(10, 5 * len(jacobians)))
    
    # Ensure axes is iterable when there's only one Jacobian (and hence a single subplot)
    if len(jacobians) == 1:
        axes = [axes]
    # Iterate over the subplots
    for i, J_eff in enumerate(jacobians):
        im = axes[i].imshow(J_eff, cmap='viridis')
        axes[i].set_title(f'Jacobian Matrix (Task {j}, ω={omegas[j]:.3f}, FP{i+1}, Sparsity={s:.2f})')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    # Save the figure
    sparsity_str = f"{s:.2f}".replace('.', 'p')
    save_figure(fig, f"Jacobian_Matrices_for_Task_{j}_Sparsity_{sparsity_str}")
    
    plt.show()
    plt.close()


def plot_unstable_eigenvalues(unstable_freqs, j):
    """
    Plot the unstable eigenvalues for a given task on the complex plane.
    """
    from _1_config import omegas, COLORS, MARKERS, s
    
    # Create the figure
    fig = plt.figure(figsize=(10, 8))

    # Plot each fixed point's unstable eigenvalues
    for i, freqs in enumerate(unstable_freqs):
        if freqs:   # check that there are unstable frequencies
            freqs_array = np.array(freqs)
            plt.scatter(np.real(freqs_array), np.imag(freqs_array),
                        color=COLORS[i], marker=MARKERS[i], label=f'FP{i+1}')
    
    # Add decorations
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Unstable Eigenvalues for Task {j} (ω={omegas[j]:.3f}, Sparsity={s:.2f})')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    sparsity_str = f"{s:.2f}".replace('.', 'p')
    save_figure(fig, f"Unstable_Eigenvalues_for_Task_{j}_Sparsity_{sparsity_str}")
    
    plt.show()
    plt.close()


def plot_frequency_comparison(all_unstable_eig_freq, omegas):
    """
    Plot a comparison of target frequencies and maximum unstable mode frequencies (imaginary components) for all tasks.

    Notable arguments:
        all_unstable_eig_freq: list of lists of lists, where the outer list is over tasks,
            the middle list is over fixed points,
            and the inner list contains unstable eigenvalues for each fixed point
    """
    # Import sparsity value from config
    from _1_config import s
    
    # Create the figure and axes
    fig = plt.figure(figsize=(12, 6))

    # Initialize lists to store data for plotting
    task_numbers = []
    target_frequencies = []
    unstable_frequencies = []

    # Collect data for all tasks
    for j, task_freqs in enumerate(all_unstable_eig_freq):
        target_freq = omegas[j]
        # Find the maximum imaginary component across all fixed points in this task
        max_imag = 0
        for fp_freqs in task_freqs:
            if fp_freqs:
                current_max = max(abs(np.imag(ev)) for ev in fp_freqs)
                max_imag = max(max_imag, current_max)
        if max_imag > 0:    # only consider tasks with unstable frequencies
            task_numbers.append(j)
            target_frequencies.append(target_freq)
            unstable_frequencies.append(max_imag)
    
    # Plot the data
    plt.scatter(task_numbers, target_frequencies, color='black', label='Target Frequency', s=100)
    plt.scatter(task_numbers, unstable_frequencies, color='red', label='Max Unstable Mode Frequency', s=100)

    # Add decorations
    plt.xlabel('Task #')
    plt.ylabel('Frequency')
    plt.title(f'Comparison of Target Frequencies and Maximum Unstable Mode Frequencies (Sparsity={s:.2f})')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    sparsity_str = f"{s:.2f}".replace('.', 'p')
    save_figure(fig, f"Frequency_Comparison_Sparsity_{sparsity_str}")
    
    plt.show()
    plt.close()


# =============================================================================
# PLOTTING EXECUTION FUNCTIONS
# =============================================================================
def analyze_jacobians_visualization(point_type="fixed", all_jacobians=None):
    """
    Unified function to plot Jacobian matrices and check for equal Jacobians for either fixed points or slow points.
    
    Arguments:
        point_type      : str, either "fixed" or "slow" to specify which type of points
        all_jacobians   : list of lists of Jacobians, one list per task
    """
    from _1_config import TEST_INDICES, num_tasks, JACOBIAN_TOL, omegas
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    print(f"\n{point_name}: Plotting Jacobian matrices for selected tasks...")
    for j in TEST_INDICES:
        omega_j = omegas[j]
        if all_jacobians[j]:    # Check if there are any Jacobians for this task
            plot_jacobian_matrices(all_jacobians[j], j)

    print(f"\n{point_name}: Checking for Equal Jacobians within Tasks:")
    for j in range(num_tasks):
        task_jacobians = all_jacobians[j]
        num_jacobians = len(task_jacobians)
        if num_jacobians > 1:   # Only check this if there are multiple Jacobians for this task
            # Track which Jacobians have been matched
            matched = [False] * num_jacobians
            equal_groups = []
            for i in range(num_jacobians):
                if not matched[i]:  # Skip if this Jacobian has already been matched
                    current_group = [i]
                    matched[i] = True
                    for k in range(i + 1, num_jacobians):
                        if not matched[k]:
                            # Check if the Jacobians are equal within the tolerance
                            if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
                                current_group.append(k)
                                matched[k] = True
                    if len(current_group) > 1:  # Only add groups with more than one equal Jacobian
                        equal_groups.append(current_group)
            if equal_groups:
                print(f"\nTask {j}:")
                print(f"  Total number of Jacobians: {num_jacobians}")
                for group_idx, group in enumerate(equal_groups):
                    print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")


def analyze_unstable_frequencies_visualization(point_type="fixed", all_unstable_eig_freq=None):
    """
    Unified function to plot unstable eigenvalues and frequency comparisons for either fixed points or slow points.
    
    Arguments:
        point_type              : str, either "fixed" or "slow" to specify which type of points
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies
    """
    from _1_config import TEST_INDICES, omegas
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point name based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    
    print(f"\n{point_name}: Plotting unstable eigenvalues for selected tasks...")
    for j in TEST_INDICES:
        omega_j = omegas[j]
        if all_unstable_eig_freq[j]:
            plot_unstable_eigenvalues(all_unstable_eig_freq[j], j)

    print(f"\n{point_name}: Plotting comparisons between target frequencies and unstable mode frequencies...")
    plot_frequency_comparison(all_unstable_eig_freq, omegas)