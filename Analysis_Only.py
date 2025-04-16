# -------------------------------
# -------------------------------
# 0. IMPORTS
# -------------------------------
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.optimize import root
from scipy.linalg import eig

import os
from tqdm import tqdm
import time
from datetime import datetime
import pickle




# -------------------------------
# -------------------------------
# 1. RNN SETUP
# -------------------------------
# -------------------------------

# -------------------------------
# 1.1. Set Random Seed & Parameters
# -------------------------------

np.random.seed(42)
torch.manual_seed(42)

# Network dimensions and task parameters (CHANGE THESE FOR TEST RUNS)
N = 200         # number of neurons (STANDARD VALUE IS 200)
I = 1           # input dimension (scalar input)
num_tasks = 51  # number of different sine-wave tasks (STANDARD VALUE IS 51)

# Frequencies: equally spaced between 0.1 and 0.6 rad/s
omegas = np.linspace(0.1, 0.6, num_tasks)

# Static input offset for each task: j/51 + 0.25, j=0,...,50
static_inputs = np.linspace(0, num_tasks-1, num_tasks) / num_tasks + 0.25

# Time parameters (in seconds) (NEED TO CHOOSE THESE CAREFULLY)
dt = 0.02        # integration time step
T_drive = 4.0   # driving phase duration (to set network state)
T_train = 24.0   # training phase duration with static input (target generation) (OMEGA = 0.1 NEEDS 63 SECONDS TO GO THROUGH A WHOLE CYCLE)
num_steps_drive = int(T_drive/dt)
num_steps_train = int(T_train/dt)
time_drive = np.arange(0, T_drive, dt)
time_train = np.arange(0, T_train, dt)
time_full  = np.concatenate([time_drive, T_drive + time_train])




# -------------------------------
# 1.2. Define RNN Parameters
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# -------------------------------
# 1.3. Useful Global Variables
# -------------------------------

BATCH_SIZE = 10

# Optimizer parameters
OPTIMIZER_LR = 0.6
OPTIMIZER_MAX_ITER = 10
OPTIMIZER_HISTORY_SIZE = 10
OPTIMIZER_LINE_SEARCH_FN = "strong_wolfe"

# Training parameters
NUM_EPOCHS = 50
LOSS_THRESHOLD = 1e-4

# Plotting parameters
TEST_INDICES = np.linspace(0, omegas.shape[0] - 1, 5, dtype=int)
TEST_INDICES_EXTREMES = [0, omegas.shape[0] - 1]
COLORS = ['b', 'g', 'r', 'c', 'm']
MARKERS = ['o', 's', '^', 'v', '<']

# Jacobians, fixed points, slow points, and unstable mode frequencies parameters
NUM_ATTEMPTS = 10
TOL = 1e-2
MAXITER = 1000
NUM_ATTEMPTS_SLOW = 10
TOL_SLOW = 1e-2
MAXITER_SLOW = 1000
JACOBIAN_TOL = 1e-2
EVALUE_TOL = 1e-3
SEARCH_BATCH_SIZE = 5

# Create output directory for saving files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), 'Outputs', f'Outputs_{timestamp}')
os.makedirs(output_dir, exist_ok=True)




# -------------------------------
# 1.4. Saving Functions
# -------------------------------

def generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train):
    """
    Generate a filename with timestamp and parameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{variable_name}_{timestamp}_Neuron_Number_{N}_Task_Number_{num_tasks}_Time_Steps_{dt}_Driving_Time_{T_drive}_Training_Time_{T_train}.pkl"


def save_variable(variable, variable_name, N, num_tasks, dt, T_drive, T_train, output_dir=None):
    """
    Save a variable to a pickle file with a descriptive filename in the Outputs folder.
    """
    # Use the current working directory to build the base directory for saving files
    base_dir = os.getcwd()

    # Create timestamp in the same format as generate_filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Allow the caller to specify an output directory; default to a subfolder 'Outputs_{timestamp}' in the current working directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'Outputs',f'Outputs_{timestamp}')
    else:
        output_dir = os.path.join(base_dir, output_dir)

    # Create the output directory if it doesn't exist (exist_ok avoids error if it does)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train)
    filepath = os.path.join(output_dir, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")




# -------------------------------
# -------------------------------
# 2. RNN TRAINING
# -------------------------------
# -------------------------------

# -------------------------------
# 2.1. Trajectory Simulation Function
# -------------------------------




# -------------------------------
# 2.2. Training Run Function
# -------------------------------




# -------------------------------
# 2.3. Execute Training Procedure
# -------------------------------

# Define class to store trajectories during the training phase and the final point from driving phase
class TrainingState:
    def __init__(self):
        self.traj_states = []   # list of length num_tasks, each element is a (num_steps_train, N) tensor
        self.fixed_point_inits = []   # list of length num_tasks, each element is a (N,) tensor

# Instance of the training state class
state = TrainingState()




# -------------------------------
# 2.4. Save Results from Training Procedure
# -------------------------------




# -------------------------------
# 2.5. (EXTRA) Save Results from Training Procedure
# -------------------------------

# Import J_param, B_param, b_x_param, w_param, b_z_param, and state from the saved file

# Import the state file into the state instance




# -------------------------------
# -------------------------------
# 3. DYNAMICS RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 3.1. Produced Trajectories vs. Target Signals Plots
# -------------------------------




# -------------------------------
# 3.2. Parameter Matrices Plots
# -------------------------------




# -------------------------------
# -------------------------------
# 4. FIXED POINTS, SLOW POINTS, AND UNSTABLE MODES RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 4.1. Jacobians, Fixed Points, Slow Points, and Unstable Mode Frequencies Functions
# -------------------------------

def fixed_point_func(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute f(x) = -x + J*tanh(x) + B*u + b_x for given x and fixed u.
    Vectorized implementation for better performance.

    Arguments:
        x_np: Input vector (numpy array, shape (N,))
        u_val: Constant input value (float)
        J_np: Recurrent weight matrix (numpy array, shape (N, N))
        B_np: Input weight matrix (numpy array, shape (N, I))
        b_x_np: Bias vector (numpy array, shape (N,))
    
    Returns:
        f_x: Output vector (numpy array, shape (N,))
    """
    x = x_np
    # Note that np.array([u_val]) has shape (1,), so the dot product has shape (N,1), which means that it must be flattened to (N,)
    return -x + np.dot(J_np, np.tanh(x)) + np.dot(B_np, np.array([u_val])).flatten() + b_x_np


def slow_point_func(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute q(x) = 0.5 * F(x) \cdot F(x), where F(x) = -x + J*tanh(x) + B*u + b_x
    
    Arguments:
        x_np: Input vector (numpy array, shape (N,))
        u_val: Constant input value (float)
        J_np: Recurrent weight matrix (numpy array, shape (N, N))
        B_np: Input weight matrix (numpy array, shape (N, I))
        b_x_np: Bias vector (numpy array, shape (N,))
    
    Returns:
        q_x: Output scalar (float)
    """
    F_x = fixed_point_func(x_np, u_val, J_np, B_np, b_x_np)
    return 0.5 * np.dot(F_x, F_x)


def jacobian_fixed_point(x_star, J_np):
    """
    Compute Jacobian: J_eff = -I + J * diag(1 - tanh(x_star)^2).
    Vectorized implementation for better performance.

    Arguments:
        x_star: Fixed point (numpy array, shape (N,))
        J_np: Recurrent weight matrix (numpy array, shape (N, N))
    
    Returns:
        J_eff: Jacobian matrix evaluated at x_star (numpy array, shape (N, N))
    """
    diag_term = 1 - np.tanh(x_star)**2  # This gives a vector of derivaties of shape (N,)
    # Note that [np.newaxis, :] reshapes the vector to a row vector of shape (1, N) so that it can be broadcast
    return -np.eye(len(x_star)) + J_np * diag_term[np.newaxis, :]


def find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS, tol=TOL):
    """
    Find multiple fixed points for a given task by trying different initial conditions.
    
    Arguments:
        x0_guess: Initial guess for fixed point for the given task
        u_const: Constant input for the given task
        J_trained, B_trained, b_x_trained: Network parameters
        num_attempts: Number of different initial conditions to try
        tol: Tolerance for considering two fixed points as distinct
        
    Returns:
        List of distinct fixed points found for the given task
    """
    fixed_points = []
    
    # Try the original initial condition
    try:
        sol = root(fixed_point_func, x0_guess, args=(u_const, J_trained, B_trained, b_x_trained),
                  method='lm', options={'maxiter': MAXITER})
        if sol.success:
            fixed_points.append(sol.x)
    except Exception as e:
        print(f"Fixed point search failed for initial guess: {str(e)}")
    
    # Try perturbed initial conditions
    for attempt in tqdm(range(num_attempts - 1), desc="Finding fixed points", leave=False):
        # Create a perturbed initial condition
        x0_perturbed = x0_guess + np.random.normal(0, 0.5, size=x0_guess.shape) # 0.5 is the default for this scenario
        try:
            sol = root(fixed_point_func, x0_perturbed, args=(u_const, J_trained, B_trained, b_x_trained),
                      method='lm', options={'maxiter': MAXITER})
            if sol.success:
                # Check if this fixed point is distinct from previous ones
                is_distinct = True
                for fp in fixed_points:
                    if np.linalg.norm(sol.x - fp) < tol:
                        is_distinct = False
                        break
                if is_distinct:
                    fixed_points.append(sol.x)
        except Exception as e:
            print(f"Fixed point search failed for attempt {attempt+1}: {str(e)}")
    
    return fixed_points


def find_slow_points(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=NUM_ATTEMPTS_SLOW, tol=TOL_SLOW):
    """
    Find multiple slow points for a given task by trying different initial conditions.

    Arguments:
        x0_guess: Initial guess for fixed point for the given task
        u_const: Constant input for the given task
        J_trained, B_trained, b_x_trained: Network parameters
        num_attempts: Number of different initial conditions to try
        tol: Tolerance for considering two fixed points as distinct
        
    Returns:
        List of distinct slow points found for the given task
    """
    slow_points = []
    
    # Try the original initial condition
    try:
        sol = root(slow_point_func, x0_guess, args=(u_const, J_trained, B_trained, b_x_trained),
                  method='lm', options={'maxiter': MAXITER_SLOW})
        if sol.success:
            slow_points.append(sol.x)
    except Exception as e:
        print(f"Slow point search failed for initial guess: {str(e)}")
    
    # Try perturbed initial conditions
    for attempt in tqdm(range(num_attempts - 1), desc="Finding slow points", leave=False):
        # Create a perturbed initial condition
        x0_perturbed = x0_guess + np.random.normal(0, 0.5, size=x0_guess.shape) # 0.5 is the default for this scenario
        try:
            sol = root(slow_point_func, x0_perturbed, args=(u_const, J_trained, B_trained, b_x_trained),
                      method='lm', options={'maxiter': MAXITER_SLOW})
            if sol.success:
                # Check if this slow point is distinct from previous ones
                is_distinct = True
                for sp in slow_points:
                    if np.linalg.norm(sol.x - sp) < tol:
                        is_distinct = False
                        break
                if is_distinct:
                    slow_points.append(sol.x)
        except Exception as e:
            print(f"Slow point search failed for attempt {attempt+1}: {str(e)}")
    
    return slow_points


def analyze_fixed_points(fixed_points, J_trained):
    """
    Analyze fixed points for a given task and compute their properties in a memory-efficient way.

    Arguments:
        fixed_points: List of fixed points (numpy array, shape (N,))
        J_trained: Recurrent weight matrix (numpy array, shape (N, N))
        static_inputs: List of constant input values (numpy array, shape (I,))
    
    Returns:
        jacobians: List of Jacobian matrices of length len(fixed_points) (where each element is a numpy array, shape (N, N))
        unstable_freqs: List of lists of unstable eigenvalues, where each inner list contains all unstable eigenvalues for a fixed point
    """
    jacobians = []
    unstable_freqs = []
    
    for x_star in fixed_points:
        # Compute Jacobian at the fixed point
        J_eff = jacobian_fixed_point(x_star, J_trained)
        jacobians.append(J_eff)
        
        # Compute eigenvalues
        eigenvals, _ = eig(J_eff)
        
        # Find all unstable eigenvalues for the given fixed point (real part > 0)
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        if len(unstable_idx) > 0:
            # Store all unstable eigenvalues for this fixed point
            unstable_freqs.append(list(eigenvals[unstable_idx]))
        else:
            # If no unstable eigenvalues, append an empty list
            unstable_freqs.append([])
    
    return jacobians, unstable_freqs


def plot_unstable_eigenvalues(unstable_freqs, j, omega, save_dir=None):
    """
    Plot unstable eigenvalues on the complex plane for a specific task.

    Arguments:
        unstable_freqs: List of lists of unstable eigenvalues for the task (where each inner list contains eigenvalues for a fixed point)
        j: Task index
        omega: Angular frequency
        save_dir: Directory to save the plot (if None, plot is only displayed)
    """
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Plot each fixed point's eigenvalues
    for i, freqs in enumerate(unstable_freqs):
        if freqs:  # if there are any unstable eigenvalues
            # Convert to numpy array for easier handling
            freqs_array = np.array(freqs)
            # Plot real vs imaginary parts
            plt.scatter(np.real(freqs_array), np.imag(freqs_array), 
                       color=colors[i], marker=markers[i], 
                       label=f'FP{i+1}')
    
    # Add reference lines and labels
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Unstable Eigenvalues for Task {j} (ω={omega:.3f})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if a directory is provided
    if save_dir is not None:
        filename = generate_filename(f"unstable_eigenvalues_task_{j}", N, num_tasks, dt, T_drive, T_train)
        filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
        plt.savefig(filepath)
        print(f"Saved unstable eigenvalues plot for task {j} to {filepath}")
    
    plt.show()


def plot_jacobian_matrices(jacobians, j, omega, save_dir=None):
    """
    Plot the Jacobian matrices for a specific task.

    Arguments:
        jacobians: List of Jacobian matrices for task j (where each element is a numpy array, shape (N, N))
        j: Task index
        omega: Angular frequency
        save_dir: Directory to save the plot (if None, plot is only displayed)
    """
    # Create the figure and axes
    fig, axes = plt.subplots(len(jacobians), 1, figsize=(10, 5*len(jacobians)))
    
    # Ensure axes is iterable when there's a single subplot
    if len(jacobians) == 1:
        axes = [axes]
    
    # Iterate over the subplots
    for i, J_eff in enumerate(jacobians):
        im = axes[i].imshow(J_eff, cmap='viridis')
        axes[i].set_title(f'Jacobian Matrix (Task {j}, ω={omega:.3f}, FP{i+1})')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir is not None:
        filename = generate_filename(f"jacobian_matrices_task_{j}", N, num_tasks, dt, T_drive, T_train)
        filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
        plt.savefig(filepath)
        print(f"Saved Jacobian matrices plot for task {j} to {filepath}")
    
    plt.show()


def plot_frequency_comparison(all_unstable_eig_freq, omegas, save_dir=None):
    """
    Plot a comparison between target frequencies and unstable eigenvalues' imaginary components.
    
    Arguments:
        all_unstable_eig_freq: List of lists of lists of unstable eigenvalues for all tasks
        omegas: Array of target frequencies for each task
        save_dir: Directory to save the plot (if None, plot is only displayed)
    """
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Initialize lists to store data points
    task_numbers = []
    target_frequencies = []
    unstable_frequencies = []
    
    # Collect data for all tasks
    for j, task_freqs in enumerate(all_unstable_eig_freq):
        target_freq = omegas[j]
        
        # Find the maximum imaginary component across all fixed points in this task
        max_imag = 0
        for fp_freqs in task_freqs:
            if fp_freqs:  # if there are unstable eigenvalues
                current_max = max(abs(np.imag(ev)) for ev in fp_freqs)
                max_imag = max(max_imag, current_max)
        
        if max_imag > 0:  # only add points if there were unstable eigenvalues
            task_numbers.append(j)
            target_frequencies.append(target_freq)
            unstable_frequencies.append(max_imag)
    
    # Plot the data
    plt.scatter(task_numbers, target_frequencies, color='black', label='Target Frequency', s=100)
    plt.scatter(task_numbers, unstable_frequencies, color='red', label='Max Unstable Mode Frequency', s=100)
    
    # Add labels and title
    plt.xlabel('Task #')
    plt.ylabel('Frequency (radians)')
    plt.title('Comparison of Target Frequencies and Maximum Unstable Mode Frequencies')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if a directory is provided
    if save_dir is not None:
        filename = generate_filename("frequency_comparison", N, num_tasks, dt, T_drive, T_train)
        filepath = os.path.join(save_dir, filename.replace('.pkl', '.png'))
        plt.savefig(filepath)
        print(f"Saved frequency comparison plot to {filepath}")
    
    plt.show()




# -------------------------------
# 4.2.1. Fixed Points: Find Jacobians, Fixed Points, and Unstable Mode Frequencies
# -------------------------------

# Extract trained parameters as NumPy arrays
J_trained = J_param.detach().cpu().numpy()
B_trained = B_param.detach().cpu().numpy()
b_x_trained = b_x_param.detach().cpu().numpy()

# Initialize lists to store multiple fixed points and their properties
all_fixed_points = []  # List of lists of fixed points, one list per task, and the inner lists contain the fixed points for each task
all_jacobians = []     # List of lists of Jacobians, one list per task, and the inner lists contain the Jacobians for each fixed point
all_unstable_eig_freq = []  # List of lists of lists of unstable frequencies, one list per task, one middle list per fixed point for that task, and the inner list contains the unstable frequencies for each fixed point

# Track fixed point search time
start_time = time.time()
print("\nStarting fixed point search...")

# Process tasks in batches for memory efficiency
batch_size = BATCH_SIZE  # Adjust based on available memory
for batch_start in range(0, num_tasks, batch_size):
    batch_end = min(batch_start + batch_size, num_tasks)
    
    for j in range(batch_start, batch_end):
        u_const = static_inputs[j]
        x0_guess = state.fixed_point_inits[j]
        
        # Find multiple fixed points
        task_fixed_points = find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained)
        all_fixed_points.append(task_fixed_points)
        
        # Analyze fixed points
        task_jacobians, task_unstable_freqs = analyze_fixed_points(task_fixed_points, J_trained)
        all_jacobians.append(task_jacobians)
        all_unstable_eig_freq.append(task_unstable_freqs)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Calculate search time
fixed_point_time = time.time() - start_time
print(f"\nFixed point search completed in {fixed_point_time:.2f} seconds")


# -------------------------------
# Save Results from Fixed Point Analysis

print("\nSaving results...")

# Save fixed point analysis results
save_variable(all_fixed_points, "all_fixed_points", N, num_tasks, dt, T_drive, T_train, output_dir)
save_variable(all_jacobians, "all_jacobians", N, num_tasks, dt, T_drive, T_train, output_dir)
save_variable(all_unstable_eig_freq, "all_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train, output_dir)




# -------------------------------
# 4.2.2. Slow Points: Find Jacobians, Slow Points, and Unstable Mode Frequencies
# -------------------------------

# Extract trained parameters as NumPy arrays
J_trained = J_param.detach().cpu().numpy()
B_trained = B_param.detach().cpu().numpy()
b_x_trained = b_x_param.detach().cpu().numpy()

# Initialize lists to store multiple fixed points and their properties
all_slow_points = []  # List of lists of slow points, one list per task, and the inner lists contain the slow points for each task
all_slow_jacobians = []     # List of lists of Jacobians, one list per task, and the inner lists contain the Jacobians for each slow point
all_slow_unstable_eig_freq = []  # List of lists of lists of unstable frequencies, one list per task, one middle list per slow point for that task, and the inner list contains the unstable frequencies for each slow point

# Track slow point search time
start_time = time.time()
print("\nStarting slow point search...")

# Process tasks in batches for memory efficiency
batch_size = BATCH_SIZE  # Adjust based on available memory
for batch_start in range(0, num_tasks, batch_size):
    batch_end = min(batch_start + batch_size, num_tasks)
    
    for j in range(batch_start, batch_end):
        u_const = static_inputs[j]
        x0_guess = state.fixed_point_inits[j]
        
        # Find multiple slow points
        task_slow_points = find_slow_points(x0_guess, u_const, J_trained, B_trained, b_x_trained)
        all_slow_points.append(task_slow_points)
        
        # Analyze slow points
        task_jacobians, task_unstable_freqs = analyze_fixed_points(task_slow_points, J_trained)
        all_slow_jacobians.append(task_jacobians)
        all_slow_unstable_eig_freq.append(task_unstable_freqs)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Calculate search time
slow_point_time = time.time() - start_time
print(f"\nSlow point search completed in {slow_point_time:.2f} seconds")


# -------------------------------
# Save Results from Slow Point Analysis

print("\nSaving results...")

# Save fixed point analysis results
save_variable(all_slow_points, "all_slow_points", N, num_tasks, dt, T_drive, T_train, output_dir)
save_variable(all_slow_jacobians, "all_slow_jacobians", N, num_tasks, dt, T_drive, T_train, output_dir)
save_variable(all_slow_unstable_eig_freq, "all_slow_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train, output_dir)




# -------------------------------
# 4.3.1. Fixed Points: Fixed Point Summaries
# -------------------------------

print("\nSummary of Fixed Points Found:")

# Initialize dictionaries to store counts
fixed_points_count = {}  # key: number of fixed points, value: count of tasks with that many fixed points
unstable_eigs_count = {}  # key: number of unstable eigenvalues, value: count of fixed points with that many unstable eigenvalues


# 1. Iterate through all tasks and print basic information
print("\n1. Fixed Points per Task:")
for j in range(num_tasks):
    # Count the number of fixed points found for this task
    num_fps = len(all_fixed_points[j])
    print(f"Task {j}: {num_fps} fixed points found")
    
    # Update the count for this number of fixed points
    fixed_points_count[num_fps] = fixed_points_count.get(num_fps, 0) + 1
    
    # Count unstable eigenvalues for each fixed point in this task
    for fp_freqs in all_unstable_eig_freq[j]:
        num_unstable = len(fp_freqs)
        # Update the count for this number of unstable eigenvalues
        unstable_eigs_count[num_unstable] = unstable_eigs_count.get(num_unstable, 0) + 1


# 2. Print summary of tasks grouped by number of fixed points
print("\n2. Tasks Grouped by Number of Fixed Points:")
for num_fps in sorted(fixed_points_count.keys()):
    count = fixed_points_count[num_fps]
    print(f"  {count} tasks have {num_fps} fixed point{'s' if num_fps != 1 else ''}")


# 3. Print summary of fixed points grouped by number of unstable eigenvalues
print("\n3. Fixed Points Grouped by Number of Unstable Eigenvalues:")
for num_unstable in sorted(unstable_eigs_count.keys()):
    count = unstable_eigs_count[num_unstable]
    print(f"  {count} fixed point{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")


# Select the tasks to summarise
test_js = TEST_INDICES
# Print detailed information for selected tasks
print("\nDetailed Information for Selected Tasks:")
for j in test_js:
    print(f"\nTask {j}:")
    print(f"Number of distinct fixed points found: {len(all_fixed_points[j])}")
    for i, (fp, freqs) in enumerate(zip(all_fixed_points[j], all_unstable_eig_freq[j])):
        print(f"  Fixed point {i+1}:")
        print(f"    Number of unstable eigenvalues: {len(freqs)}")
        if len(freqs) > 0:
            print("    Unstable eigenvalues:")
            for ev in freqs:
                print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
        print(f"    Norm: {np.linalg.norm(fp):.4f}")




# -------------------------------
# 4.3.2. Slow Points: Slow Point Summaries
# -------------------------------

print("\nSummary of Slow Points Found:")

# Initialize dictionaries to store counts
slow_points_count = {}  # key: number of slow points, value: count of tasks with that many slow points
slow_unstable_eigs_count = {}  # key: number of unstable eigenvalues, value: count of slow points with that many unstable eigenvalues


# 1. Iterate through all tasks and print basic information
print("\n1. Slow Points per Task:")
for j in range(num_tasks):
    # Count the number of slow points found for this task
    num_sps = len(all_slow_points[j])
    print(f"Task {j}: {num_sps} slow points found")
    
    # Update the count for this number of slow points
    slow_points_count[num_sps] = slow_points_count.get(num_sps, 0) + 1
    
    # Count unstable eigenvalues for each slow point in this task
    for sp_freqs in all_slow_unstable_eig_freq[j]:
        num_unstable = len(sp_freqs)
        # Update the count for this number of unstable eigenvalues
        slow_unstable_eigs_count[num_unstable] = slow_unstable_eigs_count.get(num_unstable, 0) + 1


# 2. Print summary of tasks grouped by number of slow points
print("\n2. Tasks Grouped by Number of Slow Points:")
for num_sps in sorted(slow_points_count.keys()):
    count = slow_points_count[num_sps]
    print(f"  {count} tasks have {num_sps} slow point{'s' if num_sps != 1 else ''}")


# 3. Print summary of slow points grouped by number of unstable eigenvalues
print("\n3. Slow Points Grouped by Number of Unstable Eigenvalues:")
for num_unstable in sorted(slow_unstable_eigs_count.keys()):
    count = slow_unstable_eigs_count[num_unstable]
    print(f"  {count} slow point{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")


# Select the tasks to summarise
test_js = TEST_INDICES
# Print detailed information for selected tasks
print("\nDetailed Information for Selected Tasks:")
for j in test_js:
    print(f"\nTask {j}:")
    print(f"Number of distinct slow points found: {len(all_slow_points[j])}")
    for i, (sp, freqs) in enumerate(zip(all_slow_points[j], all_slow_unstable_eig_freq[j])):
        print(f"  Slow point {i+1}:")
        print(f"    Number of unstable eigenvalues: {len(freqs)}")
        if len(freqs) > 0:
            print("    Unstable eigenvalues:")
            for ev in freqs:
                print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
        print(f"    Norm: {np.linalg.norm(sp):.4f}")




# -------------------------------
# 4.4.1. Fixed Points: Jacobian Plots and Equal Jacobians Check
# -------------------------------

# Select the tasks to plot
test_js = TEST_INDICES
# Plot Jacobian matrices for selected tasks
print("\nPlotting Jacobian matrices for selected tasks...")
for j in test_js:
    omega_j = omegas[j]
    
    if all_jacobians[j]:  # if any fixed points were found
        # Plot Jacobian matrices
        plot_jacobian_matrices(all_jacobians[j], j, omega_j, save_dir=output_dir)


# Check for equal Jacobians within each task
print("\nChecking for Equal Jacobians within Tasks:")
for j in range(num_tasks):
    # Get the Jacobians for this task
    task_jacobians = all_jacobians[j]
    num_jacobians = len(task_jacobians)
    
    if num_jacobians > 1:  # Only check if there are multiple Jacobians
        # Create a list to track which Jacobians have been matched
        matched = [False] * num_jacobians
        equal_groups = []
        
        # Compare each pair of Jacobians
        for i in range(num_jacobians):
            if not matched[i]:  # Skip if this Jacobian has already been matched
                current_group = [i]
                matched[i] = True
                
                for k in range(i+1, num_jacobians):
                    if not matched[k]:
                        # Check if the Jacobians are equal within tolerance
                        if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
                            current_group.append(k)
                            matched[k] = True
                
                if len(current_group) > 1:  # Only store groups with more than one Jacobian
                    equal_groups.append(current_group)
        
        # Print results if any equal groups were found
        if equal_groups:
            print(f"\nTask {j}:")
            print(f"  Total number of Jacobians: {num_jacobians}")
            for group_idx, group in enumerate(equal_groups):
                print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")




# -------------------------------
# 4.4.2. Slow Points: Jacobian Plots and Equal Jacobians Check
# -------------------------------

# Select the tasks to plot
test_js = TEST_INDICES
# Plot Jacobian matrices for selected tasks
print("\nPlotting Jacobian matrices for selected tasks...")
for j in test_js:
    omega_j = omegas[j]
    
    if all_slow_jacobians[j]:  # if any slow points were found
        # Plot Jacobian matrices
        plot_jacobian_matrices(all_slow_jacobians[j], j, omega_j, save_dir=output_dir)


# Check for equal Jacobians within each task
print("\nChecking for Equal Jacobians within Tasks:")
for j in range(num_tasks):
    # Get the Jacobians for this task
    task_jacobians = all_slow_jacobians[j]
    num_jacobians = len(task_jacobians)
    
    if num_jacobians > 1:  # Only check if there are multiple Jacobians
        # Create a list to track which Jacobians have been matched
        matched = [False] * num_jacobians
        equal_groups = []
        
        # Compare each pair of Jacobians
        for i in range(num_jacobians):
            if not matched[i]:  # Skip if this Jacobian has already been matched
                current_group = [i]
                matched[i] = True
                
                for k in range(i+1, num_jacobians):
                    if not matched[k]:
                        # Check if the Jacobians are equal within tolerance
                        if np.all(np.abs(task_jacobians[i] - task_jacobians[k]) < JACOBIAN_TOL):
                            current_group.append(k)
                            matched[k] = True
                
                if len(current_group) > 1:  # Only store groups with more than one Jacobian
                    equal_groups.append(current_group)
        
        # Print results if any equal groups were found
        if equal_groups:
            print(f"\nTask {j}:")
            print(f"  Total number of Jacobians: {num_jacobians}")
            for group_idx, group in enumerate(equal_groups):
                print(f"  Group {group_idx+1}: {len(group)} equal Jacobians (indices: {[idx+1 for idx in group]})")




# -------------------------------
# 4.5.1. Fixed Points: Unstable Mode Frequency Summaries and Plots
# -------------------------------

# Select the tasks to plot
test_js = TEST_INDICES
# Set the plotting parameters
colors = COLORS
markers = MARKERS

# Plot unstable eigenvalues for selected tasks
print("\nPlotting unstable eigenvalues for selected tasks...")
for j in test_js:
    omega_j = omegas[j]
    
    if all_unstable_eig_freq[j]:  # if any fixed points were found
        # Plot unstable eigenvalues
        plot_unstable_eigenvalues(all_unstable_eig_freq[j], j, omega_j, save_dir=output_dir)


# Plot the comparisons between target frequencies and unstable mode frequencies
print("\nPlotting comparisons between target frequencies and unstable mode frequencies...")
plot_frequency_comparison(all_unstable_eig_freq, omegas, save_dir=output_dir)




# -------------------------------
# 4.5.2. Slow Points: Unstable Mode Frequency Summaries and Plots
# -------------------------------

# Select the tasks to plot
test_js = TEST_INDICES
# Set the plotting parameters
colors = COLORS
markers = MARKERS

# Plot unstable eigenvalues for selected tasks
print("\nPlotting unstable eigenvalues for selected tasks...")
for j in test_js:
    omega_j = omegas[j]
    
    if all_slow_unstable_eig_freq[j]:  # if any slow points were found
        # Plot unstable eigenvalues
        plot_unstable_eigenvalues(all_slow_unstable_eig_freq[j], j, omega_j, save_dir=output_dir)


# Plot the comparisons between target frequencies and unstable mode frequencies
print("\nPlotting comparisons between target frequencies and unstable mode frequencies...")
plot_frequency_comparison(all_slow_unstable_eig_freq, omegas, save_dir=output_dir)




# -------------------------------
# -------------------------------
# 5. PCA RESULTS
# -------------------------------
# -------------------------------

# -------------------------------
# 5.1.1. Fixed Points: Perform PCA on Trajectories and Fixed Points
# -------------------------------

# Track PCA computation time
start_time = time.time()
print("\nStarting PCA computation...")

# Concatenate all states from all tasks (from training phase) to perform PCA
all_states = np.concatenate([traj for traj in state.traj_states], axis=0)   # resulting shape is (num_tasks * num_steps_train, N)

# Perform PCA with 3 components
pca = PCA(n_components=3)
proj_all = pca.fit_transform(all_states)

# For plotting, also project each trajectory and each fixed point into PCA space
proj_trajs = []
start = 0
for traj in tqdm(state.traj_states, desc="Projecting trajectories"):
    T = traj.shape[0]   # number of time steps in the trajectory
    proj_traj = proj_all[start:start+T] # extracts the projected trajectory from the PCA space, resulting shape is (num_steps_train, 3)
    proj_trajs.append(proj_traj)
    start += T

# Project all fixed points from all tasks
all_fixed_points_flat = []
for task_fps in all_fixed_points:
    all_fixed_points_flat.extend(task_fps)
if all_fixed_points_flat:  # Check if there are any fixed points
    all_fixed_points_flat = np.array(all_fixed_points_flat)
    if all_fixed_points_flat.ndim == 1:  # If it's a 1D array, reshape it
        all_fixed_points_flat = all_fixed_points_flat.reshape(1, -1)
    proj_fixed = pca.transform(all_fixed_points_flat)  # projects all fixed points from all tasks into the PCA space, resulting shape is (num_total_fixed_points, 3)
else:
    proj_fixed = np.array([]).reshape(0, 3)  # Empty array with correct shape for plotting
# Note that np.array(all_fixed_points_flat) has shape (num_total_fixed_points, N)

# Calculate PCA computation time
pca_time = time.time() - start_time
print(f"\nPCA computation completed in {pca_time:.2f} seconds")




# -------------------------------
# 5.1.2. Slow Points: Perform PCA on Trajectories and Fixed Points
# -------------------------------

# Track PCA computation time
start_time_slow = time.time()
print("\nStarting PCA computation...")


# Project all slow points from all tasks
all_slow_points_flat = []
for task_sps in all_slow_points:
    all_slow_points_flat.extend(task_sps)
if all_slow_points_flat:  # Check if there are any slow points
    all_slow_points_flat = np.array(all_slow_points_flat)
    if all_slow_points_flat.ndim == 1:  # If it's a 1D array, reshape it
        all_slow_points_flat = all_slow_points_flat.reshape(1, -1)
    proj_slow = pca.transform(all_slow_points_flat)  # projects all slow points from all tasks into the PCA space, resulting shape is (num_total_slow_points, 3)
else:
    proj_slow = np.array([]).reshape(0, 3)  # Empty array with correct shape for plotting
# Note that np.array(all_slow_points_flat) has shape (num_total_slow_points, N)

# Calculate PCA computation time
pca_time_slow = time.time() - start_time_slow
print(f"\nPCA computation completed in {pca_time_slow:.2f} seconds")




# -------------------------------
# 5.2.1. Fixed Points: PCA Trajectory and Fixed Points Plot
# -------------------------------

# Create the figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories in blue
for traj in proj_trajs:
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='blue', alpha=0.5)
    
# Plot the fixed points as green circles
ax.scatter(proj_fixed[:,0], proj_fixed[:,1], proj_fixed[:,2], color='green', s=50, label="Fixed Points")

# For each fixed point, plot all unstable modes as red lines
fixed_point_idx = 0
for j, task_fps in enumerate(all_fixed_points):
    for x_star in task_fps:
        # Compute Jacobian and eigenvalues at the fixed point
        J_eff = jacobian_fixed_point(x_star, J_trained)
        eigenvals, eigenvecs = eig(J_eff)
        
        # Find all unstable eigenvalues
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        if len(unstable_idx) > 0:
            # For each unstable eigenvalue, plot its eigenvector direction
            for idx in unstable_idx:
                v = eigenvecs[:, idx].real  # take real part for plotting direction
                # Scale vector for visualisation
                scale = 0.5  
                # Project the unstable eigenvector into PCA space
                v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_fixed[fixed_point_idx]
                # Plot a line centered on the fixed point
                line = np.array([proj_fixed[fixed_point_idx] - v_proj, proj_fixed[fixed_point_idx] + v_proj])
                ax.plot(line[:,0], line[:,1], line[:,2], color='red', linewidth=2, alpha=0.5)
        fixed_point_idx += 1
    
ax.set_title('PCA of Network Trajectories and Fixed Points')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()

# Save the PCA plot
filename = generate_filename("pca_plot", N, num_tasks, dt, T_drive, T_train)
filepath = os.path.join(output_dir, filename.replace('.pkl', '.png'))
plt.savefig(filepath)
print(f"Saved PCA plot to {filepath}")

plt.show()


# -------------------------------
# 5.2.2. Slow Points: PCA Trajectory and Slow Points Plot
# -------------------------------

# Create the figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories in blue
for traj in proj_trajs:
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='blue', alpha=0.5)
    
# Plot the slow points as green circles
ax.scatter(proj_slow[:,0], proj_slow[:,1], proj_slow[:,2], color='green', s=50, label="Slow Points")

# For each slow point, plot all unstable modes as red lines
slow_point_idx = 0
for j, task_sps in enumerate(all_slow_points):
    for x_star in task_sps:
        # Compute Jacobian and eigenvalues at the slow point
        J_eff = jacobian_fixed_point(x_star, J_trained)
        eigenvals, eigenvecs = eig(J_eff)
        
        # Find all unstable eigenvalues
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        if len(unstable_idx) > 0:
            # For each unstable eigenvalue, plot its eigenvector direction
            for idx in unstable_idx:
                v = eigenvecs[:, idx].real  # take real part for plotting direction
                # Scale vector for visualisation
                scale = 0.5  
                # Project the unstable eigenvector into PCA space
                v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_slow[slow_point_idx]
                # Plot a line centered on the slow point
                line = np.array([proj_slow[slow_point_idx] - v_proj, proj_slow[slow_point_idx] + v_proj])
                ax.plot(line[:,0], line[:,1], line[:,2], color='red', linewidth=2, alpha=0.5)
        slow_point_idx += 1
    
ax.set_title('PCA of Network Trajectories and Slow Points')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()

# Save the PCA plot
filename = generate_filename("pca_plot_slow", N, num_tasks, dt, T_drive, T_train)
filepath = os.path.join(output_dir, filename.replace('.pkl', '.png'))
plt.savefig(filepath)
print(f"Saved PCA plot to {filepath}")

plt.show()




# -------------------------------
# 5.3. Save Results from PCA
# -------------------------------


print("\nSaving results...")

# Save PCA results
pca_results = {
    'proj_trajs': proj_trajs,
    'proj_fixed': proj_fixed,
    'proj_slow': proj_slow
}
save_variable(pca_results, "pca_results", N, num_tasks, dt, T_drive, T_train, output_dir)