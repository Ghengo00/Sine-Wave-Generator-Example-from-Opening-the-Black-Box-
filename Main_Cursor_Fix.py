import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import root
from scipy.linalg import eig
from tqdm import tqdm
import time
import pickle
from datetime import datetime
import os

# -------------------------------
# 1. Set random seed & parameters
# -------------------------------
np.random.seed(42)
torch.manual_seed(42)

# Network dimensions and task parameters
N = 200         # number of neurons
I = 1           # input dimension (scalar input)
num_tasks = 51  # 51 different sine-wave tasks

# Frequencies: equally spaced between 0.1 and 0.6 rad/s
omegas = np.linspace(0.1, 0.6, num_tasks)

# Static input offset for each task: j/51 + 0.25, j=0,...,50 (j/51+0.25)
static_inputs = np.linspace(0, num_tasks-1, num_tasks) / num_tasks + 0.25

# Time parameters (in seconds)
dt = 0.02       # integration time step
T_drive = 12.0   # driving phase duration (to set network state)
T_train = 24.0   # training phase duration with static input (target generation)
num_steps_drive = int(T_drive/dt)
num_steps_train = int(T_train/dt)
time_drive = np.arange(0, T_drive, dt)
time_train = np.arange(0, T_train, dt)
time_full  = np.concatenate([time_drive, T_drive + time_train])

# -------------------------------
# 2. Define the RNN and its parameters
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Randomly initialize recurrent weight matrix J with scaling ~1/sqrt(N)
J_param = torch.nn.Parameter(torch.randn(N, N, device=device) / np.sqrt(N))
# Randomly initialize input weight matrix B of shape (N, I)
B_param = torch.nn.Parameter(torch.randn(N, I, device=device) / np.sqrt(N))
# Bias for network neurons, shape (N,)
b_x_param = torch.nn.Parameter(torch.zeros(N, device=device))

# Output weights and bias: readout is scalar.
w_param = torch.nn.Parameter(torch.randn(N, device=device) / np.sqrt(N))
b_z_param = torch.nn.Parameter(torch.tensor(0.0, device=device))

# Collect parameters in a list for optimization
params = [J_param, B_param, b_x_param, w_param, b_z_param]

# -------------------------------
# 3. Simulation function for one trajectory
# -------------------------------
def simulate_trajectory(x0, u_seq, J, B, b_x, w, b_z, dt):
    """
    Simulate the RNN dynamics with Euler integration.
    Vectorized implementation for better performance.
    
    Arguments:
      x0    : initial state (torch tensor, shape (N,))
      u_seq : input sequence (torch tensor, shape (T, I))
      J, B, b_x, w, b_z : network parameters (torch tensors)
      dt    : time step
      
    Returns:
      xs : (T+1, N) tensor of states over time
      zs : (T+1,) tensor of outputs computed as z = w^T tanh(x) + b_z
    """
    T = u_seq.shape[0]
    xs = torch.zeros(T+1, x0.shape[0], device=x0.device)
    zs = torch.zeros(T, device=x0.device)
    xs[0] = x0
    
    # Pre-compute B*u for all time steps
    Bu = torch.matmul(B, u_seq.transpose(0, 1)).transpose(0, 1)  # shape (T, N)
    
    # Main simulation loop
    for t in range(T):
        x = xs[t]
        # Compute nonlinear activation
        r = torch.tanh(x)
        # Compute readout
        zs[t] = torch.dot(w, r) + b_z
        # Euler integration: dx/dt = -x + J tanh(x) + B u + b_x
        xs[t+1] = x + dt * (-x + torch.matmul(J, r) + Bu[t] + b_x)
    
    return xs, zs

# -------------------------------
# 4. Training procedure and visualization
# -------------------------------
def run_batch(J, B, b_x, w, b_z):
    """
    Run a batch of tasks with improved memory efficiency and GPU utilization.
    """
    loss_total = 0.0
    traj_states = []  # store states from training phase for later PCA
    fixed_point_inits = []  # store final state from drive phase as an initial guess
    
    # Pre-allocate tensors for all tasks
    x0 = torch.zeros(N, device=device)
    
    # Process tasks in smaller batches to manage memory
    batch_size = 10  # Adjust based on available memory
    for batch_start in range(0, num_tasks, batch_size):
        batch_end = min(batch_start + batch_size, num_tasks)
        batch_loss = 0.0
        
        # Pre-compute input sequences for the batch
        u_drive_batch = []
        u_train_batch = []
        target_train_batch = []
        
        for j in range(batch_start, batch_end):
            omega = omegas[j]
            u_offset = static_inputs[j]
            
            # Build input sequences for both phases
            u_drive = torch.tensor(np.sin(omega*time_drive) + u_offset, 
                                 dtype=torch.float32, device=device).view(-1, 1)
            u_train = torch.full((num_steps_train, 1), u_offset, 
                               dtype=torch.float32, device=device)
            target_train = torch.tensor(np.sin(omega * time_train), 
                                      dtype=torch.float32, device=device)
            
            u_drive_batch.append(u_drive)
            u_train_batch.append(u_train)
            target_train_batch.append(target_train)
        
        # Process each task in the batch
        for j, (u_drive, u_train, target_train) in enumerate(zip(
            u_drive_batch, u_train_batch, target_train_batch)):
            
            # Simulate drive phase
            xs_drive, _ = simulate_trajectory(x0, u_drive, J, B, b_x, w, b_z, dt)
            x_drive_final = xs_drive[-1]
            
            # Save initial state for fixed point search
            fixed_point_inits.append(x_drive_final.detach().cpu().numpy())
            
            # Simulate training phase
            xs_train, zs_train = simulate_trajectory(x_drive_final, u_train, J, B, b_x, w, b_z, dt)
            
            # Compute loss
            loss = torch.mean((zs_train - target_train)**2)
            batch_loss += loss
            
            # Store trajectory states
            traj_states.append(xs_train.detach().cpu().numpy())
            
            # Clear intermediate tensors
            del xs_drive, xs_train, zs_train
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Accumulate batch loss
        loss_total += batch_loss
        
    # Average loss over all tasks
    loss_total /= num_tasks
    return loss_total, traj_states, fixed_point_inits

# Define LBFGS optimizer with more conservative parameters
optimizer = optim.LBFGS(params, lr=0.6, max_iter=10, history_size=10, line_search_fn="strong_wolfe")

num_epochs = 50  # number of training epochs
loss_history = []
best_loss = float('inf')
best_params = None
loss_threshold = 1e-4  # threshold for early stopping

class TrainingState:
    def __init__(self):
        self.traj_states = []
        self.fixed_point_inits = []

state = TrainingState()

# Track training time
start_time = time.time()
print("Starting training...")

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    # Clear memory between epochs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def closure():  # required by the LBFGS optimizer to re-evaluate the model function multiple times per step
        optimizer.zero_grad()
        loss, state.traj_states, state.fixed_point_inits = run_batch(J_param, B_param, b_x_param, w_param, b_z_param)
        loss.backward() # computes the gradient of the loss with respect to the model parameters
        return loss
    
    try:
        loss_val = optimizer.step(closure)
        loss_history.append(loss_val.item())    # returns the computed loss value for the epoch
        
        # Save best parameters
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_params = [p.detach().clone() for p in params]
            
        # Print epoch and loss information
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val.item():.4f}")
            
        # Check if loss is below threshold
        if loss_val.item() < loss_threshold:
            print(f"\nTraining converged with loss {loss_val.item():.4f} below threshold {loss_threshold}")
            break
            
    except RuntimeError as e:
        print(f"\nOptimization failed at epoch {epoch+1}: {str(e)}")
        break

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Restore best parameters if available
if best_params is not None:
    for p, best_p in zip(params, best_params):
        p.data.copy_(best_p)

# Ensure we have the final states for analysis
if state.traj_states is None or state.fixed_point_inits is None:
    print("Running final batch to get states for analysis...")
    _, state.traj_states, state.fixed_point_inits = run_batch(J_param, B_param, b_x_param, w_param, b_z_param)

# Plot produced trajectories vs. target signals for selected tasks
test_js = [0, 10, 20, 30, 40, 50]
fig, axes = plt.subplots(nrows=len(test_js), ncols=1, figsize=(10, 3 * len(test_js)))
if len(test_js) == 1:
    axes = [axes]  # Ensure axes is iterable when there's a single subplot

for ax, j in zip(axes, test_js):
    omega = omegas[j]
    u_offset = static_inputs[j]
    
    # Build the training input and compute the target sine signal.
    u_train_test = torch.full((num_steps_train, 1), u_offset, dtype=torch.float32, device=device)
    target_train_test = np.sin(omega * time_train)
    
    # Initialize state using driving phase.
    x0_test = torch.zeros(N, device=device)
    u_drive_test = torch.tensor(np.sin(omega*time_drive) + u_offset, 
                                dtype=torch.float32, device=device).view(-1, 1)
    xs_drive_test, _ = simulate_trajectory(x0_test, u_drive_test, J_param, B_param, 
                                           b_x_param, w_param, b_z_param, dt)
    x_drive_final_test = xs_drive_test[-1]
    
    # Simulate training phase starting from the drive-phase final state.
    _, zs_train_test = simulate_trajectory(x_drive_final_test, u_train_test, J_param, B_param, 
                                           b_x_param, w_param, b_z_param, dt)
    produced_traj = zs_train_test.detach().cpu().numpy()
    
    # Plot the produced trajectory and the target signal.
    ax.plot(time_train, produced_traj, label="Produced Output", linewidth=2)
    ax.plot(time_train, target_train_test, 'k--', label="Target Signal", linewidth=1.5)
    ax.set_title(f"Task {j}: omega = {omega:.3f}, u_offset = {u_offset:.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.legend(loc="upper right")
    
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Post-Training Analysis: Fixed point search & Unstable Mode Frequencies
# -------------------------------
def fixed_point_func(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute f(x) = -x + J*tanh(x) + B*u + b_x for given x and fixed u.
    Vectorized implementation for better performance.
    """
    x = x_np
    return -x + np.dot(J_np, np.tanh(x)) + np.dot(B_np, np.array([u_val])).flatten() + b_x_np

def jacobian_fixed_point(x_star, J_np):
    """
    Compute Jacobian: J_eff = -I + J * diag(1 - tanh(x_star)^2)
    Vectorized implementation for better performance.
    """
    diag_term = 1 - np.tanh(x_star)**2
    return -np.eye(len(x_star)) + J_np * diag_term[np.newaxis, :]

# Extract trained parameters as NumPy arrays
J_trained = J_param.detach().cpu().numpy()
B_trained = B_param.detach().cpu().numpy()
b_x_trained = b_x_param.detach().cpu().numpy()

def find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained, num_attempts=10, tol=1e-6):
    """
    Find multiple fixed points by trying different initial conditions.
    
    Arguments:
        x0_guess: Initial guess for fixed point
        u_const: Constant input
        J_trained, B_trained, b_x_trained: Network parameters
        num_attempts: Number of different initial conditions to try
        tol: Tolerance for considering two fixed points as distinct
        
    Returns:
        List of distinct fixed points found
    """
    fixed_points = []
    
    # Try the original initial condition
    try:
        sol = root(fixed_point_func, x0_guess, args=(u_const, J_trained, B_trained, b_x_trained),
                  method='lm', options={'maxiter': 1000})
        if sol.success:
            fixed_points.append(sol.x)
    except Exception as e:
        print(f"Fixed point search failed for initial guess: {str(e)}")
    
    # Try perturbed initial conditions
    for attempt in tqdm(range(num_attempts - 1), desc="Finding fixed points", leave=False):
        # Create a perturbed initial condition
        x0_perturbed = x0_guess + np.random.normal(0, 0.5, size=x0_guess.shape)
        try:
            sol = root(fixed_point_func, x0_perturbed, args=(u_const, J_trained, B_trained, b_x_trained),
                      method='lm', options={'maxiter': 1000})
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

def analyze_fixed_points(fixed_points, J_trained, static_inputs):
    """
    Analyze fixed points and compute their properties in a memory-efficient way.
    """
    jacobians = []
    unstable_freqs = []
    
    for x_star in fixed_points:
        # Compute Jacobian at the fixed point
        J_eff = jacobian_fixed_point(x_star, J_trained)
        jacobians.append(J_eff)
        
        # Compute eigenvalues
        eigenvals, _ = eig(J_eff)
        
        # Find unstable eigenvalues
        idx_complex = np.where((np.abs(np.imag(eigenvals)) > 1e-3) & (np.real(eigenvals) > 0))[0]
        if len(idx_complex) > 0:
            # Sort by imaginary part magnitude and take the largest
            sorted_idx = idx_complex[np.argsort(np.abs(np.imag(eigenvals[idx_complex])))]
            ev = eigenvals[sorted_idx[-1]]  # take the one with largest imaginary part
            unstable_freqs.append(np.abs(np.imag(ev)))
        else:
            unstable_freqs.append(0.0)
    
    return jacobians, unstable_freqs

# Initialize lists to store multiple fixed points and their properties
all_fixed_points = []  # List of lists, one list per task
all_jacobians = []     # List of lists of Jacobians
all_unstable_eig_freq = []  # List of lists of unstable frequencies

# Track fixed point search time
start_time = time.time()
print("\nStarting fixed point search...")

# Process tasks in batches for memory efficiency
batch_size = 5  # Adjust based on available memory
for batch_start in range(0, num_tasks, batch_size):
    batch_end = min(batch_start + batch_size, num_tasks)
    
    for j in range(batch_start, batch_end):
        u_const = static_inputs[j]
        x0_guess = state.fixed_point_inits[j]
        
        # Find multiple fixed points
        task_fixed_points = find_fixed_points(x0_guess, u_const, J_trained, B_trained, b_x_trained)
        all_fixed_points.append(task_fixed_points)
        
        # Analyze fixed points
        task_jacobians, task_unstable_freqs = analyze_fixed_points(task_fixed_points, J_trained, static_inputs)
        all_jacobians.append(task_jacobians)
        all_unstable_eig_freq.append(task_unstable_freqs)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

fixed_point_time = time.time() - start_time
print(f"\nFixed point search completed in {fixed_point_time:.2f} seconds")

# Print summary of fixed points found
print("\nSummary of Fixed Points Found:")
for j in range(num_tasks):
    print(f"\nTask {j} (omega = {omegas[j]:.3f}):")
    print(f"Number of distinct fixed points found: {len(all_fixed_points[j])}")
    for i, (fp, freq) in enumerate(zip(all_fixed_points[j], all_unstable_eig_freq[j])):
        print(f"  Fixed point {i+1}:")
        print(f"    Unstable mode frequency: {freq:.4f}")
        print(f"    Norm: {np.linalg.norm(fp):.4f}")

# Plot unstable mode frequencies for the first fixed point of each task
first_fixed_point_freqs = [freqs[0] if freqs else 0.0 for freqs in all_unstable_eig_freq]
plt.figure(figsize=(8,5))
plt.plot(omegas, first_fixed_point_freqs, 'o-', label='|Imag(eigenvalue)| (unstable mode)')
plt.plot(omegas, omegas, 'k--', label='Target frequency')
plt.xlabel('Target Frequency (rad/s)')
plt.ylabel('Frequency from Linearization (rad/s)')
plt.title('Comparison of Target Frequencies and Unstable Mode Frequency')
plt.legend()
plt.show()

# -------------------------------
# Additional Analysis: Jacobian and Parameter Visualization for Selected Tasks
# -------------------------------
test_js = [0, 10, 20, 30, 40, 50]
colors = ['b', 'g', 'r', 'c', 'm', 'y']
markers = ['o', 's', '^', 'v', '<', '>']

# Print detailed information about unstable eigenvalues for all tasks
print("\nDetailed Analysis of Unstable Eigenvalues:")
for j in range(num_tasks):
    print(f"\nTask {j} (omega = {omegas[j]:.3f}):")
    for i, (J_eff, freqs) in enumerate(zip(all_jacobians[j], all_unstable_eig_freq[j])):
        eigenvals, _ = eig(J_eff)
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        num_unstable = len(unstable_idx)
        
        print(f"\n  Fixed point {i+1}:")
        print(f"  Number of unstable eigenvalues: {num_unstable}")
        if num_unstable > 0:
            unstable_eigenvals = eigenvals[unstable_idx]
            print("  Unstable eigenvalues:")
            for ev in unstable_eigenvals:
                print(f"    Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")

# Plot unstable eigenvalues for selected tasks
plt.figure(figsize=(12, 8))
for idx, j in enumerate(test_js):
    for i, (J_eff, freqs) in enumerate(zip(all_jacobians[j], all_unstable_eig_freq[j])):
        eigenvals, _ = eig(J_eff)
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        unstable_eigenvals = eigenvals[unstable_idx]
        
        if len(unstable_eigenvals) > 0:
            plt.scatter(np.real(unstable_eigenvals), np.imag(unstable_eigenvals), 
                       color=colors[idx], marker=markers[i], 
                       label=f'Task {j} FP{i+1} (ω={omegas[j]:.3f})')

plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Unstable Eigenvalues of Jacobian for Selected Tasks')
plt.legend()
plt.grid(True)
plt.show()

# Create subplots for Jacobian visualizations
num_rows = len(test_js)
fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))
fig.suptitle('Jacobian and Parameter Matrices for Selected Tasks', fontsize=16)

# Plot Jacobians and parameters for each selected task
for idx, j in enumerate(test_js):
    if all_jacobians[j]:  # If any fixed points were found
        # Plot first Jacobian for this task
        im = axes[idx, 0].imshow(all_jacobians[j][0], cmap='viridis')
        axes[idx, 0].set_title(f'Jacobian Matrix (Task {j}, ω={omegas[j]:.3f}, FP1)')
        plt.colorbar(im, ax=axes[idx, 0])
        
        # Plot J_param (only for first row)
        if idx == 0:
            im = axes[idx, 1].imshow(J_param.detach().cpu().numpy(), cmap='viridis')
            axes[idx, 1].set_title('J_param Matrix')
            plt.colorbar(im, ax=axes[idx, 1])
        else:
            axes[idx, 1].axis('off')
    else:
        axes[idx, 0].axis('off')
        axes[idx, 1].axis('off')

# Add B_param and b_x_param visualizations
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
fig2.suptitle('Parameter Matrices', fontsize=16)

# Plot B_param
im = axes2[0].imshow(B_param.detach().cpu().numpy(), cmap='viridis')
axes2[0].set_title('B_param Matrix')
plt.colorbar(im, ax=axes2[0])

# Plot b_x_param
im = axes2[1].imshow(b_x_param.detach().cpu().numpy().reshape(-1, 1), cmap='viridis')
axes2[1].set_title('b_x_param Vector')
plt.colorbar(im, ax=axes2[1])

# Print information about state.traj_states and state.fixed_point_inits
print("\nState Information:")
print(f"state.traj_states:")
print(f"  Type: {type(state.traj_states)}")
print(f"  Length: {len(state.traj_states)}")
print(f"  Shape of first trajectory: {state.traj_states[0].shape}")
print(f"  Data type of first trajectory: {state.traj_states[0].dtype}")

print(f"\nstate.fixed_point_inits:")
print(f"  Type: {type(state.fixed_point_inits)}")
print(f"  Length: {len(state.fixed_point_inits)}")
print(f"  Shape of first fixed point: {state.fixed_point_inits[0].shape}")
print(f"  Data type of first fixed point: {state.fixed_point_inits[0].dtype}")

plt.tight_layout()
plt.show()

# -------------------------------
# 6. PCA and Visualization
# -------------------------------
# Track PCA computation time
start_time = time.time()
print("\nStarting PCA computation...")

# Concatenate all states from all tasks (from training phase) to perform PCA.
all_states = np.concatenate([traj for traj in state.traj_states], axis=0)
pca = PCA(n_components=3)
proj_all = pca.fit_transform(all_states)

# For plotting, also project each trajectory and each fixed point into PCA space.
proj_trajs = []
start = 0
for traj in tqdm(state.traj_states, desc="Projecting trajectories"):
    T = traj.shape[0]
    proj_traj = proj_all[start:start+T]
    proj_trajs.append(proj_traj)
    start += T

# Project all fixed points from all tasks
all_fixed_points_flat = []
for task_fps in all_fixed_points:
    all_fixed_points_flat.extend(task_fps)
proj_fixed = pca.transform(np.array(all_fixed_points_flat))

pca_time = time.time() - start_time
print(f"\nPCA computation completed in {pca_time:.2f} seconds")

# Plot trajectories (blue) and fixed points (green circles) with unstable eigen-directions (red lines)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for traj in proj_trajs:
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='blue', alpha=0.5)
    
# Plot fixed points as green circles
ax.scatter(proj_fixed[:,0], proj_fixed[:,1], proj_fixed[:,2], color='green', s=50, label="Fixed Points")

# For each fixed point, plot the unstable mode as a red line.
fixed_point_idx = 0
for j, task_fps in enumerate(all_fixed_points):
    for x_star in task_fps:
        u_const = static_inputs[j]
        J_eff = jacobian_fixed_point(x_star, J_trained)
        eigenvals, eigenvecs = eig(J_eff)
        idx_complex = np.where((np.abs(np.imag(eigenvals)) > 1e-3) & (np.real(eigenvals) > 0))[0]
        if len(idx_complex) > 0:
            # Sort by imaginary part magnitude and take the largest
            sorted_idx = idx_complex[np.argsort(np.abs(np.imag(eigenvals[idx_complex])))]
            v = eigenvecs[:, sorted_idx[-1]].real  # take real part for plotting direction
            # Scale vector for visualisation
            scale = 0.5  
            # Project the unstable eigenvector into PCA space
            v_proj = pca.transform((x_star + scale * v).reshape(1, -1))[0] - proj_fixed[fixed_point_idx]
            # Plot a line centered on the fixed point
            line = np.array([proj_fixed[fixed_point_idx] - v_proj, proj_fixed[fixed_point_idx] + v_proj])
            ax.plot(line[:,0], line[:,1], line[:,2], color='red', linewidth=2)
        fixed_point_idx += 1
    
ax.set_title('PCA of Network Trajectories and Fixed Points')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
plt.show()

# Save all important variables
print("\nSaving results...")

def generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train):
    """
    Generate a filename with timestamp and parameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{variable_name}_{timestamp}_Neuron_Number_{N}_Task_Number_{num_tasks}_Time_Steps_{dt}_Driving_Time_{T_drive}_Training_Time_{T_train}.pkl"

def save_variable(variable, variable_name, N, num_tasks, dt, T_drive, T_train):
    """
    Save a variable to a pickle file with a descriptive filename in the Outputs folder.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create Outputs folder if it doesn't exist
    outputs_dir = os.path.join(script_dir, 'Outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Created Outputs directory at: {outputs_dir}")
    
    # Generate filename and save in Outputs folder
    filename = generate_filename(variable_name, N, num_tasks, dt, T_drive, T_train)
    filepath = os.path.join(outputs_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(variable, f)
    print(f"Saved {variable_name} to {filepath}")

# Save network parameters
save_variable(J_param.detach().cpu().numpy(), "J_param", N, num_tasks, dt, T_drive, T_train)
save_variable(B_param.detach().cpu().numpy(), "B_param", N, num_tasks, dt, T_drive, T_train)
save_variable(b_x_param.detach().cpu().numpy(), "b_x_param", N, num_tasks, dt, T_drive, T_train)
save_variable(w_param.detach().cpu().numpy(), "w_param", N, num_tasks, dt, T_drive, T_train)
save_variable(b_z_param.detach().cpu().numpy(), "b_z_param", N, num_tasks, dt, T_drive, T_train)

# Save state information
state_dict = {
    'traj_states': state.traj_states,
    'fixed_point_inits': state.fixed_point_inits
}
save_variable(state_dict, "state", N, num_tasks, dt, T_drive, T_train)

# Save fixed point analysis results
save_variable(all_fixed_points, "all_fixed_points", N, num_tasks, dt, T_drive, T_train)
save_variable(all_jacobians, "all_jacobians", N, num_tasks, dt, T_drive, T_train)
save_variable(all_unstable_eig_freq, "all_unstable_eig_freq", N, num_tasks, dt, T_drive, T_train)

# Save PCA results
pca_results = {
    'proj_trajs': proj_trajs,
    'proj_fixed': proj_fixed,
    'pca_components': pca.components_,
    'pca_mean': pca.mean_
}
save_variable(pca_results, "pca_results", N, num_tasks, dt, T_drive, T_train)

print("All results saved successfully!")