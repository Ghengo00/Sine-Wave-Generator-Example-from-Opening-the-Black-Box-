# ------------------------------------------------------------
# ------------------------------------------------------------
# A. SET-UP
# ------------------------------------------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# 0. IMPORTS
# ------------------------------------------------------------
from tqdm import tqdm
import time
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.optimize import root, least_squares
from scipy.linalg import eig
from sklearn.decomposition import PCA

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# ------------------------------------------------------------
# 1.1 TASK PARAMETERS (3-Bit Flip-Flop Task)
# ------------------------------------------------------------
N = 1000                                # neurons (standard is 1,000)
O = 3                                   # outputs (=inputs) (stnadard is 3)

dt = 1e-3                               # time step (standard is 1e-3)
T_train = 20.0                          # seconds to simulate during training (standard is 20.0)
T_test = 4.0                            # seconds to simulate during testing (standard is 4.0)

g = 1.5                                 # spectral radius (standard is 1.5)

pulse_prob = 4.0                        # expected pulses per second per line (standard is 4.0)
pulse_amp = 1.0                         # ±1 flip amplitude (standard is 1.0)

alpha = 1.0                             # FORCE ridge parameter (P = I/alpha at start) (standard is 1.0)

# Create output directory for saving files
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), 'Outputs', f'3_Bit_Outputs_{RUN_TIMESTAMP}')
os.makedirs(output_dir, exist_ok=True)




# ------------------------------------------------------------
# 1.2 WEIGHTS
# ------------------------------------------------------------
J_param   = g*torch.randn(N,N,device=device)/np.sqrt(N)     # recurrent weights, spectral radius g achieved by scaling by g / sqrt(N)
B_param   = torch.randn(N,O,device=device)/np.sqrt(O)       # input weights, couples the external pulses to the network
w_param   = torch.zeros(N,O,device=device)                  # read-out weights - this is the only thing that FORCE will train
b_z_param = torch.zeros(O,device=device)                    # biases, kept 0, only included to match the sine wave generator example
P         = torch.eye(N,device=device)/alpha                # RLS state, P = I / alpha is the inverse correlation matrix used by FORCE

# ATTENTION: Normally, you would not use this (set fb_scale to 1.0)
# However, here I changed it to try to improve RNN training by giving it stronger, orthogonal feedback weights
# Use a stronger feedback weight to facilitate training
fb_scale = 1.0                                                          # scaling parameter, tune as needed (standard is 5.0)
W_fb_param = fb_scale * torch.randn(N, O, device=device) / np.sqrt(N)   # feedback weights, sends the current output back into the network
# Make columns of the feedback weight orthonormal so they do not cross-talk
# W_fb_param, _ = torch.linalg.qr(W_fb_param)     # keeps scale≈fb_scale
# W_fb_param = W_fb_param[:, :O].contiguous()




# ------------------------------------------------------------
# 1.3 TRAINING PARAMETERS
# ------------------------------------------------------------
force_max_iter = 1                                  # maximum number of FORCE passes to run
force_tol      = 1e-2                               # loss tolerance for early stopping


# ATTENTION: Normally, you would not use this (set lam to 0.0)
# However, here I changed it to try to improve RNN training by giving it a slower leak
lam = 0.0                                           # leak rate, tune as needed (standard is 1.0)

# ATTENTION: Normally, you would not use this (set TEACHER_FORCING_STEPS to 0)
# However, here I changed it to try to improve RNN training in the early stages
TEACHER_FORCING_STEPS = min(0, int(T_train/dt))  # number of steps to use teacher forcing (standard is 5000)




# ------------------------------------------------------------
# 1.4 ROOT-FINDING PARAMETERS
# ------------------------------------------------------------
NUM_ATTEMPTS_FIXED_TRAJECTORIES   = 3      # number of candidate initialisations for fixed points to pull from trajectories
NUM_ATTEMPTS_SLOW_TRAJECTORIES    = 3      # numer of candidate initialisations for slow points to pull from trajectories
NUM_ATTEMPS_FIXED_PERTURBATIONS   = 1       # number of perturbations to apply to each candidate initialisation from trajectory
NUM_ATTEMPS_SLOW_PERTURBATIONS    = 1       # number of perturbations to apply to each candidate initialisation from trajectory

MAXITER_FIXED        = 1000    # maximum number of iterations for finding fixed points
MAXITER_SLOW         = 1000    # maximum number of iterations for finding slow points

SCALE_OF_PERTURBATIONS_FIXED      = 0.1     # scale of perturbations to apply to candidate initialisations from trajectory for fixed points
SCALE_OF_PERTURBATIONS_SLOW       = 0.1     # scale of perturbations to apply to candidate initialisations from trajectory for slow points

TOL_FIXED_ROOT       = 1e-3    # tolerance for finding fixed points as roots of F(x)
TOL_SLOW_ROOT        = 1e-3    # tolerance for finding slow points as roots of grad_q(x)
TOL_FIXED            = 1e-2    # tolerance for ascertaining two fixed points are distinct
TOL_SLOW             = 1e-2    # tolerance for ascertaining two slow points are distinct




# -------------------------------
# 1.5 SAVING FUNCTIONS
# -------------------------------
def generate_filename(variable_name):
    """
    Generate a filename with timestamp.
    """

    return f"{variable_name}_{RUN_TIMESTAMP}.pkl"


def save_variable(variable, variable_name, output_directory=output_dir):
    """
    Save a variable to a pickle file with a descriptive filename in the Outputs folder.
    """

    # Generate filename with descriptive parameters
    filename = generate_filename(variable_name)
    filepath = os.path.join(output_directory, filename)
    
    # Use a try-except block to handle potential I/O errors
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Saved {variable_name} to {filepath}")
    except Exception as e:
        print(f"Error saving {variable_name} to {filepath}: {e}")




# ------------------------------------------------------------
# 1.5 PLOTTING FUNCTIONS
# ------------------------------------------------------------
def plot_test_run(u_test, z_test, z_target_test, dt, to_save=True):
    """
    Plots the test run outputs for the network.

    Parameters:
    u_test: input sequence for testing, dimensions (steps, O)
    z_test: output sequence from testing, dimensions (steps, O)
    z_target_test: target output sequence, dimensions (steps, O)
    dt: time step
    to_save: boolean, if True, saves the plot as a png file using save_variable()
    """
    
    t = np.arange(u_test.shape[0]) * dt
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for k in range(3):
        ax[k].plot(t, u_test[:, k].cpu(), '--', color=colors[k], alpha=0.4, label='input')
        ax[k].plot(t, z_test[:, k], color=colors[k], label='output')
        ax[k].plot(t, z_target_test[:, k].cpu(), '--', color='lightgrey', label='target')
        ax[k].set_ylabel(f'bit {k}')
        ax[k].legend(loc='upper right')
    ax[-1].set_xlabel('time (s)')
    plt.tight_layout()
    
    if to_save:
        # Save the plot as a PNG file and then log its path with save_variable()
        filename = f"3_Bit_{RUN_TIMESTAMP}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath)
    
    plt.show()




# ------------------------------------------------------------
# ------------------------------------------------------------
# B. TRAIN RNN
# ------------------------------------------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2. TASK GENERATOR
# ------------------------------------------------------------
def generate_pulse_task(dt, T, prob, amp):
    """
    Generates a pulse task with the given parameters.

    Parameters:
    dt: time step
    T: total time
    prob: expected number of pulses per second
    amp: amplitude of the pulses
    
    Returns:
    u: input sequence
    z_tgt: target output
    Note that the outputs are returned on device, because after this they are fed directly into simulate()
    """

    steps = int(T/dt)
    u = torch.zeros(steps,O)        # input sequence, dimensions (steps,O)
    z_tgt = torch.zeros(steps,O)    # target output, dimensions (steps,O)
    mem = torch.zeros(O)            # memory of the last pulse, dimensions (3,)

    for t in range(steps):
        # 1) Generate a pulse
        # 1) a) Randomly determine the 3 pulse values
        # Note that the probability of a pulse is the expected number of pulses per second times the time step
        mask = torch.rand(O) < prob * dt    # Boolean mask, dimensions (3,)
        
        # 1) b) Generate the pulse values
        # The pulses are generated only in the channels where the mask is True, and their sign is chosen randomly
        pulse = amp * mask * torch.sign(torch.randn(O)) # (3,)

        # 2) Update input sequence with new pulse
        u[t]  = pulse

        # 3) Update target output sequence
        # 3) a) If there is a pulse...
        if mask.any():
            # Replace the value of the memory tensor in the indices in which there was a pulse with the value of the new pulse
            # Note that this means the memory tensor will contain the last pulse value from every channel, not the last value from every channel
            mem[mask] = pulse[mask]
        # 3) b) Update the target output sequence using the memory tensor
        z_tgt[t] = mem

    return u.to(device), z_tgt.to(device)




# ------------------------------------------------------------
# 3.1 FORCE UPDATE
# ------------------------------------------------------------
@torch.no_grad()
def force_step(r, z_hat, z_tgt):
    """
    Performs a single First-Order Reduced and Controlled Error (FORCE) update step on P and w_param.
    
    The FORCE algorithm is used because modifying the output weights modifies the reservir state online.
    Keeping the instantaneous error small is done to prevent the network from drifting into regions where the linear read-out would no longer be sufficient.
    FORCE is essentially RLS applied to an echo-state / reservoir system with immediate feedback.
    
    The Recursive Least Squares (RLS) algorithm is a recursive algorithm that updates the inverse correlation matrix P and the weights w_param.
    This uses the Sherman-Morrison formula, which allows for efficient rank-1 updates (O(N^2) instead of O(N^3)).
    The update is done using the following equations:
    P = P - (P @ r @ r.T @ P) / (1 + r.T @ P @ r)
    w_param = w_param - (P @ r @ (z_hat - z_tgt)) / (1 + r.T @ P @ r)
    The update is done in place, so the values of P and w_param are updated directly.
    The update is done using the outer product of the vectors, which is more efficient than using matrix multiplication.

    Parameters:
    r: current state of the network, dimensions (N,)
    z_hat: current output of the network, dimensions (O,)
    z_tgt: target output, dimensions (O,)
    """

    global w_param, P                         # global variables to be updated

    err = z_hat - z_tgt                       # instantaneous error between current and target output, dimensions (3,)
    Pr = P @ r                                # dimesnions (N,)
    k = Pr / (1. + (r * Pr).sum())            # Sherman-Morrison gain vector, dimesions (N,)
    P -= torch.ger(k, Pr)                     # rank-1 update, torch.ger(a,b) = a[:,None] @ b[None,:] produces an outer product
    w_param -= torch.ger(k, err)              # rank-1 update, torch.ger(a,b) = a[:,None] @ b[None,:] produces an outer product




# ------------------------------------------------------------
# 3.2 SIMULATION CORE
# ------------------------------------------------------------
@torch.no_grad()
def simulate(u_seq, z_tgt, train=True, teacher_forcing_steps=TEACHER_FORCING_STEPS, record_trajectory=False):
    """
    Simulates the network with the given input sequence u_seq.
    The simulation is done using the Euler method, which is a simple numerical method for solving ordinary differential equations.

    Parameters:
    u_seq: input sequence, dimensions (steps,O)
    z_tgt: target output sequence, dimensions (steps,O)
    train: boolean, if True, the network is trained using the FORCE algorithm, if False, the network is run in inference mode with fixed weights
    teacher_forcing_steps: number of steps to use teacher forcing
    record_trajectory: boolean, if True, the trajectory of the network state is recorded and returned

    Returns:
    zs: output sequence, dimensions (steps,O)
    Note that the outputs are returned on device, because after this they are only used for the loss calculation
    """

    steps = u_seq.shape[0]
    x  = torch.zeros(N, device=device)
    zs = torch.zeros(steps, O, device=device)
    xs_traj = [] if record_trajectory else None

    for t in range(steps):
        r  = torch.tanh(x)
        z  = r @ w_param + b_z_param
        zs[t] = z

        if train:
            force_step(r, z, z_tgt[t])
            # ATTENTION: Normally, you would not use this (set TEACHER_FORCING_STEPS to 0)
            # However, here I changed it to try to improve RNN training in the early stages
            z_fb = z_tgt[t] if t < teacher_forcing_steps else z
        else:
            z_fb = z
        # ATTENTION: Normally, you should set lam = 0.0
        # However, here I changed it to try to improve RNN training by giving it a slower leak
        x = (1. - lam * dt) * x + dt * (J_param @ r + B_param @ u_seq[t] + W_fb_param @ z_fb)
        
        if record_trajectory:
            xs_traj.append(x.clone())
    return (zs.cpu(), xs_traj) if record_trajectory else zs.cpu()




# ------------------------------------------------------------
# 3.3 TRAINING
# ------------------------------------------------------------
def train_force(max_iter=force_max_iter, tol=force_tol):
    """
    Trains the network using FORCE until max_iter iterations or until the MSE falls below tol.
    Every 10 iterations, generates and plots a test run with current network parameters.
    Stores and restores the parameters that yield the smallest loss.
    
    Parameters:
    max_iter: int - maximum number of FORCE passes to run
    tol: float - loss tolerance for early stopping
    """

    global w_param, P, trajectories  # make trajectories accessible outside this function
    trajectories = {'trajectories_x': [], 'final_values_x': []}
    best_loss = float('inf')
    best_w = w_param.clone()
    best_P = P.clone()

    pbar = tqdm(range(max_iter), desc="Training iterations", unit="iter")
    for i in pbar:
        start_time = time.time()

        u_seq, z_tgt = generate_pulse_task(dt, T_train, pulse_prob, pulse_amp)
        zs, xs = simulate(u_seq, z_tgt, train=True, record_trajectory=True)
        trajectories['trajectories_x'].append(xs)           # dimensions (steps, N)
        trajectories['final_values_x'].append(xs[-1])       # dimensions (N,)
        loss = torch.mean((zs - z_tgt.cpu()) ** 2).item()

        # Save the parameters if they yield a lower loss
        if loss < best_loss:
            best_loss = loss
            best_w = w_param.clone()
            best_P = P.clone()
            # THE FOLLOWING IS A CHECK - remove when done
            plt.figure()
            plt.imshow(w_param.cpu().numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Visualization of w at iteration {i} | New Best Loss = {loss:.3e} | Norm = {torch.norm(best_w):.3e}")
            plt.xlabel("Output dimension")
            plt.ylabel("Neuron index")
            plt.show()
            plt.close()
            # Generate a test run with the current network parameters
            u_test, z_target_test = generate_pulse_task(dt, T_test, pulse_prob, pulse_amp)
            z_test = simulate(u_test, z_target_test, train=False)
            plot_test_run(u_test, z_test, z_target_test, dt, to_save=False)

        iter_time = time.time() - start_time
        pbar.set_postfix(loss=f'{loss:.3e}', time=f'{iter_time:.3f}s')

        # Early stopping if the loss is below the tolerance
        if loss < tol:
            break

    # Restore the best parameters
    w_param = best_w.clone()
    P = best_P.clone()
    # Save the trajectories dictionary to file
    # Note that the dictionary has two keys, and the value of each key is a list of length number_of_iterations
    # Each element in the list of key 'trajectories_x' is a tensor of shape (steps, N)
    # Each element in the list of key 'final_values_x' is a tensor of shape (N,)
    save_variable(trajectories, "trajectories")


train_force(force_max_iter, force_tol)




# ------------------------------------------------------------
# 4.1 TESTING
# ------------------------------------------------------------
u_test, z_target_test = generate_pulse_task(dt, T_test, pulse_prob, pulse_amp)
z_test = simulate(u_test, z_target_test, train=False)




# ------------------------------------------------------------
# 4.2 PLOTTING
# ------------------------------------------------------------
plot_test_run(u_test, z_test, z_target_test, dt)




# ------------------------------------------------------------
# ------------------------------------------------------------
# C. ANALYSE DYNAMICS
# ------------------------------------------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5.1 FUNCTIONS TO CALCULATE F, q, AND GRADIENTS
# ------------------------------------------------------------
def calculate_F(x, u, J, B, W_fb, w):
    """
    Compute F(x) = -x + J*tanh(x) + B*u + b_x for given x and fixed u.

    Arguments:
        x: Input vector (numpy array, shape (N,))
        u: Input vector (numpy array, shape (O,))
        J: Recurrent weight matrix (numpy array, shape (N, N))
        B: Input weight matrix (numpy array, shape (N, O))
        W_fb: Feedback weight matrix (numpy array, shape (N, O))
        w: Readout weight matrix (numpy array, shape (N, O))
    
    Returns:
        F_x: Output vector (numpy array, shape (N,))
    """

    # Compute the activation
    r = np.tanh(x)  # (N,)
    # Compute the output (/feedback)
    z = r @ w       # (O,)

    # Compute F(x)
    F_x = -x + J @ r + B @ u + W_fb @ z

    return F_x


def calculate_q(F_x):
    """
    Compute q(x) = 0.5 * ||F(x)||^2.

    Arguments:
        F_x: Input vector (numpy array, shape (N,))
    
    Returns:
        q_x: Output scalar (float)
    """

    # Compute the norm of F(x)
    norm_F_x = np.linalg.norm(F_x)
    # Compute q(x)
    q_x = 0.5 * norm_F_x ** 2

    return q_x


def calculate_grad_F(x, u, J, B, W_fb, w):
    """
    Compute the gradient of F(x) with respect to x.

    Arguments:
        x: Input vector (numpy array, shape (N,))
        u: Input vector (numpy array, shape (O,))
        J: Recurrent weight matrix (numpy array, shape (N, N))
        B: Input weight matrix (numpy array, shape (N, O))
        W_fb: Feedback weight matrix (numpy array, shape (N, O))
        w: Readout weight matrix (numpy array, shape (N, O))
    
    Returns:
        grad_F_x: Gradient vector (numpy array, shape (N,N))
    """

    # Compute the activation
    r = np.tanh(x)        # (N,)
    # Compute the output (/feedback)
    z = r @ w             # (O,)
    
    # Compute the Jacobian of the activation
    sech2 = 1 - r**2      # shape (N,) - memory-efficient compared to np.diag(1 - r ** 2)

    # Compute the gradient of F(x)
    grad_F_x = -np.eye(len(x)) + (J + W_fb @ w.T) * sech2[np.newaxis, :]

    return grad_F_x


def calculate_grad_q(F_x, grad_F_x):
    """
    Compute the gradient of q(x) with respect to x.

    Arguments:
        F_x: Input vector (numpy array, shape (N,))
    
    Returns:
        grad_q_x: Gradient vector (numpy array, shape (N,))
    """

    # Compute the gradient of q(x)
    grad_q_x = F_x @ grad_F_x

    return grad_q_x




# ------------------------------------------------------------
# 5.2 FUNCTIONS TO FIND FIXED AND SLOW POINTS
# ------------------------------------------------------------
# Find fixed points by finding roots of F(x)
def find_fixed_points_by_roots_of_F(initial_guess, u, J, B, W_fb, w,
                                    num_candidate_pts = NUM_ATTEMPTS_FIXED_TRAJECTORIES,
                                    num_perturbs_of_candidates = NUM_ATTEMPS_FIXED_PERTURBATIONS,
                                    max_iter = MAXITER_FIXED,
                                    perturbs_scale = SCALE_OF_PERTURBATIONS_FIXED,
                                    tol_root = TOL_FIXED_ROOT,
                                    tol_distinct=TOL_FIXED):
    """
    Find fixed points by finding roots of F(x) = 0.

    Arguments:
        initial_guess: Initial guess for the fixed point (numpy array, shape (N,))
        u: Input vector (numpy array, shape (O,))
        J: Recurrent weight matrix (numpy array, shape (N, N))
        B: Input weight matrix (numpy array, shape (N, O))
        W_fb: Feedback weight matrix (numpy array, shape (N, O))
        w: Readout weight matrix (numpy array, shape (N, O))
        num_candidate_pts: Number of candidate points to generate
        num_perturbs_of_candidates: Number of perturbations to apply to each candidate point
        max_iter: Maximum number of iterations for the root-finding algorithm
        perturbs_scale: Scale of the perturbations
        tol_root: Tolerance for the root-finding algorithm
        tol_distinct: Tolerance for determining if two fixed points are distinct

    Returns:
        fixed_points: List of fixed points found (list of numpy arrays, each shape (N,))
    """

    fixed_points = []

    # Define a function to check if a found fixed point is new
    def is_new(fp, points, tol):
        for pt in points:
            if np.linalg.norm(fp - pt) < tol:
                return False
        return True

    # Initialise a list of candidate initial guesses
    candidate_guesses = [np.zeros_like(initial_guess), np.array(initial_guess)]

    # STEP 1) FIND CANDIDATE SOLUTIONS
    # STEP 1)A) ADD CANDIDATES FROM TRAJECTORIES
    # If the trajectories dictionary exists, add candidate guesses from final_values_x and, if we have more space, from trajectories_x
    try:
        final_vals = trajectories['final_values_x']
        trajs = trajectories['trajectories_x']
        num_final = len(final_vals)
        n = min(num_final, num_candidate_pts - 2)
        # Append the last n elements from final_values_x
        for cand in final_vals[-n:]:
            cand_np = cand.cpu().numpy() if hasattr(cand, 'cpu') else np.array(cand)
            candidate_guesses.append(cand_np)
        # If we have fewer candidates than num_candidate_pts-2, supplement with random rows from all trajectories_x
        if n < num_candidate_pts - 2:
            num_needed = (num_candidate_pts - 2) - n
            # Concatenate all trajectories_x into a single tensor
            all_traj = torch.cat(trajs, dim=0) if isinstance(trajs[0], torch.Tensor) else np.concatenate(trajs, axis=0)
            all_traj_np = all_traj.cpu().numpy() if hasattr(all_traj, 'cpu') else np.array(all_traj)
            # Randomly select num_needed rows from all_traj_np
            indices = np.random.choice(all_traj_np.shape[0], size=num_needed, replace=False)
            for idx in indices:
                candidate_guesses.append(all_traj_np[idx])
    except NameError:
        # trajectories not defined; skip this step
        pass

    # STEP 1)B) PERTURB CANDIDATES TO OBTAIN MORE CANDIDATES
    # Add random perturbations to candidate guesses to create more candidates
    base_candidates = candidate_guesses.copy()
    for base in base_candidates:
        for _ in range(num_perturbs_of_candidates):
            perturbation = np.random.normal(scale = perturbs_scale, size = base.shape)
            candidate_guesses.append(base + perturbation)

    # STEP 2) TRY FINDING FIXED POINTS FROM CHOSEN CANDIDATES
    # Try finding a fixed point from each candidate initial guess
    for guess in tqdm(candidate_guesses, desc="Finding fixed points", unit="point"):
        sol = root(lambda x: calculate_F(x, u, J, B, W_fb, w), guess,
                   method='lm', options={'maxiter': max_iter})
        if sol.success and np.linalg.norm(calculate_F(sol.x, u, J, B, W_fb, w)) < tol_root:
            fp = sol.x
            if is_new(fp, fixed_points, tol_distinct):
                fixed_points.append(fp)
                
    return fixed_points


# Find slow points by finding roots of grad_q(x) using the Gauss-Newton approximation of the Hessian
def find_slow_points_by_roots_of_grad_q(initial_guess, u, J, B, W_fb, w,
                                         num_candidate_pts=NUM_ATTEMPTS_SLOW_TRAJECTORIES,
                                         num_perturbs_of_candidates=NUM_ATTEMPS_SLOW_PERTURBATIONS,
                                         max_iter=MAXITER_SLOW,
                                         perturbs_scale=SCALE_OF_PERTURBATIONS_SLOW,
                                         tol_root=TOL_SLOW_ROOT,
                                         tol_distinct=TOL_SLOW):
    """
    Find slow points by finding roots of grad_q(x) = 0.
    We use the Gauss-Newton approximation of the Hessian 
    (i.e. H ≈ grad_F(x).T @ grad_F(x)) in our iterative procedure.
    
    Arguments:
        initial_guess: Initial guess for the slow point (numpy array, shape (N,))
        u: Input vector (numpy array, shape (O,))
        J: Recurrent weight matrix (numpy array, shape (N, N))
        B: Input weight matrix (numpy array, shape (N, O))
        W_fb: Feedback weight matrix (numpy array, shape (N, O))
        w: Readout weight matrix (numpy array, shape (N, O))
        num_candidate_pts: Number of candidate points to generate from trajectories.
        num_perturbs_of_candidates: Number of perturbations to apply to each candidate.
        max_iter: Maximum number of iterations for the Gauss-Newton procedure.
        perturbs_scale: Scale of the random perturbations.
        tol_root: Tolerance for the norm of grad_q(x) to decide convergence.
        tol_distinct: Tolerance to consider two slow points as distinct.
    
    Returns:
        slow_points: List of slow points found (list of numpy arrays, each shape (N,))
    """

    slow_points = []

    # Function to check if a found slow point is new
    def is_new(pt, points, tol):
        for existing in points:
            if np.linalg.norm(pt - existing) < tol:
                return False
        return True

    # STEP 1) FIND CANDIDATE SOLUTIONS
    # STEP 1)A) ADD CANDIDATES FROM TRAJECTORIES
    # Generate candidate initial guesses
    candidate_guesses = [np.zeros_like(initial_guess), np.array(initial_guess)]
    try:
        final_vals = trajectories['final_values_x']
        trajs = trajectories['trajectories_x']
        num_final = len(final_vals)
        n = min(num_final, num_candidate_pts - 2)
        # Append the last n elements from final_values_x
        for cand in final_vals[-n:]:
            cand_np = cand.cpu().numpy() if hasattr(cand, 'cpu') else np.array(cand)
            candidate_guesses.append(cand_np)
        # If we have fewer candidates than num_candidate_pts-2, supplement with random rows from all trajectories_x
        if n < num_candidate_pts - 2:
            num_needed = (num_candidate_pts - 2) - n
            all_traj = torch.cat(trajs, dim=0) if isinstance(trajs[0], torch.Tensor) else np.concatenate(trajs, axis=0)
            all_traj_np = all_traj.cpu().numpy() if hasattr(all_traj, 'cpu') else np.array(all_traj)
            # Randomly select num_needed rows from all_traj_np
            indices = np.random.choice(all_traj_np.shape[0], size=num_needed, replace=False)
            for idx in indices:
                candidate_guesses.append(all_traj_np[idx])
    except NameError:
        # trajectories not defined; only use the provided initial_guess
        pass

    # STEP 1)B) PERTURB CANDIDATES TO OBTAIN MORE CANDIDATES
    # Augment candidate guesses by applying random perturbations
    base_candidates = candidate_guesses.copy()
    for base in base_candidates:
        for _ in range(num_perturbs_of_candidates):
            perturbation = np.random.normal(scale = perturbs_scale, size = base.shape)
            candidate_guesses.append(base + perturbation)

    # STEP 2) TRY FINDING FIXED POINTS FROM CHOSEN CANDIDATES
    # Iterate over each candidate using a Gauss-Newton method
    for guess in tqdm(candidate_guesses, desc="Finding slow points", unit="point"):
        x_current = np.array(guess)
        for iteration in range(max_iter):
            # Compute F(x) and its Jacobian grad_F(x)
            F_x = calculate_F(x_current, u, J, B, W_fb, w)
            grad_F_x = calculate_grad_F(x_current, u, J, B, W_fb, w)
            # Evaluate grad_q(x) = F(x) @ grad_F(x)
            f_val = calculate_grad_q(F_x, grad_F_x)
            
            # Check convergence: if the norm of grad_q is below tolerance, break
            if np.linalg.norm(f_val) < tol_root:
                break

            # --- Gauss-Newton approximation ---
            # Approximate the Hessian of q(x) using H ≈ grad_F(x).T @ grad_F(x)
            H_approx = grad_F_x.T @ grad_F_x
            try:
                # Solve for the update: delta = H_approx^{-1} * grad_q(x)
                delta = np.linalg.solve(H_approx, f_val)
            except np.linalg.LinAlgError:
                # If H_approx is singular, use least squares to find delta
                delta = np.linalg.lstsq(H_approx, f_val, rcond=None)[0]
            # Update the current solution estimate
            x_current = x_current - delta
        
        F_x  = calculate_F(x_current, u, J, B, W_fb, w)
        grad_F_x = calculate_grad_F(x_current, u, J, B, W_fb, w)
        f_val = calculate_grad_q(F_x, grad_F_x)

        # After iterations, accept the candidate if converged and is new
        if np.linalg.norm(f_val) < tol_root and is_new(x_current, slow_points, tol_distinct):
            slow_points.append(x_current)

    return slow_points




# ------------------------------------------------------------
# 6 FIND FIXED AND SLOW POINTS
# ------------------------------------------------------------
# Find fixed points
fixed_points = find_fixed_points_by_roots_of_F(
    initial_guess = np.random.normal(size=(N,)),
    u = torch.zeros(O).cpu().numpy(),
    J = J_param.cpu().numpy(),
    B = B_param.cpu().numpy(),
    W_fb = W_fb_param.cpu().numpy(),
    w = w_param.cpu().numpy()
)

# Save the fixed points to file
save_variable(fixed_points, "fixed_points")

# Check if any fixed points were found
if len(fixed_points) == 0:
    print("No fixed points found.")
else:
    # Print the number of fixed points found and the norm of each
    print(f"Number of fixed points found: {len(fixed_points)}")
    for i, fp in enumerate(fixed_points):
        print(f"Fixed point {i}: norm = {np.linalg.norm(fp):.3e}")

# Find slow points
slow_points = find_slow_points_by_roots_of_grad_q(
    initial_guess = np.random.normal(size=(N,)),
    u = torch.zeros(O).cpu().numpy(),
    J = J_param.cpu().numpy(),
    B = B_param.cpu().numpy(),
    W_fb = W_fb_param.cpu().numpy(),
    w = w_param.cpu().numpy()
)

# Save the slow points to file
save_variable(slow_points, "slow_points")

# Check if any slow points were found
if len(slow_points) == 0:
    print("No slow points found.")
else:
    # Print the number of slow points found and the norm of each
    print(f"Number of slow points found: {len(slow_points)}")
    for i, sp in enumerate(slow_points):
        print(f"Slow point {i}: norm = {np.linalg.norm(sp):.3e}")