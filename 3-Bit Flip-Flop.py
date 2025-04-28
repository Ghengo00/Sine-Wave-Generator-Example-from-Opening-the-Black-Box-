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


NUM_ATTEMPTS_FIXED   = 50      # number of candidate initialisations for fixed points
NUM_ATTEMPTS_SLOW    = 50      # numer of candidate initialisations for slow points
TOL_FIXED            = 1e-2    # tolerance for finding distinct fixed points
TOL_SLOW             = 1e-2    # tolerance for finding distinct slow points
MAXITER_FIXED        = 5000    # maximum number of iterations for finding fixed points
MAXITER_SLOW         = 5000    # maximum number of iterations for finding slow points




# -------------------------------
# 1.4. SAVING FUNCTIONS
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
def simulate(u_seq, z_tgt, train=True, teacher_forcing_steps=TEACHER_FORCING_STEPS):
    """
    Simulates the network with the given input sequence u_seq.
    The simulation is done using the Euler method, which is a simple numerical method for solving ordinary differential equations.

    Parameters:
    u_seq: input sequence, dimensions (steps,O)
    train: boolean, if True, the network is trained using the FORCE algorithm, if False, the network is run in inference mode with fixed weights
    
    Returns:
    zs: output sequence, dimensions (steps,O)
    Note that the outputs are returned on device, because after this they are only used for the loss calculation
    """

    steps = u_seq.shape[0]
    x  = torch.zeros(N,device=device)
    zs = torch.zeros(steps,O,device=device)
    
    for t in range(steps):
        r  = torch.tanh(x)
        z  = r @ w_param + b_z_param            # (3,)
        zs[t] = z
        
        if train:
            force_step(r, z, z_tgt[t])
            # ATTENTION: Normally, you would not use this (set TEACHER_FORCING_STEPS to 0)
            # However, here I changed it to try to improve RNN training in the early stages
            z_fb = z_tgt[t] if t < teacher_forcing_steps else z
        else:
            z_fb = z

        # Update the state of the network
        # ATTENTION: Normally, this should be:
        # x += dt * (J_param @ r + B_param @ u_seq[t] + W_fb_param @ z)
        # However, here I changed it to try to improve RNN training by giving it a slower leak
        x = (1. - lam * dt) * x + dt * (J_param @ r + B_param @ u_seq[t] + W_fb_param @ z_fb)

    return zs.cpu()




# ------------------------------------------------------------
# 3.3 TRAINING
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


def train_force(max_iter=force_max_iter, tol=force_tol):
    """
    Trains the network using FORCE until max_iter iterations or until the MSE falls below tol.
    Every 10 iterations, generates and plots a test run with current network parameters.
    Stores and restores the parameters that yield the smallest loss.
    
    Parameters:
    max_iter: int - maximum number of FORCE passes to run
    tol: float - loss tolerance for early stopping
    """

    global w_param, P  # access global parameters updated by FORCE
    best_loss = float('inf')
    best_w = w_param.clone()
    best_P = P.clone()

    pbar = tqdm(range(max_iter), desc="Training iterations", unit="iter")
    for i in pbar:
        start_time = time.time()
        
        u_seq, z_tgt = generate_pulse_task(dt, T_train, pulse_prob, pulse_amp)
        zs = simulate(u_seq, z_tgt, train=True)
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
            u_test, z_target_test = generate_pulse_task(dt, T_test, pulse_prob, pulse_amp)
            z_test = simulate(u_test, z_target_test, train=False)
            plot_test_run(u_test, z_test, z_target_test, dt, to_save=False)

        iter_time = time.time() - start_time
        pbar.set_postfix(loss=f'{loss:.3e}', time=f'{iter_time:.3f}s')

        # Early stopping if the loss is below the tolerance
        if loss < tol:
            break

    # Restore the best parameters before finishing training
    w_param = best_w.clone()
    P = best_P.clone()


train_force(force_max_iter, force_tol)

# Save the final parameters
save_variable(w_param.cpu().numpy(), 'w')
save_variable(P.cpu().numpy(), 'P')




# ------------------------------------------------------------
# 4.1 TESTING
# ------------------------------------------------------------
u_test, z_target_test = generate_pulse_task(dt, T_test, pulse_prob, pulse_amp)
z_test = simulate(u_test, z_target_test, train=False)




# ------------------------------------------------------------
# 4.2 PLOTTING
# ------------------------------------------------------------
plot_test_run(u_test, z_test, z_target_test, dt)