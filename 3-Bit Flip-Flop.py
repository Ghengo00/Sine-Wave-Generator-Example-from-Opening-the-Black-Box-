# ------------------------------------------------------------
# 0. IMPORTS
# ------------------------------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# ------------------------------------------------------------
# 1. PARAMETERS (3-Bit Flip-Flop Task)
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




# ------------------------------------------------------------
# 1.1 WEIGHTS
# ------------------------------------------------------------
J_param   = g*torch.randn(N,N,device=device)/np.sqrt(N)     # recurrent weights, spectral radius g achieved by scaling by g / sqrt(N)
B_param   = torch.randn(N,O,device=device)/np.sqrt(O)       # input weights, couples the external pulses to the network
W_fb_param= torch.randn(N,O,device=device)/np.sqrt(N)       # feedback weights, sends the current output back into the network
w_param   = torch.zeros(N,O,device=device)                  # read-out weights - this is the only thing that FORCE will train
b_z_param = torch.zeros(O,device=device)                    # biases, kept 0, only included to match the sine wave generator example
P         = torch.eye(N,device=device)/alpha                # RLS state, P = I / alpha is the inverse correlation matrix used by FORCE




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
# 3. FORCE UPDATE
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
    Pr  = P @ r                               # dimesnions (N,)
    k   = Pr / (1. + (r * Pr).sum())          # Sherman-Morrison gain vector, dimesions (N,)
    P   = P - torch.ger(k, Pr)                # rank-1 update, torch.ger(a,b) = a[:,None] @ b[None,:] produces an outer product
    w_param -= torch.ger(k, err)              # rank-1 update, torch.ger(a,b) = a[:,None] @ b[None,:] produces an outer product




# ------------------------------------------------------------
# 4. SIMULATION CORE
# ------------------------------------------------------------
@torch.no_grad()
def simulate(u_seq, train=True):
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
        
        if train: force_step(r, z, z_tgt[t])
        # Update the state of the network
        x += dt*(-x + J_param @ r + B_param @ u_seq[t] + W_fb_param @ z)

    return zs.cpu()




# ------------------------------------------------------------
# 5. TRAINING
# ------------------------------------------------------------
u_seq, z_tgt = generate_pulse_task(dt, T_train, pulse_prob, pulse_amp)
zs = simulate(u_seq, train=True)
loss = torch.mean((zs - z_tgt.cpu())**2).item()
print(f'After one FORCE pass: MSE = {loss:.3e}')




# ------------------------------------------------------------
# 6. TESTING
# ------------------------------------------------------------
u_test, z_target_test = generate_pulse_task(dt, T_test, pulse_prob, pulse_amp)
z_test = simulate(u_test, train=False)




# ------------------------------------------------------------
# 7. PLOTTING
# ------------------------------------------------------------
t = np.arange(u_test.shape[0]) * dt
colors=['r','g','b']
fig,ax = plt.subplots(3,1,figsize=(10,6),sharex=True)
for k in range(3):
    ax[k].plot(t, u_test[:,k].cpu(),  '--', color=colors[k],alpha=.4,label='input')
    ax[k].plot(t, z_test[:,k],        color=colors[k],       label='output')
    ax[k].set_ylabel(f'bit {k}')
    ax[k].legend(loc='upper right')
ax[-1].set_xlabel('time (s)')
plt.tight_layout();  plt.show()