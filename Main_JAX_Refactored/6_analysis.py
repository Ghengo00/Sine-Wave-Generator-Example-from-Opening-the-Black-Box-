"""
Fixed point and slow point analysis functions.
"""


import numpy as np
from tqdm import tqdm
from scipy.optimize import root, least_squares
from scipy.linalg import eig
import time


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================
def compute_F(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute the evolution equation for the RNN, F(x) = -x + J * tanh(x) + B * u + b_x.

    Notable arguments:
        x_np   : current state, shape (N,)
        u_val  : scalar input value
    """
    return -x_np + J_np @ np.tanh(x_np) + (B_np * np.array([u_val])).sum(axis=1) + b_x_np


def compute_jacobian(x_np, J_np):
    """
    Compute the Jacobian of the evolution equation F(x), J_F(x) = -I + J * diag(1 - tanh(x)^2).
    """
    # diag(1 - tanh(x)^2) is a diagonal matrix with the derivative of tanh on the diagonal
    diag_term = 1 - np.tanh(x_np) ** 2

    # Compute the Jacobian J_F(x) = -I + J * diag(1 - tanh(x)^2)
    return -np.eye(len(x_np)) + J_np * diag_term[np.newaxis, :]


def compute_grad_q(x_np, u_val, J_np, B_np, b_x_np):
    """
    Compute the gradient of q(x) = |F(x)|^2, grad_q = F(x) @ J_F(x).
    """
    # Compute F(x) = -x + J * tanh(x) + B * u + b_x
    F_x = compute_F(x_np, u_val, J_np, B_np, b_x_np)

    # Compute the Jacobian J_F(x) = -I + J * diag(1 - tanh(x)^2)
    J_F = compute_jacobian(x_np, J_np)

    # Compute the gradient of q(x) = |F(x)|^2, which is grad_q = F(x) @ J_F
    grad_q = F_x @ J_F

    return np.array(grad_q)


# =============================================================================
# FIND AND ANALYZE POINTS FUNCTIONS
# =============================================================================
def find_points(
    x0_guess, u_const, J_trained, B_trained, b_x_trained,
    mode: str,
    num_attempts=None, tol=None, maxiter=None, alt_inits=None, gaussian_std=None):
    """
    Unified finder for fixed points and slow points.

    Notable arguments:
        mode        : 'fixed' for fixed point search or 'slow' for slow point search
        num_attempts: number of attempts to find a point
        tol         : tolerance for two fixed/slow points to be considered distinct
        maxiter     : maximum iterations for the solver
        alt_inits   : alternative initial conditions from training trajectories
        gaussian_std: standard deviation for Gaussian noise in perturbed initial conditions

    Returns:
        points      : list of found fixed or slow points, each point is a NumPy array of shape (N,)
    """
    from _1_config import (NUM_ATTEMPTS, TOL, MAXITER, GAUSSIAN_STD, 
                       NUM_ATTEMPTS_SLOW, TOL_SLOW, MAXITER_SLOW, 
                       GAUSSIAN_STD_SLOW, SLOW_POINT_CUT_OFF)
    
    mode = mode.lower()
    
    if mode == 'fixed':
        # Use the provided parameters or fixed point search defaults
        num_attempts = num_attempts if num_attempts is not None else NUM_ATTEMPTS
        tol = tol if tol is not None else TOL
        maxiter = maxiter if maxiter is not None else MAXITER
        gaussian_std = gaussian_std if gaussian_std is not None else GAUSSIAN_STD
        # Define the root solver for fixed points
        solver = lambda x: root(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                x, method='lm', options={'maxiter': maxiter})
        success_cond = lambda sol: sol.success
        extract = lambda sol: sol.x

    elif mode == 'slow':
        # Use the provided parameters or slow point search defaults
        num_attempts = num_attempts if num_attempts is not None else NUM_ATTEMPTS_SLOW
        tol = tol if tol is not None else TOL_SLOW
        maxiter = maxiter if maxiter is not None else MAXITER_SLOW
        gaussian_std = gaussian_std if gaussian_std is not None else GAUSSIAN_STD_SLOW
        # Define the least squares solver for slow points
        solver = lambda x: least_squares(lambda x: np.array(compute_F(x, u_const, J_trained, B_trained, b_x_trained)),
                                         x, method='lm', max_nfev=maxiter)
        success_cond = lambda sol: sol.success and np.linalg.norm(sol.fun) < SLOW_POINT_CUT_OFF
        extract = lambda sol: sol.x
    
    else:
        raise ValueError("Argument 'mode' must be either 'fixed' or 'slow'.")

    points = []

    def _attempt(x_init):
        try:
            sol = solver(x_init)
            if success_cond(sol):
                return extract(sol)
        except Exception as e:
            print(f"{mode.capitalize()}-point search failed: {e}")
        return None

    # (I) Original guess
    pt = _attempt(x0_guess)
    if pt is not None:
        points.append(pt)

    # (II) Gaussian perturbations of the original guess
    for _ in tqdm(range(num_attempts - 1), desc=f"Finding {mode} points (perturbations)"):
        pert = x0_guess + np.random.normal(0, gaussian_std, size=x0_guess.shape)
        pt = _attempt(pert)
        if pt is not None and all(np.linalg.norm(pt - p) >= tol for p in points):
            points.append(pt)

    # (III) Alternative initial conditions from training trajectories
    if alt_inits is not None:
        for cand in tqdm(alt_inits, desc=f"Finding {mode} points (initial conditions from training)"):
            pt = _attempt(cand)
            if pt is not None and all(np.linalg.norm(pt - p) >= tol for p in points):
                points.append(pt)

    return points


def analyze_points(fixed_points, J_trained):
    """
    For each fixed point, compute the Jacobian and its unstable eigenvalues.

    Returns:
        jacobians      : list of Jacobian matrices at each fixed point, each matrix is a NumPy array of shape (N, N)
        unstable_freqs : list of lists, where each inner list contains the unstable eigenvalues (with positive real part) for the corresponding fixed point
    """
    jacobians = []
    unstable_freqs = []

    for x_star in fixed_points:
        # Compute Jacobians at the fixed point
        J_eff = np.array(compute_jacobian(x_star, J_trained))
        jacobians.append(J_eff)

        # Compute eigenvalues of the Jacobian
        eigenvals, _ = eig(J_eff)

        # Find all the unstable eigenvalues for the given fixed point (i.e., those with positive real part)
        unstable_idx = np.where(np.real(eigenvals) > 0)[0]
        if len(unstable_idx) > 0:
            # Store all unstable eigenvalues for this fixed point
            unstable_freqs.append(list(eigenvals[unstable_idx]))
        else:
            # If there are no unstable eigenvalues, append an empty list
            unstable_freqs.append([])
    
    return jacobians, unstable_freqs


# =============================================================================
# EXECUTION FUNCTIONS
# =============================================================================
def find_and_analyze_points(state_traj_states, state_fixed_point_inits, params, point_type="fixed"):
    """
    Unified function to find and analyze either fixed points or slow points for all tasks.
    
    Arguments:
        state_traj_states: trajectory states from training
        state_fixed_point_inits: initial states for fixed point search
        params: trained model parameters
        point_type: str, either "fixed" or "slow" to specify which type of points to find
    
    Returns:
        all_points              : list of lists of points, one list per task
        all_jacobians           : list of lists of Jacobians, one list per task  
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies
    """
    from _1_config import (num_tasks, static_inputs, NUM_ATTEMPTS, TOL, MAXITER, GAUSSIAN_STD,
                       NUM_ATTEMPTS_SLOW, TOL_SLOW, MAXITER_SLOW, GAUSSIAN_STD_SLOW)
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set parameters based on point type
    if point_type == "fixed":
        num_attempts = NUM_ATTEMPTS
        tol = TOL
        maxiter = MAXITER
        gaussian_std = GAUSSIAN_STD
        search_name = "fixed point"
    else:  # point_type == "slow"
        num_attempts = NUM_ATTEMPTS_SLOW
        tol = TOL_SLOW
        maxiter = MAXITER_SLOW
        gaussian_std = GAUSSIAN_STD_SLOW
        search_name = "slow point"
    
    # Extract trained parameters
    J_trained = np.array(params["J"])
    B_trained = np.array(params["B"])
    b_x_trained = np.array(params["b_x"])
    
    # Initialize lists to store results
    all_points = []              # List of lists of points, one list per task
    all_jacobians = []           # List of lists of Jacobians, one list per task
    all_unstable_eig_freq = []   # List of lists of lists of unstable frequencies
    
    # Track search time
    start_time = time.time()
    print(f"\nStarting {search_name} search...")
    
    # Iterate over each task to find points
    for j in range(num_tasks):
        if j % 5 == 0:
            print(f"\n Processing Task {j} of {num_tasks}...")

        u_const = static_inputs[j]
        x0_guess = state_fixed_point_inits[j]
        
        # Build fallback initializations from training trajectories
        idxs = np.random.choice(len(state_traj_states),
                                size=min(num_tasks, num_attempts), replace=False)
        fallback_inits = []
        for idx in idxs:
            traj = state_traj_states[idx]                   # shape (num_steps_train+1, N)
            row = np.random.randint(0, traj.shape[0])
            fallback_inits.append(traj[row])                # shape (N,)
        
        # Find the points
        task_points = find_points(
            x0_guess, u_const, J_trained, B_trained, b_x_trained,
            mode=point_type,
            num_attempts=num_attempts, 
            tol=tol, 
            maxiter=maxiter, 
            alt_inits=fallback_inits, 
            gaussian_std=gaussian_std
        )
        all_points.append(task_points)
        
        # Analyze the points to compute Jacobians and unstable eigenvalues
        task_jacobians, task_unstable_freqs = analyze_points(task_points, J_trained)
        all_jacobians.append(task_jacobians)
        all_unstable_eig_freq.append(task_unstable_freqs)
    
    # Calculate search time
    search_time = time.time() - start_time
    print(f"{search_name.capitalize()} search completed in {search_time:.2f} seconds")
    
    return all_points, all_jacobians, all_unstable_eig_freq


def generate_point_summaries(point_type="fixed", all_points=None, all_unstable_eig_freq=None):
    """
    Unified function to generate summaries for either fixed points or slow points.
    
    Arguments:
        point_type              : str, either "fixed" or "slow" to specify which type of points
        all_points              : list of lists of points, one list per task
        all_unstable_eig_freq   : list of lists of lists of unstable frequencies
    """
    from _1_config import num_tasks, TEST_INDICES
    
    # Validate point_type parameter
    if point_type not in ["fixed", "slow"]:
        raise ValueError("point_type must be either 'fixed' or 'slow'")
    
    # Set point names based on point type
    point_name = "Fixed Points" if point_type == "fixed" else "Slow Points"
    point_name_lower = "fixed points" if point_type == "fixed" else "slow points"
    point_name_singular = "fixed point" if point_type == "fixed" else "slow point"
    
    print(f"\nSummary of {point_name} Found:\n")
    points_count = {}
    unstable_eigs_count = {}

    print(f"1. {point_name} per Task:")
    for j in range(num_tasks):
        # Count the number of points found for this task
        num_points = len(all_points[j])
        print(f"Task {j}: {num_points} {point_name_lower} found")
        # Update the count of tasks with this number of points
        points_count[num_points] = points_count.get(num_points, 0) + 1
        
        for point_freqs in all_unstable_eig_freq[j]:
            # Count the number of unstable eigenvalues for this point
            num_unstable = len(point_freqs)
            # Update the count of points with this number of unstable eigenvalues
            unstable_eigs_count[num_unstable] = unstable_eigs_count.get(num_unstable, 0) + 1

    print(f"\n2. Tasks Grouped by Number of {point_name}:")
    for num_points in sorted(points_count.keys()):
        count = points_count[num_points]
        print(f"  {count} tasks have {num_points} {point_name_singular}{'s' if num_points != 1 else ''}")

    print(f"\n3. {point_name} Grouped by Number of Unstable Eigenvalues:")
    for num_unstable in sorted(unstable_eigs_count.keys()):
        count = unstable_eigs_count[num_unstable]
        print(f"  {count} {point_name_singular}{'s' if count != 1 else ''} have {num_unstable} unstable eigenvalue{'s' if num_unstable != 1 else ''}")

    print("\nDetailed Information for Selected Tasks:")
    for j in TEST_INDICES:
        print(f"\nTask {j}:")
        print(f"Number of distinct {point_name_lower} found: {len(all_points[j])}")
        for i, (point, freqs) in enumerate(zip(all_points[j], all_unstable_eig_freq[j])):
            print(f"  {point_name_singular.capitalize()} {i+1}:")
            print(f"    Number of unstable eigenvalues: {len(freqs)}")
            if len(freqs) > 0:
                print("    Unstable eigenvalues:")
                for ev in freqs:
                    print(f"      Real: {np.real(ev):.4f}, Imag: {np.imag(ev):.4f}")
            print(f"    Norm: {np.linalg.norm(point):.4f}")