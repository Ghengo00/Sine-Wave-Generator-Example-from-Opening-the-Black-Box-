"""
Data generation functions.
"""


import jax.numpy as jnp
from config import omegas, static_inputs, time_drive, time_train, num_steps_drive, num_steps_train, I


def get_drive_input(task):
    """
    Get the driving phase input for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        drive_input: driving phase input, shape (num_steps_drive, I)
    """
    return (jnp.sin(omegas[task] * time_drive) + static_inputs[task])[:, None]  # shape (num_steps_drive, I)


def get_train_input(task):
    """
    Get the training phase input for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        train_input: training phase input, shape (num_steps_train, I)
    """
    return jnp.full((num_steps_train, I), static_inputs[task], dtype=jnp.float32)  # shape (num_steps_train, I)


def get_train_target(task):
    """
    Get the training phase target for a given task.
    
    Arguments:
        task: index of the task
    
    Returns:
        train_target: training phase target, shape (num_steps_train,)
    """
    return jnp.sin(omegas[task] * time_train)  # shape (num_steps_train,)


def generate_all_inputs_and_targets():
    """
    Generate all driving inputs, training inputs, and training targets for all tasks.
    
    Returns:
        drive_inputs: shape (num_tasks, num_steps_drive, I)
        train_inputs: shape (num_tasks, num_steps_train, I)  
        train_targets: shape (num_tasks, num_steps_train)
    """
    # Driving phase inputs: shape (num_tasks, num_steps_drive, I)
    drive_inputs = (
        jnp.sin(omegas[:, None] * time_drive[None, :])  # omegas[:, None] is (num_tasks, 1); time_drive[None, :] is (1, num_steps_drive)
                                                        # jnp.sin(...) is (num_tasks, num_steps_drive)
        + static_inputs[:, None]                        # static_inputs[:, None] is (num_tasks, 1); this broadcasts to (num_tasks, num_steps_drive)
    )[..., None]                                        # Add a new axis to make it (num_tasks, num_steps_drive, I)

    # Training phase inputs: shape (num_tasks, num_steps_train, I)
    train_inputs = jnp.broadcast_to(
        static_inputs[:, None, None],                   # static_inputs[:, None, None] is (num_tasks, 1, I); this broadcasts to (num_tasks, num_steps_train, I)
        (len(omegas), num_steps_train, I)
    ).astype(jnp.float32)

    # Training phase targets: shape (num_tasks, num_steps_train)
    train_targets = jnp.sin(omegas[:, None] * time_train[None, :])  # omegas[:, None] is (num_tasks, 1); time_train[None, :] is (1, num_steps_train)
    
    return drive_inputs, train_inputs, train_targets