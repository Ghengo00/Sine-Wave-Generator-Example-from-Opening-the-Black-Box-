#!/usr/bin/env python3
"""
Eigenvalue Complex Plane Plotter

This script loads results from sparsity and L1 regularization experiments
and plots eigenvalues in the complex plane with color coding based on training iterations.

Usage:
    python eigenvalue_evolution_plotter.py

Modify the FOLDER_PATH variable below to point to your experiment results folder.
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import glob
from pathlib import Path




# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Specify the path to your experiment results folder
FOLDER_PATH = "/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Outputs/Sparsity_Experiments_20250910_130454_16_Same_as_11_but_with_Jacobian_Eigenvalue_Tracking_For_3_Frequencies_Every_100_Epochs"

# Choose which eigenvalue types to plot
EIGENVALUE_TYPES = 'jacobian'  # Options: 'jacobian', 'connectivity', 'both'

# Choose which optimization phases to include
OPTIMIZATION_PHASES = 'both'  # Options: 'adam', 'lbfgs', 'both'

# Figure size and DPI settings
FIGURE_SIZE = (15, 10)
DPI = 300




# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_pickle_file(filepath):
    """Load a pickle file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_eigenvalue_data_files(folder_path):
    """Find all eigenvalue_data pickle files in the folder structure."""
    eigenvalue_files = []
    
    # Look for individual eigenvalue_data files in subfolders
    pattern1 = os.path.join(folder_path, "**/eigenvalue_data_*.pkl")
    files1 = glob.glob(pattern1, recursive=True)
    eigenvalue_files.extend(files1)
    
    # Look for complete_results files that contain eigenvalue_data
    pattern2 = os.path.join(folder_path, "**/complete_results_*.pkl")
    files2 = glob.glob(pattern2, recursive=True)
    eigenvalue_files.extend(files2)
    
    # Look for complete experiment files
    pattern3 = os.path.join(folder_path, "complete_*_experiment_*.pkl")
    files3 = glob.glob(pattern3)
    eigenvalue_files.extend(files3)
    
    return eigenvalue_files


def extract_experiment_info(filepath):
    """Extract experiment type and parameters from file path."""
    path_parts = Path(filepath).parts
    filename = Path(filepath).name
    
    # Extract sparsity or L1 regularization parameter
    if "sparsity" in filename.lower():
        # Extract sparsity value from path parts
        if "Sparsity_" in str(filepath):
            for part in path_parts:
                if part.startswith("Sparsity_"):
                    # Skip directory names that don't contain numeric sparsity values
                    # (e.g., "Sparsity_Experiments_...")
                    sparsity_part = part.replace("Sparsity_", "")
                    
                    # Check if this looks like a sparsity value (should start with digit or 'p')
                    if sparsity_part and (sparsity_part[0].isdigit() or sparsity_part.startswith('p')):
                        try:
                            sparsity_str = sparsity_part.replace("p", ".")
                            return "sparsity", float(sparsity_str)
                        except ValueError:
                            # Skip this part if it can't be converted to float
                            continue
        
        # Extract from filename
        parts = filename.split("_")
        for i, part in enumerate(parts):
            if part == "sparsity" and i + 1 < len(parts):
                try:
                    return "sparsity", float(parts[i + 1])
                except ValueError:
                    continue
                    
    elif "l1" in filename.lower():
        # Extract L1 regularization value
        parts = filename.split("_")
        for i, part in enumerate(parts):
            if part.lower() == "l1" and i + 1 < len(parts):
                try:
                    return "l1", float(parts[i + 1])
                except ValueError:
                    continue
    
    return "unknown", 0.0




# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_eigenvalue_evolution(eigenvalue_data, title_suffix="", ax=None):
    """
    Plot eigenvalue evolution in the complex plane over training iterations.
    
    Args:
        eigenvalue_data: Dictionary with optimization phases as keys
        title_suffix: Additional text for the plot title
        ax: Matplotlib axis to plot on (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    all_iterations = []
    all_eigenvalues = []
    all_labels = []
    
    # Process each optimization phase
    for phase in ['adam', 'lbfgs']:
        if phase not in eigenvalue_data or not eigenvalue_data[phase]:
            continue
            
        if OPTIMIZATION_PHASES not in ['both', phase]:
            continue
        
        phase_data = eigenvalue_data[phase]
        
        for iteration, eigenval_dict in phase_data:
            # Process each eigenvalue type
            for eigenval_type in ['jacobian', 'connectivity']:
                if eigenval_type not in eigenval_dict:
                    continue
                    
                if EIGENVALUE_TYPES not in ['both', eigenval_type]:
                    continue
                
                eigenvals = eigenval_dict[eigenval_type]
                
                if isinstance(eigenvals, dict):
                    # Handle frequency-indexed eigenvalues
                    for freq_idx, freq_eigenvals in eigenvals.items():
                        if len(freq_eigenvals) > 0:
                            for eigenval in freq_eigenvals:
                                if np.isfinite(eigenval):  # Skip NaN/inf values
                                    all_iterations.append(iteration)
                                    all_eigenvalues.append(eigenval)
                                    all_labels.append(f"{phase}_{eigenval_type}_freq{freq_idx}")
                else:
                    # Handle simple eigenvalue arrays
                    if len(eigenvals) > 0:
                        for eigenval in eigenvals:
                            if np.isfinite(eigenval):  # Skip NaN/inf values
                                all_iterations.append(iteration)
                                all_eigenvalues.append(eigenval)
                                all_labels.append(f"{phase}_{eigenval_type}")
    
    if not all_iterations:
        # Provide more detailed diagnostic information
        debug_info = []
        total_entries = 0
        empty_arrays = 0
        
        for phase in ['adam', 'lbfgs']:
            if phase not in eigenvalue_data or not eigenvalue_data[phase]:
                debug_info.append(f"No {phase} data")
                continue
                
            phase_data = eigenvalue_data[phase]
            total_entries += len(phase_data)
            
            for iteration, eigenval_dict in phase_data:
                for eigenval_type in ['jacobian', 'connectivity']:
                    if eigenval_type not in eigenval_dict:
                        continue
                    
                    eigenvals = eigenval_dict[eigenval_type]
                    if isinstance(eigenvals, dict):
                        for freq_idx, freq_eigenvals in eigenvals.items():
                            if len(freq_eigenvals) == 0:
                                empty_arrays += 1
                    else:
                        if len(eigenvals) == 0:
                            empty_arrays += 1
        
        debug_msg = f'No finite eigenvalue data found\n'
        debug_msg += f'Total entries: {total_entries}\n'
        debug_msg += f'Empty arrays: {empty_arrays}\n'
        debug_msg += f'Available types: {list(eigenvalue_data.keys())}'
        
        ax.text(0.5, 0.5, debug_msg, 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Eigenvalue Evolution in Complex Plane{title_suffix}')
        return
    
    # Convert to numpy arrays
    all_iterations = np.array(all_iterations)
    all_eigenvalues = np.array(all_eigenvalues)
    
    # Extract real and imaginary parts for complex plane plotting
    real_parts = np.real(all_eigenvalues)
    imag_parts = np.imag(all_eigenvalues)
    
    # Create color map based on iterations
    norm = Normalize(vmin=min(all_iterations), vmax=max(all_iterations))
    colors = cm.viridis(norm(all_iterations))
    
    # Create scatter plot in complex plane
    scatter = ax.scatter(real_parts, imag_parts, c=all_iterations, 
                        cmap='viridis', alpha=0.6, s=10)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Iteration', rotation=270, labelpad=15)
    
    # Add unit circle for stability reference
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle_real = np.cos(theta)
    unit_circle_imag = np.sin(theta)
    ax.plot(unit_circle_real, unit_circle_imag, 'r--', alpha=0.5, linewidth=1, 
            label='Unit circle (stability boundary)')
    
    # Customize plot
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'Eigenvalue Evolution in Complex Plane{title_suffix}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')  # Make the plot square for proper circle
    
    # Add axes through origin
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=8)
    
    return ax




# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to load data and create plots."""
    
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Folder path '{FOLDER_PATH}' does not exist.")
        print("Please modify the FOLDER_PATH variable in the script.")
        return
    
    print(f"Searching for eigenvalue data in: {FOLDER_PATH}")
    
    # Find all eigenvalue data files
    eigenvalue_files = find_eigenvalue_data_files(FOLDER_PATH)
    
    if not eigenvalue_files:
        print("No eigenvalue data files found!")
        print("Looking for files matching patterns:")
        print("  - eigenvalue_data_*.pkl")
        print("  - complete_results_*.pkl")
        print("  - complete_*_experiment_*.pkl")
        return
    
    print(f"Found {len(eigenvalue_files)} potential data files")
    
    # Process each file
    valid_files = []
    for filepath in eigenvalue_files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        data = load_pickle_file(filepath)
        if data is None:
            continue
        
        # Extract eigenvalue_data from different file types
        eigenvalue_data = None
        
        if isinstance(data, dict):
            if 'eigenvalue_data' in data:
                eigenvalue_data = data['eigenvalue_data']
            elif 'adam' in data or 'lbfgs' in data:
                eigenvalue_data = data
            elif any('eigenvalue_data' in str(v) for v in data.values() if isinstance(v, dict)):
                # Look for eigenvalue_data in nested structures
                for key, value in data.items():
                    if isinstance(value, dict) and 'eigenvalue_data' in value:
                        eigenvalue_data = value['eigenvalue_data']
                        break
        elif hasattr(data, 'get'):
            eigenvalue_data = data.get('eigenvalue_data', data)
        else:
            eigenvalue_data = data
        
        if eigenvalue_data is None or not eigenvalue_data:
            print(f"  No eigenvalue data found in this file")
            continue
        
        # Extract experiment info
        exp_type, exp_value = extract_experiment_info(filepath)
        
        valid_files.append({
            'filepath': filepath,
            'eigenvalue_data': eigenvalue_data,
            'exp_type': exp_type,
            'exp_value': exp_value
        })
        
        print(f"  Found eigenvalue data - {exp_type}: {exp_value}")
    
    if not valid_files:
        print("\nNo valid eigenvalue data found in any files!")
        return
    
    print(f"\nCreating plots for {len(valid_files)} datasets...")
    
    # Sort files by experiment value for consistent plotting
    valid_files.sort(key=lambda x: x['exp_value'])
    
    # Create individual plots for each experiment
    for i, file_info in enumerate(valid_files):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        
        title_suffix = f" ({file_info['exp_type']}: {file_info['exp_value']})"
        plot_eigenvalue_evolution(file_info['eigenvalue_data'], title_suffix, ax)
        
        # Save individual plot
        output_filename = f"eigenvalue_complex_plane_{file_info['exp_type']}_{file_info['exp_value']:.3f}.pdf"
        output_path = os.path.join(FOLDER_PATH, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_filename}")
        plt.close()
    
    print("\nPlotting complete!")
    print(f"All plots saved to: {FOLDER_PATH}")


if __name__ == "__main__":
    main()