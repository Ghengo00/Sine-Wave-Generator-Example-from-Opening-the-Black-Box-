"""
Standalone PCA Analysis Script for Existing Training Results
============================================================

This script loads previously saved training results from pickle files and runs PCA analysis
without needing to retrain the model. All parameters can be manually specified, making it
independent of config files.

Features:
- Load data from any training run directory
- Manual parameter specification (no config file dependency)
- Interactive run selection
- Flexible analysis options (fixed points, slow points)
- Skip initial time steps from trajectory analysis
- Custom output directory for results (with timestamp)
- Batch analysis with multiple skip values
- Tanh transformation option for trajectory states
- Comprehensive error handling and progress reporting

Usage Examples:
    # Interactive mode - select from available runs
    python pca_analysis_standalone.py
    
    # Specify a specific run directory
    python pca_analysis_standalone.py --run_dir /path/to/output/directory
    
    # Include slow points analysis
    python pca_analysis_standalone.py --include_slow_points
    
    # Specify custom output directory for results
    python pca_analysis_standalone.py --output_name "custom_pca_analysis"
    
    # Run multiple iterations with different trajectory truncations
    python pca_analysis_standalone.py --skip_steps 0,10,20,50 --output_dir pca_custom_analysis
    
    # Apply tanh transformation to trajectories before PCA
    python pca_analysis_standalone.py --skip_steps 0,10,20 --apply_tanh
    
    # Combine interactive mode with custom output, multiple iterations, and tanh transformation
    python pca_analysis_standalone.py --skip_steps 0,10,20 --output_dir pca_results --include_slow_points --apply_tanh
"""

import os
import sys
import argparse
import numpy as np

# Add the current directory to path for importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_variable, find_files_by_pattern, find_output_directories, 
    list_available_runs, Timer, set_custom_output_dir
)
from pca_analysis import run_pca_analysis


class TrainingDataLoader:
    """Class to handle loading and processing of training data from pickle files."""
    
    def __init__(self, run_directory):
        """
        Initialize the data loader.
        
        Arguments:
            run_directory: path to the directory containing training output files
        """
        self.run_directory = run_directory
        self.data = {}
        self.params = None
        self.state_traj_states = None
        self.state_fixed_point_inits = None
        
    def load_required_data(self):
        """Load all required data files for PCA analysis."""
        print("Loading required training data...")
        
        # Required parameter files
        param_files = {
            'J_param': 'J_param_*.pkl',
            'B_param': 'B_param_*.pkl', 
            'b_x_param': 'b_x_param_*.pkl',
            'w_param': 'w_param_*.pkl',
            'b_z_param': 'b_z_param_*.pkl'
        }
        
        # Load parameter files
        for param_name, pattern in param_files.items():
            files = find_files_by_pattern(self.run_directory, pattern)
            if not files:
                raise FileNotFoundError(f"Required parameter file not found: {pattern}")
            self.data[param_name] = load_variable(files[0])
        
        # Load state data
        state_files = find_files_by_pattern(self.run_directory, 'state_*.pkl')
        if not state_files:
            raise FileNotFoundError("Required state file not found: state_*.pkl")
        
        state_data = load_variable(state_files[0])
        self.data['state'] = state_data
        
        # Extract trajectory data
        self.state_traj_states = state_data['traj_states']
        self.state_fixed_point_inits = state_data['fixed_point_inits']
        
        print(f"✓ Loaded trajectory data: {self.state_traj_states.shape}")
        print(f"✓ Loaded fixed point initializations: {self.state_fixed_point_inits.shape}")
        
    def load_optional_data(self, include_slow_points=False):
        """Load optional analysis data (fixed points, slow points, etc.)."""
        print("Loading optional analysis data...")
        
        # Optional analysis files
        optional_files = {
            'all_fixed_points': 'all_fixed_points_*.pkl',
            'all_fixed_jacobians': 'all_fixed_jacobians_*.pkl',
            'all_fixed_unstable_eig_freq': 'all_fixed_unstable_eig_freq_*.pkl'
        }
        
        if include_slow_points:
            optional_files.update({
                'all_slow_points': 'all_slow_points_*.pkl',
                'all_slow_jacobians': 'all_slow_jacobians_*.pkl',
                'all_slow_unstable_eig_freq': 'all_slow_unstable_eig_freq_*.pkl'
            })
        
        for data_name, pattern in optional_files.items():
            files = find_files_by_pattern(self.run_directory, pattern)
            if files:
                self.data[data_name] = load_variable(files[0])
                print(f"✓ Loaded {data_name}")
            else:
                self.data[data_name] = None
                print(f"○ Optional file not found: {pattern}")
    
    def reconstruct_params(self):
        """Reconstruct the params dictionary from individual parameter files."""
        self.params = {
            "J": self.data['J_param'],
            "B": self.data['B_param'],
            "b_x": self.data['b_x_param'],
            "w": self.data['w_param'],
            "b_z": self.data['b_z_param']
        }
        print("✓ Reconstructed model parameters")
        return self.params
    
    def get_analysis_data(self, include_slow_points=False):
        """
        Get the data needed for PCA analysis.
        
        Returns:
            tuple: (state_traj_states, params, all_fixed_points, all_slow_points)
        """
        all_fixed_points = self.data.get('all_fixed_points')
        all_slow_points = self.data.get('all_slow_points') if include_slow_points else None
        
        return (
            self.state_traj_states,
            self.params,
            all_fixed_points,
            all_slow_points
        )
    
    def print_data_summary(self, include_slow_points=False):
        """Print a summary of the loaded data."""
        print("\nDATA SUMMARY")
        print("-" * 40)
        
        # Basic shapes
        print(f"Network size: {self.params['J'].shape[0]} neurons")
        print(f"Input dimension: {self.params['B'].shape[1]}")
        print(f"Number of tasks: {self.state_traj_states.shape[0]}")
        print(f"Trajectory length: {self.state_traj_states.shape[1]} time steps")
        
        # Analysis data availability
        all_fixed_points = self.data.get('all_fixed_points')
        all_slow_points = self.data.get('all_slow_points')
        
        if all_fixed_points is not None:
            total_fixed = sum(len(task_fps) for task_fps in all_fixed_points)
            print(f"Fixed points available: {total_fixed} total")
        else:
            print("Fixed points: Not available")
        
        if include_slow_points and all_slow_points is not None:
            total_slow = sum(len(task_sps) for task_sps in all_slow_points)
            print(f"Slow points available: {total_slow} total")
        elif include_slow_points:
            print("Slow points: Requested but not available")
        else:
            print("Slow points: Not requested")


def select_run_interactively():
    """Allow user to interactively select a training run."""
    print("AVAILABLE TRAINING RUNS")
    print("=" * 60)
    
    runs_info = list_available_runs()
    
    if not runs_info:
        print("No training runs found.")
        return None
    
    while True:
        try:
            choice = input(f"Select a run [0-{len(runs_info)-1}] or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            run_index = int(choice)
            if 0 <= run_index < len(runs_info):
                return runs_info[run_index]['path']
            else:
                print(f"Please enter a number between 0 and {len(runs_info)-1}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def main():
    """Main function to run standalone PCA analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run PCA analysis on existing training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--run_dir', 
        type=str, 
        default=None,
        help="Path to the training run directory containing pickle files"
    )
    parser.add_argument(
        '--include_slow_points',
        action='store_true',
        help="Include slow points in PCA analysis if available"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Custom output directory for saving results and figures"
    )
    parser.add_argument(
        '--skip_steps',
        type=str,
        default="0",
        help="Comma-separated list of initial time steps to skip (e.g., '0,10,20,50')"
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help="Custom name for the output files (default: use timestamp)"
    )
    parser.add_argument(
        '--list_runs',
        action='store_true',
        help="List available training runs and exit"
    )
    parser.add_argument(
        '--apply_tanh',
        action='store_true',
        help="Apply tanh transformation to trajectory states before PCA"
    )
    
    args = parser.parse_args()
    
    # Handle list runs option
    if args.list_runs:
        list_available_runs()
        return
    
    # Determine run directory
    if args.run_dir:
        run_directory = args.run_dir
        if not os.path.exists(run_directory):
            print(f"Error: Specified run directory does not exist: {run_directory}")
            sys.exit(1)
    else:
        # Interactive selection
        run_directory = select_run_interactively()
        if run_directory is None:
            print("No run selected. Exiting.")
            sys.exit(0)
    
    # Set custom output directory if specified
    if args.output_dir:
        set_custom_output_dir(args.output_dir)
        print(f"Custom output directory set: {args.output_dir}")
    
    # Parse skip_steps values
    try:
        skip_steps_list = [int(x.strip()) for x in args.skip_steps.split(',')]
    except ValueError:
        print(f"Error: Invalid skip_steps format. Expected comma-separated integers, got: {args.skip_steps}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("STANDALONE PCA ANALYSIS")
    print("=" * 60)
    print(f"Using run directory: {os.path.basename(run_directory)}")
    print(f"Full path: {run_directory}")
    print(f"Skip steps to analyze: {skip_steps_list}")
    print(f"Apply tanh transformation: {args.apply_tanh}")
    if args.output_dir:
        print(f"Custom output directory: {args.output_dir}")
    
    try:
        with Timer("PCA Analysis from Existing Data"):
            # Initialize data loader
            print("\n1. INITIALIZING DATA LOADER")
            print("-" * 40)
            loader = TrainingDataLoader(run_directory)
            
            # Load required data
            print("\n2. LOADING REQUIRED DATA")
            print("-" * 40)
            loader.load_required_data()
            loader.reconstruct_params()
            
            # Load optional data
            print("\n3. LOADING OPTIONAL DATA")
            print("-" * 40)
            loader.load_optional_data(include_slow_points=args.include_slow_points)
            
            # Print data summary
            loader.print_data_summary(include_slow_points=args.include_slow_points)
            
            # Get analysis data
            print("\n4. PREPARING PCA ANALYSIS")
            print("-" * 40)
            state_traj_states, params, all_fixed_points, all_slow_points = loader.get_analysis_data(
                include_slow_points=args.include_slow_points
            )
            
            # Run PCA analysis for each skip value
            print("\n5. RUNNING PCA ANALYSIS")
            print("-" * 40)
            all_pca_results = {}
            
            for i, skip_steps in enumerate(skip_steps_list):
                tanh_info = " with tanh transformation" if args.apply_tanh else ""
                print(f"\nRunning PCA analysis {i+1}/{len(skip_steps_list)} (skipping {skip_steps} initial steps{tanh_info})...")
                print("-" * 50)
                
                pca_results = run_pca_analysis(
                    state_traj_states,
                    all_fixed_points=all_fixed_points,
                    all_slow_points=all_slow_points,
                    params=params,
                    slow_point_search=args.include_slow_points and all_slow_points is not None,
                    skip_initial_steps=skip_steps,
                    apply_tanh=args.apply_tanh
                )
                
                all_pca_results[f"skip_{skip_steps}"] = pca_results
                print(f"✓ Completed analysis for skip_steps={skip_steps}")
            
            print("\n6. ANALYSIS COMPLETE")
            print("-" * 40)
            print("PCA analysis completed successfully!")
            print("Results have been saved and plots have been generated.")
            
            # Print summary of all results
            for skip_steps in skip_steps_list:
                key = f"skip_{skip_steps}"
                pca_results = all_pca_results[key]
                print(f"\nResults for skip_steps={skip_steps}:")
                
                if 'proj_trajs' in pca_results:
                    print(f"  ✓ Trajectory projections: {len(pca_results['proj_trajs'])} trajectories")
                
                if 'proj_fixed' in pca_results:
                    print(f"  ✓ Fixed point projections: {pca_results['proj_fixed'].shape[0]} points")
                    
                if 'proj_slow' in pca_results:
                    print(f"  ✓ Slow point projections: {pca_results['proj_slow'].shape[0]} points")
    
    except Exception as e:
        print(f"\nError during PCA analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
