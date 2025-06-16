# TEST_PCA_Analysis_Variations.py - README

## Overview

`TEST_PCA_Analysis_Variations.py` is a standalone script for performing comprehensive Principal Component Analysis (PCA) on previously trained Recurrent Neural Network (RNN) models. This script is designed to analyze neural network dynamics by loading saved training results and conducting PCA analysis on trajectory data and fixed/slow points without requiring model retraining.

## Purpose

This script enables post-training analysis of RNN dynamics by:
- Loading previously saved training results from pickle files
- Performing PCA on neural network state trajectories
- Analyzing fixed points and slow points in the low-dimensional PCA space
- Providing flexible analysis options with customizable parameters
- Supporting batch analysis with multiple trajectory truncation settings

## Key Features

### Data Loading
- **Independent Operation**: Works with any training run directory containing the required pickle files
- **Manual Parameter Specification**: No dependency on configuration files
- **Comprehensive Data Loading**: Loads model parameters, trajectory states, and analysis results
- **Error Handling**: Robust error handling for missing or corrupted files

### Analysis Options
- **Interactive Run Selection**: Browse and select from available training runs
- **Flexible Trajectory Analysis**: Skip initial time steps to focus on steady-state dynamics
- **Fixed and Slow Points**: Optional inclusion of fixed points and slow points analysis
- **Tanh Transformation**: Option to apply tanh transformation to trajectory states before PCA
- **Batch Processing**: Analyze multiple skip values in a single run

### Output Management
- **Custom Output Directory**: Specify custom directories for saving results
- **Timestamped Results**: Automatic timestamping of output files
- **Comprehensive Visualization**: Generated plots saved as PNG files
- **Data Preservation**: Analysis results saved as pickle files for future use

## Requirements

### Python Dependencies
```python
import os
import sys
import argparse
import numpy as np
```

### Required Modules (from script directory)
- `utils`: Utility functions for file operations and timing
- `pca_analysis`: Core PCA analysis functionality

### Required Input Files
The script expects the following pickle files in the training run directory:
- `J_param_*.pkl`: Recurrent weight matrix
- `B_param_*.pkl`: Input weight matrix  
- `b_x_param_*.pkl`: Recurrent bias vector
- `w_param_*.pkl`: Output weight vector
- `b_z_param_*.pkl`: Output bias scalar
- `state_*.pkl`: Trajectory states and fixed point initializations

### Optional Input Files
- `all_fixed_points_*.pkl`: Fixed point locations
- `all_fixed_jacobians_*.pkl`: Jacobian matrices at fixed points
- `all_fixed_unstable_eig_freq_*.pkl`: Unstable eigenfrequencies at fixed points
- `all_slow_points_*.pkl`: Slow point locations (if slow point analysis available)
- `all_slow_jacobians_*.pkl`: Jacobian matrices at slow points
- `all_slow_unstable_eig_freq_*.pkl`: Unstable eigenfrequencies at slow points

## Usage

### Basic Usage
```bash
# Interactive mode - select from available runs
python TEST_PCA_Analysis_Variations.py

# Specify a specific run directory
python TEST_PCA_Analysis_Variations.py --run_dir /path/to/training/output

# List all available training runs
python TEST_PCA_Analysis_Variations.py --list_runs
```

### Advanced Usage
```bash
# Include slow points analysis
python TEST_PCA_Analysis_Variations.py --include_slow_points

# Specify custom output directory
python TEST_PCA_Analysis_Variations.py --output_dir "custom_pca_analysis"

# Run multiple iterations with different trajectory truncations
python TEST_PCA_Analysis_Variations.py --skip_steps 0,10,20,50

# Apply tanh transformation to trajectories before PCA
python TEST_PCA_Analysis_Variations.py --apply_tanh

# Comprehensive analysis with all options
python TEST_PCA_Analysis_Variations.py \
    --run_dir /path/to/training/output \
    --include_slow_points \
    --skip_steps 0,10,20,50 \
    --output_dir pca_comprehensive_analysis \
    --apply_tanh
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_dir` | string | None | Path to training run directory containing pickle files |
| `--include_slow_points` | flag | False | Include slow points in PCA analysis if available |
| `--output_dir` | string | None | Custom output directory for saving results and figures |
| `--skip_steps` | string | "0" | Comma-separated list of initial time steps to skip |
| `--output_name` | string | None | Custom name for output files (default: use timestamp) |
| `--list_runs` | flag | False | List available training runs and exit |
| `--apply_tanh` | flag | False | Apply tanh transformation to trajectory states before PCA |

## Analysis Process

### 1. Data Loading Phase
- **Initialize TrainingDataLoader**: Sets up the data loading framework
- **Load Required Data**: Loads essential model parameters and trajectory data
- **Load Optional Data**: Loads analysis data (fixed points, slow points) if available
- **Data Validation**: Verifies data integrity and completeness

### 2. PCA Analysis Phase
- **Trajectory Processing**: Concatenates and optionally transforms trajectory data
- **PCA Computation**: Performs Principal Component Analysis with 3 components
- **Point Projection**: Projects fixed points and slow points into PCA space
- **Visualization**: Generates comprehensive 3D plots of the analysis

### 3. Results Generation
- **Save Analysis Results**: Stores PCA projections and metadata
- **Generate Plots**: Creates and saves visualization figures
- **Progress Reporting**: Provides detailed progress updates and summaries

## Output Files

### Generated Plots
- **3D PCA Visualization**: Shows trajectories, fixed points, and unstable modes in PCA space
- **Color-coded Analysis**: Different colors for trajectories, fixed points, and slow points
- **Eigenvector Visualization**: Red lines showing unstable mode directions

### Saved Data Files
- **PCA Results**: Pickle files containing projection data
- **Trajectory Projections**: Low-dimensional representations of neural dynamics
- **Point Projections**: Fixed point and slow point locations in PCA space

## Data Structure

### TrainingDataLoader Class
The `TrainingDataLoader` class handles all data loading operations:

#### Methods
- `load_required_data()`: Loads essential training data
- `load_optional_data()`: Loads analysis data based on availability
- `reconstruct_params()`: Rebuilds parameter dictionary from individual files
- `get_analysis_data()`: Prepares data for PCA analysis
- `print_data_summary()`: Displays comprehensive data overview

## Error Handling

The script includes comprehensive error handling for:
- **Missing Files**: Clear error messages for required files not found
- **Data Corruption**: Validation of loaded data structures
- **Invalid Arguments**: Argument parsing and validation
- **Analysis Failures**: Graceful handling of PCA computation errors

## Integration

This script is designed to work seamlessly with:
- **Main Training Pipeline**: Uses output from primary RNN training scripts
- **Analysis Modules**: Integrates with `pca_analysis.py` for core functionality
- **Utility Functions**: Leverages `utils.py` for file operations and timing

## Performance Considerations

- **Memory Efficiency**: Loads data incrementally to manage memory usage
- **Batch Processing**: Supports multiple analysis iterations in a single run
- **Progress Tracking**: Provides timing information and progress updates
- **File Management**: Automatic organization of output files and directories

## Customization Options

- **Analysis Parameters**: Flexible trajectory truncation and transformation options
- **Output Organization**: Customizable output directories and file naming
- **Analysis Scope**: Optional inclusion of different types of analysis data
- **Visualization Settings**: Automatic plot generation with descriptive titles

## Use Cases

1. **Post-Training Analysis**: Analyze dynamics after model training completion
2. **Comparative Studies**: Compare PCA results across different training runs
3. **Parameter Sensitivity**: Study effects of different trajectory truncations
4. **Presentation Preparation**: Generate publication-ready visualizations
5. **Data Exploration**: Interactive exploration of available training results

## Dependencies on Other Files

- **utils.py**: File operations, timing utilities, directory management
- **pca_analysis.py**: Core PCA computation and visualization functions
- **Training Output**: Pickle files from main training pipeline

## Version Information

- **Creation Date**: 2025
- **Purpose**: Standalone PCA analysis for neural network dynamics
- **Compatibility**: Designed for JAX-based RNN training pipeline
- **Python Version**: Compatible with Python 3.6+

---

*This script provides a comprehensive framework for analyzing the low-dimensional structure of recurrent neural network dynamics through Principal Component Analysis, enabling detailed investigation of learned representations and dynamical properties.*
