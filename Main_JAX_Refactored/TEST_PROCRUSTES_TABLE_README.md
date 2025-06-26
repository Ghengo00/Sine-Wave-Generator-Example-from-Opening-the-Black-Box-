# TEST_Procrustes_Table.py - README

## Overview

This script performs Procrustes analysis on neural network trajectory data to compare the similarity of trajectories across different sparsity levels. Procrustes analysis is a statistical shape analysis technique used to analyze the distribution of a set of shapes by removing the effects of translation, rotation, and scaling.

## Purpose

The script is designed to:
1. Compare neural network trajectories from different sparsity experiments
2. Generate normalized similarity metrics using two different Procrustes analysis methods
3. Create heatmap visualizations showing pairwise comparisons between sparsity levels
4. Save results and visualizations for further analysis

## Key Features

### Two Procrustes Analysis Methods

1. **Tensor Flattening Method** (`perform_procrustes_with_tensor_flattening`)
   - Flattens 3D tensors (K, T, N) to 2D matrices (K*T, N) before analysis
   - Performs single Procrustes analysis on flattened data
   - Returns disparity, R², and transformed arrays

2. **Averaging Method** (`perform_procrustes_with_averaging`)
   - Performs separate Procrustes analysis on each slice of the 3D tensor
   - Sums disparities across all slices
   - Provides alternative perspective on trajectory similarity

### Metrics Computed

- **Disparity**: Measure of dissimilarity (lower = more similar)
- **R²**: Coefficient of determination (higher = more similar)
- **Normalized values**: All metrics are normalized by baseline comparisons

## Input Requirements

Before running the script, you must populate the following variables:

```python
# Random seeds and file paths for baseline comparisons
rand_seeds = []  # List of random seeds used for baseline
file_paths_for_baselines = []  # Paths to baseline trajectory files

# Sparsity levels and corresponding file paths
sparsity_levels = []  # List of sparsity level strings/values
file_paths_to_trajs = []  # Corresponding paths to trajectory files
```

### Expected File Format

- Trajectory data should be stored as pickle (.pkl) files
- Data should be 3D numpy arrays with shape (K, T, N) where:
  - K: Number of trajectories/trials
  - T: Time steps
  - N: Number of neurons/dimensions

## Output

### Files Generated

All outputs are saved in `../Outputs/Procrustes_Analysis_{timestamp}/`:

1. **Results Dictionary** (`Procrustes_Analysis_Results_sparsity_levels_{levels}_{timestamp}.pkl`)
   - Contains all computed metrics for each sparsity level comparison
   - Keys: `"{sparsity_x}_vs_{sparsity_y}"`
   - Values: Dictionary with normalized disparity and R² values

2. **Heatmap Visualizations** (PNG files)
   - `Procrustes_disparity_flat_heatmap_sparsity_levels_{levels}_{timestamp}.png`
   - `Procrustes_R2_flat_heatmap_sparsity_levels_{levels}_{timestamp}.png`
   - `Procrustes_total_disparity_avg_heatmap_sparsity_levels_{levels}_{timestamp}.png`
   - `Procrustes_R2_avg_heatmap_sparsity_levels_{levels}_{timestamp}.png`

### Visualization Features

- Color-coded heatmaps with appropriate colormaps
- Annotated cells showing exact values
- Proper axis labeling and titles
- High-resolution output (300 DPI)

## Usage Example

```python
# 1. Set up your data paths and sparsity levels
rand_seeds = [42, 123]
file_paths_for_baselines = [
    "path/to/baseline1.pkl",
    "path/to/baseline2.pkl"
]

sparsity_levels = ["0.1", "0.3", "0.5", "0.7"]
file_paths_to_trajs = [
    "path/to/sparsity_0.1.pkl",
    "path/to/sparsity_0.3.pkl",
    "path/to/sparsity_0.5.pkl",
    "path/to/sparsity_0.7.pkl"
]

# 2. Run the script
python TEST_Procrustes_Table.py
```

## Dependencies

```python
import os
import pickle
from time import time
import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

## Key Functions

### Data I/O Functions
- `read_trajectory_data(file_path)`: Load trajectory data from pickle files
- `save_file_to_pkl(data_dict, base_name)`: Save results to pickle files
- `save_plot_to_png(plot_figure, base_name)`: Save plots as PNG images

### Analysis Functions
- `perform_procrustes_with_tensor_flattening(X, Y)`: Procrustes analysis with tensor flattening
- `perform_procrustes_with_averaging(X, Y)`: Procrustes analysis with slice-wise averaging

### Visualization Functions
- `plot_table(results, sparsity_levels, value_key)`: Create heatmap tables for different metrics

## Interpretation of Results

### Disparity Values
- **Lower values** indicate more similar trajectories
- Normalized by baseline comparison (typically random comparisons)
- Values < 1.0 suggest more similarity than baseline

### R² Values
- **Higher values** indicate better alignment/similarity
- Range: 0 to 1 (theoretical maximum)
- Normalized by baseline comparison

### Choosing Between Methods
- **Flattening method**: Better for overall trajectory similarity
- **Averaging method**: Better for time-point specific comparisons
- Both methods provide complementary perspectives on trajectory similarity

## Notes

- Ensure all trajectory files have the same tensor dimensions
- The script automatically creates output directories
- Timestamps are used to prevent file overwrites
- All plots are saved automatically with descriptive filenames
- Results are normalized relative to baseline comparisons for interpretability

## Troubleshooting

1. **Shape mismatch errors**: Ensure all trajectory tensors have identical dimensions
2. **File not found errors**: Verify all file paths in the input lists are correct
3. **Empty results**: Check that `sparsity_levels` and `file_paths_to_trajs` have the same length
4. **Memory issues**: For large datasets, consider processing subsets of the data

## Related Scripts

This script is part of a larger neural network analysis pipeline and works in conjunction with:
- `TEST_Sparsity_Experiments.py`: Generates the sparsity trajectory data
- `TEST_L1_Experiments.py`: Generates L1 regularization trajectory data
- Other analysis scripts in the pipeline
