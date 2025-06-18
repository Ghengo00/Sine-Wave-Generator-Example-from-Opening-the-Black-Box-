# Comprehensive L1 Regularization Experiments

This directory contains scripts for running comprehensive L1 regularization experiments that replicate the full analysis pipeline from `_0_main_refactored.py` while tracking Jacobian eigenvalue evolution during RNN training.

## Files

1. **`TEST_L1_Experiments.py`** - Comprehensive L1 regularization experiment script with full analysis pipeline
2. **`TEST_L1_EXPERIMENTS_README.md`** - This documentation

## What This Script Does

The comprehensive L1 regularization experiment script replicates **ALL** functionality from `_0_main_refactored.py` for each L1 regularization strength, including:

### Core Training & Analysis
1. **Train RNNs with different L1 regularization strengths** - Interactive L1 regularization value selection
2. **Track Jacobian eigenvalues during training** - Configurable sampling during optimization
3. **Post-training diagnostics** - Full trajectory and state analysis
4. **Fixed point analysis** - Find and analyze all fixed points with Jacobian eigenvalues
5. **Slow point analysis** (optional) - Find and analyze slow points if enabled
6. **PCA analysis** - Complete principal component analysis of network dynamics

### Key Differences from Sparsity Experiments
- **Fixed sparsity at s=0.0** - No structural sparsity applied to connectivity matrix
- **Variable L1 regularization strength** - Controls weight magnitude penalty during training
- **L1-specific output organization** - Results saved with L1 regularization naming conventions
- **Log-scale visualizations** - Plots use logarithmic scales appropriate for L1 regularization values
- **Scientific notation formatting** - L1 values displayed in scientific notation for clarity

### Comprehensive Visualizations
1. **Trajectory plots** - RNN output vs target signals for all tasks
2. **Parameter matrix plots** - Visualization of trained network parameters
3. **Jacobian eigenvalue evolution** - Complex plane plots showing temporal progression
4. **Fixed/slow point visualizations** - Standard analysis plots from main pipeline
5. **Comparative summary plots** - Cross-L1-regularization comparisons of key metrics
6. **Eigenvalue comparison plots** - Side-by-side eigenvalue evolution across L1 regularization strengths

### Organized Output Management
- **Timestamped output directories** - All results saved in `/Outputs/L1_Regularization_Experiments_{TIMESTAMP}/`
- **Comprehensive data saving** - All parameters, states, analysis results, and figures
- **Individual and combined results** - Both per-L1-regularization and aggregate experimental data

## Key Features

- **Complete pipeline replication** - Every analysis step from `_0_main_refactored.py` for each L1 regularization strength
- **Leverages existing analysis code** - Uses all modules from `_6_analysis.py`, `_7_visualization.py`, `_8_pca_analysis.py`
- **Configurable eigenvalue sampling** via `EIGENVALUE_SAMPLE_FREQUENCY` in `_1_config.py` (default: every 50 iterations)
- **All eigenvalues tracked** (not just unstable ones) from Jacobian at fixed points
- **Color-coded temporal evolution** showing eigenvalue trajectories through training
- **Organized output management** - All results saved in timestamped `/Outputs/L1_Regularization_Experiments_{TIMESTAMP}/` directory
- **Comprehensive comparisons** - Cross-L1-regularization analysis of training dynamics and network properties
- **Full parameter and state saving** - Complete experimental reproducibility
- **L1-specific formatting** - Scientific notation and log-scale plots for L1 regularization values

## Usage

### Main Experiment (Comprehensive Analysis)
```bash
python TEST_L1_Experiments.py
```

**This will:**
1. Prompt you to enter L1 regularization values interactively (e.g., "0.0, 1e-5, 1e-4, 1e-3")
2. Prompt you to select eigenvalue computation options (Jacobian and/or connectivity matrix eigenvalues)
3. Create a timestamped output directory: `/Outputs/L1_Regularization_Experiments_{TIMESTAMP}/`
4. For each L1 regularization strength, run the **complete analysis pipeline**:
   - Training with eigenvalue tracking
   - Post-training diagnostics
   - Parameter and state saving
   - Trajectory visualization
   - Fixed point analysis with Jacobian eigenvalues
   - Slow point analysis (if enabled in config)
   - PCA analysis
   - All standard visualizations
5. Generate comparative summary plots across all L1 regularization strengths
6. Save comprehensive experimental results

### Example L1 Regularization Values
- **No regularization**: `0.0`
- **Weak regularization**: `1e-6, 1e-5`
- **Moderate regularization**: `1e-4, 1e-3`
- **Strong regularization**: `0.01, 0.1`

## Configuration

Key parameters in `_1_config.py`:

- `EIGENVALUE_SAMPLE_FREQUENCY = 50` - How often to sample eigenvalues during training
- `NUM_EPOCHS_ADAM = 1000` - Number of Adam optimization steps
- `NUM_EPOCHS_LBFGS = 2000` - Number of L-BFGS optimization steps  
- `N = 200` - Network size (number of neurons)
- `s = 0.0` - Fixed sparsity (no structural sparsity for L1 experiments)
- `SLOW_POINT_SEARCH = False` - Whether to perform slow point analysis (computationally expensive)

## Output Structure

The experiment creates a timestamped directory containing:

### Individual L1 Regularization Results (per L1 strength)
```
/Outputs/L1_Regularization_Experiments_{TIMESTAMP}/
├── J_param_l1_reg_{l1}.pkl               # Connectivity matrices
├── B_param_l1_reg_{l1}.pkl               # Input matrices  
├── w_param_l1_reg_{l1}.pkl               # Output weights
├── state_l1_reg_{l1}.pkl                 # Network states and trajectories
├── all_fixed_points_l1_reg_{l1}.pkl      # Fixed point locations
├── all_fixed_jacobians_l1_reg_{l1}.pkl   # Jacobian matrices at fixed points
├── pca_results_l1_reg_{l1}.pkl           # PCA analysis results
├── complete_results_l1_reg_{l1}.pkl      # Combined results dictionary
└── [visualization plots for each L1 regularization strength]
```

### Comparative Analysis
```
├── complete_l1_reg_experiment_{TIMESTAMP}.pkl    # All results combined
├── l1_reg_eigenvalue_evolution_comparison.png    # Eigenvalue evolution plots
├── l1_reg_summary_plots.png                      # Cross-L1-reg comparisons
├── l1_reg_unstable_eigenvalue_distribution.png   # Unstable eigenvalue table
└── [other comparative visualizations]
```

### What Gets Generated
1. **Training curves and eigenvalue evolution** for each L1 regularization strength
2. **RNN trajectory plots** vs target signals for all tasks
3. **Parameter matrix visualizations** (J, B, b_x matrices)
4. **Fixed point analysis plots** (Jacobian eigenvalues, unstable frequencies)
5. **PCA analysis plots** (principal components, variance explained)
6. **Cross-L1-regularization comparison plots**:
   - Final loss vs L1 regularization strength (log scale)
   - Number of fixed points vs L1 regularization strength
   - Spectral radius vs L1 regularization strength
   - Slow points vs L1 regularization strength (if enabled)
7. **Eigenvalue evolution comparison** - side-by-side complex plane plots
8. **Unstable eigenvalue distribution table** - comprehensive summary statistics
9. **Complete experimental data** for further analysis

## Technical Details

### Complete Analysis Pipeline (Per L1 Regularization Strength)
The script replicates the full `_0_main_refactored.py` workflow:

1. **Parameter initialization** with specified L1 regularization strength (fixed sparsity s=0.0)
2. **Training with eigenvalue tracking**:
   - Adam optimization with periodic eigenvalue sampling
   - L-BFGS optimization (if loss threshold not met)
   - Eigenvalue computation at fixed points every N iterations
3. **Post-training diagnostics** via `run_batch_diagnostics()`
4. **Parameter and state saving** (all trained parameters and network states)
5. **Trajectory visualization** (RNN outputs vs targets for test tasks)
6. **Parameter matrix visualization** (J, B, b_x matrices)
7. **Fixed point analysis**:
   - Find all fixed points using existing robust algorithms
   - Compute Jacobian matrices at each fixed point
   - Extract ALL eigenvalues (not just unstable ones)
   - Generate fixed point summary statistics
   - Create Jacobian eigenvalue and frequency analysis plots
8. **Slow point analysis** (if `SLOW_POINT_SEARCH = True`)
9. **PCA analysis** of network dynamics and fixed points
10. **Individual result saving** for this L1 regularization strength

### Cross-L1-Regularization Comparative Analysis
After all L1 regularization strengths complete:
- **Eigenvalue evolution comparison plots** - complex plane with temporal coloring
- **Summary statistics plots** - key metrics vs L1 regularization strength
- **Unstable eigenvalue distribution table** - comprehensive analysis across all L1 values
- **Combined experimental data saving** - all results in single comprehensive file

### L1 Regularization Theory
L1 regularization adds a penalty term to the loss function proportional to the sum of absolute values of all network parameters:

```
Loss_total = Loss_task + λ * ||W||_1
```

Where:
- `λ` is the L1 regularization strength
- `||W||_1` is the L1 norm (sum of absolute values) of all weights

**Effects of L1 Regularization:**
- **Weight sparsification** - Drives small weights toward zero
- **Feature selection** - Promotes sparse solutions by eliminating unimportant connections
- **Reduced overfitting** - Regularizes the model to improve generalization
- **Network compression** - Can lead to more compact network representations

**Typical L1 Regularization Ranges:**
- `λ = 0.0`: No regularization (baseline)
- `λ ∈ [1e-6, 1e-4]`: Weak to moderate regularization
- `λ ∈ [1e-3, 1e-2]`: Strong regularization
- `λ > 0.01`: Very strong regularization (may impair learning)

### Eigenvalue Analysis Context
The experiments track how L1 regularization affects:
1. **Network dynamics** - Changes in Jacobian eigenvalues at fixed points
2. **Stability properties** - Evolution of unstable eigenvalues during training
3. **Complexity reduction** - How regularization simplifies network dynamics
4. **Learning trajectories** - Eigenvalue paths through complex plane during optimization

### Comparison with Sparsity Experiments
- **Sparsity experiments**: Structural pruning (hard zeros in connectivity matrix)
- **L1 experiments**: Soft weight reduction (continuous shrinkage toward zero)
- **Complementary insights**: Different mechanisms for achieving network simplification

## Interactive Prompts

### L1 Regularization Value Input
```
Enter L1 regularization strength values to test (≥ 0.0)
Examples: 0.0, 1e-5, 1e-4, 1e-3, 0.01
Enter values separated by commas (e.g., '0.0, 1e-5, 1e-4'): 
```

### Eigenvalue Computation Options
```
Select eigenvalue computation options:
1. Jacobian eigenvalues during training
2. Connectivity matrix eigenvalues  
3. Both
4. Neither (training only)
```

### Jacobian Frequency Selection (if applicable)
```
Enter specific frequencies for Jacobian eigenvalue computation:
Available frequencies: [0.10, 0.15, 0.20, ...]
Enter frequencies separated by commas, or press Enter for all:
```

## Output Interpretation

### Summary Plots
- **Loss vs L1 Regularization**: Shows trade-off between task performance and regularization
- **Fixed Points vs L1 Regularization**: How regularization affects network attractors
- **Spectral Radius vs L1 Regularization**: Impact on network stability properties

### Eigenvalue Evolution Plots
- **Color progression**: Blue (early training) → Red (late training)
- **Complex plane visualization**: Real vs imaginary eigenvalue components
- **Temporal tracking**: How regularization shapes eigenvalue trajectories

### Unstable Eigenvalue Table
- **Comprehensive statistics** for each L1 regularization strength
- **Fixed point counts** and **unstable eigenvalue distributions**
- **Comparative analysis** across different regularization strengths

## Best Practices

1. **Start with baseline**: Always include `L1 = 0.0` for comparison
2. **Log-spaced values**: Use logarithmically spaced L1 values (e.g., 1e-5, 1e-4, 1e-3)
3. **Monitor convergence**: Check that networks converge for all L1 values tested
4. **Resource consideration**: Eigenvalue tracking is computationally intensive
5. **Output management**: Use descriptive timestamps and organize results systematically

## Dependencies

The script requires all modules from the refactored codebase:
- `_1_config.py` - Configuration parameters
- `_2_utils.py` - Utility functions and output management
- `_3_data_generation.py` - Input/target generation
- `_4_rnn_model.py` - RNN model and loss functions
- `_5_training.py` - Optimization algorithms
- `_6_analysis.py` - Fixed point and eigenvalue analysis
- `_7_visualization.py` - Plotting and visualization
- `_8_pca_analysis.py` - Principal component analysis

Standard Python packages: `numpy`, `jax`, `matplotlib`, `tqdm`
