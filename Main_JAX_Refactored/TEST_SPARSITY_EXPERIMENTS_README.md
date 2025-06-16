# Comprehensive Sparsity Experiments

This directory contains scripts for running comprehensive sparsity experiments that replicate the full analysis pipeline from `_0_main_refactored.py` while tracking Jacobian eigenvalue evolution during RNN training.

## Files Added

1. **`TEST_Sparsity_Experiments.py`** - Comprehensive experiment script with full analysis pipeline
2. **`example_sparsity_experiment.py`** - Simple example launcher  
3. **`README_Sparsity_Experiments.md`** - This documentation

## What These Scripts Do

The comprehensive sparsity experiment script replicates **ALL** functionality from `_0_main_refactored.py` for each sparsity level, including:

### Core Training & Analysis
1. **Train RNNs with different sparsity levels** - Interactive sparsity value selection
2. **Track Jacobian eigenvalues during training** - Configurable sampling during optimization
3. **Post-training diagnostics** - Full trajectory and state analysis
4. **Fixed point analysis** - Find and analyze all fixed points with Jacobian eigenvalues
5. **Slow point analysis** (optional) - Find and analyze slow points if enabled
6. **PCA analysis** - Complete principal component analysis of network dynamics

### Comprehensive Visualizations
1. **Trajectory plots** - RNN output vs target signals for all tasks
2. **Parameter matrix plots** - Visualization of trained network parameters
3. **Jacobian eigenvalue evolution** - Complex plane plots showing temporal progression
4. **Fixed/slow point visualizations** - Standard analysis plots from main pipeline
5. **Comparative summary plots** - Cross-sparsity comparisons of key metrics
6. **Eigenvalue comparison plots** - Side-by-side eigenvalue evolution across sparsity levels

### Organized Output Management
- **Timestamped output directories** - All results saved in `/Outputs/Sparsity_Experiments_{TIMESTAMP}/`
- **Comprehensive data saving** - All parameters, states, analysis results, and figures
- **Individual and combined results** - Both per-sparsity and aggregate experimental data

## Key Features

- **Complete pipeline replication** - Every analysis step from `_0_main_refactored.py` for each sparsity level
- **Leverages existing analysis code** - Uses all modules from `_6_analysis.py`, `_7_visualization.py`, `_8_pca_analysis.py`
- **Configurable eigenvalue sampling** via `EIGENVALUE_SAMPLE_FREQUENCY` in `_1_config.py` (default: every 50 iterations)
- **All eigenvalues tracked** (not just unstable ones) from Jacobian at fixed points
- **Color-coded temporal evolution** showing eigenvalue trajectories through training
- **Organized output management** - All results saved in timestamped `/Outputs/Sparsity_Experiments_{TIMESTAMP}/` directory
- **Comprehensive comparisons** - Cross-sparsity analysis of training dynamics and network properties
- **Full parameter and state saving** - Complete experimental reproducibility

## Usage

### Main Experiment (Comprehensive Analysis)
```bash
python TEST_Sparsity_Experiments.py
```
**This will:**
1. Prompt you to enter sparsity values interactively (e.g., "0.0, 0.2, 0.5, 0.8")
2. Create a timestamped output directory: `/Outputs/Sparsity_Experiments_{TIMESTAMP}/`
3. For each sparsity level, run the **complete analysis pipeline**:
   - Training with eigenvalue tracking
   - Post-training diagnostics
   - Parameter and state saving
   - Trajectory visualization
   - Fixed point analysis with Jacobian eigenvalues
   - Slow point analysis (if enabled in config)
   - PCA analysis
   - All standard visualizations
4. Generate comparative summary plots across all sparsity levels
5. Save comprehensive experimental results

### Quick Example
```bash
python example_sparsity_experiment.py
```
This provides guided setup for running with example values [0.0, 0.3, 0.6].

## Configuration

Key parameters in `_1_config.py`:

- `EIGENVALUE_SAMPLE_FREQUENCY = 50` - How often to sample eigenvalues during training
- `NUM_EPOCHS_ADAM = 1000` - Number of Adam optimization steps
- `NUM_EPOCHS_LBFGS = 2000` - Number of L-BFGS optimization steps  
- `N = 200` - Network size (number of neurons)
- `SLOW_POINT_SEARCH = False` - Whether to perform slow point analysis (computationally expensive)

## Output Structure

The experiment creates a timestamped directory containing:

### Individual Sparsity Results (per sparsity level)
```
/Outputs/Sparsity_Experiments_{TIMESTAMP}/
├── J_param_sparsity_{s}.pkl              # Connectivity matrices
├── B_param_sparsity_{s}.pkl              # Input matrices  
├── w_param_sparsity_{s}.pkl              # Output weights
├── state_sparsity_{s}.pkl                # Network states and trajectories
├── all_fixed_points_sparsity_{s}.pkl     # Fixed point locations
├── all_fixed_jacobians_sparsity_{s}.pkl  # Jacobian matrices at fixed points
├── pca_results_sparsity_{s}.pkl          # PCA analysis results
├── complete_results_sparsity_{s}.pkl     # Combined results dictionary
└── [visualization plots for each sparsity level]
```

### Comparative Analysis
```
├── complete_sparsity_experiment_{TIMESTAMP}.pkl  # All results combined
├── sparsity_eigenvalue_evolution_comparison.png  # Eigenvalue evolution plots
├── sparsity_summary_plots.png                    # Cross-sparsity comparisons
└── [other comparative visualizations]
```

### What Gets Generated
1. **Training curves and eigenvalue evolution** for each sparsity level
2. **RNN trajectory plots** vs target signals for all tasks
3. **Parameter matrix visualizations** (J, B, b_x matrices)
4. **Fixed point analysis plots** (Jacobian eigenvalues, unstable frequencies)
5. **PCA analysis plots** (principal components, variance explained)
6. **Cross-sparsity comparison plots**:
   - Final loss vs sparsity
   - Number of fixed points vs sparsity  
   - Spectral radius vs sparsity
   - Slow points vs sparsity (if enabled)
7. **Eigenvalue evolution comparison** - side-by-side complex plane plots
8. **Complete experimental data** for further analysis

## Technical Details

### Complete Analysis Pipeline (Per Sparsity Level)
The script replicates the full `_0_main_refactored.py` workflow:

1. **Parameter initialization** with specified sparsity level
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
10. **Individual result saving** for this sparsity level

### Cross-Sparsity Comparative Analysis
After all sparsity levels complete:
- **Eigenvalue evolution comparison plots** - complex plane with temporal coloring
- **Summary metric plots** - loss, fixed points, spectral radius vs sparsity
- **Combined result saving** - all data in one comprehensive file

### Eigenvalue Computation Process
1. During training, every `EIGENVALUE_SAMPLE_FREQUENCY` iterations:
   - Find fixed points using `find_points()` from `_6_analysis.py`
   - Compute Jacobian matrices using `compute_jacobian()`  
   - Extract all eigenvalues using `np.linalg.eigvals()`
   - Store with iteration numbers for temporal tracking

2. Post-training analysis:
   - Full fixed point analysis with comprehensive eigenvalue characterization
   - All eigenvalues plotted and analyzed (stable and unstable)

### Visualization Features
- **Eigenvalue evolution**: Complex plane plots with color gradients showing training progression
- **Temporal comparison**: Side-by-side plots for different sparsity levels
- **Reference markers**: Unit circle for stability analysis
- **Comprehensive legends**: Clear identification of training phases and progress
- **Publication-ready plots**: High-quality figures suitable for research presentations

### Performance & Scalability
- **Optimized eigenvalue sampling**: Reduced fixed point search attempts during training
- **JIT compilation**: Fast training loops with JAX
- **Memory management**: Efficient storage of large eigenvalue datasets
- **Configurable sampling**: Balance between detail and computation time
- **Parallel-ready**: Multiple sparsity experiments can be run independently

## Example Expected Results

### Training Dynamics
- **Lower sparsity (s=0.0)**: Dense connectivity, potentially faster convergence, complex eigenvalue dynamics
- **Medium sparsity (s=0.5)**: Balanced connectivity, intermediate dynamics
- **Higher sparsity (s=0.8)**: Sparse connectivity, may show different convergence patterns and eigenvalue clustering

### Eigenvalue Evolution Patterns
- **Early training**: Eigenvalues may be dispersed across complex plane
- **Mid training**: Gradual movement toward stable configurations
- **Late training**: Convergence to final eigenvalue distribution
- **Color progression**: Temporal visualization shows continuous eigenvalue trajectories

### Cross-Sparsity Comparisons
- **Training efficiency**: How sparsity affects convergence speed and final loss
- **Network stability**: Spectral radius and eigenvalue distribution changes
- **Fixed point structure**: Number and stability of fixed points vs sparsity
- **Dynamic complexity**: PCA analysis revealing dimensionality vs sparsity

## Troubleshooting

### Common Issues
- **"No eigenvalue data" messages**: Fixed points may not be found at some iterations (normal behavior)
- **Long computation times**: Eigenvalue sampling is expensive; consider:
  - Increasing `EIGENVALUE_SAMPLE_FREQUENCY` (sample less often)
  - Reducing `NUM_EPOCHS_ADAM` or `NUM_EPOCHS_LBFGS` for testing
  - Testing with fewer sparsity levels initially
- **Memory issues**: Large networks or many sparsity levels may require more RAM
- **Convergence failures**: Some sparsity levels may not converge; this is scientifically interesting data

### Performance Optimization
- **Start small**: Test with 2-3 sparsity values first (e.g., [0.0, 0.5])
- **Adjust sampling frequency**: Start with `EIGENVALUE_SAMPLE_FREQUENCY = 100` for faster runs
- **Monitor disk space**: Experiments generate substantial data; ensure adequate storage
- **Use appropriate hardware**: GPU acceleration helps with JAX computations

### Expected Runtime
- **Per sparsity level**: 15-45 minutes depending on convergence and analysis depth
- **Full experiment (4-5 sparsity levels)**: 1-4 hours total
- **With slow point analysis**: Additional 50-100% runtime increase

## Scientific Applications

This framework enables investigation of:

1. **Sparsity effects on learning dynamics**: How connectivity sparsity affects training efficiency
2. **Eigenvalue evolution patterns**: Temporal dynamics of network stability during learning  
3. **Fixed point structure**: Relationship between sparsity and attractor landscape
4. **Dynamical systems analysis**: Comprehensive characterization of trained RNN dynamics
5. **Network architecture optimization**: Finding optimal sparsity levels for specific tasks

## Integration with Existing Code

This comprehensive experiment framework:

### ✅ Complete Compatibility
- **Replicates ALL functionality** from `_0_main_refactored.py` for each sparsity level
- **Uses existing analysis pipeline** without modification (`_6_analysis.py`, `_7_visualization.py`, `_8_pca_analysis.py`)
- **Preserves all visualizations** and analysis outputs from the main pipeline
- **Maintains configuration system** - all parameters from `_1_config.py` respected
- **Uses existing I/O utilities** - leverages `_2_utils.py` for organized saving

### ✅ Enhanced Features  
- **Organized output management** - timestamped directories prevent file conflicts
- **Cross-sparsity comparisons** - new comparative analysis not in original pipeline
- **Eigenvalue evolution tracking** - novel temporal analysis of training dynamics
- **Comprehensive result aggregation** - combined experimental datasets for meta-analysis

### ✅ Research Workflow Integration
- **Non-disruptive**: Does not modify any existing core files
- **Parallel-compatible**: Can run alongside other experiments  
- **Reproducible**: Complete parameter and random seed tracking
- **Extensible**: Easy to add new sparsity levels or analysis metrics
- **Publication-ready**: Generates high-quality figures and comprehensive datasets

### 🔄 Workflow Comparison

| Original `_0_main_refactored.py` | Enhanced `TEST_Sparsity_Experiments.py` |
|----------------------------------|------------------------------------------|
| Single sparsity level | Multiple user-defined sparsity levels |
| Standard output directory | Organized timestamped directories |
| Fixed point analysis only | Fixed points + eigenvalue evolution tracking |
| Individual experiment results | Individual + comparative cross-sparsity analysis |
| Static eigenvalue analysis | Dynamic eigenvalue evolution visualization |

This framework transforms the single-experiment pipeline into a comprehensive multi-condition experimental platform while preserving all original functionality and scientific rigor.
