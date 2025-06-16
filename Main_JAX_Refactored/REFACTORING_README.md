# Refactored RNN Sine Wave Generation Project

This repository contains a refactored version of the RNN sine wave generation experiment. The original monolithic `Main_JAX.py` file has been broken down into modular components for better maintainability, readability, and extensibility.

## File Structure

### Core Modules

1. **`config.py`** - Configuration and hyperparameters
   - Network architecture parameters (N, I, num_tasks)
   - Training parameters (learning rates, epochs, etc.)
   - Analysis parameters (fixed point search settings)
   - Visualization parameters (colors, markers, test indices)

2. **`data_generation.py`** - Data generation functions
   - Sine wave input generation
   - Task-specific input/target creation
   - Batch data generation utilities

3. **`rnn_model.py`** - RNN model definition
   - Parameter initialization
   - RNN dynamics (forward pass)
   - Trajectory simulation
   - Loss computation
   - Batch diagnostics

4. **`training.py`** - Training procedures
   - Adam and L-BFGS optimizer setup
   - Training step functions
   - Complete training workflow
   - Loss tracking and best model selection

5. **`analysis.py`** - Fixed/slow point analysis
   - Fixed point finding algorithms
   - Slow point search (optional)
   - Jacobian computation
   - Eigenvalue analysis
   - Point summaries and statistics

6. **`visualization.py`** - Plotting and visualization
   - Trajectory vs target plots
   - Parameter matrix visualization
   - Jacobian matrix plots
   - Eigenvalue visualizations
   - Frequency comparison plots

7. **`pca_analysis.py`** - PCA analysis and visualization
   - PCA computation on trajectories
   - Point projection into PCA space
   - 3D PCA visualizations with unstable modes

8. **`utils.py`** - Utility functions
   - File I/O operations
   - Result saving functions
   - Timing utilities
   - Output directory management

### Execution Scripts

9. **`main_refactored.py`** - Main execution script
   - Orchestrates the complete workflow
   - Imports and uses all modules
   - Provides clear execution phases
   - Comprehensive logging and timing

## Key Improvements

### Modularity
- Each file has a single, well-defined responsibility
- Clear separation of concerns
- Easy to modify individual components without affecting others

### Maintainability
- Shorter, focused files (100-300 lines vs 1,710 lines)
- Clear function and module organization
- Consistent naming conventions
- Comprehensive docstrings

### Reusability
- Functions can be imported and used independently
- Configurable parameters through config.py
- Modular design allows for easy extension

### Testing & Debugging
- Each module can be tested independently
- Easier to isolate and fix bugs
- Clear error tracking within modules

### Performance
- Only necessary modules need to be imported
- JIT compilation benefits preserved
- Efficient memory usage

## Usage

### Running the Complete Experiment
```python
python main_refactored.py
```

### Using Individual Modules
```python
from config import *
from rnn_model import init_params, simulate_trajectory
from training import train_model

# Initialize model
key = random.PRNGKey(42)
_, subkey = random.split(key)
mask, params = init_params(subkey)

# Train model
trained_params, final_loss = train_model(params, mask)
```

### Customizing Configuration
Edit `config.py` to modify:
- Network size (`N`)
- Number of tasks (`num_tasks`)
- Training parameters (`NUM_EPOCHS_ADAM`, `ADAM_LR`)
- Analysis settings (`SLOW_POINT_SEARCH`, `NUM_ATTEMPTS`)

## Dependencies

- JAX and JAX-NumPy
- Optax (for optimizers)
- NumPy
- Matplotlib
- SciPy
- scikit-learn (for PCA)
- tqdm (for progress bars)

## Output Structure

All results are saved to timestamped directories:
```
Outputs/
└── JAX_Outputs_YYYYMMDD_HHMMSS/
    ├── trained_parameters.pkl
    ├── analysis_results.pkl
    ├── pca_results.pkl
    └── figures/
        ├── trajectory_plots.png
        ├── jacobian_matrices.png
        └── pca_visualization.png
```

## Comparison with Original

| Aspect | Original Main_JAX.py | Refactored Version |
|--------|---------------------|-------------------|
| Lines of Code | 1,710 lines | 9 files, ~200 lines each |
| Modularity | Monolithic | Highly modular |
| Maintainability | Difficult | Easy |
| Testability | Hard to test | Easy unit testing |
| Reusability | Low | High |
| Readability | Complex | Clear and focused |

## Future Extensions

The modular structure makes it easy to:
- Add new analysis methods
- Implement different RNN architectures
- Add new visualization options
- Integrate different optimizers
- Add automated testing
- Create parameter sweeps
- Add experiment tracking

## Notes

- Import resolution errors in the IDE are expected due to the module structure - they will resolve when running the actual code
- All original functionality is preserved
- Performance characteristics remain the same
- JAX JIT compilation benefits are maintained