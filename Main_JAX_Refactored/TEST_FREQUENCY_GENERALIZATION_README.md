# Frequency Generalization Test Script

## Overview

The `TEST_Frequency_Generalization.py` script allows you to test how well a trained RNN generalizes to frequencies outside its training range. This script leverages the refactored codebase and provides comprehensive analysis and visualization capabilities.

## Features

- **Load trained RNN parameters** from any run directory in the Outputs folder
- **Interactive run selection** or specify run name via command line
- **Test on custom frequencies** outside the training range (typically 0.1-0.6 rad/s)
- **Automatic static input calculation** based on test frequencies
- **Comprehensive analysis** with enhanced visualizations:
  - Individual trajectory vs target plots for each frequency
  - Error vs frequency plot with shaded training range
  - Detailed summary with configuration and statistical analysis
- **Flat output structure** - all plots saved directly in main output directory
- **Enhanced summary file** with detailed configuration, results, and statistics

## Usage Examples

### 1. Interactive Mode (Recommended for beginners)
```bash
python TEST_Frequency_Generalization.py
```
This will:
- Show you all available training runs
- Let you select which run to analyze
- Prompt you to enter test frequencies

### 2. Command Line Mode - Specify Run and Frequencies
```bash
python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5
```

### 3. With Custom Output Directory
```bash
python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5 --output_dir custom_test_name
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--run_name` | Name of the training run to load | Interactive selection |
| `--test_frequencies` | Space-separated frequencies to test | Interactive input |
| `--output_dir` | Custom output directory name | Auto-generated with timestamp |

## Understanding the Results

### 1. Output Structure
All results are saved in the `Outputs` folder with a flat structure:
```
Outputs/
└── frequency_generalization_test_YYYYMMDD_HHMMSS/
    ├── trajectory_vs_target_freq_0.50.png
    ├── trajectory_vs_target_freq_1.50.png
    ├── error_vs_frequency.png
    └── frequency_generalization_summary.txt
```

### 2. Key Files
- **Individual trajectory plots**: Direct comparison between RNN output and target sine wave for each test frequency
- **Error vs frequency plot**: Shows MSE performance across all test frequencies with shaded training range
- **Comprehensive summary**: Detailed text file with configuration, results, and statistics

### 3. Enhanced Analysis
- **Training range visualization**: The error vs frequency plot clearly shows the training frequency range (typically 0.1-0.6 rad/s) as a shaded region
- **Detailed configuration info**: Summary includes all relevant parameters (number of neurons, time constants, etc.)
- **Statistical summary**: Mean, standard deviation, min/max MSE values across all test frequencies

## Example Analysis Workflow

1. **Run in interactive mode** to explore available models:
   ```bash
   python TEST_Frequency_Generalization.py
   ```

2. **Test specific frequencies** on a known run:
   ```bash
   python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5
   ```

3. **Test very low frequencies** to see extrapolation limits:
   ```bash
   python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.1 0.2 0.3
   ```

4. **Test very high frequencies** to see upper limits:
   ```bash
   python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 2.0 3.0 4.0
   ```

5. **Custom output directory** for organized testing:
   ```bash
   python TEST_Frequency_Generalization.py --run_name run_20231201_120000 --test_frequencies 0.5 1.5 2.5 --output_dir low_freq_generalization_test
   ```

## Interpreting Results

### Good Generalization Signs:
- **Gradual MSE increase**: Error increases smoothly as you move away from training range
- **Maintained structure**: RNN preserves basic sine wave pattern even at new frequencies
- **Low error within training range**: Shaded region in error plot shows consistent low MSE

### Poor Generalization Signs:
- **Sudden MSE jumps**: Large performance drops immediately outside training range
- **Chaotic outputs**: RNN produces non-sinusoidal or constant outputs
- **High variance**: Large MSE fluctuations across similar frequencies

### Key Plots:
1. **Error vs Frequency Plot**: Shows generalization capability across frequency spectrum
   - Green shaded area = training range
   - Points outside shaded area = generalization performance
2. **Individual Trajectory Plots**: Detailed view of RNN output quality for each test frequency

### Typical Expectations:
- **Very low frequencies (< 0.1 rad/s)**: May struggle due to long-term dependencies
- **Very high frequencies (> 1.0 rad/s)**: May struggle due to fast dynamics  
- **Near training range**: Often shows best generalization performance

## Troubleshooting

### Common Issues:

1. **"No runs found"**:
   - Make sure you're in the Main_JAX_Refactored directory
   - Check that the Outputs folder exists with training runs
   - Verify training runs contain parameter files (rnn_params_*.npz)

2. **"No RNN parameter files found"**:
   - The selected run directory doesn't contain the required .npz parameter files
   - Make sure the training run completed successfully

3. **Import errors**:
   - Ensure you're running from the Main_JAX_Refactored directory
   - Check that all required dependencies are installed (JAX, matplotlib, numpy)

4. **Plot display issues**:
   - The script uses a non-interactive matplotlib backend to save plots
   - All plots are saved as PNG files, not displayed on screen

### Getting Help:
- Use `python TEST_Frequency_Generalization.py --help` for command line help
- Check the script comments for detailed function documentation

## Script Structure

The script is structured to integrate seamlessly with the refactored codebase:

- **Main function**: Handles argument parsing and interactive/command-line modes
- **Run selection**: Automatically discovers available runs in the Outputs folder
- **Parameter loading**: Loads RNN parameters from .npz files and extracts configuration
- **Testing framework**: Uses `run_single_task_diagnostics` from `rnn_model.py` 
- **Visualization**: Creates trajectory vs target plots using `plot_trajectories_vs_targets`
- **Output organization**: Saves all results in the Outputs folder with clear structure

## Integration with Existing Code

This script leverages:
- `rnn_model.py`: For `run_single_task_diagnostics` function and RNN model
- `visualization.py`: For `plot_trajectories_vs_targets` plotting function
- `data_generation.py`: For generating test data with specified frequencies
- `config.py`: For configuration management
- `utils.py`: For directory utilities

The testing procedure uses the same diagnostic framework as training, ensuring consistency and comparability with training results. All outputs follow the same format and structure as the main training pipeline.
