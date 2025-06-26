# Gap Training Instructions

## Overview

The `TEST_Frequency_Generalization.py` script has been enhanced to support **Gap Training** - training RNNs on frequencies from 0.1-0.25 and 0.45-0.6 while excluding the gap region (0.25-0.45), then testing generalization across the full frequency range (0.01-1.2).

## Key Features Added

1. **Gap Training Mode**: Train on non-contiguous frequency ranges
2. **Variable Test Time**: Adjust testing duration based on frequency (ensures adequate cycles for low frequencies)
3. **Comprehensive Testing**: Test across 50+ frequencies including gap region and extrapolation ranges
4. **Enhanced Analysis**: Detailed breakdown of performance by region (training/gap/extrapolation)

## Configuration

### In `_1_config.py`:
- `TRAINING_FREQ_RANGES`: Defines the training ranges [(0.1, 0.25), (0.45, 0.6)]
- `TESTING_FREQ_RANGE`: Full testing range (0.01, 1.2)
- `VARIABLE_TEST_TIME`: Enable frequency-dependent testing time

### In `TEST_Frequency_Generalization.py`:
- `GAP_TRAINING_MODE`: Set to `True` to enable gap training mode

## Usage

### 1. Gap Training Mode
```bash
# Run gap training (GAP_TRAINING_MODE = True in the script)
python TEST_Frequency_Generalization.py --mode train

# Or use command line flag
python TEST_Frequency_Generalization.py --mode train
```

### 2. Testing Mode
```bash
# Test existing trained model
python TEST_Frequency_Generalization.py --mode test --run_name your_trained_model

# Test with custom frequencies
python TEST_Frequency_Generalization.py --mode test --test_frequencies 0.3 0.35 0.4 --run_name your_model
```

### 3. Interactive Mode
```bash
# Run without arguments for interactive prompts
python TEST_Frequency_Generalization.py
```

## What Happens During Gap Training

1. **Filtered Training Data**: Only frequencies in ranges 0.1-0.25 and 0.45-0.6 are used
2. **Custom Loss Function**: Modified to only compute loss on gap training frequencies
3. **Standard Training Pipeline**: Uses existing Adam + L-BFGS optimization
4. **Automatic Testing**: Option to immediately test the trained model on comprehensive frequency set

## Output Structure

### Gap Training Output:
```
Outputs/gap_training_YYYYMMDD_HHMMSS/
├── gap_trained_params_YYYYMMDD_HHMMSS.npz
├── gap_training_config_YYYYMMDD_HHMMSS.txt
└── [optional test results if requested]
```

### Testing Output:
```
Outputs/gap_training_test_YYYYMMDD_HHMMSS/
├── error_vs_frequency_gap_training.png
├── gap_training_test_summary.txt
└── [individual trajectory plots]
```

## Key Analysis Features

### Performance Breakdown by Region:
- **Training regions**: Performance on 0.1-0.25 and 0.45-0.6 ranges
- **Gap region**: Generalization to 0.25-0.45 (excluded during training)
- **Extrapolation**: Performance on 0.01-0.1 and 0.6-1.2 ranges

### Variable Test Time:
- Low frequencies get longer test times to capture multiple cycles
- High frequencies use standard test duration
- Ensures fair comparison across frequency ranges

### Enhanced Visualizations:
- Error vs frequency plot with highlighted training/gap regions
- Log-scale MSE visualization for better dynamic range
- Comprehensive statistical summaries

## Benefits

1. **Minimal Code Changes**: Leverages existing training infrastructure
2. **Comprehensive Testing**: Automatic generation of test frequencies across full range
3. **Detailed Analysis**: Region-specific performance metrics
4. **Flexible**: Easy to modify frequency ranges and test parameters
5. **Self-Contained**: Everything in one script, no new dependencies

## Tips

- Monitor training progress - gap training may take longer due to reduced training data
- Check the gap region performance to assess generalization capability
- Use the interactive mode for quick experiments