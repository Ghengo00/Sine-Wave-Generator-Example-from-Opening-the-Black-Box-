#!/bin/bash
#SBATCH --job-name=sparsity_experiments_safe
#SBATCH --output=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_safe_%j.out
#SBATCH --error=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_safe_%j.err
#SBATCH --chdir=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored

# notify me by e-mail when the job ends or fails:
#SBATCH --mail-user=gianluca.carrozzo.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G  # Reduced from 128G to leave memory buffer
#SBATCH --time=24:00:00  # Reduced time to force earlier completion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job starting on node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"

# Ensure the logs directory exists
mkdir -p logs

# ------------------------------------
# Initialize and activate Conda
# ------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: could not find conda.sh at $HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

if ! conda activate lr_env 2>/dev/null; then
  echo "ERROR: Conda env 'lr_env' not found. Create it first (see README)." >&2
  exit 1
fi

# Install psutil for memory monitoring
pip install --quiet psutil

# Set environment variables for safer execution
unset OMP_NUM_THREADS
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=False  # Use 32-bit for memory efficiency
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Conservative GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Don't preallocate all GPU memory
ulimit -c 0  # Disable core dumps

echo "Environment setup:"
python --version
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Memory limits: JAX 32-bit, GPU mem fraction: 0.7"

# ------------------------------------
# Run sparsity experiments with safer settings
# ------------------------------------
echo "Running SAFE sparsity experiments in $(pwd)"

echo "Starting Python script at $(date)"
python -u TEST_Sparsity_Experiments.py << EOF
0.0
y
1
1
0, 25
y
EOF
PYTHON_EXIT_CODE=$?

echo "Python script completed at $(date) with exit code: $PYTHON_EXIT_CODE"

# Check for output files
if [ -d "../Outputs" ]; then
    echo "Output directory contents:"
    ls -la ../Outputs | tail -5
fi

# Clean up
conda deactivate || true

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully on $(date)"
else
    echo "Job failed with exit code: $PYTHON_EXIT_CODE on $(date)"
    exit $PYTHON_EXIT_CODE
fi
