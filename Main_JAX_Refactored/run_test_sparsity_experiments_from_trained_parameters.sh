#!/bin/bash
#SBATCH --job-name=sparsity_experiments_from_trained_parameters
#SBATCH --output=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_from_trained_params_%j.out
#SBATCH --error=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_from_trained_params_%j.err
#SBATCH --chdir=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored

# notify me by e-mail when the job ends or fails:
#SBATCH --mail-user=gianluca.carrozzo.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
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
  echo "ERROR: could not find conda.sh at $HOME/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Activate pre-created environment
if ! conda activate lr_env 2>/dev/null; then
  echo "ERROR: Conda env 'lr_env' not found. Create it first." >&2
  exit 1
fi

# Optional module load (uncomment if cluster requires)
# module load cuda/12.2

# Lightweight dependency self-heal
if ! python -c 'import matplotlib, jax, numpy, optax' 2>/dev/null; then
  echo "Installing / updating dependencies" >&2
  pip install --no-cache-dir -r requirements.txt || { echo "Dependency installation failed" >&2; exit 2; }
fi

unset OMP_NUM_THREADS
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Environment setup:"
python --version
echo "Conda env: $CONDA_DEFAULT_ENV"
python - <<'PY'
import jax, sys
print("JAX devices:", jax.devices())
if not any(d.platform == 'gpu' for d in jax.devices()):
    print("WARNING: No GPU detected by JAX. Running on CPU.", file=sys.stderr)
PY

echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

# ------------------------------------
# Run sparsity experiments (from trained parameters)
# ------------------------------------
echo "Running TEST_Sparsity_Experiments_from_Trained_Parameters.py"

echo "Start time: $(date)"

# Provide inputs for the Python script through heredoc
# Input sequence:
# 1. Mode selection: "2" for import mode
# 2. Sparsity values: "0.0, 0.2, 0.4, 0.6, 0.8, 0.99"
# 3. Confirmation: "y" to proceed with sparsity values
python -u TEST_Sparsity_Experiments_from_Trained_Parameters.py << EOF
2
0.0, 0.2, 0.4, 0.6, 0.8, 0.99
y
EOF
PYTHON_EXIT_CODE=$?

echo "Python script completed at $(date) with exit code: $PYTHON_EXIT_CODE"
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
  echo "ERROR: Python script failed" >&2
  exit $PYTHON_EXIT_CODE
fi

# Check for output files
if [ -d "../Outputs" ]; then
    echo "Recent output directory contents:"
    ls -lt ../Outputs | head -10
fi

conda deactivate || true

echo "Job completed successfully on $(date)"