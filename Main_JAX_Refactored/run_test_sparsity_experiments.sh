#!/bin/bash
#SBATCH --job-name=sparsity_experiments
#SBATCH --output=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_%j.out
#SBATCH --error=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored/logs/sparsity_exp_%j.err
#SBATCH --chdir=/nfs/ghome/live/gcarrozzo/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored

# notify me by e-mail when the job ends or fails:
#SBATCH --mail-user=gianluca.carrozzo.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
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
# Make sure your ~/.bashrc has the conda hook block, or source it directly:
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: could not find conda.sh at $HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

# Activate pre-created environment (create it manually once outside the job)
if ! conda activate lr_env 2>/dev/null; then
  echo "ERROR: Conda env 'lr_env' not found. Create it first (see README)." >&2
  exit 1
fi

# Optional on some clusters: load a CUDA module (uncomment & adjust if needed)
# module load cuda/12.2

# Lightweight dependency self-heal (installs only if missing)
if ! python -c 'import matplotlib, jax, numpy, optax' 2>/dev/null; then
  echo "Installing / updating dependencies (matplotlib, jax, numpy, scipy, tqdm, scikit-learn, optax)" >&2
  pip install --no-cache-dir -r requirements.txt || {
    echo "Dependency installation failed" >&2; exit 2; }
fi

# Set environment variables for optimal performance
unset OMP_NUM_THREADS
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0 # Pin to a single GPU

echo "Environment setup:"
python --version
echo "Conda env: $CONDA_DEFAULT_ENV"
python - <<'PY'
import jax, sys
devs = jax.devices()
print("JAX devices:", devs)
if not any(d.platform == 'gpu' for d in devs):
  print("WARNING: No GPU detected by JAX. Running on CPU.", file=sys.stderr)
PY
echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

# ------------------------------------
# Run sparsity experiments with automated input
# ------------------------------------
echo "Running original sparsity experiments in $(pwd)"

# Run the experiment with automated inputs
echo "Starting Python script at $(date)"
echo "Running original TEST_Sparsity_Experiments.py with fixed JAX debug print"
python -u TEST_Sparsity_Experiments.py << EOF
0.0
y
2
EOF
PYTHON_EXIT_CODE=$?

echo "Python script completed at $(date) with exit code: $PYTHON_EXIT_CODE"
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
  echo "ERROR: Python script failed" >&2
  exit $PYTHON_EXIT_CODE
fi

# Check for output files
if [ -d "../Outputs" ]; then
    echo "Output directory contents:"
    ls -la ../Outputs | tail -5
fi

# ------------------------------------
# Clean up
# ------------------------------------
conda deactivate || true
echo "Job completed successfully on $(date)"