#!/bin/bash
#SBATCH --job-name=freq_generalization
#SBATCH --output=logs/freq_generalization_%j.out
#SBATCH --error=logs/freq_generalization_%j.err

# notify me by e-mail when the job ends or fails:
#SBATCH --mail-user=gianluca.carrozzo.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Job starting on node: $SLURMD_NODENAME"

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

conda activate lr_env

# ------------------------------------
# Run frequency generalization analysis
# ------------------------------------
cd "$HOME/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored" || exit 1
echo "Running frequency generalization analysis in $(pwd)"
python TEST_Frequency_Generalization.py

# ------------------------------------
# Clean up
# ------------------------------------
conda deactivate
echo "Job completed successfully on $(date)"
