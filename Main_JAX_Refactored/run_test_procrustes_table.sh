#!/bin/bash
#SBATCH --job-name=procrustes_table
#SBATCH --output=logs/procrustes_table_%j.out
#SBATCH --error=logs/procrustes_table_%j.err

# notify me by e-mail when the job ends or fails:
#SBATCH --mail-user=gianluca.carrozzo.24@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu

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
# Run Procrustes table analysis
# ------------------------------------
cd "$HOME/Sine-Wave-Generator-Example-from-Opening-the-Black-Box-/Main_JAX_Refactored" || exit 1
echo "Running Procrustes table analysis in $(pwd)"
python TEST_Procrustes_Table.py

# ------------------------------------
# Clean up
# ------------------------------------
conda deactivate
echo "Job completed successfully on $(date)"
