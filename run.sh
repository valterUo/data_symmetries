#!/bin/bash
#SBATCH --job-name=symmetry            # Job name (shown in squeue)
#SBATCH --output=slurm-%j.out          # Stdout/stderr log file (%j = job ID)
#SBATCH --cpus-per-task=100            # Give this job 100 CPU cores
#SBATCH --time=48:00:00                # Max runtime (hh:mm:ss)
#SBATCH --mem=25G                      # Total memory requested
#SBATCH --partition=compute            # (optional) specify a partition/queue

# --- Environment setup ---
# Load modules if needed (depends on your cluster)
# module load python/3.10

# Activate your Python environment
source ~/myenv/bin/activate            # or: conda activate myenv

# --- Run your script ---
echo "Starting job at $(date)"
srun python your_script.py
echo "Job finished at $(date)"