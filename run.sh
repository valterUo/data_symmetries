#!/bin/bash
#SBATCH --job-name=expressibility_sweep
#SBATCH --output=logs/slurm-%A_%a.out   # %A = master job ID, %a = array index
#SBATCH --array=1-95                    # 5 depths × 19 ansatz IDs = 95 jobs
#SBATCH --cpus-per-task=8               # CPUs per run (tweak as needed)
#SBATCH --mem=4G
#SBATCH --time=10:00:00

module load scicomp-python-env

# Activate your environment
source ~/myenv/bin/activate   # or: conda activate myenv

# --- Compute parameters from array index ---
# depth ∈ [1,5], ansatz_id ∈ [1,19]
DEPTH=$(( ($SLURM_ARRAY_TASK_ID - 1) % 5 + 1 ))
ANSATZ_ID=$(( ($SLURM_ARRAY_TASK_ID - 1) % 19 + 1 ))

echo "Running job with depth=$DEPTH, ansatz_id=$ANSATZ_ID"
#srun python expressibility.py --depth "$DEPTH" --ansatz_id "$ANSATZ_ID"
srun python entangling_capability.py --depth "$DEPTH" --ansatz_id "$ANSATZ_ID"
