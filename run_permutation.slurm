#!/bin/bash
#SBATCH --job-name=permutation_angle_encoding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --partition=cpu
#SBATCH --output=permutation_angle_%j.out
#SBATCH --error=permutation_angle_%j.err

# Set up environment variables for better performance
export OMP_NUM_THREADS=100
export MKL_NUM_THREADS=100
export NUMBA_NUM_THREADS=100

# Define virtual environment path
VENV_PATH="./venv_permutation"

# Check if virtual environment exists, create if it doesn't
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH"
    python -m venv $VENV_PATH
    
    # Activate and install packages
    source $VENV_PATH/bin/activate
    
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install pennylane
    
    echo "Virtual environment created and packages installed"
else
    echo "Virtual environment already exists at $VENV_PATH"
    source $VENV_PATH/bin/activate
    echo "Virtual environment activated"
fi

# Run the Python script
echo "Starting permutation angle encoding experiment with 100 CPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

python permutation_angle_encoding.py

echo "End time: $(date)"
echo "Job completed"