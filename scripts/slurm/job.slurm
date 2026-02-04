#!/bin/bash
#SBATCH --job-name=m2-ovo-mae
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Print some debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODENAME"
echo "Starting at: $(date)"

# Execute the command passed as arguments to sbatch
# Example: sbatch job.slurm "uv run python src/m2_ovo_mae/train_pretrain.py ..."
eval "$@"

echo "Finished at: $(date)"
