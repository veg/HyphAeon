#!/bin/bash
#SBATCH --job-name=busted_head_slurm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sergei/Projects/BUSTED-PH/logs/busted_slurm_%j.out
#SBATCH --error=/home/sergei/Projects/BUSTED-PH/logs/busted_slurm_%j.err

echo "=== Running BUSTED-PH / BUSTED+S Head Training on SLURM Compute Node ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Mem:  128GB"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

mkdir -p /home/sergei/Projects/BUSTED-PH/logs
mkdir -p /home/sergei/Projects/BUSTED-PH/busted_head_output

/home/sergei/.conda/envs/pytorch/bin/python /home/sergei/Projects/BUSTED-PH/train_busted_head.py \
  --msa_dir /home/sergei/Projects/BUSTED-PH/mammalian120/msas \
  --json_dir /home/sergei/Projects/BUSTED-PH/mammalian120/echo_msa_results_take3 \
  --output_dir /home/sergei/Projects/BUSTED-PH/busted_head_output \
  --max_samples 1000 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3

echo "=== Job Completed Successfully ==="
date
