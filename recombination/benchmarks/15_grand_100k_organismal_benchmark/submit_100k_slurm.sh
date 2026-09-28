#!/bin/bash
#SBATCH --job-name=grand100k
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --array=0-49%50
#SBATCH --output=/data/users/sergei/Projects/TOGA_MEME/recombination/benchmarks/15_grand_100k_organismal_benchmark/logs/slurm_%A_%a.out
#SBATCH --error=/data/users/sergei/Projects/TOGA_MEME/recombination/benchmarks/15_grand_100k_organismal_benchmark/logs/slurm_%A_%a.err

set -e

# Parameters
TOTAL_SIMS=100000
NUM_SHARDS=50
CHUNK_SIZE=$((TOTAL_SIMS / NUM_SHARDS)) # 2000

SHARD_ID=${SLURM_ARRAY_TASK_ID}
START_IDX=$((SHARD_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE))

BASE_DIR="/data/users/sergei/Projects/TOGA_MEME/recombination/benchmarks/15_grand_100k_organismal_benchmark"
SHARDS_DIR="${BASE_DIR}/shards"
mkdir -p "${SHARDS_DIR}" "${BASE_DIR}/logs"

DB_OUT="${SHARDS_DIR}/shard_${SHARD_ID}.db"
THREE_SEQ_BIN="/home/sergei/tools/3seq/build/3seq"
PTABLE_PATH="/home/sergei/tools/3seq/build/3seq_ptable_250"
RHIZAEON_SRC="/data/users/sergei/Projects/TOGA_MEME/axomeme_repo/rhizaeon/src"

echo "=========================================================="
echo "Starting Grand 100k Benchmark Shard ${SHARD_ID}"
echo "Node: $(hostname) | Range: [${START_IDX}, ${END_IDX})"
echo "Start time: $(date)"
echo "=========================================================="

export PYTHONPATH="${RHIZAEON_SRC}:${PYTHONPATH}"

PYTORCH_PYTHON="/home/sergei/.conda/envs/pytorch/bin/python"

"${PYTORCH_PYTHON}" "${BASE_DIR}/benchmark_worker.py" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --shard-id "${SHARD_ID}" \
  --db-output "${DB_OUT}" \
  --3seq-bin "${THREE_SEQ_BIN}" \
  --ptable "${PTABLE_PATH}" \
  --rhizaeon-src "${RHIZAEON_SRC}" \
  --timeout 60.0

echo "=========================================================="
echo "Completed Grand 100k Benchmark Shard ${SHARD_ID}"
echo "End time: $(date)"
echo "=========================================================="
