#!/bin/bash
#SBATCH --job-name=TOGA_absrel
#SBATCH --output=/home/sergei/Projects/TOGA/logs/absrel_%A_%a.out
#SBATCH --error=/home/sergei/Projects/TOGA/logs/absrel_%A_%a.err
#SBATCH --array=0-124
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --partition=general

# Load environment modules (if any are needed, but we will run the compiled hyphy-dev directly)
# Explicitly set number of threads for OpenMP
export OMP_NUM_THREADS=8

GENE_LIST="/home/sergei/Projects/TOGA/gene_list_magilla.txt"
LOCK_DIR="/home/sergei/Projects/TOGA/locks"
MSA_DIR="/home/sergei/Projects/TOGA/msa"
OUT_DIR="/home/sergei/Projects/TOGA/absrel"
LOGS_DIR="/home/sergei/Projects/TOGA/logs"

# Ensure log directory exists
mkdir -p "$LOGS_DIR"

echo "Array Task ID ${SLURM_ARRAY_TASK_ID} starting..."

# Loop through gene list and process dynamically
while read -r gene; do
    # Try to acquire an atomic lock for this gene
    if mkdir "${LOCK_DIR}/${gene}.lock" 2>/dev/null; then
        echo "[Task ${SLURM_ARRAY_TASK_ID}] Processing ${gene}..."
        
        # Run hyphy absrel with specific LIBPATH and CPU cores
        # CPU=8 specifies the thread count inside HyPhy
        # ENV="GZIP_OUTPUT=1;" forces gzipped JSON output
        /home/sergei/hyphy-dev/hyphy LIBPATH=//home/sergei/hyphy-dev/res absrel --alignment "${MSA_DIR}/${gene}.gz" --output "${OUT_DIR}/${gene}.json.gz" CPU=8 ENV="GZIP_OUTPUT=1;"
        
        echo "[Task ${SLURM_ARRAY_TASK_ID}] Finished ${gene}."
    fi
done < "$GENE_LIST"

echo "Array Task ID ${SLURM_ARRAY_TASK_ID} completed."
