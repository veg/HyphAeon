#!/usr/bin/env bash

# ==============================================================================
# Driver Script: AxoMeme Model Training & Inference Pipeline
# ==============================================================================

set -euo pipefail

# ANSI color codes for enhanced readability
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RED="\033[31m"

# Print usage if required arguments are missing
usage() {
    echo -e "Usage: $0 <DB_PATH> <MSA_DIR> <CACHE_PATH> [EPOCHS] [SUBSAMPLE] [NEW_ALIGNMENT] [INFERENCE_OUTPUT]"
    echo -e "\nPositional Arguments:"
    echo -e "  1. DB_PATH          Path to the SQLite selection database (e.g. meme_results.db)"
    echo -e "  2. MSA_DIR          Path to the MSA alignments folder containing gzipped files"
    echo -e "  3. CACHE_PATH       Path to load/save the precomputed MSA cache (e.g. msa_cache.pkl.gz)"
    echo -e "  4. EPOCHS           (Optional) Number of training epochs (default: 10)"
    echo -e "  5. SUBSAMPLE        (Optional) Limit dataset size for quick testing (default: none)"
    echo -e "  6. NEW_ALIGNMENT    (Optional) Path to a new NEXUS alignment file for inference"
    echo -e "  7. INFERENCE_OUTPUT (Optional) Path to save inference CSV (default: auto-generated)"
    exit 1
}

if [ "$#" -lt 3 ]; then
    usage
fi

DB_PATH="$1"
MSA_DIR="$2"
CACHE_PATH="$3"
EPOCHS="${4:-10}"
SUBSAMPLE="${5:-}"
NEW_ALIGNMENT="${6:-}"
INFERENCE_OUTPUT="${7:-}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

log_step() {
    local step_num="$1"
    local step_desc="$2"
    echo -e "\n${BOLD}${BLUE}==============================================================================${RESET}"
    echo -e "${BOLD}${BLUE}🚀 STEP ${step_num}: ${step_desc}${RESET}"
    echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
}

log_info() {
    echo -e "${BLUE}info:${RESET} $1"
}

log_success() {
    echo -e "${GREEN}${BOLD}success:${RESET} $1"
}

log_error() {
    echo -e "${RED}${BOLD}error:${RESET} $1" >&2
}

# Validation checks
if [ ! -f "${DB_PATH}" ]; then
    log_error "Database file not found: ${DB_PATH}"
    exit 1
fi

if [ ! -d "${MSA_DIR}" ]; then
    log_error "MSA directory not found: ${MSA_DIR}"
    exit 1
fi

t_start_total=$(date +%s)

# ------------------------------------------------------------------------------
# STEP 1: Precompute the MSA Cache
# ------------------------------------------------------------------------------
log_step "1" "Precomputing the MSA Cache"
if [ -f "${CACHE_PATH}" ]; then
    log_info "Cache file already exists: ${CACHE_PATH}"
    log_info "Skipping precomputation step (to force rebuild, delete the cache file)."
else
    log_info "Rebuilding cache at: ${CACHE_PATH}..."
    t_start=$(date +%s)
    python3 scripts/precompute_msa_cache.py \
      --db_path "${DB_PATH}" \
      --msa_dir "${MSA_DIR}" \
      --out_cache "${CACHE_PATH}"
    t_end=$(date +%s)
    log_success "MSA cache precomputed in $((t_end - t_start)) seconds."
fi

# ------------------------------------------------------------------------------
# STEP 2: Train the Model
# ------------------------------------------------------------------------------
log_step "2" "Training the Model"
log_info "Training epochs: ${EPOCHS}"

TRAIN_ARGS=(
  --db_path "${DB_PATH}"
  --msa_dir "${MSA_DIR}"
  --cache_path "${CACHE_PATH}"
  --epochs "${EPOCHS}"
)

if [ -n "${SUBSAMPLE}" ]; then
    log_info "Subsampling enabled: limiting to ${SUBSAMPLE} sites."
    TRAIN_ARGS+=(--subsample "${SUBSAMPLE}")
fi

t_start=$(date +%s)
python3 scripts/train_regression.py "${TRAIN_ARGS[@]}"
t_end=$(date +%s)
log_success "Model training complete in $((t_end - t_start)) seconds."

# ------------------------------------------------------------------------------
# STEP 3: Checkpoint Validation
# ------------------------------------------------------------------------------
log_step "3" "Checkpoint Validation"
MODEL_PATH="/Users/sergei/Projects/TOGA_MEME/MEME_transformer_joint.pt"

if [ -f "${MODEL_PATH}" ]; then
    log_success "Best model checkpoint saved at: ${MODEL_PATH}"
else
    log_error "Model checkpoint file was not found at expected path: ${MODEL_PATH}"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 4: Optional Downstream Inference
# ------------------------------------------------------------------------------
if [ -n "${NEW_ALIGNMENT}" ]; then
    log_step "4" "Downstream Inference on New Alignment"
    
    if [ ! -f "${NEW_ALIGNMENT}" ]; then
        log_error "New alignment file not found: ${NEW_ALIGNMENT}"
        exit 1
    fi
    
    INFERENCE_ARGS=(
      --alignment "${NEW_ALIGNMENT}"
      --model "${MODEL_PATH}"
      --device cpu
    )
    
    if [ -n "${INFERENCE_OUTPUT}" ]; then
        INFERENCE_ARGS+=(--output "${INFERENCE_OUTPUT}")
        log_info "Writing output predictions to: ${INFERENCE_OUTPUT}"
    else
        log_info "Writing output predictions to default path..."
    fi
    
    t_start=$(date +%s)
    python3 scripts/predict_regression_nexus.py "${INFERENCE_ARGS[@]}"
    t_end=$(date +%s)
    log_success "Inference completed successfully in $((t_end - t_start)) seconds."
fi

t_end_total=$(date +%s)
echo -e "\n${BOLD}${GREEN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN}🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!${RESET}"
echo -e "${BOLD}${GREEN}==============================================================================${RESET}"
log_info "Total elapsed time: $((t_end_total - t_start_total)) seconds."
