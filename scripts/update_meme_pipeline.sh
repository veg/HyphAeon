#!/usr/bin/env bash

# ==============================================================================
# Driver Script: End-to-End MEME Database Update Pipeline
# ==============================================================================

set -euo pipefail

# ANSI color codes for enhanced readability
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RED="\033[31m"

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="${BASE_DIR}/meme_results.db"

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

log_warning() {
    echo -e "${YELLOW}${BOLD}warning:${RESET} $1"
}

log_error() {
    echo -e "${RED}${BOLD}error:${RESET} $1" >&2
}

# Ensure we are in the correct base directory
cd "${BASE_DIR}"
log_info "Base directory: ${BASE_DIR}"
log_info "Target Database: ${DB_PATH}"

# Check for Python dependencies
if ! command -v python3 &> /dev/null; then
    log_error "python3 is not installed or not in PATH."
    exit 1
fi

t_start_total=$(date +%s)

# ------------------------------------------------------------------------------
# STEP 1: Sync Raw JSONs & Incremental Database Compile
# ------------------------------------------------------------------------------
log_step "1" "Synchronizing MEME JSONs & Incremental Compile"
SYNC_SCRIPT="scripts/pull_and_update_all.py"

if [[ -f "${SYNC_SCRIPT}" ]]; then
    log_info "Running ${SYNC_SCRIPT}..."
    t_start=$(date +%s)
    if python3 "${SYNC_SCRIPT}"; then
        t_end=$(date +%s)
        log_success "Incremental database sync completed in $((t_end - t_start)) seconds."
    else
        log_error "Sync script failed!"
        exit 1
    fi
else
    log_warning "Sync script ${SYNC_SCRIPT} not found."
    log_info "Assuming database already has the latest raw results or is updated externally."
fi

# ------------------------------------------------------------------------------
# STEP 2: Benjamini-Hochberg FDR / Q-Value Recalculation
# ------------------------------------------------------------------------------
log_step "2" "Recalculating FDR Q-Values & Significance Filters"
QVAL_SCRIPT="scratch/populate_meme_qvals.py"

if [[ -f "${QVAL_SCRIPT}" ]]; then
    log_info "Running Q-value calculation: ${QVAL_SCRIPT}..."
    t_start=$(date +%s)
    if python3 "${QVAL_SCRIPT}"; then
        t_end=$(date +%s)
        log_success "Benjamini-Hochberg FDR correction complete in $((t_end - t_start)) seconds."
    else
        log_error "Q-value calculations failed!"
        exit 1
    fi
else
    log_error "Required Q-value update helper script ${QVAL_SCRIPT} was not found!"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 3: Database Indexing & Vacuum Optimization
# ------------------------------------------------------------------------------
log_step "3" "Database Indexing & Vacuum Optimization"
OPTIMIZE_SCRIPT="scratch/optimize_meme_db.py"

if [[ -f "${OPTIMIZE_SCRIPT}" ]]; then
    log_info "Running database optimization: ${OPTIMIZE_SCRIPT}..."
    t_start=$(date +%s)
    if python3 "${OPTIMIZE_SCRIPT}"; then
        t_end=$(date +%s)
        log_success "Database indexing and vacuum optimization complete in $((t_end - t_start)) seconds."
    else
        log_error "Database optimization script failed!"
        exit 1
    fi
else
    log_error "Required database optimization helper script ${OPTIMIZE_SCRIPT} was not found!"
    exit 1
fi

# ------------------------------------------------------------------------------
# PIPELINE COMPLETE SUMMARY
# ------------------------------------------------------------------------------
t_end_total=$(date +%s)
echo -e "\n${BOLD}${GREEN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN}🎉 END-TO-END UPDATE PIPELINE COMPLETED SUCCESSFULLY!${RESET}"
echo -e "${BOLD}${GREEN}==============================================================================${RESET}"
log_info "Total elapsed time: $((t_end_total - t_start_total)) seconds."
if [[ -f "${DB_PATH}" ]]; then
    db_size=$(du -sh "${DB_PATH}" | cut -f1)
    log_info "Final Database size: ${db_size} (${DB_PATH})"
fi
