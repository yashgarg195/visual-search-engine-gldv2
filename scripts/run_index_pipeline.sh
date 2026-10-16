#!/usr/bin/env bash
# ==============================================================================
# FAISS IVF-PQ Index Build & Evaluation Pipeline Runner
# Trains IVF-PQ coarse/fine quantizers, adds all shards with explicit IDs,
# and executes ANN recall and recognition accuracy evaluations.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE

NLIST="${1:-1024}"
M="${2:-64}"
NBITS="${3:-8}"
SAMPLE_SIZE="${4:-50000}"

echo "======================================================================"
echo " 1. TRAINING FAISS IVF-PQ INDEX"
echo " nlist       : ${NLIST}"
echo " m           : ${M}"
echo " nbits       : ${NBITS}"
echo " Sample size : ${SAMPLE_SIZE}"
echo "======================================================================"

python3 "${PROJECT_ROOT}/scripts/train_ivfpq.py" \
    --nlist "${NLIST}" \
    --m "${M}" \
    --nbits "${NBITS}" \
    --sample-size "${SAMPLE_SIZE}"

echo ""
echo "======================================================================"
echo " 2. INGESTING EMBEDDING SHARDS INTO IVF-PQ INDEX"
echo "======================================================================"

python3 "${PROJECT_ROOT}/scripts/add_embeddings.py"

echo ""
echo "======================================================================"
echo " 3. EVALUATING RETRIEVAL BENCHMARK & RECOGNITION"
echo "======================================================================"

python3 "${PROJECT_ROOT}/scripts/evaluate.py" \
    --num-queries 100 \
    --nprobe-sweep 1 4 8 16 32 64

echo "Index pipeline completed successfully."
