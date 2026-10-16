#!/usr/bin/env bash
# ==============================================================================
# Resumable Large-Scale Embedding Extraction Runner
# Generates normalized CLIP embeddings in persistent shards with checkpointing.
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

SPLIT="${1:-train}"
BATCH_SIZE="${2:-64}"
SHARD_SIZE="${3:-10000}"
DEVICE="${4:-auto}"

echo "======================================================================"
echo " LAUNCHING RESUMABLE EMBEDDING PIPELINE"
echo " Split      : ${SPLIT}"
echo " Batch Size : ${BATCH_SIZE}"
echo " Shard Size : ${SHARD_SIZE}"
echo " Device     : ${DEVICE}"
echo "======================================================================"

python3 "${PROJECT_ROOT}/scripts/embed_images.py" \
    --split "${SPLIT}" \
    --batch-size "${BATCH_SIZE}" \
    --shard-size "${SHARD_SIZE}" \
    --device "${DEVICE}"

echo "Embedding pipeline completed successfully."
