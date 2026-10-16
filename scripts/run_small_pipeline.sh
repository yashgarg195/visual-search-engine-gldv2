#!/usr/bin/env bash
# ==============================================================================
# End-to-End Small-Scale Landmark Search Pipeline Runner
# Executes environment check, metadata prep, embedding extraction,
# IVF-PQ training, addition, query demonstration, and ANN evaluation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtualenv if present
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "======================================================================"
echo " 1. SYSTEM ENVIRONMENT & HARDWARE AUDIT"
echo "======================================================================"
python3 "${PROJECT_ROOT}/scripts/check_environment.py"

echo ""
echo "======================================================================"
echo " 2. ENSURING SAMPLE DATASET"
echo "======================================================================"
# If raw image data does not exist, generate synthetic GLDv2 sample for testing
if [[ ! -d "${PROJECT_ROOT}/data/raw/images/train" ]] || [[ $(find "${PROJECT_ROOT}/data/raw/images/train" -type f | wc -l) -lt 256 ]]; then
    echo "Generating representative synthetic GLDv2 sample (300 images) for local verification..."
    python3 "${PROJECT_ROOT}/scripts/generate_synthetic_sample.py" --images-per-landmark 60
else
    echo "Existing image directory found at data/raw/images. Proceeding with existing data."
fi

echo ""
echo "======================================================================"
echo " 3. PREPARING & VALIDATING METADATA"
echo "======================================================================"
python3 "${PROJECT_ROOT}/scripts/prepare_metadata.py" --splits train index --force

PRETRAINED="${1:-none}"

echo ""
echo "======================================================================"
echo " 4. GENERATING NORMALIZED CLIP EMBEDDINGS (SHARDED)"
echo "======================================================================"
python3 "${PROJECT_ROOT}/scripts/embed_images.py" \
    --split train \
    --batch-size 32 \
    --shard-size 250 \
    --pretrained "${PRETRAINED}"

echo ""
echo "======================================================================"
echo " 5. TRAINING FAISS IVF-PQ INDEX"
echo "======================================================================"
# For small sample test, use nlist=8, m=64, nbits=8 (trained with 300 vectors >= 256)
python3 "${PROJECT_ROOT}/scripts/train_ivfpq.py" \
    --nlist 8 \
    --m 64 \
    --nbits 8 \
    --sample-size 300 \
    --force

echo ""
echo "======================================================================"
echo " 6. ADDING EMBEDDINGS TO IVF-PQ INDEX WITH ID TRACKING"
echo "======================================================================"
python3 "${PROJECT_ROOT}/scripts/add_embeddings.py" --force-rebuild

echo ""
echo "======================================================================"
echo " 7. VISUAL SEARCH DEMONSTRATION & LANDMARK VOTING"
echo "======================================================================"
QUERY_IMG="${PROJECT_ROOT}/data/sample_queries/query_eiffel_tower.jpg"
if [[ -f "${QUERY_IMG}" ]]; then
    python3 "${PROJECT_ROOT}/scripts/search.py" \
        --query-image "${QUERY_IMG}" \
        --top-k 5 \
        --nprobe 8 \
        --pretrained "${PRETRAINED}"
else
    echo "Query image not found, running text search demo..."
    python3 "${PROJECT_ROOT}/scripts/search.py" \
        --query-text "Eiffel Tower in Paris" \
        --top-k 5 \
        --nprobe 8 \
        --pretrained "${PRETRAINED}"
fi

echo ""
echo "======================================================================"
echo " 8. BENCHMARK EVALUATION (ANN RECALL & RECOGNITION)"
echo "======================================================================"
python3 "${PROJECT_ROOT}/scripts/evaluate.py" \
    --num-queries 20 \
    --nprobe-sweep 1 4 8 16

echo ""
echo "======================================================================"
echo " [SUCCESS] SMALL PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================================================"
