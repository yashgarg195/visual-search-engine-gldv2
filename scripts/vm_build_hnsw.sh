#!/usr/bin/env bash
# ==============================================================================
# High-Connectivity HNSW-PQ Graph Builder for Google Cloud Compute VM
# ==============================================================================
# Builds a high-degree, deep-exploration HNSW graph (M=64, efConstruction=256)
# over 5.05M vectors to prevent graph partitioning and local minima traps.
#
# Recommended VM: c2-standard-16 or n2-standard-16 (16 vCPUs, 64 GB RAM, Ubuntu 22.04)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# 1. Environment & Parameter Defaults
M="${M:-64}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-256}"
EF_SEARCH="${EF_SEARCH:-128}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
OUTPUT_NAME="${OUTPUT_NAME:-hnswpq_M64_efC256.index}"

# Detect available CPU cores (cap at available physical/vCPUs)
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)
NUM_THREADS="${NUM_THREADS:-${NUM_CORES}}"

TOTAL_RAM_GB=$(awk '/MemTotal/ { printf "%.0f", $2/1024/1024 }' /proc/meminfo 2>/dev/null || sysctl -n hw.memsize 2>/dev/null | awk '{ printf "%.0f", $1/1024/1024/1024 }' || echo "Unknown")

echo "=============================================================================="
echo " Starting High-Connectivity HNSW-PQ Index Build on Cloud VM"
echo "=============================================================================="
echo " - System Cores:      ${NUM_CORES} (Building with ${NUM_THREADS} threads)"
echo " - System RAM:        ${TOTAL_RAM_GB} GB"
echo " - Graph Degree (M):  ${M} (128 connections/node on layer 0)"
echo " - efConstruction:    ${EF_CONSTRUCTION} (4x deeper candidate exploration)"
echo " - efSearch:          ${EF_SEARCH} (Default runtime beam width)"
echo " - Codebook Sample:   ${SAMPLE_SIZE} vectors"
echo " - Target Output:     outputs/index/${OUTPUT_NAME}"
echo "=============================================================================="

# 2. Pre-flight checks
if [ ! -d "outputs/embeddings" ]; then
    echo "ERROR: outputs/embeddings directory not found in ${PROJECT_DIR}."
    echo "Please ensure the 10 GB embedding shards are present."
    exit 1
fi

mkdir -p outputs/index

# 3. Optimize OpenMP and thread scheduling for high-throughput AVX
export OMP_NUM_THREADS="${NUM_THREADS}"
export MKL_NUM_THREADS="${NUM_THREADS}"
export FAISS_NUM_THREADS="${NUM_THREADS}"
export PYTHONUNBUFFERED=1

# 4. Run the HNSW-PQ construction script
START_TIME=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching FAISS HNSW graph builder..."

python3 scripts/train_hnsw_pq.py \
    --M "${M}" \
    --ef-construction "${EF_CONSTRUCTION}" \
    --ef-search "${EF_SEARCH}" \
    --sample-size "${SAMPLE_SIZE}" \
    --num-threads "${NUM_THREADS}" \
    --output-name "${OUTPUT_NAME}" \
    --checkpoint-interval 100

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$(awk "BEGIN {printf \"%.1f\", ${ELAPSED}/60}")

echo "=============================================================================="
echo " Build Completed Successfully in ${ELAPSED_MIN} minutes (${ELAPSED} seconds)!"
echo "=============================================================================="

# 5. Summary and sync instructions
if [ -f "outputs/index/${OUTPUT_NAME}" ]; then
    INDEX_SIZE=$(ls -lh "outputs/index/${OUTPUT_NAME}" | awk '{print $5}')
    echo " Generated Index: outputs/index/${OUTPUT_NAME} (${INDEX_SIZE})"
    echo ""
    echo " To copy this back to your local MacBook, run from your Mac:"
    echo "   gcloud compute scp <VM_NAME>:$(pwd)/outputs/index/${OUTPUT_NAME} outputs/index/hnswpq_populated.index"
    echo "   gcloud compute scp <VM_NAME>:$(pwd)/outputs/index/${OUTPUT_NAME%.*}.meta.json outputs/index/hnswpq_populated.meta.json"
    echo "=============================================================================="
fi
