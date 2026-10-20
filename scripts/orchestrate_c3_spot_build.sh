#!/usr/bin/env bash
# ==============================================================================
# End-to-End GCP Spot VM Orchestrator for High-Connectivity HNSW Build
# ==============================================================================
# 1. Provisions a c3-standard-22 (22 vCPUs, 88 GB RAM) Spot VM (<$0.25/hr)
# 2. Transfers embedding shards and code
# 3. Builds HNSW graph (M=64, efConstruction=256) across 22 threads (~20 mins)
# 4. Downloads the resulting index (~2.8 GB) back to local Mac
# 5. Automatically DESTROYS the VM upon completion (or failure)
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="hnsw-c3-spot-${TIMESTAMP}"
ZONES=("asia-southeast1-a" "asia-southeast1-b" "asia-southeast1-c" "us-central1-a")
MACHINE_TYPE="c3-standard-22"
BOOT_DISK_SIZE="120GB"

ACTIVE_ZONE=""

cleanup() {
    if [ -n "${ACTIVE_ZONE}" ] && [ -n "${INSTANCE_NAME}" ]; then
        echo ""
        echo "=============================================================================="
        echo " [CLEANUP] Deleting Spot VM ${INSTANCE_NAME} in ${ACTIVE_ZONE}..."
        echo "=============================================================================="
        gcloud compute instances delete "${INSTANCE_NAME}" --zone="${ACTIVE_ZONE}" --quiet || true
        echo " [CLEANUP] VM deleted. Zero ongoing charges."
    fi
}
trap cleanup EXIT INT TERM

echo "=============================================================================="
echo " Launching Automated C3-Standard-22 Spot Build Pipeline"
echo "=============================================================================="
echo " - Machine:  ${MACHINE_TYPE} (22 vCPUs, 88 GB RAM, DDR5, Sapphire Rapids)"
echo " - Pricing:  SPOT Preemptible (~$0.25/hr -> ~$0.09 total)"
echo " - Graph:    M=64, efConstruction=256 (5.05M vectors)"
echo "=============================================================================="

# 1. Provision Spot VM with multi-zone fallback
VM_CREATED=0
for ZONE in "${ZONES[@]}"; do
    echo "[$(date '+%H:%M:%S')] Attempting to provision Spot VM in ${ZONE}..."
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --provisioning-model=SPOT \
        --instance-termination-action=DELETE \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type=pd-ssd \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --quiet; then
        ACTIVE_ZONE="${ZONE}"
        VM_CREATED=1
        echo " Successfully provisioned ${INSTANCE_NAME} in ${ZONE}!"
        break
    else
        echo " Spot capacity unavailable in ${ZONE}. Trying next zone..."
    fi
done

if [ "${VM_CREATED}" -ne 1 ]; then
    echo "ERROR: Failed to allocate Spot c3-standard-22 across all candidate zones."
    exit 1
fi

# 2. Wait for SSH to become ready
echo "[$(date '+%H:%M:%S')] Waiting for instance SSH to initialize..."
SSH_READY=0
for i in {1..40}; do
    if gcloud compute ssh "${INSTANCE_NAME}" --zone="${ACTIVE_ZONE}" --command="echo 'SSH ready'" --quiet 2>/dev/null; then
        echo " SSH is ready!"
        SSH_READY=1
        break
    fi
    sleep 5
done

if [ "${SSH_READY}" -ne 1 ]; then
    echo "ERROR: Instance SSH failed to become ready in time."
    exit 1
fi

# 3. Setup Remote Environment
echo "[$(date '+%H:%M:%S')] Installing Python, Faiss, and build tools on VM..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ACTIVE_ZONE}" --command="
    sudo apt-get update -y && \
    sudo apt-get install -y python3-pip rsync && \
    pip3 install --upgrade pip && \
    pip3 install faiss-cpu numpy pandas pyarrow
" --quiet

# 4. Transfer Code and Embeddings
echo "[$(date '+%H:%M:%S')] Syncing project code and 10GB embeddings to VM..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ACTIVE_ZONE}" --command="mkdir -p ~/visual-search-engine-gldv2/outputs" --quiet

gcloud compute scp --recurse --zone="${ACTIVE_ZONE}" \
    src scripts requirements.txt \
    "${INSTANCE_NAME}":~/visual-search-engine-gldv2/ --quiet

echo "[$(date '+%H:%M:%S')] Uploading embedding shards to VM..."
gcloud compute scp --recurse --zone="${ACTIVE_ZONE}" \
    outputs/embeddings \
    "${INSTANCE_NAME}":~/visual-search-engine-gldv2/outputs/ --quiet

# 5. Execute Build on VM
echo "[$(date '+%H:%M:%S')] Launching high-connectivity graph construction on VM..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ACTIVE_ZONE}" --command="
    cd ~/visual-search-engine-gldv2 && \
    chmod +x scripts/vm_build_hnsw.sh && \
    M=64 EF_CONSTRUCTION=256 EF_SEARCH=128 NUM_THREADS=22 ./scripts/vm_build_hnsw.sh
"

# 6. Download the Completed Index Back to Mac
echo "[$(date '+%H:%M:%S')] Downloading generated index back to local Mac..."
mkdir -p "${PROJECT_DIR}/outputs/index"

gcloud compute scp --zone="${ACTIVE_ZONE}" \
    "${INSTANCE_NAME}":~/visual-search-engine-gldv2/outputs/index/hnswpq_M64_efC256.index \
    "${PROJECT_DIR}/outputs/index/hnswpq_populated.index" --quiet

gcloud compute scp --zone="${ACTIVE_ZONE}" \
    "${INSTANCE_NAME}":~/visual-search-engine-gldv2/outputs/index/hnswpq_M64_efC256.meta.json \
    "${PROJECT_DIR}/outputs/index/hnswpq_populated.meta.json" --quiet

echo "=============================================================================="
echo " Index successfully synced to: outputs/index/hnswpq_populated.index"
ls -lh "${PROJECT_DIR}/outputs/index/hnswpq_populated.index"
echo "=============================================================================="
