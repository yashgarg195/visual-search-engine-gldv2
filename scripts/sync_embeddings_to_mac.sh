#!/usr/bin/env bash
# ==============================================================================
# Robust Background Shard Sync from GCP VM to Local MacBook
# ==============================================================================

set -uo pipefail

VM_NAME="instance-20261016-195121"
ZONE="asia-southeast1-c"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

mkdir -p "${OUTPUT_DIR}"

while true; do
    ACTIVE_VM_INFO=$(gcloud compute instances list --format="csv[no-heading](name,zone,status)" 2>/dev/null | grep "RUNNING" | grep "instance-" | head -n 1 || true)
    if [ -n "${ACTIVE_VM_INFO}" ]; then
        VM_NAME=$(echo "${ACTIVE_VM_INFO}" | cut -d',' -f1)
        ZONE=$(echo "${ACTIVE_VM_INFO}" | cut -d',' -f2)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting sync pass from ${VM_NAME} (${ZONE})..."
        gcloud compute scp --recurse --zone="${ZONE}" "${VM_NAME}:visual-search-engine-gldv2/outputs/embeddings" "${OUTPUT_DIR}" 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync pass finished for ${VM_NAME}."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No active running VM found, waiting for watchdog..."
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for 3 minutes..."
    sleep 180
done
