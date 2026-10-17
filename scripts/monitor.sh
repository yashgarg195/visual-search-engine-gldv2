#!/usr/bin/env bash
# ==============================================================================
# Universal GLDv2 Dashboard & Live Streamer
# Automatically finds the active VM regardless of zone/name, reports progress,
# and streams the live logs seamlessly.
# ==============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

clear
echo "======================================================================"
echo "          GLDv2 OVERNIGHT EXTRACTION DASHBOARD (UNIVERSAL)            "
echo "======================================================================"
echo "Local Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Local Sync Status
LOCAL_SHARDS=$(ls -1 outputs/embeddings/*.npy 2>/dev/null | wc -l | tr -d ' ')
LOCAL_VECTORS=$(grep -o '"total_vectors": [0-9]*' outputs/embeddings/embedding_manifest.json 2>/dev/null | awk '{print $2}' || echo "0")
echo "[Local MacBook Storage]"
echo "  - Completed Shards on Mac : ${LOCAL_SHARDS}"
echo "  - Total Vectors Preserved : ${LOCAL_VECTORS} (~$((LOCAL_VECTORS * 100 / 4132914))% of 4.13M GLDv2)"
echo ""

# 2. Find Currently Active VM
echo "[Scanning Google Cloud for Active GPU Instance...]"
ACTIVE_VM_INFO=$(gcloud compute instances list --format="csv[no-heading](name,zone,status)" | grep "RUNNING" | head -n 1)

if [[ -z "${ACTIVE_VM_INFO}" ]]; then
    echo "  [STATUS] No RUNNING VM found. Checking stopped instances..."
    LAST_VM=$(gcloud compute instances list --format="csv[no-heading](name,zone,status)" | head -n 1)
    echo "  [INFO] Last known VM: ${LAST_VM}"
    echo "  [WATCHDOG] The autonomous watchdog daemon is active and handling restart/reprovisioning."
    echo ""
    echo "Displaying latest watchdog activity log:"
    echo "----------------------------------------------------------------------"
    tail -n 15 outputs/watchdog.log 2>/dev/null || echo "No watchdog log yet."
    echo "----------------------------------------------------------------------"
    exit 0
fi

VM_NAME=$(echo "${ACTIVE_VM_INFO}" | cut -d',' -f1)
VM_ZONE=$(echo "${ACTIVE_VM_INFO}" | cut -d',' -f2)

echo "  - Active VM Name : ${VM_NAME}"
echo "  - Active Zone    : ${VM_ZONE}"
echo "  - Status         : RUNNING"
echo ""

# 3. Remote GPU and Process Health
echo "[Remote GPU & Process Status]"
gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" --quiet --command="nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo "  GPU check in progress..."
echo ""

echo "======================================================================"
echo " STREAMING LIVE LOGS FROM ${VM_NAME} (Ctrl+C anytime to exit)"
echo "======================================================================"
# 4. Stream live logs continuously without needing a raw pty terminal
gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" --tunnel-through-iap --quiet --command="
SESSION=''
if tmux has-session -t gldv2_splits 2>/dev/null; then
    SESSION='gldv2_splits'
elif tmux has-session -t gldv2_fast 2>/dev/null; then
    SESSION='gldv2_fast'
elif tmux has-session -t gldv2 2>/dev/null; then
    SESSION='gldv2'
fi

if [[ -n \"\$SESSION\" ]]; then
    tmux capture-pane -pt \"\${SESSION}:0\" -S -25
    echo '--- Streaming live output (Ctrl+C to stop) ---'
    tmux pipe-pane -o -t \"\${SESSION}:0\" 'cat'
elif [ -f ~/visual-search-engine-gldv2/pipeline.log ]; then
    tail -n 30 -f ~/visual-search-engine-gldv2/pipeline.log
else
    echo 'Waiting for pipeline process to initialize...'
fi
"
