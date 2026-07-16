#!/bin/bash

# --- Configuration ---
# Hostfile path
HOSTFILE="./hostfile"
# GSPO experiment script path
EXPERIMENT_SCRIPT="./scripts/src_ma.sh"

# --- Main Script ---

# Check if hostfile exists
if [ ! -f "$HOSTFILE" ]; then
    echo "Error: Hostfile '$HOSTFILE' not found!"
    exit 1
fi

# Extract all IP addresses from hostfile into an array
# awk extracts the first column (IP address) of each line, filtering empty lines and lines starting with #
mapfile -t HOST_IPS < <(awk 'NF > 0 && !/^[[:space:]]*#/ {print $1}' "$HOSTFILE")

# Get total number of hosts and Chief node IP (default is the first in the list)
HOST_NUM=${#HOST_IPS[@]}
CHIEF_IP=${HOST_IPS[0]}

echo "========================================================"
echo "GSPO experiment will be launched on ${HOST_NUM} hosts..."
echo "Chief IP set to: ${CHIEF_IP}"
echo "========================================================"
echo ""

# ========================================
# Clean up python3 processes on all nodes
# ========================================
echo "========================================================"
echo "🔄 Cleaning up python3 and VLLM related processes on all hosts..."
echo "========================================================"
echo ""

# Create temporary directory to store background task PIDs
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Execute cleanup command on all hosts in parallel
for i in "${!HOST_IPS[@]}"; do
    CURRENT_IP=${HOST_IPS[$i]}
    
    (
        echo "[node${i}] Cleaning up ${CURRENT_IP}..."
        
        # Execute pkill command, first try SIGTERM, then SIGKILL
        # Clean up python3 and VLLM related processes
        # Use bracket trick [v]llm and [V]LLM to avoid pkill matching itself
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${CURRENT_IP}" \
            "pkill -9 python3 2>/dev/null; pkill -9 -f '[v]llm' 2>/dev/null; pkill -9 -f '[V]LLM' 2>/dev/null; sleep 1; pkill -9 python3 2>/dev/null; pkill -9 -f '[v]llm' 2>/dev/null; pkill -9 -f '[V]LLM' 2>/dev/null || true" 2>&1
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[node${i}] ✓ Cleanup command executed successfully"
        else
            echo "[node${i}] ⚠️  SSH connection or execution failed (exit code: $EXIT_CODE)"
        fi
    ) &
    
    # Save background process PID
    echo $! > "$TMP_DIR/pid_$i"
done

echo "Waiting for all hosts to complete cleanup..."
echo ""

# Wait for all background tasks to complete
for pidfile in "$TMP_DIR"/pid_*; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        wait $pid 2>/dev/null
    fi
done

echo "✅ All hosts cleanup completed."
echo ""

echo "========================================================"
echo "⏳ Waiting 30 seconds to ensure all processes have fully terminated..."
echo "========================================================"
sleep 30
echo ""

# ========================================
# Verify process cleanup status
# ========================================
echo "========================================================"
echo "🔍 Verifying status of all hosts..."
echo "========================================================"
echo ""

ALL_CLEAN=true
for i in "${!HOST_IPS[@]}"; do
    CURRENT_IP=${HOST_IPS[$i]}
    # Check for python3 and VLLM related processes
    PROC_COUNT=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${CURRENT_IP}" \
        "ps aux | grep -E '[p]ython3|[v]llm|[V]LLM' | wc -l" 2>/dev/null || echo "0")
    
    if [ "$PROC_COUNT" -gt 0 ]; then
        echo "[node${i}] Checking ${CURRENT_IP}... ⚠️  Still has ${PROC_COUNT} python3 processes"
        ALL_CLEAN=false
    else
        echo "[node${i}] Checking ${CURRENT_IP}... ✓ Cleaned up"
    fi
done

if [ "$ALL_CLEAN" = false ]; then
    echo ""
    echo "⚠️  Warning: Some nodes still have residual processes, cleaning up again..."
    
    for i in "${!HOST_IPS[@]}"; do
        CURRENT_IP=${HOST_IPS[$i]}
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${CURRENT_IP}" \
            "pkill -9 python3 2>/dev/null; pkill -9 -f '[v]llm' 2>/dev/null; pkill -9 -f '[V]LLM' 2>/dev/null; sleep 2; pkill -9 python3 2>/dev/null; pkill -9 -f '[v]llm' 2>/dev/null; pkill -9 -f '[V]LLM' 2>/dev/null || true" 2>&1 &
    done
    wait
    
    echo "⏳ Waiting another 20 seconds..."
    sleep 20
fi

echo ""
echo "✅ All nodes verification completed"
echo ""

# ========================================
# Clean up port usage on Master node
# ========================================
echo "========================================================"
echo "🔧 Cleaning up port usage on Master node ${CHIEF_IP}..."
echo "========================================================"

ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${CHIEF_IP}" \
    "fuser -k 29500/tcp 2>/dev/null || true; sleep 2" 2>&1

echo "✓ Port cleanup completed"
echo ""

echo "========================================================"
echo "🚀 Starting GSPO tasks..."
echo "========================================================"
echo ""

# Initialize process ID array for waiting later
pids=()

# ========================================
# Phase 1: Start Master node first (INDEX=0)
# ========================================
echo "Phase 1: Starting Master node ${CHIEF_IP}..."

MASTER_CMD="HOST_NUM=${HOST_NUM} CHIEF_IP=${CHIEF_IP} INDEX=0 bash ${EXPERIMENT_SCRIPT}"
ssh -n "${CHIEF_IP}" "${MASTER_CMD}" &
pids+=($!)

echo "⏳ Waiting 30 seconds for Master node to initialize TCPStore..."
sleep 30

# Test if Master node port is ready
echo "🔍 Checking if Master node port 29500 is ready..."
RETRY_COUNT=0
MAX_RETRIES=10

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if timeout 5 bash -c "echo >/dev/tcp/${CHIEF_IP}/29500" 2>/dev/null; then
        echo "✓ Master node port 29500 is ready"
        break
    else
        echo "⏳ Port not ready, retrying in 5 seconds... ($((RETRY_COUNT+1))/${MAX_RETRIES})"
        sleep 5
        ((RETRY_COUNT++))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "⚠️  Warning: Master node port check timed out, continuing to start Worker nodes..."
fi

echo ""

# ========================================
# Phase 2: Start all Worker nodes (INDEX=1 to N-1)
# ========================================
echo "Phase 2: Starting Worker nodes..."

# Iterate over all host IPs except Master node
for i in "${!HOST_IPS[@]}"; do
    # Skip Master node (INDEX=0)
    if [ $i -eq 0 ]; then
        continue
    fi
    
    # Current host IP
    CURRENT_IP=${HOST_IPS[$i]}
    # Current host index (INDEX)
    INDEX=$i

    echo "Starting GSPO task on host ${CURRENT_IP} (INDEX=${INDEX})..."

    # Build command to execute on remote host
    CMD="HOST_NUM=${HOST_NUM} CHIEF_IP=${CHIEF_IP} INDEX=${INDEX} bash ${EXPERIMENT_SCRIPT}"

    # Use ssh to execute command in background on remote host
    # -n option prevents ssh from reading stdin
    # & puts command in background for parallel execution
    ssh -n "${CURRENT_IP}" "${CMD}" &
    
    # Save the PID of the last background process
    pids+=($!)
    
    # Pause briefly after every 5 nodes to avoid connection congestion
    if [ $((i % 5)) -eq 0 ]; then
        echo "⏳ Waiting 5 seconds to avoid connection congestion..."
        sleep 5
    fi
done

echo "----------------------------------------"
echo "All GSPO tasks have been started. Now waiting for all tasks to complete..."

# Wait for all background ssh processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All GSPO experiment tasks have been completed!"
