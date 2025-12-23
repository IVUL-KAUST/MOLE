#!/bin/bash
MODEL_NAME=$1
MAX_LEN=$2
VERSION="2.0"
SERVED_MODEL_NAME=$MODEL_NAME
if [ $MODEL_NAME == "MOLE" ]; then
    VERSION="1.0"
    SERVED_MODEL_NAME="Qwen2.5-3B-Instruct"
fi
# Function to check if vLLM server is running
check_vllm_server() {
    curl -s http://localhost:8787/v1/models > /dev/null 2>&1
    return $?
}

# Function to start vLLM server
start_vllm_server() {
    echo "🚀 Starting vLLM server..."
    
    # Start vLLM server in background
    uv run vllm serve $SERVED_MODEL_NAME \
        --host localhost \
        --port 8787 \
        --max-model-len $MAX_LEN \
        --max-num-seqs 50 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --disable-log-requests \
        > vllm_server.log 2>&1 &
    
    # Store the PID
    VLLM_PID=$!
    echo $VLLM_PID > vllm_server.pid
    echo "📝 vLLM server started with PID: $VLLM_PID"
}

# Function to wait for vLLM server to be ready
wait_for_server() {
    echo "⏳ Waiting for vLLM server to be ready..."
    local max_attempts=150  # Wait up to 5 minutes (60 * 5 seconds)
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if check_vllm_server; then
            echo "✅ vLLM server is ready!"
            return 0
        fi
        
        echo "⏳ Server not ready yet, waiting... (attempt $((attempt + 1))/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ Timeout: vLLM server failed to start within expected time"
    return 1
}

# Function to cleanup on exit
cleanup() {
    if [ -f vllm_server.pid ]; then
        local pid=$(cat vllm_server.pid)
        echo "🧹 Cleaning up vLLM server (PID: $pid)..."
        kill $pid 2>/dev/null || true
        rm -f vllm_server.pid
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Main execution
echo "🔍 Checking if vLLM server is already running..."

if check_vllm_server; then
    echo "✅ vLLM server is already running"
else
    start_vllm_server
    if ! wait_for_server; then
        echo "❌ Failed to start vLLM server. Exiting."
        exit 1
    fi
fi

echo "🚀 Starting evaluation..."
for schema_name in ar en ru jp fr multi model tool s2orc bib; do
    echo "🚀 Starting evaluation for $schema_name..."
    uv run src/evaluate.py \
        --split test \
        --backend vllm \
        --model $MODEL_NAME \
        --schema_name $schema_name \
        --version $VERSION \
        --max_model_len $MAX_LEN \
        --max_output_len 2048 
done
echo "✅ Evaluation completed!"
