#!/bin/bash
# Quick start script for running agent with multiple vLLM servers
# Usage: ./run_agent.sh [output_dir] [base_port] [num_servers] [dataset_name] [browser_backend] [model_path] [max_turns]
#
# max_turns: 每问最大轮次，默认 50；不传则使用 deploy_agent.py 默认值
#
# To reduce verbose logs and see progress bar clearly, redirect logs to file:
#   bash run_agent.sh ... 2>&1 | tee output.log | grep -E "(Worker|%|it/s)" 
# Or simply: bash run_agent.sh ... > agent.log 2>&1 &
# Then monitor progress: tail -f agent.log | grep -E "Worker.*%"

OUTPUT_DIR=${1:-"results/browsecomp-plus/OpenResearcher_dense"}
BASE_PORT=${2:-8001}
NUM_SERVERS=${3:-2}
DATASET_NAME=${4:-"browsecomp-plus"}
BROWSER_BACKEND=${5:-"local"}
MODEL_INPUT=${6:-"OpenResearcher/OpenResearcher-30B-A3B"}
MAX_TURNS=${7:-50}


SEARCH_URL="http://localhost:8000"

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# Convert model repo ID to local path (EXACT same logic as start_nemotron_servers.sh)
# This ensures consistency: if start_nemotron_servers.sh downloads to local path,
# run_agent.sh will use the exact same local path for vLLM server API calls
# Logic matches start_nemotron_servers.sh lines 37-52
LOCAL_DIR=""
if [[ "$MODEL_INPUT" == *"/"* ]]; then
    # repo id，例如 OpenResearcher/OpenResearcher-30B-A3B
    LOCAL_DIR="$MODEL_INPUT"  # 相对项目根目录
else
    # 本地目录（相对或绝对）
    LOCAL_DIR="$MODEL_INPUT"
fi

# 计算绝对路径，供 vLLM server API 使用（与 start_nemotron_servers.sh 完全一致）
if [[ "$LOCAL_DIR" = /* ]]; then
    MODEL="$LOCAL_DIR"
else
    MODEL="$PROJECT_ROOT/$LOCAL_DIR"
fi

# Verify model path exists (if it's a repo ID that should have been downloaded)
if [[ "$MODEL_INPUT" == *"/"* ]] && [[ "$MODEL_INPUT" != /* ]]; then
    if [ ! -d "$MODEL" ]; then
        echo "Warning: Model path not found: $MODEL"
        echo "This model should have been downloaded by start_nemotron_servers.sh"
        echo "Please ensure vLLM server is started with the same model path."
        echo ""
        echo "You can either:"
        echo "  1. Start vLLM server first: bash scripts/start_nemotron_servers.sh ..."
        echo "  2. Or the model will be auto-downloaded when server starts"
    fi
fi

# Build comma-separated server URLs
SERVER_URLS=""
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    URL="http://localhost:${PORT}/v1"

    if [ -n "$SERVER_URLS" ]; then
        SERVER_URLS="${SERVER_URLS},${URL}"
    else
        SERVER_URLS="${URL}"
    fi
done

echo "=========================================="
echo "Starting Agent with Multiple vLLM Servers"
echo "=========================================="
echo "Model Input: $MODEL_INPUT"
echo "Model Path: $MODEL"
echo "Number of Servers: $NUM_SERVERS"
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo "Search Service: $SEARCH_URL"
echo "Dataset: $DATASET_NAME"
echo "Browser Backend: $BROWSER_BACKEND"
echo "Max turns per question: $MAX_TURNS"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if using browsecomp-plus dataset (needs local data path)
if [ "$DATASET_NAME" = "browsecomp_plus" ]; then
    # Try to find parquet files - they might be in data/ subdirectory or root directory
    if [ -d "${SCRIPT_DIR}/Tevatron/browsecomp-plus/data" ] && [ -n "$(ls -A ${SCRIPT_DIR}/Tevatron/browsecomp-plus/data/*.parquet 2>/dev/null)" ]; then
        DATA_PATH="${SCRIPT_DIR}/Tevatron/browsecomp-plus/data/*.parquet"
    elif [ -n "$(ls -A ${SCRIPT_DIR}/Tevatron/browsecomp-plus/*.parquet 2>/dev/null)" ]; then
        DATA_PATH="${SCRIPT_DIR}/Tevatron/browsecomp-plus/*.parquet"
    else
        echo "Error: No parquet files found in Tevatron/browsecomp-plus/"
        echo "Please check if the dataset was downloaded correctly."
        echo "Expected location: ${SCRIPT_DIR}/Tevatron/browsecomp-plus/"
        exit 1
    fi
    echo "Using local BrowseComp-Plus dataset: $DATA_PATH"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --data_path "$DATA_PATH" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_turns "$MAX_TURNS" \
        --max_concurrency_per_worker 32
else
    # HuggingFace datasets or OpenAI BrowseComp (no local data_path needed)
    echo "Using dataset: $DATASET_NAME"
    echo "Available datasets: browsecomp, gaia, xbench"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_turns "$MAX_TURNS" \
        --max_concurrency_per_worker 32
fi
