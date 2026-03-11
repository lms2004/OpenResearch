#!/bin/bash
# 启动 vLLM Qwen3-Embedding-8B 服务，供 bench_qwen_embedding.py 调用。
# 与 scripts/start_search_service.sh dense 使用同一模型（Qwen/Qwen3-Embedding-8B）。
# Usage: ./scripts/start_embed_service.sh [port] [CUDA_VISIBLE_DEVICES]
# Example: ./scripts/start_embed_service.sh 8010 0,1

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PORT="${1:-8010}"
export CUDA_VISIBLE_DEVICES="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

source .venv/bin/activate

# 与 start_search_service.sh dense 一致
export DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting vLLM Embedding Service${NC}"
echo -e "${GREEN}================================${NC}"
echo "Model: ${DENSE_MODEL_NAME}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Benchmark with: python scripts/bench_qwen_embedding.py --embed_url http://localhost:${PORT}/v1"
echo -e "${GREEN}================================${NC}"
echo ""

# vLLM 0.13+ 支持 --task embed 启动 embedding 模型（与 deploy_vllm_service.py 一样用 vllm serve）
exec vllm serve "${DENSE_MODEL_NAME}" \
    --task embed \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code
