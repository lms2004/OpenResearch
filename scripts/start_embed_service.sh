#!/bin/bash
# 部署 vLLM Qwen3-Embedding-8B 模型服务（仅启动服务，不生成向量）。
# 与 scripts/start_search_service.sh dense 使用同一模型（Qwen/Qwen3-Embedding-8B）。
#
# 用法: bash scripts/start_embed_service.sh [port] [CUDA_VISIBLE_DEVICES]
# 示例: bash scripts/start_embed_service.sh 8010 0           # 单卡，默认充分用显存
#       bash scripts/start_embed_service.sh 8010 0,1         # 双卡 tensor parallel
#       GPU_MEMORY_UTILIZATION=0.95 bash scripts/start_embed_service.sh
#
# 部署后，在另一终端运行生成向量: bash scripts/run_embed_bench.sh
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PORT="${1:-8010}"
export CUDA_VISIBLE_DEVICES="${2:-0}"

# 多卡：按 CUDA_VISIBLE_DEVICES 数量做 tensor parallel；单卡用 --gpu-memory-utilization 吃满显存
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

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
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo ""
echo "生成向量请另开终端运行: bash scripts/run_embed_bench.sh"
echo -e "${GREEN}================================${NC}"
echo ""

# vLLM 用 --runner pooling 启动 embedding 模型；多卡用 --tensor-parallel-size，单卡用 --gpu-memory-utilization 吃满
VLLM_CMD=(vllm serve "${DENSE_MODEL_NAME}"
    --runner pooling
    --host 0.0.0.0
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
)
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    VLLM_CMD+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi
exec "${VLLM_CMD[@]}"
