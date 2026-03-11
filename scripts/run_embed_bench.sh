#!/bin/bash
# 一键：启动 embedding 服务（若未运行）→ 从原始语料生成向量 + 进度条 + 输出生成速度
# Usage: ./scripts/run_embed_bench.sh [max_docs] [port] [cuda_devices]
# Example: ./scripts/run_embed_bench.sh 2000 8010 0

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MAX_DOCS="${1:-2000}"
PORT="${2:-8010}"
CUDA_DEVICES="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export EMBED_URL="http://localhost:${PORT}/v1"

# 检查语料是否存在
CORPUS_PATTERN="Tevatron/browsecomp-plus-corpus/data/*.parquet"
CORPUS_ABS_GLOB="${PROJECT_ROOT}/${CORPUS_PATTERN}"
if ! ls $CORPUS_ABS_GLOB 1>/dev/null 2>&1; then
    echo -e "${RED}Error: Corpus not found at ${CORPUS_PATTERN}${NC}"
    echo "Run ./setup.sh to download the corpus."
    exit 1
fi

# 检查 embedding 服务是否已在运行
wait_for_server() {
    local url="$1"
    local max_attempts="${2:-60}"
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  一键：原始文档 → 向量 + 速度统计 + 进度条${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  max_docs: $MAX_DOCS"
echo "  embed port: $PORT"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo ""

if wait_for_server "http://localhost:${PORT}/v1/models" 3 2>/dev/null; then
    echo -e "${YELLOW}Embedding 服务已在 localhost:${PORT} 运行，直接跑 benchmark。${NC}"
else
    echo -e "${YELLOW}正在后台启动 vLLM Embedding 服务 (port=${PORT})...${NC}"
    mkdir -p "$PROJECT_ROOT/logs"
    LOG_FILE="$PROJECT_ROOT/logs/embed_server_${PORT}.log"
    nohup bash "$SCRIPT_DIR/start_embed_service.sh" "$PORT" "$CUDA_DEVICES" > "$LOG_FILE" 2>&1 &
    EMBED_PID=$!
    echo "  Server PID: $EMBED_PID, log: $LOG_FILE"
    echo "  等待服务就绪..."
    if ! wait_for_server "http://localhost:${PORT}/v1/models" 120; then
        echo -e "${RED}Error: Embedding 服务启动超时。查看日志: $LOG_FILE${NC}"
        kill $EMBED_PID 2>/dev/null || true
        exit 1
    fi
    echo -e "${GREEN}  Embedding 服务已就绪。${NC}"
fi

echo ""
echo -e "${GREEN}开始从语料生成向量（带进度条）...${NC}"
echo ""

python scripts/bench_qwen_embedding.py \
    --max_docs "$MAX_DOCS" \
    --embed_url "$EMBED_URL" \
    --corpus_parquet "$CORPUS_PATTERN"

echo ""
echo -e "${GREEN}完成。${NC}"
