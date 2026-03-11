#!/bin/bash
# 请求生成 embedding：从原始语料读取文档，向已部署的 embedding 服务请求生成向量，带进度条并输出速度。
# 前提：已在本机部署 embedding 服务，例如: bash scripts/start_embed_service.sh 8010
#
# 用法: bash scripts/run_embed_bench.sh [max_docs] [port] [--concurrency N] [--batch_size N] [--save_vectors PATH] ...
# 示例: bash scripts/run_embed_bench.sh              # 默认全量
#       bash scripts/run_embed_bench.sh 2000        # 仅 2000 条
#       bash scripts/run_embed_bench.sh 0 8010 --concurrency 8 --batch_size 64
#       bash scripts/run_embed_bench.sh 0 8010 --save_vectors results/embeddings.npy
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

MAX_DOCS="${1:-0}"
PORT="${2:-8010}"
# 消费前两个位置参数，避免把 0 或 port 当作额外参数传给 Python
if [ $# -ge 2 ]; then shift 2; elif [ $# -ge 1 ]; then shift; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

source .venv/bin/activate

export EMBED_URL="http://localhost:${PORT}/v1"

CORPUS_PATTERN="fineweb/sample/10BT/*.parquet"
CORPUS_ABS_GLOB="${PROJECT_ROOT}/${CORPUS_PATTERN}"
if ! ls $CORPUS_ABS_GLOB 1>/dev/null 2>&1; then
    echo -e "${RED}Error: Corpus not found at ${CORPUS_PATTERN}${NC}"
    echo "请将 fineweb/sample/10BT 下载到项目根目录下，或设置 CORPUS_PARQUET_PATH。"
    exit 1
fi

# 可选：检查服务是否可达（避免跑很久再报错）
if ! curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
    echo -e "${RED}Error: Embedding 服务未就绪 (localhost:${PORT})${NC}"
    echo "请先部署: bash scripts/start_embed_service.sh ${PORT}"
    exit 1
fi

echo -e "${GREEN}从语料生成向量 (max_docs=$([ "$MAX_DOCS" = "0" ] && echo "全量" || echo "$MAX_DOCS"), embed_url=$EMBED_URL)...${NC}"
echo ""

python scripts/bench_qwen_embedding.py \
    --max_docs "$MAX_DOCS" \
    --embed_url "$EMBED_URL" \
    --corpus_parquet "$CORPUS_PATTERN" \
    "$@"

echo ""
echo -e "${GREEN}完成。${NC}"
