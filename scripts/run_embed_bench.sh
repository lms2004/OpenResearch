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

# 先加载语料（在 Python 中），再在发起请求时检查服务；不在此处提前 curl 检查
echo -e "${GREEN}从语料生成向量 (max_docs=$([ "$MAX_DOCS" = "0" ] && echo "全量" || echo "$MAX_DOCS"), embed_url=$EMBED_URL)...${NC}"
echo ""

# 显式传入模型名：优先使用用户传入的 --model；否则用 DENSE_MODEL_NAME；再否则从服务 /v1/models 自动探测
MODEL_ARG=()
if printf '%s\n' "$@" | grep -Eq -- '^(--model$|--model=)'; then
    MODEL_ARG=()
else
    MODEL_NAME="${DENSE_MODEL_NAME:-}"
    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME="$(python - <<'PY'
import json
import os
import sys
import urllib.request

base = os.environ.get("EMBED_URL", "http://localhost:8010/v1").rstrip("/")
url = base + "/models"
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        data = json.loads(r.read().decode("utf-8"))
    models = data.get("data") or []
    if models and isinstance(models, list) and isinstance(models[0], dict):
        mid = models[0].get("id")
        if mid:
            print(mid)
            sys.exit(0)
except Exception:
    pass
print("Qwen/Qwen3-Embedding-8B")
PY
)"
    fi
    MODEL_ARG=(--model "$MODEL_NAME")
fi

# 显式传入维度：优先使用用户传入的 --dimensions；否则用 EMBED_DIMENSIONS
DIM_ARG=()
if printf '%s\n' "$@" | grep -Eq -- '^(--dimensions$|--dimensions=)'; then
    DIM_ARG=()
else
    if [ -n "${EMBED_DIMENSIONS:-}" ]; then
        DIM_ARG=(--dimensions "${EMBED_DIMENSIONS}")
    fi
fi

python scripts/bench_qwen_embedding.py \
    --max_docs "$MAX_DOCS" \
    --embed_url "$EMBED_URL" \
    --corpus_parquet "$CORPUS_PATTERN" \
    "${MODEL_ARG[@]}" \
    "${DIM_ARG[@]}" \
    "$@"

echo ""
echo -e "${GREEN}完成。${NC}"
