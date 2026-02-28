#!/bin/bash
# 步骤 3：使用 GPT-OSS 对 data_synthesis 问题集进行 Deep Search，生成轨迹。
# 前置：步骤 1 已部署 vLLM（GPT-OSS），步骤 2 已生成 problems.jsonl。
# 可选：本地检索时先启动 scripts/start_search_service.sh dense 8000

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PROBLEMS_JSONL="${PROBLEMS_JSONL:-$SCRIPT_DIR/artifacts/problems.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/trajectories}"
SEARCH_URL="${SEARCH_URL:-http://localhost:8000}"
VLLM_SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8001/v1}"
BROWSER_BACKEND="${BROWSER_BACKEND:-local}"
MODEL_NAME="${MODEL_NAME:-OpenResearcher/OpenResearcher-30B-A3B}"

if [ ! -f "$PROBLEMS_JSONL" ]; then
  echo "Error: Problems file not found: $PROBLEMS_JSONL"
  echo "Run step 2 first: python data_synthesis/02_extract_problems.py --dataset_dir ... --output $PROBLEMS_JSONL"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
source .venv/bin/activate 2>/dev/null || true

echo "=========================================="
echo "Data Synthesis — Deep Search (Step 3)"
echo "=========================================="
echo "Problems: $PROBLEMS_JSONL"
echo "Output:   $OUTPUT_DIR"
echo "vLLM:     $VLLM_SERVER_URL"
echo "Search:   $SEARCH_URL ($BROWSER_BACKEND)"
echo "Model:    $MODEL_NAME"
echo "=========================================="

python deploy_agent.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_NAME" \
  --search_url "$SEARCH_URL" \
  --dataset_name synthesis \
  --data_path "$PROBLEMS_JSONL" \
  --browser_backend "$BROWSER_BACKEND" \
  --reasoning_effort high \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --max_concurrency_per_worker 32

echo "Done. Trajectories written to $OUTPUT_DIR"
