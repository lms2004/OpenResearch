#!/bin/bash
# 步骤 4：LLM-as-Judge 评估轨迹答案是否正确。
# 前置：步骤 3 已在 artifacts/trajectories 生成 *.jsonl。
# 需配置 .env：OPENAI_API_KEY（及可选 OPENAI_BASE_URL、OPENAI_MODEL）

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/artifacts/trajectories}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Trajectories dir not found: $INPUT_DIR"
  echo "Run step 3 first: bash data_synthesis/03_run_deepsearch.sh"
  exit 1
fi

source .venv/bin/activate 2>/dev/null || true

echo "=========================================="
echo "Data Synthesis — LLM-as-Judge (Step 4)"
echo "=========================================="
echo "Input: $INPUT_DIR"
echo "=========================================="

python eval.py --input_dir "$INPUT_DIR"

echo "Done. See $INPUT_DIR/evaluated.jsonl and summary above."
