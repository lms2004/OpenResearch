#!/bin/bash
# 启动多个 vLLM 服务：按 TP_SIZE 将 GPU 分组，每组启动一个服务（一个端口一个进程）
# 并在启动前自动配置 HF 镜像 + 自动下载模型到本地（目录已存在则跳过/断点续传）
#
# 用法: ./scripts/start_nemotron_servers.sh [tensor_parallel_size] [base_port] [cuda_visible_devices] [model_repo_or_local_dir] [gpu_memory_utilization] [max_model_len]
# 示例: ./scripts/start_nemotron_servers.sh 2 8001 "0,1,2,3,4,5" "OpenResearcher/OpenResearcher-30B-A3B" 0.9
# 带 max seq len: ./scripts/start_nemotron_servers.sh 2 8001 "0,1,2,3" "OpenResearcher/OpenResearcher-30B-A3B" 0.9 16384
# 单卡示例: ./scripts/start_nemotron_servers.sh 1 8001 "0" "OpenResearcher/OpenResearcher-30B-A3B" 0.75
#
# 默认本地模型目录（相对项目根目录）：OpenResearcher/OpenResearcher-30B-A3B

set -e

TP_SIZE=${1:-2}
BASE_PORT=${2:-8001}
CUDA_VISIBLE_DEVICES=${3:-0,1,2,3}
MODEL_INPUT=${4:-"OpenResearcher/OpenResearcher-30B-A3B"}
GPU_MEMORY_UTILIZATION=${5:-0.9}
MAX_MODEL_LEN=${6:-}

# 获取脚本目录与项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# -----------------------------
# 1) 自动配置 HuggingFace 镜像
# -----------------------------
export HF_ENDPOINT="https://hf-mirror.com"

# -----------------------------
# 2) 自动下载模型到本地（目录存在则跳过/断点续传）
#    先查本地：绝对路径或 PROJECT_ROOT/<model> 存在则用本地；否则视为 HF repo，从默认 HF 位置下载到同名本地目录
#    - 绝对路径（以 / 开头）且目录存在 → 本地模型，不下载
#    - 相对路径（如 openai/gpt-oss-20b）且 PROJECT_ROOT/openai/gpt-oss-20b 存在 → 本地模型，不下载
#    - 否则视为 HuggingFace repo（如 openai/gpt-oss-20b），下载到 PROJECT_ROOT/openai/gpt-oss-20b
# -----------------------------
MODEL_REPO=""
LOCAL_DIR="$MODEL_INPUT"

if [[ "$MODEL_INPUT" = /* ]]; then
    # 绝对路径：必须是已存在的本地目录
    if [[ ! -d "$MODEL_INPUT" ]]; then
        echo "Error: Local model path does not exist: $MODEL_INPUT"
        echo "Use HF repo to download first, e.g.: $0 2 8001 0,1,2,3 openai/gpt-oss-20b 0.7 16000"
        exit 1
    fi
    LOCAL_ABS="$MODEL_INPUT"
    MODEL_REPO=""
elif [[ -d "$PROJECT_ROOT/$MODEL_INPUT" ]]; then
    # 相对路径且存在：本地目录
    LOCAL_ABS="$PROJECT_ROOT/$MODEL_INPUT"
    MODEL_REPO=""
elif [[ -d "$MODEL_INPUT" ]]; then
    # 当前目录下的相对路径且存在
    LOCAL_ABS="$(cd "$MODEL_INPUT" && pwd)"
    MODEL_REPO=""
else
    # 视为 HuggingFace repo（如 OpenResearcher/OpenResearcher-30B-A3B）
    LOCAL_DIR="$MODEL_INPUT"
    if [[ "$LOCAL_DIR" = /* ]]; then
        LOCAL_ABS="$LOCAL_DIR"
    else
        LOCAL_ABS="$PROJECT_ROOT/$LOCAL_DIR"
    fi
    MODEL_REPO="$MODEL_INPUT"
fi

# 若是 repo 模式，则执行 huggingface-cli download（支持断点续传；目录存在会复用已下载文件）
if [ -n "$MODEL_REPO" ]; then
    echo "=========================================="
    echo "本地模型不存在，从 HF 默认位置下载"
    echo "=========================================="
    echo "HF repo:    $MODEL_REPO"
    echo "HF_ENDPOINT: $HF_ENDPOINT"
    echo "下载到:     $LOCAL_ABS"
    echo "=========================================="
    mkdir -p "$(dirname "$LOCAL_ABS")"

    # 说明：
    # - --resume-download：断点续传/已存在文件会复用
    # - 如果目录已存在且文件齐全，会很快跳过
    huggingface-cli download --resume-download "$MODEL_REPO" --local-dir "$LOCAL_ABS"
    echo ""
else
    echo "=========================================="
    echo "使用本地模型（跳过下载）"
    echo "=========================================="
    echo "Model path: $LOCAL_ABS"
    echo "=========================================="
    echo ""
fi

# 最终用于 vLLM 的模型路径（本地）
MODEL="$LOCAL_ABS"

# -----------------------------
# 3) 解析/探测 GPU 列表
# -----------------------------
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPU_ARRAY=($(seq 0 $((NUM_GPUS-1))))
fi

TOTAL_GPUS=${#GPU_ARRAY[@]}

# -----------------------------
# 4) 校验：总 GPU 数必须能被 TP_SIZE 整除
# -----------------------------
if [ $((TOTAL_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: Total GPUs ($TOTAL_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
    exit 1
fi

NUM_SERVERS=$((TOTAL_GPUS / TP_SIZE))

echo "=========================================="
echo "Starting Multiple vLLM Servers"
echo "=========================================="
echo "Model (local): $MODEL"
echo "Total GPUs: $TOTAL_GPUS (${GPU_ARRAY[@]})"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Number of Servers: $NUM_SERVERS"
echo "Base Port: $BASE_PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
[ -n "$MAX_MODEL_LEN" ] && echo "Max Model Length: $MAX_MODEL_LEN"
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p logs

# -----------------------------
# 5) 启动每个 vLLM 服务（后台运行）
# -----------------------------
PIDS=()
for i in $(seq 0 $((NUM_SERVERS-1))); do
    START_IDX=$((i * TP_SIZE))
    END_IDX=$((START_IDX + TP_SIZE - 1))

    SERVER_GPUS=""
    for j in $(seq $START_IDX $END_IDX); do
        if [ -n "$SERVER_GPUS" ]; then
            SERVER_GPUS="$SERVER_GPUS,${GPU_ARRAY[$j]}"
        else
            SERVER_GPUS="${GPU_ARRAY[$j]}"
        fi
    done

    PORT=$((BASE_PORT + i))
    LOG_FILE="logs/vllm_server_${PORT}.log"

    echo "Starting Server $((i+1))/$NUM_SERVERS:"
    echo "  - GPUs: $SERVER_GPUS"
    echo "  - Port: $PORT"
    echo "  - Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SERVER_GPUS python scripts/deploy_vllm_service.py \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor_parallel_size "$TP_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        $([ -n "$MAX_MODEL_LEN" ] && echo "--max_model_len $MAX_MODEL_LEN") \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo ""

    sleep 2
done

echo "=========================================="
echo "All servers started!"
echo "=========================================="
echo ""
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo ""
echo "To stop all servers:"
echo "  kill ${PIDS[@]}"
echo ""
echo "PIDs saved to logs/server_pids.txt"
echo "To stop all servers later:"
echo "  kill \$(cat logs/server_pids.txt)"
echo "=========================================="

# 保存 PID
echo "${PIDS[@]}" > logs/server_pids.txt

# 等待任意一个服务退出
wait -n

# 任意一个退出，就停掉全部
echo ""
echo "One server exited. Stopping all servers..."
kill ${PIDS[@]} 2>/dev/null || true
wait || true
echo "All servers stopped."
