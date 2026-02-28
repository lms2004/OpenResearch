# 数据合成流程 (Data Synthesis Pipeline)

本目录用于从 **OpenResearcher-Dataset** 抽取问题与标准答案，使用 **GPT-OSS** 进行 Deep Search 生成轨迹，并用 **LLM-as-Judge** 评估轨迹答案是否正确。

## 流程概览

| 步骤 | 说明 | 产出 |
|------|------|------|
| **1** | 部署 GPT-OSS | vLLM 服务（OpenAI 兼容 API） |
| **2** | 解析 OpenResearcher-Dataset | `problems.jsonl`（qid, question, answer） |
| **3** | GPT-OSS Deep Search | 轨迹结果目录（含 `*.jsonl`） |
| **4** | LLM-as-Judge 评估 | `evaluated.jsonl` + 准确率汇总 |

---

## 步骤 1：部署 GPT-OSS

使用 vLLM 部署 GPT-OSS 模型（如 `openai/gpt-oss-20b` 或你本地的 120B 等），提供 OpenAI 兼容接口，供后续 Deep Search 调用。

```bash
# 在项目根目录执行
# 单卡示例：1 个 server，端口 8001
bash scripts/start_nemotron_servers.sh 1 8001 0

# 多卡示例：2 个 server，端口 8001/8002，TP=2
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3
```

**注意**：若使用官方 GPT-OSS 模型，需在 `scripts/start_nemotron_servers.sh` 或环境变量中指定 `MODEL` 为对应模型路径或 HuggingFace 名称（如 `openai/gpt-oss-20b`）。当前脚本默认是 OpenResearcher-30B-A3B，做数据合成时请改为 GPT-OSS 模型。

确认服务就绪：

```bash
curl http://localhost:8001/v1/models
```

---

## 步骤 2：解析问题与最终答案

从 **OpenResearcher-Dataset** 中解析出 **问题 (question)** 与 **最终答案 (answer)**，供 Deep Search 使用（问题作为输入）和 Judge 使用（答案作为标准答案）。

### 2.1 本地目录（Parquet）

若数据集在本地目录 `OpenResearcher-Dataset`，且含有 `seed_*/` 下的 parquet 文件：

```bash
# 激活环境后，在项目根目录执行
python data_synthesis/02_extract_problems.py \
  --dataset_dir /mnt/c/Users/lms/Desktop/OpenResearcher/OpenResearcher-Dataset \
  --output data_synthesis/artifacts/problems.jsonl \
  --all-seeds
```

- `--dataset_dir`：OpenResearcher-Dataset 根目录（其下应有 `seed_42/`, `seed_43/` 等，内含 `*.parquet`）。
- `--output`：输出的 JSONL 路径，每行 `{"qid": int, "question": str, "answer": str}`。
- `--all-seeds`：使用所有 `seed_*`；也可用 `--seeds 42 43` 指定部分 seed。

### 2.2 使用 HuggingFace 上的数据集

若使用 HuggingFace 上的 `OpenResearcher/OpenResearcher-Dataset`：

```bash
python data_synthesis/02_extract_problems.py \
  --from_hf OpenResearcher/OpenResearcher-Dataset \
  --split train \
  --output data_synthesis/artifacts/problems.jsonl \
  --max_samples 1000
```

- `--max_samples`：可选，限制抽取条数，便于先小规模跑通。

---

## 步骤 3：GPT-OSS Deep Search

使用步骤 1 的 GPT-OSS 服务，对步骤 2 生成的 `problems.jsonl` 中的每个 **question** 做 Deep Search，得到每条问题的轨迹（含多轮推理与工具调用）。

### 3.1 启动检索服务（BrowseComp-Plus 本地检索时）

若使用本地 BM25/Dense 检索（与 BrowseComp-Plus 一致）：

```bash
# 终端 A：检索服务
bash scripts/start_search_service.sh dense 8000
```

若使用 Serper 网页搜索，可跳过本步，并在下面命令中把 `browser_backend` 改为 `serper`。

### 3.2 运行 Agent（Deep Search）

```bash
# 在项目根目录执行
bash data_synthesis/03_run_deepsearch.sh
```

脚本会：

- 使用 `data_synthesis/artifacts/problems.jsonl` 作为问题来源；
- 调用本机 vLLM 服务（默认 `http://localhost:8001/v1`）作为 GPT-OSS 推理后端；
- 将轨迹写入 `data_synthesis/artifacts/trajectories/`（其下为 `node_*_shard_*.jsonl` 等）。

如需自定义端口、检索方式或输出目录，可编辑 `data_synthesis/03_run_deepsearch.sh` 中的变量。

---

## 步骤 4：LLM-as-Judge 评估

用 **LLM-as-Judge** 判断每条轨迹中的**最终答案**是否与步骤 2 中的 **answer** 一致。

```bash
python eval.py --input_dir data_synthesis/artifacts/trajectories
```

需要：

- `OPENAI_API_KEY`（或第三方兼容 API 的 key）；
- 可选：`OPENAI_BASE_URL`、`OPENAI_MODEL`（见项目根目录 `.env` 或 `.env.template`）。

评估结果会写入 `data_synthesis/artifacts/trajectories/evaluated.jsonl`，并在终端打印准确率等汇总表。

也可使用封装脚本（默认 `input_dir=data_synthesis/artifacts/trajectories`）：

```bash
bash data_synthesis/04_run_judge.sh
```

---

## 目录结构建议

```
data_synthesis/
├── README.md                 # 本说明
├── 02_extract_problems.py    # 步骤 2：从 OpenResearcher-Dataset 抽取 (qid, question, answer)
├── 03_run_deepsearch.sh     # 步骤 3：运行 GPT-OSS Deep Search
├── 04_run_judge.sh          # 步骤 4：运行 LLM-as-Judge
└── artifacts/
    ├── problems.jsonl       # 步骤 2 产出：问题 + 标准答案
    └── trajectories/       # 步骤 3 产出：轨迹 JSONL；步骤 4 会在此生成 evaluated.jsonl
```

首次运行前可创建 artifacts 目录：

```bash
mkdir -p data_synthesis/artifacts data_synthesis/artifacts/trajectories
```
