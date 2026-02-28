# 步骤 1：部署 GPT-OSS

数据合成时需要用 **GPT-OSS**（或兼容的 Deep Search 模型）作为轨迹生成后端。本仓库通过 **vLLM** 提供 OpenAI 兼容 API。

## 使用本仓库脚本（推荐）

在项目根目录执行：

```bash
# 单机单卡：1 个 server，端口 8001，使用 GPU 0
bash scripts/start_nemotron_servers.sh 1 8001 0

# 单机多卡：2 个 server（8001、8002），每个 TP=2，使用 GPU 0,1,2,3
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3
```

默认加载的模型在 `scripts/start_nemotron_servers.sh` 中配置（当前为 OpenResearcher-30B-A3B）。  
若要做**数据合成**，建议改为 GPT-OSS 官方模型（如 `openai/gpt-oss-20b`）或你自建的 120B 等：修改脚本中的 `MODEL` 或通过环境变量传入。

## 验证服务

```bash
curl http://localhost:8001/v1/models
```

返回模型列表即表示服务就绪。后续步骤 3 将使用 `--vllm_server_url http://localhost:8001/v1` 调用该服务。

## 使用已有 API 或其它部署方式

若你已在别处部署了 GPT-OSS（或 OpenAI 兼容接口），无需在本机起 vLLM。在步骤 3 中设置环境变量即可：

```bash
export VLLM_SERVER_URL="https://your-api/v1"
bash data_synthesis/03_run_deepsearch.sh
```
