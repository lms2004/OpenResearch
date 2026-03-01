# SFT 数据集格式与 Think 训练说明

本文档说明**最终训练数据集格式**、**think（reasoning）如何参与训练**，以及**与 TRL SFTTrainer 的兼容性**。

---

## 一、最终数据集格式（tools_sft.jsonl / eval_sft.jsonl）

每行一条 JSON，包含两列：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list[dict]` | 多轮对话，每条消息见下表 |
| `tools` | `str` | 工具 JSON Schema 字符串（与 `data_utils.get_browser_tools_json_schema()` 一致） |

### messages 中每条消息的格式

- **system / user**：`{"role": "system"|"user", "content": "..."}`  
- **tool**：`{"role": "tool", "content": "...", "name": "..."}`（可选 `name`）  
- **assistant（纯文本或带 think）**：`{"role": "assistant", "content": "..."}`，若有 think 则增加 **`"reasoning_content": "思考内容"`**  
- **assistant（带 tool_calls）**：`{"role": "assistant", "tool_calls": [...]}`，若该条前有思考则增加 **`"reasoning_content": "思考内容"`**

### 带 think 的示例（一条 assistant 消息）

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Question: 某问题"},
    {
      "role": "assistant",
      "content": "根据检索结果，答案是 xxx。",
      "reasoning_content": "先理解问题，再查资料，最后归纳。"
    }
  ],
  "tools": "[{\"type\":\"function\",\"function\":{...}}]"
}
```

**说明**：`reasoning_content` 可选；若存在且非空，训练时会通过 chat template 渲染成 `<think>...</think>` 再拼在 assistant 的 `content` 前。

---

## 二、Think 如何参与训练（流程）

1. **数据来源**  
   - `parse.py` 会在 assistant 消息里写入 `reasoning_content`（或保留 content 中的 `<think>...</think>`）。  
   - `sft_dataset.py` 的 `normalize_messages()` 会把 `reasoning_content` 保留到写入 tools_sft.jsonl 的每条 assistant 消息中。

2. **Chat Template 渲染**  
   - `train.py` 使用 `datasets/templates/all_assistant.jinja` 作为 tokenizer 的 chat_template。  
   - 对每条 **assistant** 消息：
     - 若有 `message.reasoning_content`，模板会输出：`<think>\n` + reasoning_content + `\n</think>\n\n` + content；  
     - 若没有 `reasoning_content` 但 `content` 里含 `<think>...</think>`，模板会从 content 里拆出 think 再拼。  
   - 因此**最终送入模型的 assistant 文本** = `<think>思考内容</think>` + 换行 + 正文/工具调用等。

3. **Tokenization 与 Loss**  
   - TRL 的 SFTTrainer 对每条样本调用 `tokenizer.apply_chat_template(messages, tools=tools, ...)` 得到**一整段字符串**（含 `<think>...</think>`），再 tokenize。  
   - 模板里 assistant 部分用 `{% generation %} ... {% endgeneration %}` 标出，配合 **`assistant_only_loss=True`**，只有 **assistant 对应的 token** 参与 loss 计算。  
   - 因此 **think 段（`<think>...</think>`）和后面的正文、tool_calls 一样，都会参与训练**，模型会学习「先输出 think 再输出回答/调用」。

---

## 三、与 TRL 的兼容性

- **数据格式**：TRL SFTTrainer 支持「每条样本 = `messages` + 可选 `tools`」的格式；本仓库的 tools_sft.jsonl 正是该格式，且 `tools` 为 JSON 字符串，与 `apply_chat_template(..., tools=...)` 所需一致。  
- **Chat Template**：使用自定义 Jinja 模板（all_assistant.jinja），通过 `tokenizer.chat_template = ...` 注入，TRL 会使用该 template 做格式化。  
- **Loss 范围**：`assistant_only_loss=True` 与模板中的 `{% generation %}` 配合，只对 assistant 段（含 think）算 loss，system/user/tool 不参与。  
- **结论**：当前「带 reasoning_content 的 messages + tools + all_assistant.jinja + SFTTrainer」**完全适配 TRL**，无需改 TRL 源码；只需保证 sft_dataset 写出带 `reasoning_content` 的 messages（已实现）。

---

## 四、如何生成带 think 的训练数据

1. 确保 materialized 数据中 assistant 消息已有 `reasoning_content`（由 `parse.py` 或上游流水线写入）。  
2. 运行：

   ```bash
   cd train/datasets
   python sft_dataset.py --max-tokens 8192
   ```

   会重新生成 `tools_sft.jsonl`、`eval_sft.jsonl`（及 eval_sft_with_tokens.jsonl），其中 messages 已包含 `reasoning_content`。  
3. 按原有命令启动训练即可，例如：

   ```bash
   cd train
   python train.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct-1M --train_jsonl datasets/data/converted_gpt_oss_search_correct.tools_sft.jsonl --output_dir outputs/sft_full --max_seq_length 4096 --bf16 ...
   ```

无需改 `train.py` 或 TRL 配置；think 会通过「数据中的 reasoning_content → 模板中的 `<think>...</think>` → assistant_only_loss」自动参与训练。
