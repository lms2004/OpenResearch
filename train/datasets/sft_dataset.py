import json
from pathlib import Path

# 定义本数据集中会用到的工具 schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "browser.search",
            "description": "Search the web and return search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    }
                },
                "required": ["query"]
            },
            "return": {
                "type": "string",
                "description": "Search results as a stringified summary."
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser.open",
            "description": "Open a URL or a result by index/cursor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "URL or document identifier."
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Cursor/index of the result set."
                    }
                },
                "required": []
            },
            "return": {
                "type": "string",
                "description": "Opened page content or metadata."
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser.find",
            "description": "Find text pattern in the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cursor": {
                        "type": "integer",
                        "description": "Cursor/index of the opened page."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Substring or regex pattern to search for."
                    }
                },
                "required": ["pattern"]
            },
            "return": {
                "type": "string",
                "description": "Find results as text."
            }
        }
    }
]

TOOLS_JSON_SCHEMA_STR = json.dumps(tools, ensure_ascii=False)

input_path = Path(
    "/mnt/c/Users/lms/Desktop/Megatron-LM/examples/openresearcher/data/converted_gpt_oss_search_correct.example.jsonl"
)
output_path = input_path.with_suffix(".tools_sft.jsonl")


def normalize_messages(raw_messages):
    """
    把 pretty.json / 原始 messages 转成类似：
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "name": "...", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    id_to_tool_name = {}
    normalized = []

    for m in raw_messages:
        role = m.get("role")
        tool_calls = m.get("tool_calls")

        # 带 tool_calls 的 assistant：只保留 role + tool_calls，arguments 解析成 dict
        if role == "assistant" and tool_calls:
            new_tool_calls = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                # 原数据里 arguments 多半是 JSON str，这里解析成 dict
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except Exception:
                        parsed_args = args
                else:
                    parsed_args = args

                new_tc = {
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": parsed_args,
                    },
                }
                new_tool_calls.append(new_tc)

                tc_id = tc.get("id")
                if tc_id:
                    id_to_tool_name[tc_id] = fn.get("name")

            normalized.append(
                {
                    "role": "assistant",
                    "tool_calls": new_tool_calls,
                }
            )

        # tool 消息：根据 tool_call_id 找回 name 字段
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            name = id_to_tool_name.get(tool_call_id)
            new_msg = {
                "role": "tool",
                "content": m.get("content", ""),
            }
            if name:
                new_msg["name"] = name
            normalized.append(new_msg)

        # 其他（system / user / 普通 assistant）：只保留 role + content
        else:
            content = m.get("content") or ""
            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return normalized


def convert():
    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            raw_messages = sample["messages"]

            messages = normalize_messages(raw_messages)

            out_obj = {
                "messages": messages,
                # 工具按照 transformers / TRL 推荐存成 JSON str
                "tools": TOOLS_JSON_SCHEMA_STR,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Converted {n} samples -> {output_path}")


if __name__ == "__main__":
    convert()