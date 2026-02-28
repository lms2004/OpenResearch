#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test if vLLM model service is compatible with the project's tool-call format.

The agent uses:
  - tokenizer.apply_chat_template(messages, tools=tools) to build a prompt
  - /v1/completions with that prompt (not /chat/completions)
  - Parsing of <tool_call>{"name": "...", "arguments": {...}}</tool_call> in the raw output

This script sends one test prompt and checks that the response contains
parseable <tool_call>...</tool_call> with valid JSON inside.

Usage:
  # Local model (tokenizer + API model id from local path)
  python scripts/test_vllm_tool_call.py --base_url http://localhost:8001/v1 \
    --model /workspace/OpenResearch/openai/gpt-oss-20b

  # Explicit API model id (if vLLM exposes a different id than --model)
  python scripts/test_vllm_tool_call.py --base_url http://localhost:8001/v1 \
    --model /path/to/model --api_model /workspace/OpenResearch/openai/gpt-oss-20b

  # HuggingFace model (requires tokenizer to be downloadable)
  python scripts/test_vllm_tool_call.py --base_url http://localhost:8001/v1 --model openai/gpt-oss-20b
"""
import argparse
import asyncio
import json
import re
import sys

try:
    import httpx
except ImportError:
    print("Please install: pip install httpx")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Please install: pip install transformers")
    sys.exit(1)

# Reuse project tool definitions (run from project root)
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from data_utils import DEVELOPER_CONTENT, TOOL_CONTENT


def _resolve_tokenizer_name(model_name: str) -> str:
    """Same logic as utils/openai_generator.OpenAIAsyncGenerator._init_tokenizer"""
    name = model_name or ""
    if name.startswith("openai/"):
        name = name.replace("openai/", "OpenGPT-X/")
    return name or model_name


def _get_tokenizer_load_path(model_name: str, tokenizer_path: str | None) -> str:
    """
    Resolve where to load the tokenizer from.
    Prefer local directory over HuggingFace to avoid 401 for gated/private repos.
    """
    if tokenizer_path and os.path.isdir(tokenizer_path):
        return tokenizer_path
    raw = (model_name or "").strip()
    if not raw:
        return _resolve_tokenizer_name(raw)
    # Absolute path
    if os.path.isabs(raw) and os.path.isdir(raw):
        return raw
    # Relative to project root (e.g. openai/gpt-oss-20b when run from project root)
    cand = os.path.join(_project_root, raw)
    if os.path.isdir(cand):
        return cand
    # Fall back to HF id (may require auth for openai/ or OpenGPT-X/)
    return _resolve_tokenizer_name(model_name)


def _build_prompt(model_name: str, question: str, tokenizer_path: str | None = None) -> str:
    """Build the same prompt as deploy_agent.run_one (tokenizer + tools)."""
    load_path = _get_tokenizer_load_path(model_name, tokenizer_path)
    print(f"Loading tokenizer from: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    tools = json.loads(TOOL_CONTENT)
    system_prompt = DEVELOPER_CONTENT + "\n\nToday's date: 2025-01-01"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _parse_tool_call_from_content(content: str) -> dict | None:
    """Extract first <tool_call>...</tool_call> and parse JSON inside. Same logic as deploy_agent."""
    if "<tool_call>" in content and "</tool_call>" in content:
        m = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if m:
            text = m.group(1).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    if "</tool_call>" in content:
        m = re.search(r"^(.*?)</tool_call>", content, re.DOTALL)
        if m:
            text = m.group(1).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    return None


async def test_tool_call(
    base_url: str,
    model: str,
    question: str,
    tokenizer_path: str | None = None,
    api_model: str | None = None,
) -> bool:
    base_url = base_url.rstrip("/")
    prompt = _build_prompt(model, question, tokenizer_path)

    # Model id for API: explicit --api_model > from GET /models > same as tokenizer source
    api_model_id = api_model
    if not api_model_id:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{base_url}/models")
                if r.is_success and r.json().get("data"):
                    api_model_id = r.json()["data"][0]["id"]
                    print(f"Using API model id from server: {api_model_id}")
        except Exception as e:
            print(f"Warning: could not GET /models: {e}")
    if not api_model_id:
        api_model_id = _get_tokenizer_load_path(model, tokenizer_path)
        print(f"Using API model id: {api_model_id}")

    payload = {
        "model": api_model_id,
        "prompt": prompt,
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    print("Sending /completions request (single turn)...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{base_url}/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"},
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.text}")
        return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

    choices = data.get("choices", [])
    if not choices:
        print("No choices in response")
        return False

    text = choices[0].get("text", "")
    print("\n--- Model output (first 1500 chars) ---")
    print(text[:1500] + ("..." if len(text) > 1500 else ""))
    print("--- End ---\n")

    parsed = _parse_tool_call_from_content(text)
    if parsed is None:
        print("Result: FAIL – no valid <tool_call>...</tool_call> with JSON found in output.")
        print("The agent expects the model to output e.g.:")
        print('  <tool_call>\n  {"name": "browser.search", "arguments": {"query": "..."}}\n  </tool_call>')
        return False

    name = parsed.get("name", "")
    args = parsed.get("arguments", parsed.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    expected_tools = {"browser.search", "browser.open", "browser.find"}
    if name not in expected_tools:
        print(f"Result: WARN – parsed tool name '{name}' is not one of {expected_tools} (may still work).")
    else:
        print("Result: OK – tool-call format is compatible.")

    print(f"Parsed tool_call: name={name!r}, arguments={args}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test vLLM service for tool-call compatibility")
    parser.add_argument("--base_url", default="http://localhost:8001/v1", help="vLLM OpenAI API base URL")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name or local path (for tokenizer and default API model id)")
    parser.add_argument("--tokenizer_path", default=None, help="Local path to load tokenizer (overrides --model for tokenizer only)")
    parser.add_argument("--api_model", default=None, help="Model id to send to vLLM API (if different from --model, e.g. full path)")
    parser.add_argument(
        "--question",
        default="Search for the current population of Tokyo and tell me the number.",
        help="User question that should trigger a tool call (e.g. browser.search)",
    )
    args = parser.parse_args()

    ok = asyncio.run(test_tool_call(
        args.base_url,
        args.model,
        args.question,
        tokenizer_path=args.tokenizer_path,
        api_model=args.api_model,
    ))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
