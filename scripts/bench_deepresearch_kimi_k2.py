#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单次 Deep Research 速度基准：使用 Kimi-K2 API 模型，测量轮次速度、端到端时延等指标。

Kimi-K2 使用 OpenAI 兼容接口，支持 chat/completions + tools。
本脚本运行一条问题的完整 deep research 多轮对话，并输出：
  - 端到端时延 (s)
  - 轮次数、平均单轮时延 (s)、轮次速度 (rounds/min)
  - 首轮/末轮时延、总 tool calls 等

Usage:
  # 使用 .env 中的 OPENAI_BASE_URL + OPENAI_API_KEY + OPENAI_MODEL（与 .env.template 一致）
  # 例如 OPENAI_BASE_URL=https://api.moonshot.cn/v1, OPENAI_MODEL=kimi-k2
  cp .env.template .env   # 编辑 .env 填入 API Key 与 base URL
  python scripts/bench_deepresearch_kimi_k2.py --question "东京当前人口是多少？"

  # 指定单条问题与最大轮次
  python scripts/bench_deepresearch_kimi_k2.py --question "Find the PBS station in Tucson." --max_rounds 20

  # 从数据集取第一条问题
  python scripts/bench_deepresearch_kimi_k2.py --dataset browsecomp-plus --data_path "data/*.parquet" --max_rounds 10

  # 工具服务：与 run_agent.sh 一致，默认 http://localhost:8000（先运行 scripts/start_search_service.sh dense 8000）
  # 可通过 .env 的 SEARCH_URL 或 --search_url 覆盖

  # 命令行覆盖 .env 中的 base_url / model
  python scripts/bench_deepresearch_kimi_k2.py --base_url https://api.moonshot.cn/v1 --model kimi-k2 --question "你好"

  # 用 5 条样本测 Kimi（problems.jsonl 来自 02_extract_problems.py）
  # 1) 下载数据集: hf download OpenResearcher/OpenResearcher-Dataset --repo-type dataset --local-dir ./OpenResearcher-Dataset
  # 2) 抽取问题: python data_synthesis/02_extract_problems.py --dataset_dir ./OpenResearcher-Dataset --output data_synthesis/artifacts/problems.jsonl --max_samples 5
  # 3) 跑 5 条:
  python scripts/bench_deepresearch_kimi_k2.py --problems_file data_synthesis/artifacts/problems.jsonl --max_samples 5 --max_rounds 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print("Please install: pip install httpx")
    sys.exit(1)

# 从项目根目录导入
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

# 从 .env 加载 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL（与 .env.template 一致）
try:
    import dotenv
    dotenv.load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

from data_utils import DEVELOPER_CONTENT, TOOL_CONTENT

# 可选：Browser 相关（若未安装 openai_harmony/browser 则仅做单轮 API 测速）
try:
    from deploy_agent import BrowserPool
    _BROWSER_AVAILABLE = True
except Exception as e:
    _BROWSER_AVAILABLE = False
    BrowserPool = None  # type: ignore


# ---------- 配置（与 .env.template 一致：OPENAI_BASE_URL + OPENAI_API_KEY + OPENAI_MODEL）----------
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "kimi-k2")  # 使用 Kimi-K2 时在 .env 中设为 kimi-k2 或官方模型名


def get_tools() -> List[dict]:
    return json.loads(TOOL_CONTENT)


def build_system_content() -> str:
    import datetime
    return DEVELOPER_CONTENT + "\n\nToday's date: " + datetime.datetime.now().strftime("%Y-%m-%d")


# ---------- Kimi-K2 Chat Completions 客户端（无 tokenizer 依赖）----------
class KimiK2ChatClient:
    """仅使用 chat/completions 的轻量客户端，便于测速与兼容 Kimi-K2 API。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat_completions(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        r = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self.client.aclose()


# ---------- 单轮计时与指标 ----------
class RoundMetrics:
    def __init__(self) -> None:
        self.round_latencies: List[float] = []  # 每轮 API 时延 (s)
        self.tool_call_count_per_round: List[int] = []
        self.e2e_start: Optional[float] = None
        self.e2e_end: Optional[float] = None

    def record_round(self, latency_s: float, num_tool_calls: int = 0) -> None:
        self.round_latencies.append(latency_s)
        self.tool_call_count_per_round.append(num_tool_calls)

    def start_e2e(self) -> None:
        self.e2e_start = time.perf_counter()

    def end_e2e(self) -> None:
        self.e2e_end = time.perf_counter()

    @property
    def e2e_latency_s(self) -> float:
        if self.e2e_start is None or self.e2e_end is None:
            return 0.0
        return self.e2e_end - self.e2e_start

    @property
    def num_rounds(self) -> int:
        return len(self.round_latencies)

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_call_count_per_round)

    def summary(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "端到端时延_s": round(self.e2e_latency_s, 3),
            "轮次数": self.num_rounds,
            "总_tool_calls": self.total_tool_calls,
        }
        if self.round_latencies:
            avg = sum(self.round_latencies) / len(self.round_latencies)
            s["平均单轮时延_s"] = round(avg, 3)
            s["首轮时延_s"] = round(self.round_latencies[0], 3)
            s["末轮时延_s"] = round(self.round_latencies[-1], 3)
            # 轮次速度：rounds per minute（按平均单轮时延算）
            s["轮次速度_rounds_per_min"] = round(60.0 / avg, 2) if avg else 0
        # 端到端轮次速度：总轮数 / (端到端分钟)
        if self.e2e_latency_s and self.e2e_latency_s > 0 and self.num_rounds:
            s["端到端轮次速度_rounds_per_min"] = round(60.0 * self.num_rounds / self.e2e_latency_s, 2)
        return s


# ---------- 从 OpenAI 格式 message 中解析 tool_calls ----------
def _parse_tool_arguments(args_str: str) -> dict:
    """解析 function.arguments：先尝试 JSON，失败时对常见畸形（如 id 未加引号）做容错。"""
    if not args_str or not isinstance(args_str, str):
        return {}
    s = args_str.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 容错：模型常返回 "id": /path/without/quotes，用正则提取
    out = {}
    # 匹配 "id": "value" 或 "id": value（value 为无引号字符串，到 , 或 } 为止）
    m = re.search(r'"id"\s*:\s*"([^"]*)"', s)
    if m:
        out["id"] = m.group(1).strip()
        return out
    m = re.search(r'"id"\s*:\s*([^,}\]]+)', s)
    if m:
        out["id"] = m.group(1).strip()
        return out
    # 匹配 "query": "value"
    m = re.search(r'"query"\s*:\s*"([^"]*)"', s)
    if m:
        out["query"] = m.group(1).strip()
    m = re.search(r'"query"\s*:\s*([^,}\]]+)', s)
    if m:
        out["query"] = m.group(1).strip()
    return out if out else {}


def extract_tool_calls_from_message(message: dict) -> List[dict]:
    """OpenAI 格式: message.tool_calls = [{ id, type, function: { name, arguments } }]"""
    raw = message.get("tool_calls") or []
    out = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _parse_tool_arguments(args)
        elif args is None:
            args = {}
        if not name:
            continue
        out.append({
            "id": tc.get("id", ""),
            "type": tc.get("type", "function"),
            "function": {"name": name, "arguments": args or {}},
        })
    return out


def is_final_answer(content: str) -> bool:
    if not content:
        return False
    c = content.lower()
    if "<answer>" in c and "</answer>" in c:
        return True
    if "exact answer:" in c and "confidence:" in c:
        return True
    if "final answer:" in c or ("answer:" in c and "tool" not in c):
        return True
    return False


# ---------- 主流程：单次 Deep Research 多轮 + 计时 ----------
async def run_one_deep_research(
    question: str,
    client: KimiK2ChatClient,
    browser_pool: Optional[Any],
    qid: Any = 0,
    max_rounds: int = 30,
    verbose: bool = True,
) -> tuple:
    """Returns (messages, metrics)."""
    tools = get_tools()
    system_content = build_system_content()
    messages: List[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
    metrics = RoundMetrics()
    metrics.start_e2e()

    if verbose:
        print("\n--- Deep Research 开始 ---")
        print(f"Question: {question[:400]}{'...' if len(question) > 400 else ''}\n")

    for round_num in range(1, max_rounds + 1):
        # 发给 API 的 messages：只保留 role/content/tool_calls/tool_call_id（火山引擎等要求 tool 消息带 name，arguments 为字符串）
        api_messages = []
        for m in messages:
            msg_out = {
                "role": m["role"],
                "content": m.get("content") or "",
            }
            if "tool_calls" in m:
                # 确保 function.arguments 为 JSON 字符串（火山引擎 / 豆包要求）
                normalized_tool_calls = []
                for tc in m["tool_calls"]:
                    fn = (tc.get("function") or {}).copy()
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        fn["arguments"] = json.dumps(args, ensure_ascii=False)
                    elif args is None:
                        fn["arguments"] = "{}"
                    normalized_tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": fn,
                    })
                msg_out["tool_calls"] = normalized_tool_calls
            if "tool_call_id" in m:
                msg_out["tool_call_id"] = m["tool_call_id"]
            if m.get("role") == "tool" and "name" in m:
                msg_out["name"] = m["name"]
            # 火山引擎：assistant 仅 tool_calls 时 content 可为 null
            if m.get("role") == "assistant" and msg_out.get("tool_calls") and not (msg_out.get("content") or "").strip():
                msg_out["content"] = None
            api_messages.append(msg_out)

        t0 = time.perf_counter()
        try:
            resp = await client.chat_completions(
                messages=api_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=8192,
            )
        except Exception as e:
            if verbose:
                print(f"[Round {round_num}] API error: {e}")
            metrics.record_round(time.perf_counter() - t0, 0)
            metrics.end_e2e()
            return messages, metrics
        latency = time.perf_counter() - t0

        choices = resp.get("choices") or []
        if not choices:
            metrics.record_round(latency, 0)
            metrics.end_e2e()
            return messages, metrics

        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        tool_calls = extract_tool_calls_from_message(msg)

        metrics.record_round(latency, len(tool_calls))

        if verbose:
            # 原始模型返回（便于排查 tool_calls/arguments 解析问题）
            raw_display = dict(msg)
            if isinstance(raw_display.get("content"), str) and len(raw_display["content"]) > 600:
                raw_display["content"] = raw_display["content"][:600] + "...[truncated]"
            raw_json = json.dumps(raw_display, ensure_ascii=False, indent=2)
            if len(raw_json) > 3500:
                raw_json = raw_json[:3500] + "\n...[truncated]"
            print(f"  [Raw API message]\n{raw_json}")
            print()

        if verbose:
            print(f"[Round {round_num}] 时延={latency:.2f}s, tool_calls={len(tool_calls)}")
            if content:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                print(f"  [Assistant] {preview}")
            for tc in (tool_calls or []):
                fn = (tc.get("function") or {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"raw": args[:200]}
                print(f"  [Tool Call] {name} args={json.dumps(args, ensure_ascii=False)[:300]}")
            print()

        # 追加 assistant 消息
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if is_final_answer(content):
                if verbose:
                    print("检测到最终答案，结束。")
                    if content:
                        print(f"  [Final Answer] {content[:800]}{'...' if len(content) > 800 else ''}\n")
            break

        # 执行 tool calls
        if not browser_pool:
            # 无 browser 时用占位结果继续，便于仅测 API 轮次时延
            for tc in tool_calls:
                fn_name = (tc.get("function") or {}).get("name") or "unknown"
                placeholder = "[Benchmark mode: browser not available, skipping tool execution]"
                if verbose:
                    print(f"  [Tool Result] {fn_name} -> {placeholder[:120]}{'...' if len(placeholder) > 120 else ''}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": fn_name,
                    "content": placeholder,
                })
            continue

        browser_pool.init_session(qid)
        try:
            for tc in tool_calls:
                tid = tc.get("id", "")
                fn = (tc.get("function") or {}).get("name") or ""
                args = (tc.get("function") or {}).get("arguments") or {}
                if isinstance(args, str):
                    args = _parse_tool_arguments(args)
                elif not isinstance(args, dict):
                    args = {}
                # browser.search / browser.open / browser.find
                actual_name = fn.split(".")[-1] if "." in fn else fn
                try:
                    result = await browser_pool.call_tool(qid, actual_name, args)
                except Exception as e:
                    result = f"Error: {e}"
                if verbose:
                    result_preview = (result or "ok")[:400] + ("..." if len(result or "") > 400 else "")
                    print(f"  [Tool Result] {fn or actual_name} -> {result_preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "name": fn or "unknown",
                    "content": result or "ok",
                })
        finally:
            browser_pool.cleanup(qid)

    metrics.end_e2e()
    if verbose:
        print("--- Deep Research 结束 ---\n")
    return messages, metrics


# ---------- 入口与报表 ----------
def load_problems_from_jsonl(path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load problems from JSONL (qid, question, answer). Same format as 02_extract_problems.py output."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("question") is None:
                    continue
                problems.append(obj)
                if max_samples is not None and len(problems) >= max_samples:
                    break
            except json.JSONDecodeError:
                continue
    return problems


def load_one_question_from_dataset(dataset_name: str, data_path: Optional[str]) -> Optional[str]:
    try:
        from data_utils import load_dataset
        data = load_dataset(dataset_name, data_path=data_path)
        if not data:
            return None
        return data[0].get("question")
    except Exception as e:
        print(f"Warning: load_dataset failed: {e}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kimi-K2 API 单次 Deep Research 速度基准（轮次速度、端到端时延等）"
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI 兼容 API base URL (default: .env OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="Model name (default: .env OPENAI_MODEL, e.g. kimi-k2)",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (default: .env OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question for deep research. If not set, use --dataset/--data_path to load one.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to take first question (e.g. browsecomp-plus)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path for dataset (e.g. data/*.parquet for browsecomp-plus)",
    )
    parser.add_argument(
        "--problems_file",
        type=str,
        default=None,
        help="JSONL from 02_extract_problems.py (each line: qid, question, answer). Run multiple samples.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="When using --problems_file, max number of problems to run (default: 5). Ignored for single-question mode.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=20,
        help="Max conversation rounds (default: 20)",
    )
    parser.add_argument(
        "--search_url",
        type=str,
        default=os.getenv("SEARCH_URL", "http://localhost:8000"),
        help="工具/搜索服务 URL，与 run_agent.sh 一致 (default: env SEARCH_URL 或 http://localhost:8000)",
    )
    parser.add_argument(
        "--no_browser",
        action="store_true",
        help="Do not run browser tools (only measure API round latency with placeholder tool results)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    # ----- 多样本模式：从 problems.jsonl 读题 -----
    if args.problems_file:
        if not os.path.isfile(args.problems_file):
            print(f"Error: --problems_file not found: {args.problems_file}")
            sys.exit(1)
        problems = load_problems_from_jsonl(args.problems_file, max_samples=args.max_samples)
        if not problems:
            print(f"No problems loaded from {args.problems_file}")
            sys.exit(1)
        print(f"Loaded {len(problems)} problem(s) from {args.problems_file}. Running Kimi bench...")
    else:
        problems = None

    # ----- 单题模式：确定 question -----
    question = args.question
    if not question and args.dataset:
        question = load_one_question_from_dataset(args.dataset, args.data_path)
    if problems is None and not question:
        question = "Search for the current population of Tokyo and report the number."
        print(f"No question provided, using default: {question[:80]}...")

    api_key = args.api_key
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Set in .env or use --api_key.")

    browser_pool = None
    if not args.no_browser and _BROWSER_AVAILABLE:
        browser_pool = BrowserPool(args.search_url, browser_backend="local")
        print("Browser pool enabled (real tool execution).")
    else:
        if args.no_browser:
            print("Browser disabled (--no_browser); tool calls will use placeholder.")
        else:
            print("Browser not available (install openai_harmony, browser); tool calls will use placeholder.")

    client = KimiK2ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=300.0,
    )

    async def _run() -> None:
        if problems is not None:
            # 多样本：逐条跑，汇总指标
            all_summaries: List[Dict[str, Any]] = []
            for i, prob in enumerate(problems):
                qid = prob.get("qid", i)
                q = prob.get("question", "")
                if not q:
                    continue
                print(f"\n[{i+1}/{len(problems)}] qid={qid} question={q[:60]}...")
                messages, metrics = await run_one_deep_research(
                    question=q,
                    client=client,
                    browser_pool=browser_pool,
                    qid=qid,
                    max_rounds=args.max_rounds,
                    verbose=not args.quiet,
                )
                summary = metrics.summary()
                summary["qid"] = qid
                all_summaries.append(summary)
                if not args.quiet:
                    print(f"  -> 端到端={summary.get('端到端时延_s')}s, 轮次={summary.get('轮次数')}")

            await client.aclose()

            # 汇总
            n = len(all_summaries)
            print("\n" + "=" * 60)
            print(f"Deep Research 速度指标 (Kimi-K2 API) — 共 {n} 条样本")
            print("=" * 60)
            for i, s in enumerate(all_summaries):
                print(f"  [{i+1}] qid={s.get('qid')} 端到端时延_s={s.get('端到端时延_s')} 轮次数={s.get('轮次数')} 总_tool_calls={s.get('总_tool_calls')}")
            if n > 0:
                avg_e2e = sum(s.get("端到端时延_s", 0) for s in all_summaries) / n
                avg_rounds = sum(s.get("轮次数", 0) for s in all_summaries) / n
                total_tools = sum(s.get("总_tool_calls", 0) for s in all_summaries)
                print("  ---")
                print(f"  平均端到端时延_s: {round(avg_e2e, 3)}")
                print(f"  平均轮次数: {round(avg_rounds, 2)}")
                print(f"  总_tool_calls: {total_tools}")
            print("=" * 60)
            out_path = os.getenv("BENCH_OUTPUT_JSON")
            if out_path:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"samples": all_summaries, "n": n}, f, ensure_ascii=False, indent=2)
                print(f"Metrics written to {out_path}")
            return

        # 单题
        messages, metrics = await run_one_deep_research(
            question=question,
            client=client,
            browser_pool=browser_pool,
            qid=0,
            max_rounds=args.max_rounds,
            verbose=not args.quiet,
        )
        await client.aclose()

        # 输出速度指标
        summary = metrics.summary()
        print("\n" + "=" * 60)
        print("Deep Research 速度指标 (Kimi-K2 API)")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        # 可选：写 JSON
        out_path = os.getenv("BENCH_OUTPUT_JSON")
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"metrics": summary, "num_messages": len(messages)}, f, ensure_ascii=False, indent=2)
            print(f"Metrics written to {out_path}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
