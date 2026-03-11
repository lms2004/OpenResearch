#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen-Embedding 向量生成速度基准脚本。

从语料读取文档，向已部署的 vLLM embedding 服务请求生成向量，统计耗时与吞吐（文档/秒、向量/秒等）。

两个脚本分工：
  1. 部署服务: bash scripts/start_embed_service.sh [port] [cuda_devices]
  2. 生成向量: bash scripts/run_embed_bench.sh [max_docs] [port]
  或直接: python scripts/bench_qwen_embedding.py --embed_url http://localhost:8010/v1 ...

Usage:
  # 使用默认语料与默认 embedding 服务地址
  python scripts/bench_qwen_embedding.py

  # 指定参数、保存向量
  python scripts/bench_qwen_embedding.py --max_docs 5000 --save_vectors results/embeddings.npy
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
import time
from typing import List, Tuple

try:
    import httpx
except ImportError:
    print("Please install: pip install httpx")
    sys.exit(1)

try:
    import duckdb
except ImportError:
    print("Please install: pip install duckdb")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 从项目根目录导入
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
os.chdir(_project_root)


def load_corpus_texts(parquet_path: str, max_docs: int | None) -> List[Tuple[str, str]]:
    """从 parquet 语料中加载 (docid, text) 列表，与 backend.Corpus 使用的 schema 一致。"""
    if not parquet_path.strip():
        raise ValueError("corpus_parquet_path is required")
    con = duckdb.connect(database=":memory:", read_only=False)
    # read_parquet 支持 glob
    query = f"SELECT docid, text FROM read_parquet('{parquet_path}')"
    if max_docs is not None:
        query += f" LIMIT {int(max_docs)}"
    rows = con.execute(query).fetchall()
    con.close()
    return [(str(docid), (text or "").strip()) for docid, text in rows]


def _parquet_id_column(con, path_escaped: str) -> str:
    """检测 parquet 的 id 列名：FineWeb 用 id，Tevatron 等用 docid。"""
    try:
        names = [
            row[0] for row in con.execute(
                f"SELECT column_name FROM (DESCRIBE read_parquet('{path_escaped}'))"
            ).fetchall()
        ]
        if "id" in names:
            return "id"
        if "docid" in names:
            return "docid"
    except Exception:
        pass
    return "docid"


def load_corpus_texts_batched(
    file_paths: List[str],
    max_docs: int | None,
    progress: bool = True,
) -> List[Tuple[str, str]]:
    """分批从多个 parquet 文件加载 (id, text)，带进度条。支持 FineWeb(id,text) 与 Tevatron(docid,text)。"""
    if not file_paths:
        return []
    con = duckdb.connect(database=":memory:", read_only=False)
    result: List[Tuple[str, str]] = []
    iterator = file_paths
    if progress and tqdm:
        iterator = tqdm(file_paths, desc="加载语料", unit="file")
    id_col = "docid"
    for path in iterator:
        path_escaped = path.replace("'", "''")
        if not result:
            id_col = _parquet_id_column(con, path_escaped)
        rows = con.execute(
            f"SELECT \"{id_col}\", text FROM read_parquet('{path_escaped}')"
        ).fetchall()
        for docid, text in rows:
            result.append((str(docid), (text or "").strip()))
        if max_docs is not None and len(result) >= max_docs:
            result = result[: max_docs]
            break
    con.close()
    return result


def truncate_text(text: str, max_length: int) -> str:
    """简单按字符截断；若需按 token 截断可接 tokenizer。"""
    if max_length <= 0 or len(text) <= max_length:
        return text
    return text[:max_length]


async def request_embeddings(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    texts: List[str],
    max_text_len: int,
) -> List[List[float]]:
    """单次请求：对一批文本请求 /v1/embeddings，返回 list of embedding vectors。"""
    truncated = [truncate_text(t, max_text_len) for t in texts]
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": truncated}
    resp = await client.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    # OpenAI 格式: data["data"] = [ {"object":"embedding","embedding":[...], "index":0}, ... ]
    out = []
    for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
        out.append(item["embedding"])
    return out


async def run_benchmark(
    base_url: str,
    model: str,
    docid_texts: List[Tuple[str, str]],
    batch_size: int,
    concurrency: int,
    max_text_len: int,
    save_path: str | None,
    progress: bool = True,
) -> dict:
    """并发批量请求 embedding，统计耗时与吞吐，可选保存向量。"""
    total_docs = len(docid_texts)
    all_embeddings: List[List[float]] = []
    all_docids: List[str] = []
    batch_starts = list(range(0, total_docs, batch_size))
    pbar = tqdm(total=total_docs, desc="生成向量", unit="doc", disable=not progress) if tqdm else None

    sem = asyncio.Semaphore(concurrency)

    async def do_batch(start: int) -> List[Tuple[int, List[List[float]]]]:
        async with sem:
            batch = docid_texts[start : start + batch_size]
            if not batch:
                return []
            texts = [t for _, t in batch]
            async with httpx.AsyncClient() as client:
                vecs = await request_embeddings(client, base_url, model, texts, max_text_len)
            if pbar:
                pbar.update(len(batch))
            return [(start, vecs)]

    start_time = time.perf_counter()
    tasks = [do_batch(i) for i in batch_starts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if pbar:
        pbar.close()

    # 按起始下标排序以保证与 docid_texts 顺序一致
    ordered: List[Tuple[int, List[List[float]]]] = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        for item in r:
            ordered.append(item)
    ordered.sort(key=lambda x: x[0])

    for idx, vecs in ordered:
        all_embeddings.extend(vecs)
        for j in range(idx, min(idx + len(vecs), total_docs)):
            all_docids.append(docid_texts[j][0])

    elapsed = time.perf_counter() - start_time
    num_vectors = len(all_embeddings)

    stats = {
        "total_docs": total_docs,
        "num_vectors": num_vectors,
        "elapsed_sec": round(elapsed, 4),
        "docs_per_sec": round(total_docs / elapsed, 4) if elapsed > 0 else 0,
        "vectors_per_sec": round(num_vectors / elapsed, 4) if elapsed > 0 else 0,
        "batch_size": batch_size,
        "concurrency": concurrency,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        import numpy as np
        arr = np.array(all_embeddings, dtype=np.float32)
        np.save(save_path, arr)
        stats["saved_vectors_path"] = save_path
        # 可选：同时保存 docid 列表便于对齐
        meta_path = save_path.rsplit(".", 1)[0] + "_docids.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_docids))
        stats["saved_docids_path"] = meta_path

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen-Embedding vector generation speed via vLLM /v1/embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus_parquet",
        type=str,
        default=os.getenv(
            "CORPUS_PARQUET_PATH",
            os.path.join(_project_root, "fineweb", "sample", "10BT", "*.parquet"),
        ),
        help="Parquet 语料路径（支持 glob）。默认 fineweb/sample/10BT（FineWeb id+text）",
    )
    parser.add_argument(
        "--embed_url",
        type=str,
        default=os.getenv("EMBED_URL", "http://localhost:8010/v1"),
        help="vLLM embedding 服务 base URL（默认 8010 避免与检索服务 8000 冲突）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DENSE_MODEL_NAME", "Qwen/Qwen3-Embedding-8B"),
        help="Embedding 模型名，需与 vLLM 启动时 --model 一致",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="最多参与 benchmark 的文档数（默认 0=全量）；传正整数如 2000 可限制条数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每批请求的文档数（默认 32）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="并发请求数（默认 4）",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=8192,
        help="单条文本最大字符数，与 DenseSearcher max_length 对齐（默认 8192）",
    )
    parser.add_argument(
        "--save_vectors",
        type=str,
        default=None,
        metavar="PATH",
        help="将生成的向量保存为 .npy 文件，并同时保存 docids 到同名 _docids.txt",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="禁用进度条",
    )
    args = parser.parse_args()

    if not args.corpus_parquet or not args.corpus_parquet.strip():
        parser.error("请提供 --corpus_parquet 或设置环境变量 CORPUS_PARQUET_PATH")

    # 解析相对路径为相对项目根
    corpus_path = args.corpus_parquet
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(_project_root, corpus_path)

    print("Loading corpus...")
    # 展开 glob，打印实际加载的 parquet 文件
    matched = sorted(glob.glob(corpus_path))
    if not matched:
        print(f"No parquet files matched: {corpus_path}")
        sys.exit(1)
    print("加载的语料文件:")
    for f in matched:
        print(f"  {f}")
    print("")

    max_docs_arg = None if args.max_docs == 0 else args.max_docs
    docid_texts = load_corpus_texts_batched(
        matched,
        max_docs_arg,
        progress=not args.no_progress,
    )
    if not docid_texts:
        print("No documents found. Check corpus path and max_docs.")
        sys.exit(1)
    print(f"Loaded {len(docid_texts)} documents. Running benchmark (batch_size={args.batch_size}, concurrency={args.concurrency})...")

    try:
        stats = asyncio.run(
            run_benchmark(
                base_url=args.embed_url,
                model=args.model,
                docid_texts=docid_texts,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                max_text_len=args.max_text_len,
                save_path=args.save_vectors,
                progress=not args.no_progress,
            )
        )
    except httpx.ConnectError as e:
        print(f"\n连接失败: {e}")
        print("提示: 请先部署 embedding 服务: bash scripts/start_embed_service.sh 8010")
        sys.exit(1)

    print("\n========== Qwen-Embedding 向量生成速度 ==========")
    print(f"  文档总数:     {stats['total_docs']}")
    print(f"  向量总数:     {stats['num_vectors']}")
    print(f"  总耗时(s):    {stats['elapsed_sec']}")
    print(f"  文档/秒:      {stats['docs_per_sec']}")
    print(f"  向量/秒:      {stats['vectors_per_sec']}")
    if stats.get("saved_vectors_path"):
        print(f"  向量已保存:   {stats['saved_vectors_path']}")
        print(f"  docids 保存: {stats.get('saved_docids_path', '')}")
    print("================================================\n")


if __name__ == "__main__":
    main()
