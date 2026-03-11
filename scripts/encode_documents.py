#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据具体文档生成向量：调用已运行的检索服务（dense）的 /encode_documents 接口。

前置：已启动 dense 检索服务，例如：
  bash scripts/start_search_service.sh dense 8000 0

支持两种输入方式：
  1. 直接传入文本：--texts "doc1" "doc2" 或 --input_file 每行一条文本
  2. 传入语料中的 docid：--docids id1 id2 或 --docids_file 每行一个 docid

Usage:
  # 对两条文本生成向量
  python scripts/encode_documents.py --search_url http://localhost:8000 --texts "First document content." "Second document content."

  # 从文件读取文本（每行一条），生成向量并保存
  python scripts/encode_documents.py --search_url http://localhost:8000 --input_file docs.txt --save_vectors out.npy

  # 指定语料中的 docid 列表（从语料解析正文后编码）
  python scripts/encode_documents.py --search_url http://localhost:8000 --docids doc_001 doc_002 --save_vectors out.npy

  # 从文件读取 docid 列表（每行一个）
  python scripts/encode_documents.py --search_url http://localhost:8000 --docids_file docids.txt --save_vectors out.npy
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import httpx
except ImportError:
    print("Please install: pip install httpx")
    sys.exit(1)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(description="Call search service /encode_documents to generate vectors for given documents.")
    parser.add_argument("--search_url", type=str, default=os.getenv("SEARCH_URL", "http://localhost:8000"), help="Search service base URL")
    parser.add_argument("--texts", type=str, nargs="*", help="Document texts to encode (space-separated)")
    parser.add_argument("--input_file", type=str, help="File with one document per line (texts)")
    parser.add_argument("--docids", type=str, nargs="*", help="Document IDs from corpus to encode")
    parser.add_argument("--docids_file", type=str, help="File with one docid per line")
    parser.add_argument("--save_vectors", type=str, metavar="PATH", help="Save embeddings to .npy file")
    args = parser.parse_args()

    texts = None
    docids = None

    if args.texts is not None and len(args.texts) > 0:
        texts = list(args.texts)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    if args.docids is not None and len(args.docids) > 0:
        docids = list(args.docids)
    if args.docids_file:
        with open(args.docids_file, "r", encoding="utf-8") as f:
            docids = [line.rstrip("\n") for line in f if line.strip()]

    if (texts is None or len(texts) == 0) and (docids is None or len(docids) == 0):
        parser.error("Provide at least one of: --texts, --input_file, --docids, --docids_file")

    if texts is not None and docids is not None:
        parser.error("Provide only one of: texts/input_file OR docids/docids_file")

    url = args.search_url.rstrip("/") + "/encode_documents"
    payload = {"texts": texts} if texts is not None else {"docids": docids}

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    embeddings = data.get("embeddings", [])
    print(f"Encoded {len(embeddings)} document(s), embedding dim={len(embeddings[0]) if embeddings else 0}")

    if args.save_vectors and embeddings:
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(args.save_vectors)) or ".", exist_ok=True)
        np.save(args.save_vectors, np.array(embeddings, dtype=np.float32))
        print(f"Saved to {args.save_vectors}")


if __name__ == "__main__":
    main()
