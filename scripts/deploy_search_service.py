import os
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

import os
import re
import glob
from typing import List, Optional, Dict, Any
from threading import Lock

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
import duckdb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import setup_jar

LUCENE_EXTRA_DIR = os.getenv("LUCENE_EXTRA_DIR")
MAX_SNIPPET_LEN = int(os.getenv("MAX_SNIPPET_LEN", "180"))
setup_jar(LUCENE_EXTRA_DIR)

from backend import Corpus, get_searcher, BaseSearcher
from jnius import autoclass

_ANALYZER = None
_UH = None
_QP = None
_INIT_LOCK = Lock()

_SMART_DBL_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
}

def _drop_unpaired_quotes(q: str) -> str:
    for k, v in _SMART_DBL_QUOTES.items():
        q = q.replace(k, '"')
    out, in_quote = [], False
    for ch in q:
        if ch == '"':
            in_quote = not in_quote
            out.append(ch)
        else:
            out.append(ch)
    if in_quote:
        # drop dangling quote(s)
        return "".join(out).replace('"', "")
    return "".join(out)

def highlight_snippet_en(query: str, content: str, max_passages: int = 1) -> str:
    """Use Lucene UnifiedHighlighter without searcher to highlight content."""
    global _ANALYZER, _UH, _QP

    if _ANALYZER is None:
        with _INIT_LOCK:
            if _ANALYZER is None:
                StandardAnalyzer = autoclass(
                    "org.apache.lucene.analysis.standard.StandardAnalyzer"
                )
                UnifiedHighlighter = autoclass(
                    "org.apache.lucene.search.uhighlight.UnifiedHighlighter"
                )
                DefaultPassageFormatter = autoclass(
                    "org.apache.lucene.search.uhighlight.DefaultPassageFormatter"
                )
                QueryParser = autoclass(
                    "org.apache.lucene.queryparser.classic.QueryParser"
                )

                _ANALYZER = StandardAnalyzer()
                fmt = DefaultPassageFormatter("", "", " … ", False)
                _UH = (
                    UnifiedHighlighter.builderWithoutSearcher(_ANALYZER)
                    .withFormatter(fmt)
                    .build()
                )
                _QP = QueryParser("content", _ANALYZER)

    safe_query = _drop_unpaired_quotes(query)
    q = _QP.parse(safe_query)
    snippets = _UH.highlightWithoutSearcher("content", q, content, max_passages)
    if snippets is None:
        return content
    return str(snippets).strip()


PARQUET_PATH = os.getenv("CORPUS_PARQUET_PATH")
assert PARQUET_PATH
corpus = Corpus(PARQUET_PATH)

searcher: BaseSearcher = get_searcher(corpus=corpus)
logger.info(f"Searcher '{type(searcher).__name__}' initialized successfully.")


class SearchRequest(BaseModel):
    query: str
    topn: int = 10

class SearchResponseItem(BaseModel):
    url: str
    title: str
    summary: str

class FetchRequest(BaseModel):
    url: str

class FetchResponse(BaseModel):
    title: str
    content: str


class EncodeDocumentsRequest(BaseModel):
    """Request to encode documents into vectors. Provide either texts or docids (from corpus)."""
    texts: Optional[List[str]] = Field(default=None, description="Raw document texts to encode")
    docids: Optional[List[str]] = Field(default=None, description="Document IDs from corpus; resolved to text via corpus")


class EncodeDocumentsResponse(BaseModel):
    embeddings: List[List[float]]


_FM_PATTERN = re.compile(
    r"---\s*"
    r"(?:title:\s*(?P<title>.+?)\s*)"
    r"(?:author:\s*(?P<author>.+?)\s*)?"
    r"(?:date:\s*(?P<date>.+?)\s*)?"
    r"---\s*"
    r"(?P<content>.*?)(?=---|$)",
    re.DOTALL
)

def _parse_content(docid: str, raw_content: str) -> Dict[str, Any]:
    matches = list(_FM_PATTERN.finditer(raw_content))
    if matches:
        m = matches[0]
        title = (m.group("title") or f"Doc {docid}").strip()
        body = (m.group("content") or "").strip()
        # simplified content
        content = body
    else:
        title = f"Doc {docid}"
        content = raw_content.strip()
    return {"title": title, "content": content}


app = FastAPI(
    title="Unified Search Service",
    version="0.3.0",
    description="Search over a configured index and return results.",
)

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Unified Search Service",
        "active_searcher": type(searcher).__name__,
        "endpoints": ["/search", "/get_content", "/encode_documents"],
    }

@app.post("/search", response_model=Dict[str, List[SearchResponseItem]], tags=["search"])
def search(request: SearchRequest):
    hits = searcher.search(request.query, topn=request.topn)
    
    results = []
    for h in hits:
        if not h.text:
            logger.warning(f"No text found for docid {h.docid}, skipping.")
            continue

        url = corpus.get_url_from_id(h.docid)
        if not url:
            logger.warning(f"No URL found for docid {h.docid}, skipping.")
            continue
        
        parsed = _parse_content(h.docid, h.text)
        try:
            summary = highlight_snippet_en(request.query, parsed["content"], max_passages=50)[:MAX_SNIPPET_LEN]
        except Exception:
            summary = parsed["content"][:MAX_SNIPPET_LEN]
            

        results.append(
            SearchResponseItem(
                url=url,
                title=parsed["title"],
                summary=summary,
            )
        )
    return {"results": results}

@app.post("/get_content", response_model=FetchResponse, tags=["content"])
def get_content(request: FetchRequest):
    docid = corpus.get_id_from_url(request.url)
    if not docid:
        raise HTTPException(status_code=404, detail=f"URL not found in corpus: {request.url}")

    text = corpus.get_text_from_id(docid)
    if not text:
        raise HTTPException(status_code=404, detail=f"Content not found for docid: {docid} (URL: {request.url})")

    parsed = _parse_content(docid, text)
    
    return FetchResponse(
        title=parsed["title"],
        content=parsed["content"],
    )


@app.post("/encode_documents", response_model=EncodeDocumentsResponse, tags=["search"])
def encode_documents(request: EncodeDocumentsRequest):
    """Encode given documents into vectors (dense searcher only). Provide either texts or docids."""
    if not hasattr(searcher, "encode_documents"):
        raise HTTPException(
            status_code=501,
            detail="encode_documents is only available when using dense searcher (SEARCHER_TYPE=dense)",
        )
    if request.texts is not None and request.docids is not None:
        raise HTTPException(status_code=400, detail="Provide either texts or docids, not both.")
    if request.texts is not None:
        texts = [t or "" for t in request.texts]
    elif request.docids is not None:
        texts = []
        for docid in request.docids:
            t = corpus.get_text_from_id(str(docid))
            if t is None:
                raise HTTPException(status_code=404, detail=f"Document id not found in corpus: {docid}")
            texts.append(t)
    else:
        raise HTTPException(status_code=400, detail="Provide either texts or docids.")

    if not texts:
        return EncodeDocumentsResponse(embeddings=[])

    arr = searcher.encode_documents(texts)
    embeddings = arr.tolist()
    return EncodeDocumentsResponse(embeddings=embeddings)