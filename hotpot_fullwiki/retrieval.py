"""Budgeted, deterministic Pyserini BM25 retrieval for HotpotQA FullWiki."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import config
from .common import atomic_json, digest
from .loader import RuntimeExample


class Searcher(Protocol):
    def search(self, query: str, k: int): ...
    def doc(self, docid: str): ...


class RetrievalError(RuntimeError):
    pass


def _fallback_sentences(text: str) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [part.strip() for part in parts if part.strip()]


def parse_beir_document(raw: str) -> tuple[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RetrievalError("BEIR document is not JSON") from exc
    title = str(value.get("title", "")).strip()
    text = str(value.get("text", value.get("contents", ""))).strip()
    if not title or not text:
        raise RetrievalError("BEIR document lacks title or text")
    return title, text


def pyserini_searcher() -> Searcher:
    from inherit_mas import release_config
    release_config.ensure_java_home()
    # Pyserini imports an optional OpenAI encoder eagerly; no API call is made.
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    os.environ.setdefault("PYSERINI_CACHE", str(release_config.pyserini_cache()))
    try:
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher.from_prebuilt_index(config.BEIR_INDEX)
        searcher.set_bm25(config.BM25_K1, config.BM25_B)
        return searcher
    except Exception as exc:
        raise RetrievalError(
            f"Pyserini backend unavailable: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class RetrievalSession:
    retriever: "BM25Retriever"
    example: RuntimeExample
    call_budget: int = config.RETRIEVAL_CALLS_PER_CANDIDATE

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str) -> dict[str, Any]:
        if len(self.calls) >= self.call_budget:
            return {"ok": False, "error": "retrieval_budget_exhausted", "results": []}
        try:
            result = self.retriever.search(query, self.example)
        except (RetrievalError, ValueError):
            raise
        except Exception as exc:
            raise RetrievalError(
                f"BM25 query failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.calls.append(result)
        return result

    @property
    def retrieved_titles(self) -> set[str]:
        return {row["title"] for call in self.calls for row in call.get("results", [])}


class BM25Retriever:
    def __init__(self, *, cache_dir: str | Path, searcher: Searcher | None = None,
                 top_k: int = config.RETRIEVAL_TOP_K):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)
        self._searcher = searcher
        self._lock = threading.Lock()
        self.fingerprint = {
            "implementation": "pyserini-lucene-bm25/fullwiki-v1",
            "index": config.BEIR_INDEX,
            "archive": config.BEIR_INDEX_ARCHIVE,
            "archive_md5": config.BEIR_INDEX_MD5,
            "k1": config.BM25_K1,
            "b": config.BM25_B,
            "top_k": self.top_k,
            "sentence_mapping": "task-canonical-title-override/fallback-regex-v1",
        }
        self.fingerprint_digest = digest(self.fingerprint)

    @property
    def searcher(self) -> Searcher:
        if self._searcher is None:
            self._searcher = pyserini_searcher()
        return self._searcher

    def preflight(self, query: str, example: RuntimeExample) -> dict[str, Any]:
        """Run and validate one real retrieval before any model call is allowed."""
        try:
            result = self.search(query, example)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"BM25 preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        rows = result.get("results")
        if not isinstance(rows, list) or not rows:
            raise RetrievalError("BM25 preflight returned no indexed documents")
        if not any(row.get("title") and row.get("sentences") for row in rows):
            raise RetrievalError("BM25 preflight returned no indexed sentences")
        return {
            "schema": "hotpot_fullwiki_retriever_preflight/1",
            "query": result["query"],
            "retriever_digest": result["retriever_digest"],
            "result_count": len(rows),
            "first_title": rows[0]["title"],
            "cache_hit": bool(result.get("cache_hit", False)),
        }

    def _path(self, query: str, example_id: str) -> Path:
        key = digest({"query": query, "example_id": example_id,
                      "retriever": self.fingerprint_digest})
        return self.cache_dir / f"{key}.json"

    def search(self, query: str, example: RuntimeExample) -> dict[str, Any]:
        query = " ".join(str(query).split())
        if not query:
            raise ValueError("BM25 query cannot be empty")
        path = self._path(query, example.id)
        if path.exists():
            value = json.loads(path.read_text())
            value["cache_hit"] = True
            return value
        with self._lock:
            if path.exists():
                value = json.loads(path.read_text())
                value["cache_hit"] = True
                return value
            started = time.perf_counter()
            rows = []
            for hit in self.searcher.search(query, self.top_k):
                docid = str(hit.docid)
                stored = self.searcher.doc(docid)
                if stored is None:
                    raise RetrievalError(f"index hit {docid} has no stored document")
                title, text = parse_beir_document(stored.raw())
                sentences = example.sentences_for_title(title) or _fallback_sentences(text)
                rows.append({
                    "docid": docid,
                    "score": float(hit.score),
                    "title": title,
                    "sentences": [{"sentence_id": i, "text": sentence}
                                  for i, sentence in enumerate(sentences)],
                })
            value = {
                "schema": "hotpot_fullwiki_retrieval/1",
                "query": query,
                "retriever_digest": self.fingerprint_digest,
                "results": rows,
                "wall_s": time.perf_counter() - started,
                "cache_hit": False,
            }
            atomic_json(path, value)
            return value
