"""Public task, retrieval, and offline-score bridge for native EvoMAS."""
from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from hotpot.scorer import aggregate, score_example
from hotpot_fullwiki.loader import RuntimeExample, ScorerExample, load_examples, open_snapshot
from hotpot_fullwiki.manifests import load as load_manifest
from hotpot_fullwiki.retrieval import BM25Retriever, RetrievalSession

SCHEMA = "hotpot-native-public-v1"
QUESTION_ID_RE = re.compile(r"^Question ID:\s*([^\s]+)\s*$", re.MULTILINE)


class RemoteBM25Retriever:
    """Small client for the single pinned BM25 service shared by both baselines."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.fingerprint_digest = self._get_json("/health")["retriever_digest"]
        expected = os.environ.get("EVOMAS_BM25_DIGEST")
        if expected and self.fingerprint_digest != expected:
            raise RuntimeError("native EvoMAS BM25 service fingerprint changed")

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=30) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"BM25 service request failed: {exc}") from exc

    def search(self, query: str, example: RuntimeExample) -> dict[str, Any]:
        body = json.dumps({"task_id": example.id, "query": query}).encode()
        request = urllib.request.Request(
            self.base_url + "/search", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"BM25 service search failed: {exc}") from exc


def task_prompt(example: RuntimeExample) -> str:
    """Return the identical gold-free prompt used by both native baselines."""
    return (
        "Answer this HotpotQA FullWiki question using the retrieval tool.\n"
        f"Question ID: {example.id}\n"
        f"Question: {example.question}\n\n"
        "Return JSON only with this schema:\n"
        '{"answer":"short answer",'
        '"supporting_facts":[["exact article title", sentence_id], ...]}'
    )


def task_id_from_prompt(prompt: str) -> str:
    match = QUESTION_ID_RE.search(str(prompt))
    if not match:
        raise ValueError("Hotpot task prompt lacks a Question ID marker")
    return match.group(1)


def public_rows(manifest_path: str | Path, *, snapshot=None) -> list[dict[str, Any]]:
    """Materialize public tasks; scorer-only fields are discarded by construction."""
    manifest = load_manifest(manifest_path)
    snap = snapshot or open_snapshot()
    if manifest["dataset_revision"] != snap.digest:
        raise ValueError("manifest/snapshot digest mismatch")
    examples = load_examples(manifest["rows"], snap)
    return [
        {
            "id": item.id,
            "question": item.question,
            "level": item.level,
            "qtype": item.qtype,
            "query": task_prompt(item.runtime()),
        }
        for item in examples
    ]


def scorer_examples(manifest_path: str | Path, *, snapshot=None) -> list[ScorerExample]:
    manifest = load_manifest(manifest_path)
    snap = snapshot or open_snapshot()
    if manifest["dataset_revision"] != snap.digest:
        raise ValueError("manifest/snapshot digest mismatch")
    return load_examples(manifest["rows"], snap)


def stage_evomas_dataset(manifest_path: str | Path, output_path: str | Path, *, snapshot=None) -> dict:
    """Write EvoMAS's native JSON format without answer/supporting-fact gold."""
    rows = public_rows(manifest_path, snapshot=snapshot)
    tasks = [
        {
            "id": row["id"],
            "query": row["query"],
            "gt": "__SEALED_OFFLINE__",
            "tag": ["hotpotqa", row["level"], row["qtype"]],
            "source": "HOTPOTQA_FULLWIKI",
            "metadata": {"schema": SCHEMA, "level": row["level"], "qtype": row["qtype"]},
        }
        for row in rows
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2, ensure_ascii=True) + "\n")
    return {"n": len(tasks), "ids": [row["id"] for row in rows]}


def runtime_examples(manifest_path: str | Path, *, snapshot=None) -> dict[str, RuntimeExample]:
    manifest = load_manifest(manifest_path)
    snap = snapshot or open_snapshot()
    return {item.id: item.runtime() for item in load_examples(manifest["rows"], snap)}


def new_retrieval_session(
    example: RuntimeExample,
    *,
    cache_dir: str | Path,
    searcher=None,
    call_budget: int = 4,
) -> RetrievalSession:
    remote = os.environ.get("HOTPOT_BM25_URL", "").strip()
    if remote and searcher is None:
        retriever = RemoteBM25Retriever(remote)
    else:
        retriever = BM25Retriever(cache_dir=cache_dir, searcher=searcher)
    return RetrievalSession(retriever, example, call_budget=call_budget)


def parse_prediction(value: Any) -> dict[str, Any]:
    """Parse a final answer without guessing citations from free-form prose."""
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value or "").strip()
        parsed = None
        # EvoMAS serializes some otherwise-structured Python dictionaries with
        # single quotes. Accept only an exact safe literal, never prose-derived
        # answer or citation guesses.
        try:
            literal = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            literal = None
        if isinstance(literal, dict) and "answer" in literal:
            parsed = literal

        # Native runtimes may surround an explicit JSON result with messages or
        # append another object. Decode each object start instead of using a
        # greedy regex that merges multiple objects into invalid JSON.
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if parsed is not None or character != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "answer" in obj:
                parsed = obj
                break
        if parsed is None:
            return {"answer": text, "supporting_facts": []}

    answer = str(parsed.get("answer", "")).strip()
    facts = []
    for fact in parsed.get("supporting_facts", []) or []:
        if (
            isinstance(fact, (list, tuple))
            and len(fact) == 2
            and isinstance(fact[0], str)
            and isinstance(fact[1], int)
            and not isinstance(fact[1], bool)
        ):
            facts.append([fact[0], fact[1]])
    return {"answer": answer, "supporting_facts": facts}


def score_records(
    examples: Iterable[ScorerExample],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Official deterministic scoring, called only after a run is sealed."""
    rows = []
    for example in examples:
        prediction = parse_prediction(predictions.get(example.id, ""))
        metrics = score_example(
            prediction["answer"], example.answer,
            prediction["supporting_facts"], example.supporting_facts,
        )
        rows.append({"task_id": example.id, "prediction": prediction, "metrics": metrics})
    return {"n": len(rows), "aggregate": aggregate([row["metrics"] for row in rows]), "rows": rows}


def assert_public_payload(payload: Any) -> None:
    """Fail if a staged online payload contains either scorer-only key."""
    forbidden = {"answer", "supporting_facts", "gold", "gold_answer", "gold_supporting_facts"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            bad = forbidden.intersection(value)
            if bad:
                raise ValueError(f"online payload contains forbidden scorer field {sorted(bad)[0]}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
