"""Gold-free structured calls and judge shared by external graph adapters."""
from __future__ import annotations

import json
from typing import Any, Callable

from . import config
from .schemas import SchemaError, extract_json

JUDGE_CONTRACT = r'''{
  "quality_score": 0,
  "answer_likelihood": 0,
  "evidence_coverage": 0,
  "citation_grounding": 0,
  "critique": "specific skeptical audit",
  "keep": ["specific component"],
  "fix": ["specific missing or weak component"]
}'''


class StructuredCallError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def validate_judgment(value: Any) -> dict:
    required = {"quality_score", "answer_likelihood", "evidence_coverage",
                "citation_grounding", "critique", "keep", "fix"}
    if not isinstance(value, dict) or set(value) != required:
        raise SchemaError(f"judgment fields must be exactly {sorted(required)}")
    out = {}
    for field in ("quality_score", "answer_likelihood", "evidence_coverage", "citation_grounding"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 100:
            raise SchemaError(f"{field} must be in [0,100]")
        out[field] = float(number)
    if not isinstance(value["critique"], str) or not value["critique"].strip():
        raise SchemaError("critique must be non-empty")
    for field in ("keep", "fix"):
        if not isinstance(value[field], list) or not all(isinstance(x, str) for x in value[field]):
            raise SchemaError(f"{field} must be a string list")
        out[field] = value[field][:8]
    out["critique"] = value["critique"][:2000]
    return out


def structured_call(models, *, role: str, system: str, user: str,
                    validator: Callable[[Any], Any], contract: str,
                    max_tokens: int) -> tuple[Any, list[dict]]:
    attempts = []
    for attempt in range(2):
        result = models.structured(role=role, system=system, user=user, max_tokens=max_tokens)
        record = {"attempt": attempt, "text": result.text,
                  "usage": [row.to_dict() for row in result.usage], "error": ""}
        try:
            value = validator(extract_json(result.text))
            attempts.append(record)
            return value, attempts
        except ValueError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)
            if attempt == 0:
                system = "Repair one JSON response. Address the exact validator error. Return JSON only, no commentary."
                user = f"Validator error: {record['error']}\nContract:\n{contract}\nOriginal:\n{result.text}"
    raise StructuredCallError(attempts[-1]["error"], attempts)


def judge(models, question: str, graph: dict | None, prediction: dict,
          trace: dict) -> tuple[dict, list[dict]]:
    system = (
        "You are an independent, skeptical HotpotQA judge. You never see the gold answer or official supporting "
        "facts. Evaluate only the public question, retrieved observations, workflow, and proposed answer. Treat "
        "fluent unsupported answers as low quality. Check whether both hops are resolved, whether the answer follows "
        "from evidence, and whether every cited title/sentence was observed. Scores are 0-100. A score above 80 "
        "requires explicit evidence for both hops and no material uncertainty. Return JSON only."
    )
    compact = {"dirty_nodes": trace.get("dirty_nodes", []),
               "live_tokens": trace.get("live_tokens", 0),
               "reused_tokens": trace.get("reused_tokens", 0),
               "search_calls": trace.get("search_calls", 0),
               "retrieved_titles": trace.get("retrieved_titles", []),
               "nodes": {nid: {"role": row.get("role"), "output": row.get("output"),
                               "error": row.get("error"), "reused": row.get("reused")}
                         for nid, row in trace.get("nodes", {}).items()}}
    user = json.dumps({"question": question, "graph": graph, "prediction": prediction,
                       "execution": compact, "contract": JUDGE_CONTRACT}, ensure_ascii=True)
    return structured_call(models, role="judge", system=system, user=user,
                           validator=validate_judgment, contract=JUDGE_CONTRACT,
                           max_tokens=config.JUDGE_MAX_TOKENS)
