"""Diagnostic judge retained from the evaluated GPT Single ReAct harness."""
from __future__ import annotations

from typing import Any, Callable

from hotpot_fullwiki import config
from hotpot_fullwiki.graph import GraphError
from hotpot_fullwiki.schemas import SchemaError, extract_json
from .judge_prompts import JUDGE_CONTRACT, judge_prompt, repair_prompt


class StructuredCallError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def validate_judgment(value: Any) -> dict:
    if not isinstance(value, dict):
        raise SchemaError("judgment must be an object")
    required = {"quality_score", "answer_likelihood", "evidence_coverage",
                "citation_grounding", "critique", "keep", "fix"}
    if set(value) != required:
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
        except (SchemaError, GraphError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)
            if attempt == 0:
                system, user = repair_prompt(result.text, record["error"], contract)
    raise StructuredCallError(attempts[-1]["error"], attempts)


def _trace_for_judge(trace: dict) -> dict:
    return {"dirty_nodes": trace.get("dirty_nodes", []),
            "live_tokens": trace.get("live_tokens", 0),
            "reused_tokens": trace.get("reused_tokens", 0),
            "search_calls": trace.get("search_calls", 0),
            "retrieved_titles": trace.get("retrieved_titles", []),
            "nodes": {nid: {"role": row.get("role"), "output": row.get("output"),
                            "error": row.get("error"), "reused": row.get("reused")}
                      for nid, row in trace.get("nodes", {}).items()}}


def judge(models, question: str, graph: dict | None, prediction: dict,
          trace: dict) -> tuple[dict, list[dict]]:
    system, user = judge_prompt(question, graph, prediction, _trace_for_judge(trace))
    return structured_call(models, role="judge", system=system, user=user,
                           validator=validate_judgment, contract=JUDGE_CONTRACT,
                           max_tokens=config.JUDGE_MAX_TOKENS)
