"""Strict output and typed-DAG schemas for HotpotQA FullWiki."""
from __future__ import annotations

import json
from typing import Any


class SchemaError(ValueError):
    pass


def extract_json(text: str) -> Any:
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise SchemaError("response contains no JSON object")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaError("response JSON is malformed") from exc


def validate_prediction(value: Any) -> dict:
    if not isinstance(value, dict):
        raise SchemaError("prediction must be an object")
    if set(value) - {"answer", "supporting_facts", "confidence", "rationale"}:
        raise SchemaError("prediction contains unknown fields")
    answer = value.get("answer")
    facts = value.get("supporting_facts")
    confidence = value.get("confidence", 0.0)
    if not isinstance(answer, str) or not answer.strip():
        raise SchemaError("answer must be a non-empty string")
    if not isinstance(facts, list):
        raise SchemaError("supporting_facts must be a list")
    normalized = []
    for fact in facts:
        if not (isinstance(fact, (list, tuple)) and len(fact) == 2
                and isinstance(fact[0], str) and isinstance(fact[1], int)
                and not isinstance(fact[1], bool) and fact[1] >= 0):
            raise SchemaError("each supporting fact must be [title, nonnegative sentence_id]")
        normalized.append([fact[0], fact[1]])
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise SchemaError("confidence must be in [0,1]")
    return {"answer": answer.strip(), "supporting_facts": normalized,
            "confidence": float(confidence), "rationale": str(value.get("rationale", ""))[:1000]}


def prediction_from_text(text: str) -> dict:
    return validate_prediction(extract_json(text))


def invalid_prediction(error: str) -> dict:
    return {"answer": "", "supporting_facts": [], "confidence": 0.0,
            "rationale": "", "valid": False, "error": str(error)}
