"""Offline official scoring and mechanism diagnostics after a run is sealed."""
from __future__ import annotations

from typing import Iterable

from hotpot.scorer import score_example


def score_prediction(example, prediction: dict | None) -> dict:
    prediction = prediction or {}
    return score_example(str(prediction.get("answer", "")), example.answer,
                         prediction.get("supporting_facts", []), example.supporting_facts)


def score_condition(example, record: dict) -> dict:
    rows = []
    for candidate in record["candidates"]:
        if candidate.get("status") != "complete":
            metrics = score_prediction(example, None)
        else:
            metrics = score_prediction(example, candidate["prediction"])
        retrieved = set(candidate.get("execution", {}).get("retrieved_titles", []))
        gold_titles = {title for title, _ in example.supporting_facts}
        recall = len(retrieved & gold_titles) / len(gold_titles) if gold_titles else 0.0
        rows.append({"candidate_index": candidate["candidate_index"], "metrics": metrics,
                     "gold_title_recall": recall})
    selected = record.get("selected_candidate")
    selected_metrics = next((row["metrics"] for row in rows
                             if row["candidate_index"] == selected), score_prediction(example, None))
    oracle = max(rows, key=lambda row: row["metrics"]["joint_f1"]) if rows else None
    return {"task_id": example.id, "condition": record["condition"],
            "candidates": rows, "selected": selected_metrics,
            "oracle_any_candidate": oracle["metrics"] if oracle else score_prediction(example, None),
            "oracle_candidate_index": oracle["candidate_index"] if oracle else None}


def mean_metric(scored: Iterable[dict], field: str, view: str = "selected") -> float:
    rows = list(scored)
    return sum(row[view][field] for row in rows) / len(rows) if rows else 0.0
