"""Shared candidate records, rewards, and strict structured validators."""
from __future__ import annotations

from typing import Any

from ..controller import judge, structured_call
from ..schemas import SchemaError
from .graph import graph_digest, validate_graph


def validate_selection(value: Any, pool_ids: set[str], k: int = 2) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"selected", "rationale"}:
        raise SchemaError("selection requires selected and rationale")
    selected = value["selected"]
    if (not isinstance(selected, list) or len(selected) != k or
            len(set(selected)) != k or not all(item in pool_ids for item in selected)):
        raise SchemaError(f"selected must contain {k} distinct pool ids")
    if not isinstance(value["rationale"], str):
        raise SchemaError("selection rationale must be a string")
    return selected


def graph_validator(max_nodes: int):
    return lambda value: validate_graph(value, max_nodes=max_nodes)


def evomas_reward(judgment: dict, execution: dict) -> float:
    """Official EvoMAS reward scale: Metrics - 1e-6(tokens + 1000*time)."""
    cost = int(execution.get("live_tokens", 0)) + 1000 * float(execution.get("wall_s", 0.0))
    return float(judgment["quality_score"]) - 1e-6 * cost


def candidate_record(index: int, graph: dict, result, judgment: dict,
                     judge_attempts: list[dict], proposal: dict,
                     proposal_attempts: list[dict], *, reward: float | None = None) -> dict:
    hard = int(not result.prediction.get("valid", False)) + sum(
        bool(node.get("error")) for node in result.trace.get("nodes", {}).values())
    return {"status": "complete", "candidate_index": index, "graph": graph,
            "graph_digest": graph_digest(graph), "proposal": proposal,
            "proposal_attempts": proposal_attempts, "prediction": result.prediction,
            "execution": result.trace, "judgment": judgment,
            "judge_attempts": judge_attempts, "hard_failures": hard,
            "reward": float(reward) if reward is not None else None}


def selected_by_reward(candidates: list[dict]) -> int | None:
    complete = [row for row in candidates if row.get("status") == "complete"]
    if not complete:
        return None
    def key(row: dict) -> tuple[float, int, int]:
        reward = row.get("reward")
        return (float(reward) if reward is not None else float("-inf"),
                -int(row["hard_failures"]), -int(row["candidate_index"]))

    return max(complete, key=key)["candidate_index"]


def invalid_record(index: int, status: str, attempts: list[dict], proposal: dict) -> dict:
    return {"status": status, "candidate_index": index,
            "proposal": proposal, "proposal_attempts": attempts}


__all__ = ["candidate_record", "evomas_reward", "graph_validator", "invalid_record",
           "judge", "selected_by_reward", "structured_call", "validate_selection"]
