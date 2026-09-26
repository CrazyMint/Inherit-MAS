"""Five-round, gold-blind task-time synthesis/refinement controller."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .edits import apply_transaction, legal_edit_menu, materialize_menu_choice
from .executor import execute_graph
from .prompts import (GRAPH_GRAMMAR, JUDGE_GRAMMAR, MENU_EDIT_GRAMMAR, judge_prompt,
                      refinement_prompt, repair_prompt, synthesis_prompt)
from .schema import GraphError, graph_digest, validate_graph, validate_judgment
from .snapshots import SnapshotStore, canonical, digest

MAX_ROUNDS = 5
META_MAX_TOKENS = 5000
JUDGE_MAX_TOKENS = 3000


@dataclass(frozen=True)
class PublicTask:
    id: str
    task: str


class StructuredCallError(GraphError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise GraphError("no JSON object found")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(canonical(value))
    os.replace(tmp, path)


def _model_usage(result) -> list[dict]:
    return [u.to_dict() for u in result.usage]


def _structured_meta(models, *, role: str, system: str, user: str, validator: Callable[[Any], Any], contract: str,
                     max_tokens: int, model: str = "gpt-5.4-mini"):
    attempts = []
    original_system, original_user = system, user
    current_system, current_user = system, user
    for attempt in range(2):
        result = models.chat(model=model, role=role, system=current_system, user=current_user, max_tokens=max_tokens)
        attempts.append({"text": result.text, "usage": _model_usage(result)})
        try:
            value = extract_json(result.text)
            value = validator(value)
            return value, attempts
        except Exception as exc:
            if attempt:
                raise StructuredCallError(str(exc), attempts) from exc
            current_system, current_user = repair_prompt(
                result.text, str(exc), contract,
                original_system=original_system, original_user=original_user,
            )
    raise RuntimeError("unreachable")


def hard_failure_count(execution_trace: dict) -> int:
    failures = 0
    for node in execution_trace.get("nodes", {}).values():
        if node.get("error") or node.get("parse_valid") is False:
            failures += 1
    return failures


def selection_key(round_record: dict) -> tuple:
    judgment = round_record["judgment"]
    # Cost is outside quality; it is only the fourth deterministic tie-break.
    return (
        judgment.get("strict_quality_score", judgment["quality_score"]),
        judgment["safety_score"],
        -round_record["hard_failures"],
        -round_record["round"],
        -round_record["execution"]["live_tokens"],
    )


def accepts_candidate(parent: dict, candidate: dict, margin: float = 5.0) -> bool:
    pjudgment, cjudgment = parent["judgment"], candidate["judgment"]
    pq = pjudgment.get("strict_quality_score", pjudgment["quality_score"])
    cq = cjudgment.get("strict_quality_score", cjudgment["quality_score"])
    ps, cs = pjudgment["safety_score"], cjudgment["safety_score"]
    if candidate["hard_failures"] > parent["hard_failures"]:
        return False
    quality_improves = cq >= pq + margin and cs >= ps
    safety_improves = cs >= ps + margin and cq >= pq
    return bool(quality_improves or safety_improves)


def best_round(rounds: list[dict]) -> dict:
    if not rounds:
        raise ValueError("best_round requires at least one round")
    incumbent = rounds[0]
    for candidate in rounds[1:]:
        if accepts_candidate(incumbent, candidate):
            incumbent = candidate
    return incumbent


def can_early_stop(record: dict) -> bool:
    # Disabled by protocol: all valid trajectories attempt all five rounds.
    return False


def validate_menu_choice(graph: dict, menu: dict, value: Any, excluded: list[int]) -> dict:
    # Report the state-dependent error first. Otherwise a missing rationale can
    # consume the only repair before the model learns that the index is stale.
    if isinstance(value, dict) and value.get("edit_index") in excluded:
        raise GraphError("edit_index was already tried against this incumbent; choose a different allowed index")
    choice, tx = materialize_menu_choice(graph, menu, value)
    return {"choice": choice, "transaction": tx}


class TaskController:
    def __init__(self, *, models, run_root: str | Path, judge_model: str = "gpt-5.4-mini"):
        self.models = models
        self.run_root = Path(run_root)
        self.judge_model = judge_model

    def run(self, task: PublicTask) -> dict:
        task_dir = self.run_root / "tasks" / task.id.replace("#", "_")
        state_path = task_dir / "record.json"
        cache = SnapshotStore(task_dir / "node_cache")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("task") != {"id": task.id, "task": task.task}:
                raise RuntimeError("task checkpoint identity mismatch")
            if state.get("judge_model") != self.judge_model:
                raise RuntimeError("task checkpoint judge-model mismatch")
            if state.get("status") == "complete":
                return state
        else:
            state = {"version": 1, "task": {"id": task.id, "task": task.task}, "judge_model": self.judge_model,
                     "status": "running", "rounds": [], "pending": None}
            _atomic_json(state_path, state)

        while len(state["rounds"]) < MAX_ROUNDS:
            if state.get("pending") is None:
                try:
                    pending = self._propose(task, state["rounds"])
                except StructuredCallError as exc:
                    state.setdefault("proposal_failures", []).append({
                        "after_round": state["rounds"][-1]["round"] if state["rounds"] else None,
                        "error": str(exc),
                        "attempts": exc.attempts,
                    })
                    _atomic_json(state_path, state)
                    break
                state["pending"] = pending
                _atomic_json(state_path, state)
            pending = state["pending"]
            if "execution" not in pending:
                parent = None
                parent_snaps = None
                if pending["parent_round"] is not None:
                    parent_record = next(r for r in state["rounds"] if r["round"] == pending["parent_round"])
                    parent = parent_record["graph"]
                    parent_snaps = parent_record["snapshots"]
                result = execute_graph(
                    pending["graph"], task, models=self.models, cache=cache,
                    parent_graph=parent, parent_snapshots=parent_snaps,
                )
                pending["execution"] = {"prediction": result.prediction, "trace": result.trace, "snapshots": result.snapshots}
                _atomic_json(state_path, state)
            if "judgment" not in pending:
                execution = pending["execution"]
                jsys, juser = judge_prompt(task.task, pending["graph"], execution["prediction"], execution["trace"])
                judgment, judge_attempts = _structured_meta(
                    self.models, role="judge", system=jsys, user=juser, validator=validate_judgment,
                    contract=JUDGE_GRAMMAR, max_tokens=JUDGE_MAX_TOKENS,
                    model=self.judge_model,
                )
                pending["judgment"] = judgment
                pending["judge_attempts"] = judge_attempts
                _atomic_json(state_path, state)

            execution = pending["execution"]
            record = {
                "round": len(state["rounds"]),
                "parent_round": pending["parent_round"],
                "graph": pending["graph"],
                "graph_digest": graph_digest(pending["graph"]),
                "proposal": pending["proposal"],
                "proposal_attempts": pending["proposal_attempts"],
                "prediction": execution["prediction"],
                "execution": execution["trace"],
                "snapshots": execution["snapshots"],
                "judgment": pending["judgment"],
                "judge_attempts": pending["judge_attempts"],
                "hard_failures": hard_failure_count(execution["trace"]),
            }
            state["rounds"].append(record)
            state["pending"] = None
            _atomic_json(state_path, state)
            if can_early_stop(record):
                break

        state["status"] = "complete"
        if state["rounds"]:
            chosen = best_round(state["rounds"])
            state["selected_round"] = chosen["round"]
            state["selected_prediction"] = chosen["prediction"]
            state["task_error"] = ""
        else:
            state["selected_round"] = None
            state["selected_prediction"] = []
            state["task_error"] = "initial_synthesis_invalid"
        state["completed_at"] = time.time()
        state["record_digest"] = digest({k: v for k, v in state.items() if k != "record_digest"})
        _atomic_json(state_path, state)
        return state

    def _propose(self, task: PublicTask, rounds: list[dict]) -> dict:
        if not rounds:
            system, user = synthesis_prompt(task.task)
            graph, attempts = _structured_meta(
                self.models, role="synthesizer", system=system, user=user, validator=validate_graph,
                contract=GRAPH_GRAMMAR, max_tokens=META_MAX_TOKENS,
            )
            return {"parent_round": None, "graph": graph, "proposal": {"type": "initial_synthesis"}, "proposal_attempts": attempts}
        incumbent = best_round(rounds)
        menu = legal_edit_menu(incumbent["graph"])
        excluded = sorted({
            record["proposal"]["choice"]["edit_index"]
            for record in rounds
            if record.get("parent_round") == incumbent["round"]
            and isinstance(record.get("proposal"), dict)
            and isinstance(record["proposal"].get("choice"), dict)
        })
        system, user = refinement_prompt(task.task, incumbent["graph"], incumbent["judgment"],
                                         incumbent["execution"], menu, excluded)
        proposal, attempts = _structured_meta(
            self.models, role="refiner", system=system, user=user,
            validator=lambda value: validate_menu_choice(incumbent["graph"], menu, value, excluded),
            contract=MENU_EDIT_GRAMMAR, max_tokens=META_MAX_TOKENS,
        )
        tx = proposal["transaction"]
        graph = apply_transaction(incumbent["graph"], tx)
        return {"parent_round": incumbent["round"], "graph": graph, "proposal": proposal,
                "proposal_attempts": attempts}
