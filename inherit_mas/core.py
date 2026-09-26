"""Benchmark-neutral task-time workflow evolution.

The controller deliberately knows nothing about HotpotQA titles/hops, WorkBench
domains/actions, gold answers, or official scores.  A benchmark adapter may expose
only public task/interface information and artifacts observed during execution.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol


GENERAL_JUDGMENT_CONTRACT = r'''{
  "quality_score": 0,
  "completion_likelihood": 0,
  "requirement_coverage": 0,
  "artifact_grounding": 0,
  "critique": "brief skeptical audit",
  "keep": ["component worth preserving"],
  "fix": ["highest-value repair"]
}'''

EDIT_CONTRACT = r'''{
  "edit_index": 0,
  "rationale": "why this one atomic edit addresses the audit",
  "replacement": "only for an edit whose menu entry requires replacement",
  "subtask": "only for an add-node edit that requires it",
  "system_prompt": "only for an add-node edit that requires it"
}'''


class AdapterError(ValueError):
    pass


class ProposalFailure(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


class BenchmarkAdapter(Protocol):
    """Public, gold-free boundary between a benchmark and Inherit-MAS."""

    name: str
    workflow_contract: str
    interface_contract: str

    def task_id(self, task: Any) -> str: ...
    def task_text(self, task: Any) -> str: ...
    def audit(self, prediction: Any, trace: dict) -> dict: ...
    def compact_trace(self, trace: dict) -> dict: ...


@dataclass(frozen=True)
class EvolutionHooks:
    validate_graph: Callable[[Any], dict]
    graph_digest: Callable[[dict], str]
    legal_edit_menu: Callable[[dict], list[dict]]
    apply_menu_edit: Callable[[dict, list[dict], Any], tuple[dict, dict]]
    create_cache: Callable[[], Any]
    execute: Callable[[dict, Any, Any, dict | None, Any], Any]


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start:end + 1])
        raise


def _repair_prompt(original: str, error: str, contract: str, *,
                   original_system: str, original_user: str) -> tuple[str, str]:
    return ("Repair one JSON response. Address the exact validator error. Return JSON only.",
            json.dumps({"validator_error": error, "contract": contract,
                        "invalid_response": original,
                        "original_system": original_system,
                        "original_user": original_user}, ensure_ascii=True))


def structured_call(models, *, role: str, system: str, user: str,
                    validator: Callable[[Any], Any], contract: str,
                    max_tokens: int) -> tuple[Any, list[dict]]:
    attempts: list[dict] = []
    original_system, original_user = system, user
    for attempt in range(2):
        result = models.structured(role=role, system=system, user=user,
                                   max_tokens=max_tokens)
        record = {"attempt": attempt, "text": result.text,
                  "usage": [row.to_dict() for row in result.usage], "error": ""}
        try:
            value = validator(_extract_json(result.text))
            attempts.append(record)
            return value, attempts
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)
            if attempt == 0:
                system, user = _repair_prompt(
                    result.text, record["error"], contract,
                    original_system=original_system, original_user=original_user)
    raise ProposalFailure(attempts[-1]["error"], attempts)


def validate_judgment(value: Any) -> dict:
    required = {"quality_score", "completion_likelihood", "requirement_coverage",
                "artifact_grounding", "critique", "keep", "fix"}
    if not isinstance(value, dict) or set(value) != required:
        raise AdapterError(f"judgment fields must be exactly {sorted(required)}")
    out: dict[str, Any] = {}
    for field in ("quality_score", "completion_likelihood", "requirement_coverage",
                  "artifact_grounding"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 100:
            raise AdapterError(f"{field} must be a number in [0,100]")
        out[field] = float(number)
    if not isinstance(value["critique"], str) or not value["critique"].strip():
        raise AdapterError("critique must be non-empty")
    out["critique"] = value["critique"].strip()[:1200]
    for field in ("keep", "fix"):
        if not isinstance(value[field], list) or not all(isinstance(x, str) for x in value[field]):
            raise AdapterError(f"{field} must be a string list")
        out[field] = [x.strip() for x in value[field] if x.strip()][:4]
    return out


def selection_key(candidate: dict) -> tuple:
    """Generic candidate ordering without benchmark-specific evidence thresholds."""
    judgment = candidate["judgment"]
    audit = candidate["artifact_audit"]
    return (-int(candidate.get("hard_failures", 0)),
            int(bool(audit.get("output_valid", False))),
            judgment["quality_score"], judgment["requirement_coverage"],
            judgment["artifact_grounding"], -int(candidate["candidate_index"]))


def selected_index(candidates: list[dict]) -> int | None:
    complete = [row for row in candidates if row.get("status") == "complete"]
    return max(complete, key=selection_key)["candidate_index"] if complete else None


def initial_candidate_index(candidates: list[dict]) -> int | None:
    """Return the first successfully executed initial synthesis."""
    for row in candidates:
        if (row.get("status") == "complete"
                and row.get("proposal", {}).get("type") == "initial_synthesis"):
            return row["candidate_index"]
    return None


def _available_menu(graph: dict, menu: list[dict], excluded: set[int]) -> list[dict]:
    """Hide tried choices and show current values so the proposer can avoid no-ops."""
    by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    shown = []
    for edit in menu:
        if edit["edit_index"] in excluded:
            continue
        row = dict(edit)
        if edit.get("op") in {"edit_prompt", "edit_subtask"}:
            node = by_id.get(edit.get("target"), {})
            field = "system_prompt" if edit["op"] == "edit_prompt" else "subtask"
            row["current_value"] = node.get(field, "")
            row["constraint"] = "replacement must differ from current_value"
        shown.append(row)
    return shown


def _synthesis_prompt(adapter: BenchmarkAdapter, task: Any, *, escape: dict | None) -> tuple[str, str]:
    system = (
        "Synthesize a small typed multi-agent workflow for the supplied task and interface. "
        "Use only the declared roles, ports, and tools. Decompose only when useful, keep information "
        "flow explicit, and end at the declared output node. Return JSON only."
    )
    payload = {"benchmark": adapter.name, "task": adapter.task_text(task),
               "interface_contract": adapter.interface_contract,
               "workflow_contract": adapter.workflow_contract}
    if escape:
        system += " This is a stagnation escape; produce a meaningfully different valid workflow."
        payload["stagnation_escape"] = escape
    return system, json.dumps(payload, ensure_ascii=True)


def _judge_prompt(adapter: BenchmarkAdapter, task: Any, prediction: Any,
                  audit: dict) -> tuple[str, str]:
    system = (
        "You are an independent, skeptical, gold-free evaluator of a task workflow. Derive the task's "
        "requirements from the public request and interface contract. Use only the proposed output and "
        "observed artifacts in the audit. Penalize unsupported claims, missing requirements, invalid "
        "actions, and fluent guessing. Do not assume any benchmark-specific number of sources, steps, "
        "agents, or actions. Scores are 0-100. Return compact JSON only."
    )
    user = json.dumps({"benchmark": adapter.name, "task": adapter.task_text(task),
                       "interface_contract": adapter.interface_contract,
                       "proposed_output": prediction, "artifact_audit": audit,
                       "contract": GENERAL_JUDGMENT_CONTRACT}, ensure_ascii=True)
    return system, user


def _refinement_prompt(adapter: BenchmarkAdapter, task: Any, graph: dict,
                       incumbent: dict, menu: list[dict]) -> tuple[str, str]:
    system = (
        "Choose exactly one available, prevalidated atomic workflow edit. Use the gold-free audit and "
        "execution diagnostics, preserve useful components, and address the highest-value reported defect. "
        "Do not emit a compound edit or copy a current value unchanged. Return compact JSON only."
    )
    user = json.dumps({"benchmark": adapter.name, "task": adapter.task_text(task),
                       "interface_contract": adapter.interface_contract, "graph": graph,
                       "audit": incumbent["judgment"],
                       "artifact_audit": incumbent["artifact_audit"],
                       "execution": adapter.compact_trace(incumbent["execution"]),
                       "available_edits": menu, "contract": EDIT_CONTRACT}, ensure_ascii=True)
    return system, user


def _judge(models, adapter: BenchmarkAdapter, task: Any, prediction: Any,
           audit: dict, max_tokens: int) -> tuple[dict, list[dict]]:
    system, user = _judge_prompt(adapter, task, prediction, audit)
    return structured_call(models, role="inherit_mas_judge", system=system, user=user,
                           validator=validate_judgment,
                           contract=GENERAL_JUDGMENT_CONTRACT, max_tokens=max_tokens)


def _escape_reason(candidates: list[dict]) -> str:
    complete = [row for row in candidates if row.get("status") == "complete"]
    if not complete:
        return "no_complete_candidate"
    if not any(row["artifact_audit"].get("output_valid", False) for row in complete):
        return "no_valid_output"
    initial = complete[0]["judgment"]["quality_score"]
    if max(row["judgment"]["quality_score"] for row in complete) <= initial + 1:
        return "judge_stagnation"
    return ""


def _record(index: int, graph: dict, result: Any, judgment: dict,
            judge_attempts: list[dict], proposal: dict, proposal_attempts: list[dict],
            adapter: BenchmarkAdapter) -> dict:
    audit = adapter.audit(result.prediction, result.trace)
    hard = int(not audit.get("output_valid", False)) + sum(
        bool(node.get("error")) for node in result.trace.get("nodes", {}).values())
    return {"status": "complete", "candidate_index": index, "graph": graph,
            "proposal": proposal, "proposal_attempts": proposal_attempts,
            "prediction": result.prediction, "execution": result.trace,
            "snapshots": result.snapshots, "judgment": judgment,
            "judge_attempts": judge_attempts, "artifact_audit": audit,
            "hard_failures": hard}


def run_evolution(task: Any, *, adapter: BenchmarkAdapter, hooks: EvolutionHooks,
                  models: Any, max_candidates: int, meta_max_tokens: int,
                  judge_max_tokens: int, refiner_max_tokens: int) -> dict:
    """Run one gold-free evolution trajectory using an adapter-independent policy."""
    candidates: list[dict] = []
    tried_by_parent: dict[str, set[int]] = {}
    cache = hooks.create_cache()

    for index in range(max_candidates):
        try:
            complete = [row for row in candidates if row.get("status") == "complete"]
            parent_graph = parent_snapshots = None
            if not complete:
                system, user = _synthesis_prompt(adapter, task, escape=None)
                graph, proposal_attempts = structured_call(
                    models, role="inherit_mas_synthesizer", system=system, user=user,
                    validator=hooks.validate_graph, contract=adapter.workflow_contract,
                    max_tokens=meta_max_tokens)
                proposal = {"type": "initial_synthesis"}
            elif index == max_candidates - 1 and _escape_reason(complete):
                incumbent = max(complete, key=selection_key)
                reason = _escape_reason(complete)
                system, user = _synthesis_prompt(adapter, task, escape={
                    "reason": reason, "prior_audit": incumbent["judgment"],
                    "prior_graph": incumbent["graph"]})
                graph, proposal_attempts = structured_call(
                    models, role="inherit_mas_resynthesizer", system=system, user=user,
                    validator=hooks.validate_graph, contract=adapter.workflow_contract,
                    max_tokens=meta_max_tokens)
                proposal = {"type": "stagnation_resynthesis", "reason": reason}
            else:
                incumbent = max(complete, key=selection_key)
                parent_graph, parent_snapshots = incumbent["graph"], incumbent["snapshots"]
                parent_id = hooks.graph_digest(parent_graph)
                full_menu = hooks.legal_edit_menu(parent_graph)
                excluded = tried_by_parent.setdefault(parent_id, set())
                menu = _available_menu(parent_graph, full_menu, excluded)
                if not menu:
                    raise ProposalFailure("no untried atomic edits", [])
                system, user = _refinement_prompt(adapter, task, parent_graph, incumbent, menu)
                available = {row["edit_index"] for row in menu}

                def validate_edit(value):
                    if not isinstance(value, dict) or type(value.get("edit_index")) is not int:
                        raise AdapterError("proposal needs an integer edit_index")
                    if value["edit_index"] not in available:
                        raise AdapterError("edit_index is not in available_edits")
                    child, applied = hooks.apply_menu_edit(parent_graph, full_menu, value)
                    return {"graph": child, "applied": applied,
                            "edit_index": value["edit_index"]}

                edit, proposal_attempts = structured_call(
                    models, role="inherit_mas_refiner", system=system, user=user,
                    validator=validate_edit, contract=EDIT_CONTRACT,
                    max_tokens=refiner_max_tokens)
                excluded.add(edit["edit_index"])
                graph = edit["graph"]
                proposal = {"type": "atomic_edit", **edit["applied"]}

            result = hooks.execute(graph, task, cache, parent_graph, parent_snapshots)
            audit = adapter.audit(result.prediction, result.trace)
            judgment, judge_attempts = _judge(models, adapter, task, result.prediction,
                                               audit, judge_max_tokens)
            candidates.append(_record(index, graph, result, judgment, judge_attempts,
                                      proposal, proposal_attempts, adapter))
        except ProposalFailure as exc:
            candidates.append({"status": "invalid_proposal", "candidate_index": index,
                               "proposal_attempts": exc.attempts})

    chosen = selected_index(candidates)
    selected = next((row for row in candidates if row.get("candidate_index") == chosen), None)
    return {"schema": "inherit_mas_trajectory/1", "benchmark": adapter.name,
            "task_id": adapter.task_id(task), "candidates": candidates,
            "selected_candidate": chosen,
            "selected_prediction": selected.get("prediction") if selected else None,
            "settings": {"max_candidates": max_candidates,
                         "selector": "generic-validity-then-gold-free-quality-v1",
                         "escape": "generic-invalid-output-or-stagnation-v1"}}


__all__ = ["AdapterError", "BenchmarkAdapter", "EvolutionHooks",
           "GENERAL_JUDGMENT_CONTRACT", "ProposalFailure", "run_evolution",
           "initial_candidate_index", "selected_index", "selection_key", "structured_call",
           "validate_judgment"]
