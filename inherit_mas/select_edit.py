"""Inherit-MAS controller: Select and Edit workflow inheritance.

Each refinement round builds on the most recent completed candidate. The judge
receives the executed workflow and its per-node outputs and names the nodes
behind its critique. Select keeps the useful nodes of that candidate and
discards the rest; the kept workflow must pass the benchmark validator. Edit
applies one validated edit from the adapter menu, extended with inserting a
node between two nodes joined by an existing edge. Execution then inherits
stored node results outside the affected region. Initial synthesis, the
final-round restart, the final ranking over all candidates, and the candidate
cap are shared with ``core``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .core import (
    EDIT_CONTRACT,
    GENERAL_JUDGMENT_CONTRACT,
    AdapterError,
    BenchmarkAdapter,
    EvolutionHooks,
    ProposalFailure,
    _available_menu,
    _escape_reason,
    _record,
    _synthesis_prompt,
    selected_index,
    selection_key,
    structured_call,
    validate_judgment,
)


SELECT_CONTRACT = r'''{
  "discard": ["id of a node to drop; leave the list empty to keep every node"],
  "rationale": "why the kept nodes should be inherited"
}'''


@dataclass(frozen=True)
class ComponentHooks:
    """Benchmark operations that the Select and Edit steps need."""

    discardable_nodes: Callable[[dict], list[str]]
    prune: Callable[[dict, list[str]], dict]
    insert_menu: Callable[[dict], list[dict]]
    apply_insert: Callable[[dict, dict, Any], tuple[dict, dict]]


def _judge_prompt(adapter: BenchmarkAdapter, task: Any, prediction: Any, audit: dict,
                  graph: dict, trace: dict) -> tuple[str, str]:
    system = (
        "You are an independent, skeptical, gold-free evaluator of a task workflow. Derive the task's "
        "requirements from the public request and interface contract. Use only the proposed output, "
        "the observed artifacts in the audit, and the workflow's recorded node outputs. Penalize "
        "unsupported claims, missing requirements, invalid actions, and fluent guessing. Do not assume "
        "any benchmark-specific number of sources, steps, agents, or actions. When the node outputs "
        "show which workflow nodes contributed useful or useless work, name those node ids in the "
        "critique and in the keep/fix strings. Scores are 0-100. Return compact JSON with exactly "
        "the contract's fields and no others."
    )
    user = json.dumps({"benchmark": adapter.name, "task": adapter.task_text(task),
                       "interface_contract": adapter.interface_contract,
                       "proposed_output": prediction, "artifact_audit": audit,
                       "workflow": graph, "node_outputs": adapter.compact_trace(trace),
                       "contract": GENERAL_JUDGMENT_CONTRACT}, ensure_ascii=True)
    return system, user


def _judge(models, adapter: BenchmarkAdapter, task: Any, prediction: Any, audit: dict,
           graph: dict, trace: dict, max_tokens: int) -> tuple[dict, list[dict]]:
    system, user = _judge_prompt(adapter, task, prediction, audit, graph, trace)
    return structured_call(models, role="inherit_mas_judge", system=system, user=user,
                           validator=validate_judgment,
                           contract=GENERAL_JUDGMENT_CONTRACT, max_tokens=max_tokens)


def _select_prompt(adapter: BenchmarkAdapter, task: Any, parent: dict,
                   discardable: list[str]) -> tuple[str, str]:
    system = (
        "Select which nodes of the latest workflow the next candidate inherits. Use the gold-free "
        "audit and execution diagnostics. Discard a node only when the audit or its recorded output "
        "shows that it contributes nothing useful or harms the result, and keep every other node. "
        "The kept workflow must remain valid under the workflow contract. Return compact JSON only."
    )
    user = json.dumps({"benchmark": adapter.name, "task": adapter.task_text(task),
                       "interface_contract": adapter.interface_contract,
                       "workflow_contract": adapter.workflow_contract,
                       "graph": parent["graph"], "audit": parent["judgment"],
                       "artifact_audit": parent["artifact_audit"],
                       "execution": adapter.compact_trace(parent["execution"]),
                       "discardable_nodes": discardable, "contract": SELECT_CONTRACT},
                      ensure_ascii=True)
    return system, user


def validate_selection(value: Any, graph: dict, discardable: list[str],
                       components: ComponentHooks) -> dict:
    """Check a discard set and return the kept workflow it induces."""

    if not isinstance(value, dict) or set(value) != {"discard", "rationale"}:
        raise AdapterError("selection fields must be exactly discard and rationale")
    discard = value["discard"]
    if not isinstance(discard, list) or not all(isinstance(nid, str) for nid in discard):
        raise AdapterError("discard must be a list of node ids")
    if len(discard) != len(set(discard)):
        raise AdapterError("discard must not repeat a node")
    unknown = sorted(set(discard) - set(discardable))
    if unknown:
        raise AdapterError(f"discard names nodes outside discardable_nodes: {unknown}")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise AdapterError("rationale must be non-empty")
    kept = components.prune(graph, discard) if discard else graph
    return {"discard": discard, "rationale": rationale.strip(), "graph": kept}


def _edit_prompt(adapter: BenchmarkAdapter, task: Any, kept_graph: dict, parent: dict,
                 selection: dict, menu: list[dict]) -> tuple[str, str]:
    system = (
        "Choose exactly one available, prevalidated atomic workflow edit. Use the gold-free audit and "
        "execution diagnostics, preserve useful components, and address the highest-value reported defect. "
        "Do not emit a compound edit or copy a current value unchanged. Return compact JSON only."
    )
    user = json.dumps({"benchmark": adapter.name, "task": adapter.task_text(task),
                       "interface_contract": adapter.interface_contract, "graph": kept_graph,
                       "selection": {"discarded": selection["discard"],
                                     "rationale": selection["rationale"]},
                       "audit": parent["judgment"],
                       "artifact_audit": parent["artifact_audit"],
                       "execution": adapter.compact_trace(parent["execution"]),
                       "available_edits": menu, "contract": EDIT_CONTRACT}, ensure_ascii=True)
    return system, user


def combined_menu(kept_graph: dict, hooks: EvolutionHooks,
                  components: ComponentHooks) -> tuple[list[dict], list[dict]]:
    """Return (adapter menu, adapter menu + insertions indexed after it)."""

    base = hooks.legal_edit_menu(kept_graph)
    inserts = [
        {**{key: value for key, value in row.items() if key != "edit_index"},
         "edit_index": len(base) + offset}
        for offset, row in enumerate(components.insert_menu(kept_graph))
    ]
    return base, list(base) + inserts


def apply_choice(kept_graph: dict, value: Any, *, base: list[dict], menu: list[dict],
                 available: set[int], hooks: EvolutionHooks,
                 components: ComponentHooks) -> dict:
    if not isinstance(value, dict) or type(value.get("edit_index")) is not int:
        raise AdapterError("proposal needs an integer edit_index")
    index = value["edit_index"]
    if index not in available:
        raise AdapterError("edit_index is not in available_edits")
    if index >= len(base):
        child, applied = components.apply_insert(kept_graph, menu[index], value)
    else:
        child, applied = hooks.apply_menu_edit(kept_graph, base, value)
    return {"graph": hooks.validate_graph(child), "applied": applied, "edit_index": index}


def _stage(attempts: list[dict], stage: str) -> list[dict]:
    return [{**attempt, "stage": stage} for attempt in attempts]


def run_select_edit(task: Any, *, adapter: BenchmarkAdapter, hooks: EvolutionHooks,
                       components: ComponentHooks, models: Any, max_candidates: int,
                       meta_max_tokens: int, judge_max_tokens: int,
                       refiner_max_tokens: int) -> dict:
    """Run one gold-free trajectory with Select and Edit workflow inheritance."""

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
                parent = complete[-1]
                parent_graph, parent_snapshots = parent["graph"], parent["snapshots"]
                discardable = components.discardable_nodes(parent_graph)
                if discardable:
                    system, user = _select_prompt(adapter, task, parent, discardable)
                    try:
                        selection, select_attempts = structured_call(
                            models, role="inherit_mas_selector", system=system, user=user,
                            validator=lambda raw: validate_selection(
                                raw, parent_graph, discardable, components),
                            contract=SELECT_CONTRACT, max_tokens=refiner_max_tokens)
                    except ProposalFailure as exc:
                        raise ProposalFailure(str(exc), _stage(exc.attempts, "select"))
                    select_attempts = _stage(select_attempts, "select")
                else:
                    # No node can be dropped without invalidating the workflow: keep all.
                    selection = {"discard": [], "graph": parent_graph,
                                 "rationale": "no node can be discarded without invalidating the workflow"}
                    select_attempts = []
                kept = selection["graph"]
                base, full_menu = combined_menu(kept, hooks, components)
                excluded = tried_by_parent.setdefault(hooks.graph_digest(kept), set())
                menu = _available_menu(kept, full_menu, excluded)
                if not menu:
                    raise ProposalFailure("no untried atomic edits", select_attempts)
                system, user = _edit_prompt(adapter, task, kept, parent, selection, menu)
                available = {row["edit_index"] for row in menu}
                try:
                    edit, edit_attempts = structured_call(
                        models, role="inherit_mas_refiner", system=system, user=user,
                        validator=lambda raw: apply_choice(
                            kept, raw, base=base, menu=full_menu, available=available,
                            hooks=hooks, components=components),
                        contract=EDIT_CONTRACT, max_tokens=refiner_max_tokens)
                except ProposalFailure as exc:
                    raise ProposalFailure(str(exc), select_attempts + _stage(exc.attempts, "edit"))
                excluded.add(edit["edit_index"])
                graph = edit["graph"]
                proposal_attempts = select_attempts + _stage(edit_attempts, "edit")
                proposal = {"type": "select_edit",
                            "parent_index": int(parent["candidate_index"]),
                            "select_called": bool(discardable),
                            "discard": selection["discard"],
                            "select_rationale": selection["rationale"],
                            "edit_index": edit["edit_index"],
                            "insertion": edit["edit_index"] >= len(base),
                            **edit["applied"]}

            result = hooks.execute(graph, task, cache, parent_graph, parent_snapshots)
            audit = adapter.audit(result.prediction, result.trace)
            judgment, judge_attempts = _judge(models, adapter, task, result.prediction,
                                              audit, graph, result.trace, judge_max_tokens)
            candidates.append(_record(index, graph, result, judgment, judge_attempts,
                                      proposal, proposal_attempts, adapter))
        except ProposalFailure as exc:
            candidates.append({"status": "invalid_proposal", "candidate_index": index,
                               "proposal_attempts": exc.attempts})

    chosen = selected_index(candidates)
    selected = next((row for row in candidates if row.get("candidate_index") == chosen), None)
    return {"schema": "inherit_mas_select_edit_trajectory/1", "benchmark": adapter.name,
            "task_id": adapter.task_id(task), "candidates": candidates,
            "selected_candidate": chosen,
            "selected_prediction": selected.get("prediction") if selected else None,
            "settings": {"max_candidates": max_candidates,
                         "parent": "latest-complete-candidate-v1",
                         "judge": "graph-aware-node-attribution-v1",
                         "select": "meta-model-discard-set-v1",
                         "edit": "one-validated-edit-with-node-insertion-v1",
                         "selector": "generic-validity-then-gold-free-quality-v1",
                         "escape": "generic-invalid-output-or-stagnation-v1"}}


__all__ = ["SELECT_CONTRACT", "ComponentHooks", "apply_choice", "combined_menu",
           "run_select_edit", "validate_selection"]
