from __future__ import annotations

import json
from dataclasses import dataclass, field

from .core import AdapterError, EvolutionHooks
from .select_edit import ComponentHooks, run_select_edit


class _Usage:
    def to_dict(self):
        return {"total_tokens": 7, "estimated_usd": 0.001}


@dataclass
class _Reply:
    text: str
    usage: list = field(default_factory=lambda: [_Usage()])


class _Models:
    """Scripted structured calls, consumed in order per role."""

    def __init__(self, script: dict[str, list]):
        self.script = {role: list(rows) for role, rows in script.items()}
        self.calls: list[dict] = []

    def structured(self, *, role, system, user, max_tokens):
        self.calls.append({"role": role, "system": system, "user": json.loads(user)})
        return _Reply(json.dumps(self.script[role].pop(0)))


class _Adapter:
    name = "fake"
    workflow_contract = "keep node out"
    interface_contract = "answer"

    def task_id(self, task):
        return "t"

    def task_text(self, task):
        return "task"

    def audit(self, prediction, trace):
        return {"output_valid": True, "artifacts": []}

    def compact_trace(self, trace):
        return {"nodes": sorted(trace["nodes"])}


@dataclass
class _Run:
    prediction: str
    trace: dict
    snapshots: dict


def _g(*names):
    return {"nodes": [{"id": name} for name in names]}


def _ids(graph):
    return [node["id"] for node in graph["nodes"]]


def _validate(graph):
    names = _ids(graph) if isinstance(graph, dict) and isinstance(graph.get("nodes"), list) else []
    if "out" not in names or len(names) < 2:
        raise AdapterError("graph must keep out and one other node")
    return _g(*names)


def _menu(graph):
    rows = [{"op": "append", "target": name, "needs": "none"}
            for name in ("p", "q", "r") if name not in _ids(graph)]
    return [{"edit_index": index, **row} for index, row in enumerate(rows)]


def _apply(graph, menu, value):
    entry = menu[value["edit_index"]]
    return _g(*_ids(graph), entry["target"]), {"choice": entry}


def _setup():
    executions: list[dict] = []

    def execute(graph, task, cache, parent_graph, parent_snapshots):
        executions.append({"graph": graph, "parent_graph": parent_graph})
        return _Run("answer", {"nodes": {nid: {} for nid in _ids(graph)}}, {})

    hooks = EvolutionHooks(validate_graph=_validate, graph_digest=repr,
                           legal_edit_menu=_menu, apply_menu_edit=_apply,
                           create_cache=lambda: None, execute=execute)
    inserted: list[dict] = []

    def apply_insert(graph, entry, value):
        if not value.get("subtask"):
            raise AdapterError("insert needs a subtask")
        inserted.append(entry)
        return _g(*_ids(graph), entry["new_id"]), {"choice": entry}

    components = ComponentHooks(
        discardable_nodes=lambda graph: [nid for nid in _ids(graph) if nid != "out"],
        prune=lambda graph, discard: _validate(
            _g(*[nid for nid in _ids(graph) if nid not in discard])),
        insert_menu=lambda graph: ([] if "x" in _ids(graph) else
                                   [{"op": "insert", "target": "out", "new_id": "x",
                                     "needs": "subtask_and_prompt"}]),
        apply_insert=apply_insert,
    )
    return hooks, components, executions, inserted


def _judgment(quality):
    return {"quality_score": quality, "completion_likelihood": quality,
            "requirement_coverage": quality, "artifact_grounding": quality,
            "critique": "b contributes nothing", "keep": ["a"], "fix": ["b"]}


def _run(models, hooks, components, max_candidates):
    return run_select_edit(
        "task", adapter=_Adapter(), hooks=hooks, components=components, models=models,
        max_candidates=max_candidates, meta_max_tokens=10, judge_max_tokens=10,
        refiner_max_tokens=10)


def test_parent_is_latest_candidate_even_when_an_earlier_one_ranks_higher():
    # Each round refines the workflow just judged, not the best-ranked candidate.
    hooks, components, executions, _ = _setup()
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "b", "out")],
        "inherit_mas_judge": [_judgment(60), _judgment(90), _judgment(30), _judgment(30)],
        "inherit_mas_selector": [{"discard": [], "rationale": "keep all"}] * 3,
        "inherit_mas_refiner": [{"edit_index": 0, "rationale": "grow"}] * 3,
    })
    record = _run(models, hooks, components, max_candidates=4)
    graphs = [row["graph"] for row in record["candidates"]]
    assert [row["parent_graph"] for row in executions[1:]] == graphs[:3]
    assert [row["proposal"]["parent_index"] for row in record["candidates"][1:]] == [0, 1, 2]
    # The returned output follows the shared final ranking over all candidates.
    assert record["selected_candidate"] == 1


def test_discard_is_applied_before_the_edit_menu_is_built():
    hooks, components, executions, _ = _setup()
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "b", "out")],
        "inherit_mas_judge": [_judgment(50), _judgment(70), _judgment(70)],
        "inherit_mas_selector": [{"discard": ["b"], "rationale": "b adds nothing"},
                              {"discard": [], "rationale": "keep all"}],
        "inherit_mas_refiner": [{"edit_index": 0, "rationale": "grow"}] * 2,
    })
    record = _run(models, hooks, components, max_candidates=3)
    refiner = next(call for call in models.calls if call["role"] == "inherit_mas_refiner")
    assert refiner["user"]["graph"] == _g("a", "out")
    assert refiner["user"]["selection"]["discarded"] == ["b"]
    assert record["candidates"][1]["graph"] == _g("a", "out", "p")
    assert record["candidates"][1]["proposal"]["discard"] == ["b"]
    # Execution inheritance compares the child with the unpruned parent it came from.
    assert executions[1]["parent_graph"] == _g("a", "b", "out")


def test_invalid_discard_is_repaired_once_then_fails_closed_with_costs_kept():
    hooks, components, _, _ = _setup()
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "out")],
        "inherit_mas_resynthesizer": [_g("a", "b", "out")],
        "inherit_mas_judge": [_judgment(50), _judgment(50)],
        "inherit_mas_selector": [{"discard": ["a"], "rationale": "x"},
                              {"discard": ["a"], "rationale": "x"}],
    })
    record = _run(models, hooks, components, max_candidates=3)
    failed = record["candidates"][1]
    assert failed["status"] == "invalid_proposal"
    assert [row["stage"] for row in failed["proposal_attempts"]] == ["select", "select"]
    assert all(row["usage"] for row in failed["proposal_attempts"])
    # The shared final-round restart still governs the final round.
    assert record["candidates"][2]["proposal"]["type"] == "stagnation_resynthesis"


def test_insertions_are_indexed_after_the_adapter_menu_and_use_the_insert_hook():
    hooks, components, _, inserted = _setup()
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "b", "out")],
        "inherit_mas_judge": [_judgment(50), _judgment(70), _judgment(70)],
        "inherit_mas_selector": [{"discard": [], "rationale": "keep all"}] * 2,
        "inherit_mas_refiner": [{"edit_index": 3, "rationale": "missing evidence",
                              "subtask": "find it", "system_prompt": "search"},
                             {"edit_index": 0, "rationale": "grow"}],
    })
    record = _run(models, hooks, components, max_candidates=3)
    menu = next(call for call in models.calls
                if call["role"] == "inherit_mas_refiner")["user"]["available_edits"]
    assert [row["edit_index"] for row in menu] == [0, 1, 2, 3]
    assert menu[3]["op"] == "insert"
    assert inserted and record["candidates"][1]["proposal"]["insertion"] is True
    assert record["candidates"][1]["graph"] == _g("a", "b", "out", "x")


def test_judge_sees_the_workflow_and_node_outputs():
    hooks, components, _, _ = _setup()
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "b", "out")],
        "inherit_mas_judge": [_judgment(50)],
    })
    _run(models, hooks, components, max_candidates=1)
    judge = next(call for call in models.calls if call["role"] == "inherit_mas_judge")
    assert judge["user"]["workflow"] == _g("a", "b", "out")
    assert judge["user"]["node_outputs"] == {"nodes": ["a", "b", "out"]}
    assert "name those node ids" in judge["system"]
    assert "exactly the contract's fields and no others" in judge["system"]


def test_select_call_is_skipped_when_no_node_can_be_discarded():
    # Keeping every node is then the only legal choice, so no model call is spent on it.
    hooks, components, _, _ = _setup()
    components = components.__class__(
        discardable_nodes=lambda graph: [], prune=components.prune,
        insert_menu=components.insert_menu, apply_insert=components.apply_insert)
    models = _Models({
        "inherit_mas_synthesizer": [_g("a", "out")],
        "inherit_mas_judge": [_judgment(50), _judgment(70), _judgment(70)],
        "inherit_mas_refiner": [{"edit_index": 0, "rationale": "grow"}] * 2,
    })
    record = _run(models, hooks, components, max_candidates=3)
    assert not any(call["role"] == "inherit_mas_selector" for call in models.calls)
    proposal = record["candidates"][1]["proposal"]
    assert proposal["select_called"] is False and proposal["discard"] == []
