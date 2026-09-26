from __future__ import annotations

import copy

import pytest

import wb_env as W
from task_evolution.edits import apply_transaction, dirty_nodes, legal_edit_menu, materialize_menu_choice
from task_evolution.schema import GraphError, graph_digest, topological_order, validate_graph, validate_judgment


def graph():
    return {
        "name": "test",
        "nodes": [
            {"id": "worker_a", "role": "worker", "subtask": "look up events", "system_prompt": "report ids",
             "tools": ["calendar.search_events"], "inputs": []},
            {"id": "worker_b", "role": "worker", "subtask": "look up email", "system_prompt": "report email",
             "tools": ["company_directory.find_email_address"], "inputs": []},
            {"id": "integrator", "role": "integrator", "subtask": "form the plan", "system_prompt": "be exact",
             "tools": [], "inputs": [{"port": "events", "from": "worker_a"}, {"port": "email", "from": "worker_b"}]},
            {"id": "executor", "role": "executor", "subtask": "", "system_prompt": "", "tools": [],
             "inputs": [{"port": "plan", "from": "integrator"}]},
        ],
        "sink": "executor",
    }


def test_valid_typed_dag_and_order():
    validate_graph(graph())
    assert topological_order(graph())[-1] == "executor"


def test_llm_cannot_receive_write_tool():
    g = graph()
    g["nodes"][0]["tools"] = [next(iter(W.SIDE_EFFECT_TOOL_NAMES))]
    with pytest.raises(GraphError, match="read-only"):
        validate_graph(g)


def test_cycle_and_orphan_are_rejected():
    g = graph()
    g["nodes"][0]["inputs"] = [{"port": "peer", "from": "worker_b"}]
    g["nodes"][1]["inputs"] = [{"port": "peer", "from": "worker_a"}]
    with pytest.raises(GraphError, match="cycle"):
        validate_graph(g)
    g = graph()
    g["nodes"].append({"id": "unused", "role": "worker", "subtask": "x", "system_prompt": "", "tools": [], "inputs": []})
    with pytest.raises(GraphError, match="terminals"):
        validate_graph(g)


def test_validated_update_and_affected_region():
    parent = graph()
    tx = {
        "base_graph_digest": graph_digest(parent),
        "rationale": "make one worker more precise",
        "operations": [{"op": "update_node", "node_id": "worker_a", "patch": {"system_prompt": "report exact ids"}}],
    }
    child = apply_transaction(parent, tx)
    assert dirty_nodes(parent, child) == {"worker_a", "integrator", "executor"}


def test_stale_or_noop_edit_rejected():
    parent = graph()
    stale = {"base_graph_digest": "bad", "rationale": "x",
             "operations": [{"op": "update_node", "node_id": "worker_a", "patch": {"system_prompt": "new"}}]}
    with pytest.raises(GraphError, match="stale"):
        apply_transaction(parent, stale)
    noop = copy.deepcopy(stale)
    noop["base_graph_digest"] = graph_digest(parent)
    noop["operations"][0]["patch"]["system_prompt"] = "report ids"
    with pytest.raises(GraphError, match="no-op"):
        apply_transaction(parent, noop)


def test_refinement_transaction_is_one_operation_and_one_update_field():
    parent = graph()
    multi_op = {"base_graph_digest": graph_digest(parent), "rationale": "too broad", "operations": [
        {"op": "update_node", "node_id": "worker_a", "patch": {"system_prompt": "a"}},
        {"op": "update_node", "node_id": "worker_b", "patch": {"system_prompt": "b"}},
    ]}
    with pytest.raises(GraphError, match="exactly one"):
        apply_transaction(parent, multi_op)
    multi_field = {"base_graph_digest": graph_digest(parent), "rationale": "too broad", "operations": [
        {"op": "update_node", "node_id": "worker_a",
         "patch": {"system_prompt": "a", "subtask": "b"}},
    ]}
    with pytest.raises(GraphError, match="exactly one field"):
        apply_transaction(parent, multi_field)


def test_legal_menu_materializes_only_prevalidated_atomic_edits():
    parent = graph()
    menu = legal_edit_menu(parent)
    assert menu["entries"] and [row["edit_index"] for row in menu["entries"]] == list(range(len(menu["entries"])))
    text_entry = next(row for row in menu["entries"] if row["kind"] == "replace_system_prompt")
    choice, tx = materialize_menu_choice(parent, menu, {
        "base_graph_digest": menu["base_graph_digest"],
        "menu_digest": menu["menu_digest"],
        "edit_index": text_entry["edit_index"],
        "rationale": "make the selected role precise",
        "replacement": "Report exact identifiers and cite tool evidence.",
    })
    assert choice["edit_index"] == text_entry["edit_index"]
    child = apply_transaction(parent, tx)
    assert child != parent and len(tx["operations"]) == 1
    for entry in menu["entries"]:
        if "operation" not in entry:
            continue
        candidate = {"base_graph_digest": menu["base_graph_digest"], "rationale": "test",
                     "operations": [entry["operation"]]}
        apply_transaction(parent, candidate)


def test_legal_menu_rejects_invented_index():
    parent = graph()
    menu = legal_edit_menu(parent)
    with pytest.raises(GraphError, match="outside"):
        materialize_menu_choice(parent, menu, {
            "base_graph_digest": menu["base_graph_digest"], "menu_digest": menu["menu_digest"],
            "edit_index": len(menu["entries"]), "rationale": "invent an edit",
        })


def test_concrete_menu_choice_ignores_redundant_replacement():
    parent = graph()
    menu = legal_edit_menu(parent)
    entry = next(row for row in menu["entries"] if "operation" in row)
    choice, tx = materialize_menu_choice(parent, menu, {
        "base_graph_digest": menu["base_graph_digest"], "menu_digest": menu["menu_digest"],
        "edit_index": entry["edit_index"], "rationale": "select the concrete edit",
        "replacement": "unused model commentary",
    })
    assert "replacement" not in choice
    apply_transaction(parent, tx)


def test_strict_judgment_caps_defects_and_normalizes_satisfied():
    value = validate_judgment({
        "obligation_checks": [{"obligation": "perform both writes", "status": "missing",
                                "evidence": "only one proposed action is present"}],
        "quality_score": 100,
        "safety_score": 100,
        "correct": [],
        "wrong": [],
        "missing": ["second write"],
        "preserve": [],
        "recommended_changes": ["add the grounded second write"],
        "satisfied": True,
    })
    assert value["quality_score"] == 100
    assert value["strict_quality_score"] == 70
    assert value["reported_satisfied"] is True
    assert value["satisfied"] is False
