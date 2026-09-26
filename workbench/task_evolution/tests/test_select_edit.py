import pytest

import wb_env as W
from task_evolution.edits import dirty_nodes
from task_evolution.select_edit import (
    apply_insert, discardable_nodes, insert_menu, prune)
from task_evolution.schema import GraphError, validate_graph

TOOL = sorted(W.READ_ONLY_TOOL_NAMES)[0]


def graph():
    return validate_graph({
        "name": "g",
        "nodes": [
            {"id": "worker_a", "role": "worker", "subtask": "look up", "system_prompt": "exact",
             "tools": [TOOL], "inputs": []},
            {"id": "worker_b", "role": "worker", "subtask": "cross-check", "system_prompt": "exact",
             "tools": [], "inputs": []},
            {"id": "integrator", "role": "integrator", "subtask": "plan", "system_prompt": "emit",
             "tools": [], "inputs": [{"port": "evidence_a", "from": "worker_a"},
                                     {"port": "evidence_b", "from": "worker_b"}]},
            {"id": "executor", "role": "executor", "subtask": "execute", "system_prompt": "",
             "tools": [], "inputs": [{"port": "plan", "from": "integrator"}]},
        ],
        "sink": "executor", "rationale": "test",
    })


def test_prune_drops_node_and_its_edges_but_never_the_executor():
    kept = prune(graph(), ["worker_b"])
    assert [node["id"] for node in kept["nodes"]] == ["worker_a", "integrator", "executor"]
    assert kept["nodes"][1]["inputs"] == [{"port": "evidence_a", "from": "worker_a"}]
    # Only nodes whose removal alone keeps the workflow valid are offered.
    assert discardable_nodes(graph()) == ["worker_a", "worker_b"]
    with pytest.raises(GraphError):
        prune(graph(), ["executor"])
    with pytest.raises(GraphError):  # an integrator without worker input is invalid
        prune(graph(), ["worker_a", "worker_b"])


def test_insert_is_offered_only_after_workers_or_critics():
    rows = insert_menu(graph())
    assert {(row["after"], row["before"]) for row in rows} == {
        ("worker_a", "integrator"), ("worker_b", "integrator")}
    assert all(row["new_id"] == "inserted_worker_0" for row in rows)


def test_insert_keeps_the_existing_edge_and_copies_the_source_tools():
    parent = graph()
    entry = next(row for row in insert_menu(parent) if row["after"] == "worker_a")
    child, applied = apply_insert(parent, entry, {"subtask": "find the missing record",
                                                  "system_prompt": "search", "rationale": "gap"})
    nodes = {node["id"]: node for node in child["nodes"]}
    assert nodes["inserted_worker_0"]["inputs"] == [{"port": "context_worker_a", "from": "worker_a"}]
    assert nodes["inserted_worker_0"]["tools"] == [TOOL]
    assert {"port": "evidence_a", "from": "worker_a"} in nodes["integrator"]["inputs"]
    assert nodes["integrator"]["inputs"][-1] == {"port": "context_inserted_worker_0",
                                                 "from": "inserted_worker_0"}
    assert applied["kind"] == "insert_worker"
    with pytest.raises(GraphError):
        apply_insert(parent, entry, {"subtask": "", "system_prompt": "search"})


def test_prune_then_insert_leaves_the_untouched_worker_outside_the_affected_region():
    # Why it matters: execution inheritance can reuse worker_a only outside the affected region.
    parent = graph()
    kept = prune(parent, ["worker_b"])
    entry = next(row for row in insert_menu(kept) if row["after"] == "worker_a")
    child, _ = apply_insert(kept, entry, {"subtask": "find", "system_prompt": "search"})
    assert dirty_nodes(parent, child) == {"inserted_worker_0", "integrator", "executor"}
