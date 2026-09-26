import pytest

from hotpot_fullwiki.graph import GraphError, validate_graph
from hotpot_fullwiki.select_edit import apply_insert, discardable_nodes, insert_menu, prune


def graph():
    return validate_graph({
        "nodes": [
            {"id": "planner", "role": "planner", "subtask": "split hops",
             "system_prompt": "plan", "tools": [], "inputs": []},
            {"id": "researcher_a", "role": "researcher", "subtask": "hop one",
             "system_prompt": "search", "tools": ["search_fullwiki"],
             "inputs": [{"from": "planner", "port": "plan"}]},
            {"id": "researcher_b", "role": "researcher", "subtask": "hop two",
             "system_prompt": "search", "tools": ["search_fullwiki"],
             "inputs": [{"from": "planner", "port": "plan"}]},
            {"id": "answerer", "role": "synthesizer", "subtask": "answer",
             "system_prompt": "cite", "tools": [],
             "inputs": [{"from": "researcher_a", "port": "evidence"},
                        {"from": "researcher_b", "port": "evidence"}]},
        ],
        "output": "answerer",
    })


def test_prune_keeps_a_valid_graph_and_never_drops_the_output():
    kept = prune(graph(), ["researcher_b"])
    assert [node["id"] for node in kept["nodes"]] == ["planner", "researcher_a", "answerer"]
    # Only nodes whose removal alone keeps the workflow valid are offered.
    assert discardable_nodes(graph()) == ["planner", "researcher_a", "researcher_b"]
    two_node = prune(graph(), ["planner", "researcher_b"])
    assert discardable_nodes(two_node) == []  # the only researcher must stay
    with pytest.raises(GraphError):
        prune(graph(), ["answerer"])
    with pytest.raises(GraphError):  # at least one retrieval-capable researcher must remain
        prune(graph(), ["researcher_a", "researcher_b"])


def test_insert_uses_typed_ports_and_keeps_the_existing_edge():
    parent = graph()
    rows = insert_menu(parent)
    assert {(row["after"], row["before"]) for row in rows} == {
        ("planner", "researcher_a"), ("planner", "researcher_b"),
        ("researcher_a", "answerer"), ("researcher_b", "answerer")}
    entry = next(row for row in rows if row["after"] == "researcher_a")
    child, applied = apply_insert(parent, entry, {"subtask": "find the bridge fact",
                                                  "system_prompt": "search", "rationale": "gap"})
    nodes = {node["id"]: node for node in child["nodes"]}
    assert nodes["researcher_0"]["inputs"] == [{"from": "researcher_a", "port": "evidence"}]
    assert nodes["researcher_0"]["tools"] == ["search_fullwiki"]
    assert nodes["answerer"]["inputs"] == [
        {"from": "researcher_a", "port": "evidence"},
        {"from": "researcher_b", "port": "evidence"},
        {"from": "researcher_0", "port": "evidence"}]
    assert applied["choice"]["op"] == "insert_researcher"
    planner_entry = next(row for row in rows if row["after"] == "planner")
    child, _ = apply_insert(parent, planner_entry, {"subtask": "s", "system_prompt": "p"})
    assert {node["id"]: node for node in child["nodes"]}["researcher_0"]["inputs"] == [
        {"from": "planner", "port": "plan"}]
    with pytest.raises(GraphError):
        apply_insert(parent, entry, {"subtask": "s"})
