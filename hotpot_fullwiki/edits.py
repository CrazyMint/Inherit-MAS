"""Atomic typed edits and conservative affected region propagation."""
from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Any

from .graph import (ACCEPTS, EMITS, MAX_NODES, GraphError, canonical_node,
                    node_map, validate_graph)


def dirty_nodes(parent: dict, child: dict) -> set[str]:
    validate_graph(parent)
    validate_graph(child)
    old, new = node_map(parent), node_map(child)
    dirty = {nid for nid in new if nid not in old or canonical_node(new[nid]) != canonical_node(old[nid])}
    children: dict[str, list[str]] = defaultdict(list)
    for node in child["nodes"]:
        for edge in node["inputs"]:
            children[edge["from"]].append(node["id"])
    queue = deque(dirty)
    while queue:
        nid = queue.popleft()
        for child_id in children[nid]:
            if child_id not in dirty:
                dirty.add(child_id)
                queue.append(child_id)
    return dirty


def _valid(graph: dict) -> bool:
    try:
        validate_graph(graph)
        return True
    except GraphError:
        return False


def legal_edit_menu(graph: dict) -> list[dict]:
    validate_graph(graph)
    nodes = node_map(graph)
    menu: list[dict] = []
    for node in graph["nodes"]:
        menu.append({"op": "edit_prompt", "target": node["id"], "needs": "replacement"})
        menu.append({"op": "edit_subtask", "target": node["id"], "needs": "replacement"})
        if node["role"] != "planner":
            candidate = copy.deepcopy(graph)
            tools = node_map(candidate)[node["id"]]["tools"]
            node_map(candidate)[node["id"]]["tools"] = [] if tools else ["search_fullwiki"]
            if _valid(candidate):
                menu.append({"op": "toggle_search", "target": node["id"], "needs": "none"})
    for source, source_node in nodes.items():
        port = EMITS[source_node["role"]]
        for target, target_node in nodes.items():
            if source == target or port not in ACCEPTS[target_node["role"]]:
                continue
            if any(edge["from"] == source for edge in target_node["inputs"]):
                continue
            candidate = copy.deepcopy(graph)
            node_map(candidate)[target]["inputs"].append({"from": source, "port": port})
            if _valid(candidate):
                menu.append({"op": "add_edge", "source": source, "target": target,
                             "port": port, "needs": "none"})
    for node in graph["nodes"]:
        for edge in node["inputs"]:
            candidate = copy.deepcopy(graph)
            node_map(candidate)[node["id"]]["inputs"].remove(edge)
            if _valid(candidate):
                menu.append({"op": "remove_edge", "source": edge["from"],
                             "target": node["id"], "port": edge["port"], "needs": "none"})
    for node in graph["nodes"]:
        if node["id"] == graph["output"]:
            continue
        candidate = copy.deepcopy(graph)
        candidate["nodes"] = [row for row in candidate["nodes"] if row["id"] != node["id"]]
        for other in candidate["nodes"]:
            other["inputs"] = [edge for edge in other["inputs"] if edge["from"] != node["id"]]
        if _valid(candidate):
            menu.append({"op": "prune_node", "target": node["id"], "needs": "none"})
    if len(graph["nodes"]) < MAX_NODES:
        index = 0
        while f"researcher_{index}" in nodes:
            index += 1
        menu.append({"op": "add_researcher", "new_id": f"researcher_{index}",
                     "target": graph["output"], "needs": "subtask_and_prompt"})
    return [{"edit_index": index, **row} for index, row in enumerate(menu)]


def apply_menu_edit(graph: dict, menu: list[dict], proposal: Any) -> tuple[dict, dict]:
    if not isinstance(proposal, dict) or type(proposal.get("edit_index")) is not int:
        raise GraphError("proposal needs an integer edit_index")
    index = proposal["edit_index"]
    if not 0 <= index < len(menu):
        raise GraphError("edit_index outside current menu")
    edit = menu[index]
    child = copy.deepcopy(graph)
    nodes = node_map(child)
    op = edit["op"]
    if op in {"edit_prompt", "edit_subtask"}:
        replacement = proposal.get("replacement")
        if not isinstance(replacement, str) or not replacement.strip():
            raise GraphError(f"{op} requires replacement")
        field = "system_prompt" if op == "edit_prompt" else "subtask"
        if replacement.strip() == nodes[edit["target"]][field].strip():
            raise GraphError("edit is a no-op")
        nodes[edit["target"]][field] = replacement.strip()
    elif op == "toggle_search":
        tools = nodes[edit["target"]]["tools"]
        nodes[edit["target"]]["tools"] = [] if tools else ["search_fullwiki"]
    elif op == "add_edge":
        nodes[edit["target"]]["inputs"].append({"from": edit["source"], "port": edit["port"]})
    elif op == "remove_edge":
        nodes[edit["target"]]["inputs"].remove({"from": edit["source"], "port": edit["port"]})
    elif op == "prune_node":
        child["nodes"] = [node for node in child["nodes"] if node["id"] != edit["target"]]
        for node in child["nodes"]:
            node["inputs"] = [edge for edge in node["inputs"] if edge["from"] != edit["target"]]
    elif op == "add_researcher":
        subtask, prompt = proposal.get("subtask"), proposal.get("system_prompt")
        if not isinstance(subtask, str) or not subtask.strip() or not isinstance(prompt, str) or not prompt.strip():
            raise GraphError("add_researcher requires subtask and system_prompt")
        child["nodes"].append({"id": edit["new_id"], "role": "researcher",
                               "subtask": subtask.strip(), "system_prompt": prompt.strip(),
                               "tools": ["search_fullwiki"], "inputs": []})
        node_map(child)[edit["target"]]["inputs"].append(
            {"from": edit["new_id"], "port": "evidence"})
    else:
        raise GraphError(f"unsupported op {op}")
    validate_graph(child)
    return child, {"choice": edit, "rationale": str(proposal.get("rationale", ""))}
