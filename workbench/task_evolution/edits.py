"""Typed graph edits and affected region computation."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import wb_env as W

from .schema import GraphError, canonical_graph, node_map, validate_graph

OPS = {"add_node", "remove_node", "update_node", "add_edge", "remove_edge", "reorder_inputs"}
REFINEMENT_OPS = OPS - {"add_node"}
PATCH_FIELDS = {"subtask", "system_prompt", "tools"}


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def node_identity(node: dict) -> str:
    payload = {
        "role": node["role"],
        "subtask": node.get("subtask", ""),
        "system_prompt": node.get("system_prompt", ""),
        "tools": node.get("tools", []),
        "inputs": node.get("inputs", []),
    }
    return hashlib.sha256(_canon(payload).encode()).hexdigest()


def validate_transaction(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) - {"base_graph_digest", "rationale", "operations"}:
        raise GraphError("edit transaction has unknown/missing fields")
    if not isinstance(value.get("base_graph_digest"), str):
        raise GraphError("base_graph_digest is required")
    if not isinstance(value.get("rationale"), str):
        raise GraphError("edit rationale is required")
    operations = value.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise GraphError("operations must contain exactly one atomic edit")
    for op in operations:
        if not isinstance(op, dict) or op.get("op") not in REFINEMENT_OPS:
            raise GraphError(f"unsupported operation {op!r}")
    return value


def apply_transaction(parent: dict, tx: dict) -> dict:
    from .schema import graph_digest

    validate_graph(parent)
    validate_transaction(tx)
    if tx["base_graph_digest"] != graph_digest(parent):
        raise GraphError("edit transaction targets a stale graph")
    child = copy.deepcopy(parent)
    for op in tx["operations"]:
        _apply_one(child, op)
    validate_graph(child)
    if canonical_graph(child) == canonical_graph(parent):
        raise GraphError("edit transaction is a no-op")
    return child


def _find(graph: dict, nid: str) -> dict:
    for node in graph["nodes"]:
        if node["id"] == nid:
            return node
    raise GraphError(f"unknown node {nid!r}")


def _apply_one(graph: dict, op: dict) -> None:
    kind = op["op"]
    if kind == "add_node":
        if set(op) != {"op", "node"} or not isinstance(op["node"], dict):
            raise GraphError("add_node requires node")
        graph["nodes"].append(copy.deepcopy(op["node"]))
    elif kind == "remove_node":
        if set(op) != {"op", "node_id"}:
            raise GraphError("remove_node requires node_id")
        if op["node_id"] == graph["sink"]:
            raise GraphError("cannot remove executor")
        _find(graph, op["node_id"])
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != op["node_id"]]
        for node in graph["nodes"]:
            node["inputs"] = [e for e in node.get("inputs", []) if e["from"] != op["node_id"]]
    elif kind == "update_node":
        if set(op) != {"op", "node_id", "patch"} or not isinstance(op["patch"], dict):
            raise GraphError("update_node requires node_id and patch")
        if not op["patch"] or set(op["patch"]) - PATCH_FIELDS:
            raise GraphError(f"update_node patch may contain only {sorted(PATCH_FIELDS)}")
        if len(op["patch"]) != 1:
            raise GraphError("update_node must change exactly one field")
        node = _find(graph, op["node_id"])
        if node["role"] == "executor":
            raise GraphError("executor is not mutable")
        node.update(copy.deepcopy(op["patch"]))
    elif kind == "add_edge":
        if set(op) != {"op", "from", "to", "port", "index"}:
            raise GraphError("add_edge requires from/to/port/index")
        _find(graph, op["from"])
        target = _find(graph, op["to"])
        index = op["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(target["inputs"]):
            raise GraphError("add_edge index out of range")
        target["inputs"].insert(index, {"port": op["port"], "from": op["from"]})
    elif kind == "remove_edge":
        if set(op) != {"op", "from", "to", "port"}:
            raise GraphError("remove_edge requires from/to/port")
        target = _find(graph, op["to"])
        before = len(target["inputs"])
        target["inputs"] = [e for e in target["inputs"] if not (e["from"] == op["from"] and e["port"] == op["port"])]
        if len(target["inputs"]) == before:
            raise GraphError("remove_edge did not match an edge")
    else:
        if set(op) != {"op", "node_id", "ports"}:
            raise GraphError("reorder_inputs requires node_id and ports")
        node = _find(graph, op["node_id"])
        ports = op["ports"]
        current = {e["port"]: e for e in node["inputs"]}
        if not isinstance(ports, list) or set(ports) != set(current) or len(ports) != len(current):
            raise GraphError("reorder_inputs must name every current port once")
        node["inputs"] = [current[p] for p in ports]


def dirty_nodes(parent: dict, child: dict) -> set[str]:
    """Changed/new child nodes plus all child descendants, using ordered labeled inputs."""
    validate_graph(parent)
    validate_graph(child)
    pnodes, cnodes = node_map(parent), node_map(child)
    dirty = {nid for nid in cnodes if nid not in pnodes or node_identity(cnodes[nid]) != node_identity(pnodes[nid])}
    children = {nid: set() for nid in cnodes}
    for node in cnodes.values():
        for edge in node["inputs"]:
            children[edge["from"]].add(node["id"])
    queue = list(dirty)
    while queue:
        for downstream in children[queue.pop()]:
            if downstream not in dirty:
                dirty.add(downstream)
                queue.append(downstream)
    return dirty


def _menu_digest(entries: list[dict], base_graph_digest: str) -> str:
    return hashlib.sha256(_canon({"base_graph_digest": base_graph_digest, "entries": entries}).encode()).hexdigest()


def legal_edit_menu(graph: dict) -> dict:
    """Enumerate deterministic, graph-valid atomic edit choices."""
    from .schema import graph_digest

    validate_graph(graph)
    base_digest = graph_digest(graph)
    entries: list[dict] = []
    nodes = node_map(graph)

    for node in graph["nodes"]:
        if node["role"] == "executor":
            continue
        entries.append({"kind": "replace_system_prompt", "node_id": node["id"],
                        "replacement_type": "nonempty_string"})
        entries.append({"kind": "replace_subtask", "node_id": node["id"],
                        "replacement_type": "nonempty_string"})

    def add_concrete(kind: str, operation: dict) -> None:
        tx = {"base_graph_digest": base_digest, "rationale": "menu validation", "operations": [operation]}
        try:
            apply_transaction(graph, tx)
        except GraphError:
            return
        entries.append({"kind": kind, "operation": operation})

    for node in graph["nodes"]:
        if node["role"] != "worker":
            continue
        current_tools = list(node["tools"])
        for tool in current_tools:
            add_concrete("remove_worker_tool", {
                "op": "update_node", "node_id": node["id"],
                "patch": {"tools": [name for name in current_tools if name != tool]},
            })
        domains = {tool.split(".", 1)[0] for tool in current_tools}
        for tool in sorted(W.READ_ONLY_TOOL_NAMES):
            if tool not in current_tools and tool.split(".", 1)[0] in domains:
                add_concrete("add_worker_tool", {
                    "op": "update_node", "node_id": node["id"],
                    "patch": {"tools": current_tools + [tool]},
                })

    for node in graph["nodes"]:
        if node["role"] != "executor":
            add_concrete("remove_node", {"op": "remove_node", "node_id": node["id"]})
        if len(node["inputs"]) > 1:
            add_concrete("reverse_inputs", {
                "op": "reorder_inputs", "node_id": node["id"],
                "ports": [edge["port"] for edge in reversed(node["inputs"])],
            })
        for edge in node["inputs"]:
            add_concrete("remove_edge", {
                "op": "remove_edge", "from": edge["from"], "to": node["id"], "port": edge["port"],
            })

    for source in graph["nodes"]:
        if source["role"] == "executor":
            continue
        for target in graph["nodes"]:
            if target["role"] == "executor" or source["id"] == target["id"]:
                continue
            if any(edge["from"] == source["id"] for edge in target["inputs"]):
                continue
            existing_ports = {edge["port"] for edge in target["inputs"]}
            port = f"context_{source['id']}"
            if port in existing_ports:
                continue
            add_concrete("add_edge", {
                "op": "add_edge", "from": source["id"], "to": target["id"],
                "port": port, "index": len(target["inputs"]),
            })

    entries.sort(key=_canon)
    indexed = [{"edit_index": index, **entry} for index, entry in enumerate(entries)]
    if not indexed:
        raise GraphError("no legal atomic edits for graph")
    return {"base_graph_digest": base_digest, "entries": indexed,
            "menu_digest": _menu_digest(indexed, base_digest)}


def materialize_menu_choice(graph: dict, menu: dict, value: Any) -> tuple[dict, dict]:
    """Validate an indexed model choice and return (normalized choice, transaction)."""
    from .schema import graph_digest

    validate_graph(graph)
    if menu != legal_edit_menu(graph):
        raise GraphError("legal edit menu is stale or malformed")
    allowed = {"base_graph_digest", "menu_digest", "edit_index", "rationale", "replacement"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise GraphError("menu choice has unknown fields")
    if value.get("base_graph_digest") != graph_digest(graph) or value.get("menu_digest") != menu["menu_digest"]:
        raise GraphError("menu choice targets a stale graph/menu")
    index = value.get("edit_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(menu["entries"]):
        raise GraphError("edit_index is outside the legal menu")
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        raise GraphError("menu choice rationale must be non-empty")
    entry = menu["entries"][index]
    if entry["edit_index"] != index:
        raise GraphError("menu index mismatch")
    if entry["kind"] in {"replace_system_prompt", "replace_subtask"}:
        replacement = value.get("replacement")
        if not isinstance(replacement, str) or not replacement.strip():
            raise GraphError("selected text edit requires a non-empty replacement")
        field = "system_prompt" if entry["kind"] == "replace_system_prompt" else "subtask"
        operation = {"op": "update_node", "node_id": entry["node_id"], "patch": {field: replacement}}
    else:
        # `replacement` is semantically irrelevant for a fully concrete menu
        # entry. Ignore it rather than rejecting an otherwise valid selection.
        operation = copy.deepcopy(entry["operation"])
    tx = {"base_graph_digest": menu["base_graph_digest"], "rationale": value["rationale"],
          "operations": [operation]}
    apply_transaction(graph, tx)
    choice = {key: value[key] for key in ("base_graph_digest", "menu_digest", "edit_index", "rationale")}
    if "replacement" in value and entry["kind"] in {"replace_system_prompt", "replace_subtask"}:
        choice["replacement"] = value["replacement"]
    return choice, tx
