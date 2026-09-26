"""WorkBench bridge for the Select/Edit controller."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from inherit_mas.core import EvolutionHooks
from inherit_mas.select_edit import ComponentHooks, run_select_edit

from .controller import PublicTask
from .executor import execute_graph
from .general_v1 import (
    ADAPTER,
    JUDGE_MAX_TOKENS,
    MAX_CANDIDATES,
    META_MAX_TOKENS,
    REFINER_MAX_TOKENS,
    ModelsBridge,
    apply_generic_edit,
    generic_menu,
)
from .schema import GraphError, graph_digest, node_map, validate_graph
from .snapshots import SnapshotStore


CONDITION = "inherit_mas_select_edit_v1"
# Worker inputs must be peer reports or critiques (schema.validate_graph).
INSERT_SOURCE_ROLES = {"worker", "critic"}


def discardable_nodes(graph: dict) -> list[str]:
    """Nodes whose removal alone leaves a valid workflow (prevalidated, like the edit menu)."""

    rows = []
    for node in graph["nodes"]:
        if node["role"] == "executor":
            continue
        try:
            prune(graph, [node["id"]])
        except GraphError:
            continue
        rows.append(node["id"])
    return rows


def prune(graph: dict, discard: list[str]) -> dict:
    validate_graph(graph)
    nodes = node_map(graph)
    for nid in discard:
        if nid not in nodes or nodes[nid]["role"] == "executor":
            raise GraphError(f"cannot discard {nid!r}")
    gone = set(discard)
    child = copy.deepcopy(graph)
    child["nodes"] = [node for node in child["nodes"] if node["id"] not in gone]
    for node in child["nodes"]:
        node["inputs"] = [edge for edge in node.get("inputs", []) if edge["from"] not in gone]
    return validate_graph(child)


def _new_id(graph: dict) -> str:
    ids = {node["id"] for node in graph["nodes"]}
    index = 0
    while f"inserted_worker_{index}" in ids:
        index += 1
    return f"inserted_worker_{index}"


def _inserted(graph: dict, entry: dict, subtask: str, prompt: str) -> dict:
    child = copy.deepcopy(graph)
    child["nodes"].append({
        "id": entry["new_id"], "role": "worker", "subtask": subtask,
        "system_prompt": prompt, "tools": list(entry["tools"]),
        "inputs": [{"port": f"context_{entry['after']}", "from": entry["after"]}],
    })
    node_map(child)[entry["before"]]["inputs"].append(
        {"port": f"context_{entry['new_id']}", "from": entry["new_id"]})
    return validate_graph(child)


def insert_menu(graph: dict) -> list[dict]:
    """Insert a worker on an existing edge u->v, keeping that edge (u->new->v)."""

    validate_graph(graph)
    nodes = node_map(graph)
    new_id = _new_id(graph)
    rows = []
    for target in graph["nodes"]:
        for edge in target["inputs"]:
            source = nodes[edge["from"]]
            if source["role"] not in INSERT_SOURCE_ROLES:
                continue
            entry = {"op": "insert_worker", "target": target["id"], "after": source["id"],
                     "before": target["id"], "new_id": new_id,
                     "tools": list(source["tools"]), "needs": "subtask_and_prompt"}
            try:
                _inserted(graph, entry, "placeholder", "placeholder")
            except GraphError:
                continue
            rows.append(entry)
    return rows


def apply_insert(graph: dict, entry: dict, value: Any) -> tuple[dict, dict]:
    subtask, prompt = value.get("subtask"), value.get("system_prompt")
    if not isinstance(subtask, str) or not subtask.strip():
        raise GraphError("insert_worker requires a non-empty subtask")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GraphError("insert_worker requires a non-empty system_prompt")
    child = _inserted(graph, entry, subtask.strip(), prompt.strip())
    choice = {key: entry[key] for key in ("op", "after", "before", "new_id", "tools")}
    return child, {"choice": choice, "kind": "insert_worker",
                   "rationale": str(value.get("rationale", ""))}


COMPONENTS = ComponentHooks(discardable_nodes=discardable_nodes, prune=prune,
                            insert_menu=insert_menu, apply_insert=apply_insert)


def run_condition(task: PublicTask, *, models, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    bridge = ModelsBridge(models)

    def execute(graph, public_task, cache, parent_graph, parent_snapshots):
        return execute_graph(graph, public_task, models=bridge, cache=cache,
                             parent_graph=parent_graph, parent_snapshots=parent_snapshots)

    hooks = EvolutionHooks(
        validate_graph=validate_graph,
        graph_digest=graph_digest,
        legal_edit_menu=generic_menu,
        apply_menu_edit=apply_generic_edit,
        create_cache=lambda: SnapshotStore(root / "node_cache"),
        execute=execute,
    )
    record = run_select_edit(
        task, adapter=ADAPTER, hooks=hooks, components=COMPONENTS, models=bridge,
        max_candidates=MAX_CANDIDATES, meta_max_tokens=META_MAX_TOKENS,
        judge_max_tokens=JUDGE_MAX_TOKENS, refiner_max_tokens=REFINER_MAX_TOKENS,
    )
    record["condition"] = CONDITION
    return record


__all__ = ["COMPONENTS", "CONDITION", "apply_insert", "discardable_nodes",
           "insert_menu", "prune", "run_condition"]
