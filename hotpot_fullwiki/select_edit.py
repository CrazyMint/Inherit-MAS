"""HotpotQA bridge for the Select/Edit controller."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from inherit_mas.core import EvolutionHooks
from inherit_mas.select_edit import ComponentHooks, run_select_edit

from . import config
from .common import atomic_json, digest
from .edits import apply_menu_edit, legal_edit_menu
from .executor import SnapshotStore, execute_graph
from .graph import ACCEPTS, EMITS, GraphError, graph_digest, node_map, validate_graph
from .revision_v3 import ADAPTER, JUDGE_MAX_TOKENS, MAX_CANDIDATES, REFINER_MAX_TOKENS


CONDITION = "inherit_mas_select_edit_v1"


def discardable_nodes(graph: dict) -> list[str]:
    """Nodes whose removal alone leaves a valid workflow (prevalidated, like the edit menu)."""

    rows = []
    for node in graph["nodes"]:
        if node["id"] == graph["output"]:
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
        if nid not in nodes or nid == graph["output"]:
            raise GraphError(f"cannot discard {nid!r}")
    gone = set(discard)
    child = copy.deepcopy(graph)
    child["nodes"] = [node for node in child["nodes"] if node["id"] not in gone]
    for node in child["nodes"]:
        node["inputs"] = [edge for edge in node["inputs"] if edge["from"] not in gone]
    return validate_graph(child)


def _new_id(graph: dict) -> str:
    ids = {node["id"] for node in graph["nodes"]}
    index = 0
    while f"researcher_{index}" in ids:
        index += 1
    return f"researcher_{index}"


def _inserted(graph: dict, entry: dict, subtask: str, prompt: str) -> dict:
    child = copy.deepcopy(graph)
    nodes = node_map(child)
    child["nodes"].append({
        "id": entry["new_id"], "role": "researcher", "subtask": subtask,
        "system_prompt": prompt, "tools": ["search_fullwiki"],
        "inputs": [{"from": entry["after"], "port": EMITS[nodes[entry["after"]]["role"]]}],
    })
    node_map(child)[entry["before"]]["inputs"].append(
        {"from": entry["new_id"], "port": EMITS["researcher"]})
    return validate_graph(child)


def insert_menu(graph: dict) -> list[dict]:
    """Insert a researcher on an existing edge u->v, keeping that edge (u->new->v)."""

    validate_graph(graph)
    nodes = node_map(graph)
    new_id = _new_id(graph)
    rows = []
    for target in graph["nodes"]:
        for edge in target["inputs"]:
            source = nodes[edge["from"]]
            if EMITS[source["role"]] not in ACCEPTS["researcher"]:
                continue
            entry = {"op": "insert_researcher", "target": target["id"],
                     "after": source["id"], "before": target["id"], "new_id": new_id,
                     "needs": "subtask_and_prompt"}
            try:
                _inserted(graph, entry, "placeholder", "placeholder")
            except GraphError:
                continue
            rows.append(entry)
    return rows


def apply_insert(graph: dict, entry: dict, value: Any) -> tuple[dict, dict]:
    subtask, prompt = value.get("subtask"), value.get("system_prompt")
    if not isinstance(subtask, str) or not subtask.strip():
        raise GraphError("insert_researcher requires a non-empty subtask")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GraphError("insert_researcher requires a non-empty system_prompt")
    child = _inserted(graph, entry, subtask.strip(), prompt.strip())
    choice = {key: entry[key] for key in ("op", "target", "after", "before", "new_id", "needs")}
    return child, {"choice": choice, "rationale": str(value.get("rationale", ""))}


COMPONENTS = ComponentHooks(discardable_nodes=discardable_nodes, prune=prune,
                            insert_menu=insert_menu, apply_insert=apply_insert)


def run_condition(example, *, models, retriever, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)

    def execute(graph, task, cache, parent_graph, parent_snapshots):
        return execute_graph(
            graph, task, models=models, retriever=retriever, cache=cache,
            parent_graph=parent_graph, parent_snapshots=parent_snapshots)

    hooks = EvolutionHooks(
        validate_graph=validate_graph,
        graph_digest=graph_digest,
        legal_edit_menu=legal_edit_menu,
        apply_menu_edit=apply_menu_edit,
        create_cache=lambda: SnapshotStore(root / "trajectory_cache"),
        execute=execute,
    )
    record = run_select_edit(
        example, adapter=ADAPTER, hooks=hooks, components=COMPONENTS, models=models,
        max_candidates=MAX_CANDIDATES, meta_max_tokens=config.META_MAX_TOKENS,
        judge_max_tokens=JUDGE_MAX_TOKENS, refiner_max_tokens=REFINER_MAX_TOKENS,
    )
    record["condition"] = CONDITION
    record["record_digest"] = digest(record)
    atomic_json(root / "record.json", record)
    return record


__all__ = ["COMPONENTS", "CONDITION", "apply_insert", "discardable_nodes",
           "insert_menu", "prune", "run_condition"]
