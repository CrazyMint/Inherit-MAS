"""WorkBench adapter over the benchmark-neutral Inherit-MAS controller."""
from __future__ import annotations

from pathlib import Path

from inherit_mas.adapters import WorkBenchAdapter
from inherit_mas.core import EvolutionHooks, run_evolution

from .controller import PublicTask
from .edits import (apply_transaction, legal_edit_menu, materialize_menu_choice)
from .executor import execute_graph
from .schema import graph_digest, validate_graph
from .snapshots import SnapshotStore
import wb_env as W


CONDITION = "inherit_mas_v1"
MAX_CANDIDATES = 5
META_MAX_TOKENS = 5000
JUDGE_MAX_TOKENS = 3000
REFINER_MAX_TOKENS = 5000
ADAPTER = WorkBenchAdapter()
ADAPTER.interface_contract = (
    ADAPTER.interface_contract
    + "\n\nDECLARED READ-ONLY TOOLS (use exact qualified names):\n"
    + "\n".join(f"- {tool.name}: {tool.signature_str} -- {tool.description}"
                for tool in W.READ_ONLY_TOOLS)
    + "\n\nDECLARED WRITE ACTION SCHEMAS (integrator/verifier output only; executor applies):\n"
    + W.write_tool_schema_block()
)


class ModelsBridge:
    """Expose the generic structured-call surface over the model client."""

    def __init__(self, inner):
        self.inner = inner

    def structured(self, *, role: str, system: str, user: str, max_tokens: int):
        return self.inner.chat(model="gpt-5.4-mini", role=role, system=system,
                               user=user, max_tokens=max_tokens)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def generic_menu(graph: dict) -> list[dict]:
    menu = legal_edit_menu(graph)
    rows = []
    for entry in menu["entries"]:
        row = dict(entry)
        kind = entry["kind"]
        if kind == "replace_system_prompt":
            row.update({"op": "edit_prompt", "target": entry["node_id"],
                        "needs": "replacement"})
        elif kind == "replace_subtask":
            row.update({"op": "edit_subtask", "target": entry["node_id"],
                        "needs": "replacement"})
        else:
            operation = entry.get("operation", {})
            row.update({"op": kind,
                        "target": operation.get("node_id", operation.get("to", "")),
                        "needs": "none"})
        rows.append(row)
    return rows


def apply_generic_edit(graph: dict, _shown_menu: list[dict], value: dict) -> tuple[dict, dict]:
    native = legal_edit_menu(graph)
    proposal = {
        "base_graph_digest": native["base_graph_digest"],
        "menu_digest": native["menu_digest"],
        "edit_index": value["edit_index"],
        "rationale": value.get("rationale", "generic atomic edit"),
    }
    if "replacement" in value:
        proposal["replacement"] = value["replacement"]
    choice, transaction = materialize_menu_choice(graph, native, proposal)
    child = apply_transaction(graph, transaction)
    entry = native["entries"][choice["edit_index"]]
    return child, {"choice": choice, "kind": entry["kind"],
                   "transaction": transaction}


def run_condition(task: PublicTask, *, models, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    bridge = ModelsBridge(models)

    def execute(graph, public_task, cache, parent_graph, parent_snapshots):
        return execute_graph(graph, public_task, models=bridge, cache=cache,
                             parent_graph=parent_graph,
                             parent_snapshots=parent_snapshots)

    hooks = EvolutionHooks(
        validate_graph=validate_graph,
        graph_digest=graph_digest,
        legal_edit_menu=generic_menu,
        apply_menu_edit=apply_generic_edit,
        create_cache=lambda: SnapshotStore(root / "node_cache"),
        execute=execute,
    )
    record = run_evolution(
        task, adapter=ADAPTER, hooks=hooks, models=bridge,
        max_candidates=MAX_CANDIDATES, meta_max_tokens=META_MAX_TOKENS,
        judge_max_tokens=JUDGE_MAX_TOKENS, refiner_max_tokens=REFINER_MAX_TOKENS,
    )
    record["condition"] = CONDITION
    return record


__all__ = ["ADAPTER", "CONDITION", "MAX_CANDIDATES", "ModelsBridge",
           "apply_generic_edit", "generic_menu", "run_condition"]
