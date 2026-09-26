"""HotpotQA adapter over the benchmark-neutral Inherit-MAS controller."""
from __future__ import annotations

from pathlib import Path

from inherit_mas.adapters import HotpotQAAdapter
from inherit_mas.core import EvolutionHooks, run_evolution

from . import config
from .common import atomic_json, digest
from .edits import apply_menu_edit, legal_edit_menu
from .executor import SnapshotStore, execute_graph
from .graph import graph_digest, validate_graph

CONDITION = "inherit_mas_v3"
MAX_CANDIDATES = 5
JUDGE_MAX_TOKENS = 500
REFINER_MAX_TOKENS = 900
ADAPTER = HotpotQAAdapter()


def run_condition(example, *, models, retriever, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)

    def execute(graph, task, cache, parent_graph, parent_snapshots):
        return execute_graph(graph, task, models=models, retriever=retriever,
                             cache=cache, parent_graph=parent_graph,
                             parent_snapshots=parent_snapshots)

    hooks = EvolutionHooks(
        validate_graph=validate_graph,
        graph_digest=graph_digest,
        legal_edit_menu=legal_edit_menu,
        apply_menu_edit=apply_menu_edit,
        create_cache=lambda: SnapshotStore(root / "trajectory_cache"),
        execute=execute,
    )
    record = run_evolution(
        example, adapter=ADAPTER, hooks=hooks, models=models,
        max_candidates=MAX_CANDIDATES,
        meta_max_tokens=config.META_MAX_TOKENS,
        judge_max_tokens=JUDGE_MAX_TOKENS,
        refiner_max_tokens=REFINER_MAX_TOKENS,
    )
    record["condition"] = CONDITION
    record["record_digest"] = digest(record)
    atomic_json(root / "record.json", record)
    return record


__all__ = ["ADAPTER", "CONDITION", "JUDGE_MAX_TOKENS", "MAX_CANDIDATES",
           "REFINER_MAX_TOKENS", "run_condition"]
