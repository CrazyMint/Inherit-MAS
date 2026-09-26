"""WorkBench v1 task loader with separate public tasks and scoring outcomes.

Loads the pinned v1 tasks_and_outcomes and splits each row into:
  * a PUBLIC ``Task`` (task text + normalized domains + derived structure) — this is
    all a MAS/agent ever sees, and
  * a scorer-only ``gold_outcome`` (the gold action list) returned in a SEPARATE
    ``gold_by_id`` map that only the scorer receives.
This public/gold split structurally prevents gold leakage into any agent, exactly
as the HotpotQA port does (loader public projection).

A combined sha256 over the exact v1 CSVs loaded is exposed as ``dataset_pin()`` so
every run can record which bytes produced its tasks.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import os
import sys
from dataclasses import dataclass, field

import wb_env as W

csv.field_size_limit(sys.maxsize)

V1_DIR = os.path.join(W.VENDOR_ROOT, "data", "processed", "tasks_and_outcomes", "v1")
DOMAIN_FILES = {
    "email": "email_tasks_and_outcomes.csv",
    "calendar": "calendar_tasks_and_outcomes.csv",
    "analytics": "analytics_tasks_and_outcomes.csv",
    "project_management": "project_management_tasks_and_outcomes.csv",
    "customer_relationship_manager": "customer_relationship_manager_tasks_and_outcomes.csv",
    "multi_domain": "multi_domain_tasks_and_outcomes.csv",
}
# Deterministic file order for the dataset pin.
SOURCE_ORDER = [
    "analytics",
    "calendar",
    "customer_relationship_manager",
    "email",
    "multi_domain",
    "project_management",
]


@dataclass(frozen=True)
class Task:
    """PUBLIC view of a WorkBench task — no gold. Safe to hand to any agent."""

    id: str
    task: str
    domains: tuple[str, ...]  # normalized (crm -> customer_relationship_manager)
    source: str  # which v1 file it came from (a single domain, or "multi_domain")
    row: int
    n_gold_actions: int  # STRUCTURE ONLY (count), not the gold actions themselves
    base_template: str = field(default="", compare=False)
    chosen_template: str = field(default="", compare=False)

    @property
    def is_no_action(self) -> bool:
        return self.n_gold_actions == 0

    @property
    def is_multi_action(self) -> bool:
        return self.n_gold_actions > 1

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def is_multi_domain(self) -> bool:
        return self.n_domains > 1


def _load_file(source: str) -> list[tuple[Task, list[str]]]:
    path = os.path.join(V1_DIR, DOMAIN_FILES[source])
    out: list[tuple[Task, list[str]]] = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            gold = ast.literal_eval(r["outcome"])
            if not isinstance(gold, list):
                raise ValueError(f"{source} row {i}: gold outcome is not a list")
            domains = W.normalize_domains(ast.literal_eval(r["domains"]))
            task = Task(
                id=f"{source}#{i:04d}",
                task=r["task"],
                domains=tuple(domains),
                source=source,
                row=i,
                n_gold_actions=len(gold),
                base_template=r.get("base_template", ""),
                chosen_template=r.get("chosen_template", ""),
            )
            out.append((task, [str(a) for a in gold]))
    return out


def load_tasks(sources: list[str] | None = None) -> tuple[list[Task], dict[str, list[str]]]:
    """Return ``(public_tasks, gold_by_id)``. ``gold_by_id`` is SCORER-ONLY.

    ``sources`` selects which v1 files to load (default: all 6). Tasks come back in
    a deterministic order (SOURCE_ORDER, then row index)."""
    srcs = SOURCE_ORDER if sources is None else [s for s in SOURCE_ORDER if s in sources]
    tasks: list[Task] = []
    gold: dict[str, list[str]] = {}
    for s in srcs:
        for t, g in _load_file(s):
            tasks.append(t)
            gold[t.id] = g
    return tasks, gold


def dataset_pin() -> dict[str, str]:
    """A combined sha256 over the exact v1 CSVs loaded."""
    h = hashlib.sha256()
    per_file = {}
    for s in SOURCE_ORDER:
        p = os.path.join(V1_DIR, DOMAIN_FILES[s])
        b = open(p, "rb").read()
        fh = hashlib.sha256(b).hexdigest()
        per_file[s] = fh
        h.update(s.encode())
        h.update(fh.encode())
    return {"combined_sha256": h.hexdigest(), "per_file_sha256": per_file, "commit": W.PINNED_COMMIT}


if __name__ == "__main__":
    tasks, gold = load_tasks()
    print(f"loaded {len(tasks)} v1 tasks; pin={dataset_pin()['combined_sha256'][:16]}")
    by_src: dict[str, int] = {}
    for t in tasks:
        by_src[t.source] = by_src.get(t.source, 0) + 1
    print("by source:", by_src)
    print("no_action:", sum(t.is_no_action for t in tasks), "multi_action:", sum(t.is_multi_action for t in tasks))
