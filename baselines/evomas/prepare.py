"""Derive native WorkBench inputs from the pinned, bundled benchmark tables."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "workbench/vendor/workbench_upstream/data/processed"
ENVIRONMENT = {"analytics": "analytics_data.csv", "calendar": "calendar_events.csv",
               "customer_relationship_manager": "customer_relationship_manager_data.csv",
               "email": "emails.csv", "project_management": "project_tasks.csv"}
TAGS = {"analytics": "Analytics", "calendar": "Calendar", "customer_relationship_manager": "CRM",
        "email": "Email", "project_management": "ProjectManagement", "multi_domain": "MultiDomain"}


def source_identity() -> dict:
    paths = [Path(__file__), *(DATA / filename for filename in ENVIRONMENT.values())]
    paths.extend(DATA / "tasks_and_outcomes/v1" / f"{domain}_tasks_and_outcomes.csv" for domain in TAGS)
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def prepare_workbench(upstream: Path) -> None:
    for domain, tag in TAGS.items():
        target = upstream / "dataset/workbench" / domain
        target.mkdir(parents=True, exist_ok=True)
        with (DATA / "tasks_and_outcomes/v1" / f"{domain}_tasks_and_outcomes.csv").open(newline="") as handle:
            rows = [{"id": index, "query": row["task"], "gt": "__SEALED_OFFLINE__",
                     "tag": [f"WorkBench-{tag}"], "source": "WorkBench"}
                    for index, row in enumerate(csv.DictReader(handle))]
        atomic_json(target / "test.json", rows)
        if domain in ENVIRONMENT:
            shutil.copyfile(DATA / ENVIRONMENT[domain], target / "data.csv")
