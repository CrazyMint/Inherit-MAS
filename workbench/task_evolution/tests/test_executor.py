from __future__ import annotations

import json

from task_evolution.controller import PublicTask
from task_evolution.executor import execute_graph
from task_evolution.models import ModelResult, Usage
from task_evolution.models import sanitized_tool_schemas
import wb_env as W
from task_evolution.snapshots import SnapshotStore
from task_evolution.tests.test_schema_edits import graph


class FakeModels:
    def __init__(self):
        self.calls = []

    def fingerprint(self, model):
        return {"model": model, "revision": "fake-v1"}

    def readonly_agent(self, *, role, system, user, tool_names, **kwargs):
        self.calls.append(role)
        event = {"tool": tool_names[0] if tool_names else "none", "args": {"q": "x"},
                 "observation": "FULL OBSERVATION BYTES", "error": ""}
        return ModelResult("worker report", [Usage("gpt-4o-mini", role, 10, 3)],
                           [{"role": "user", "content": user}], [event])

    def chat(self, *, model, role, system, user, max_tokens):
        self.calls.append(role)
        text = '{"actions":[]}' if role.startswith("integrator") else "critique"
        return ModelResult(text, [Usage(model, role, 20, 4)], [{"role": "user", "content": user}])


def test_dirty_recompute_reuses_clean_worker_and_rehydrates_artifacts(tmp_path):
    models = FakeModels()
    cache = SnapshotStore(tmp_path / "cache")
    parent = graph()
    task = PublicTask("t#1", "do a task")
    first = execute_graph(parent, task, models=models, cache=cache)
    assert set(models.calls) == {"worker:worker_a", "worker:worker_b", "integrator:integrator"}
    assert len(first.trace["artifact_manifest"]) == 2

    child = json.loads(json.dumps(parent))
    child["nodes"][0]["system_prompt"] = "changed prompt"
    models.calls.clear()
    second = execute_graph(child, task, models=models, cache=cache, parent_graph=parent, parent_snapshots=first.snapshots)
    assert "worker:worker_a" in models.calls
    assert "worker:worker_b" not in models.calls
    # The changed worker happened to produce the same bytes. The integrator is in the
    # affected region, but its resolved request dynamically reconverges and hits.
    assert "integrator:integrator" not in models.calls
    assert second.snapshots["worker_b"]["reused"] is True
    assert second.snapshots["integrator"]["reuse_reason"] == "exact_request"
    assert second.snapshots["worker_b"]["tool_events"][0]["observation"] == "FULL OBSERVATION BYTES"
    assert second.trace["reused_tokens"] == 37
    assert second.trace["live_tokens"] == 13
    assert second.snapshots["executor"]["reuse_reason"] == "write_executor_always_live"


def test_exact_request_cache_hits_on_identical_full_rerun(tmp_path):
    models = FakeModels()
    cache = SnapshotStore(tmp_path / "cache")
    task = PublicTask("t#1", "do a task")
    execute_graph(graph(), task, models=models, cache=cache)
    models.calls.clear()
    second = execute_graph(graph(), task, models=models, cache=cache)
    assert models.calls == []
    assert second.trace["reused_calls"] == 3
    assert second.trace["live_calls"] == 0


def test_wire_tool_names_are_sanitized_without_losing_identity():
    schemas, reverse = sanitized_tool_schemas([W.TOOL_BY_NAME["calendar.search_events"]])
    assert schemas[0]["function"]["name"] == "calendar_search_events"
    assert reverse["calendar_search_events"] == "calendar.search_events"
