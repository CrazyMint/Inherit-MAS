"""Offline method fidelity, immutable resume, and official scoring regressions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baselines.evoagent import runner as evoagent
from baselines.single_react import _runner, runner as single
from hotpot_fullwiki.common import atomic_json, digest
from hotpot_fullwiki.models import BudgetExhausted, ModelResult
from public_runner import common, models


@pytest.fixture(autouse=True)
def offline_identity(monkeypatch):
    monkeypatch.setattr(common, "source_identity", lambda: {"source.py": "fixed"})
    monkeypatch.setattr(models, "model_settings", lambda backbone, need_meta=True, **kwargs: {
        "worker_model": backbone, "meta_model": "gpt-5.4-mini" if need_meta else backbone})


def manifest_file(tmp_path, benchmark="workbench", rows=None):
    rows = rows or [{"id": "task#0", "task": "Public task", "gold": "must not leak"}]
    value = {"n": len(rows), "tasks" if benchmark == "workbench" else "rows": rows,
             "dataset_revision": "snapshot"}
    value["manifest_digest"] = digest(value)
    path = tmp_path / "manifest.json"
    atomic_json(path, value)
    return path


def fake_runtime(monkeypatch, fn=None):
    calls = []

    def prepare(method, benchmark, manifest, rows, out, backbone, ledger):
        def execute(row):
            calls.append(row["id"])
            return fn(row) if fn else {"task_id": row["id"],
                "prediction": [] if benchmark == "workbench" else {"answer": "Paris", "supporting_facts": [["France", 0]]},
                "usage": {"calls": 1, "total_tokens": 10, "estimated_usd": 0.2}}
        return execute
    monkeypatch.setattr(_runner, "_prepare", prepare)
    return calls


@pytest.mark.parametrize("runner", [single, evoagent])
@pytest.mark.parametrize("benchmark", ["workbench", "hotpotqa"])
@pytest.mark.parametrize("backbone", ["gpt-4o-mini", "qwen3-32b"])
def test_run_resume_all_configs(tmp_path, monkeypatch, runner, benchmark, backbone):
    path = manifest_file(tmp_path, benchmark)
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    result = runner.run(path, out, usd_cap=1, workers=1, backbone=backbone)
    assert result["status"] == "sealed"
    assert result["usage"]["total_tokens"] == 10
    assert calls == ["task#0"]
    monkeypatch.setattr(_runner, "_prepare", lambda *args: pytest.fail("resume cannot create clients"))
    assert runner.run(path, out, usd_cap=1, workers=1, backbone=backbone)["status"] == "sealed"
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.run(path, out, usd_cap=2, workers=1, backbone=backbone)


@pytest.mark.parametrize("runner", [single, evoagent])
@pytest.mark.parametrize("failure,status", [("content_filter denied", "sealed"),
    ("context_length exceeded", "sealed"), ("request_timeout", "sealed"),
    ("connection failed", "incomplete")])
def test_failure_classification(tmp_path, monkeypatch, runner, failure, status):
    def fail(row):
        raise RuntimeError(failure)
    fake_runtime(monkeypatch, fail)
    out = tmp_path / "run"
    assert runner.run(manifest_file(tmp_path), out, usd_cap=1, workers=1)["status"] == status
    assert (out / "SEALED.json").exists() == (status == "sealed")


def test_cap_stops_bounded_waves_and_never_seals(tmp_path, monkeypatch):
    path = manifest_file(tmp_path, rows=[{"id": str(i), "task": "task"} for i in range(5)])
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    report = single.run(path, out, usd_cap=0.1, workers=2)
    assert report["status"] == "cap_stop"
    assert len(calls) == 2
    assert not (out / "SEALED.json").exists()
    assert single.run(path, out, usd_cap=0.1, workers=2)["status"] == "cap_stop"


def test_mid_task_cap_and_partial_ledger_retained(tmp_path, monkeypatch):
    def fail(row):
        raise BudgetExhausted("budget exhausted")
    fake_runtime(monkeypatch, fail)
    out = tmp_path / "run"
    assert evoagent.run(manifest_file(tmp_path), out, usd_cap=1, workers=1)["status"] == "cap_stop"
    assert next((out / "units").glob("*.json")).exists()
    assert not (out / "SEALED.json").exists()


def test_resume_rejects_changed_model_deployment(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    path, out = manifest_file(tmp_path), tmp_path / "run"
    single.run(path, out, usd_cap=1, workers=1)
    monkeypatch.setattr(models, "model_settings", lambda *args, **kwargs: {"deployment": "changed"})
    with pytest.raises(RuntimeError, match="different configuration"):
        single.run(path, out, usd_cap=1, workers=1)


@pytest.mark.parametrize("mutation", ["prediction", "missing", "duplicate", "unit_status", "config", "source", "ledger"])
def test_sealed_mutation_rejected_before_gold(tmp_path, monkeypatch, mutation):
    fake_runtime(monkeypatch)
    path, out = manifest_file(tmp_path), tmp_path / "run"
    single.run(path, out, usd_cap=1, workers=1)
    unit_path = next((out / "units").glob("*.json"))
    unit = json.loads(unit_path.read_text())
    if mutation == "prediction":
        unit["record"]["prediction"] = ["changed"]
        atomic_json(unit_path, unit)
    elif mutation == "missing":
        unit_path.unlink()
    elif mutation == "duplicate":
        atomic_json(out / "units" / "duplicate.json", unit)
    elif mutation == "unit_status":
        unit["status"] = "unexpected"
        atomic_json(unit_path, unit)
    elif mutation == "config":
        value = json.loads((out / "RUN_CONFIG.json").read_text())
        value["max_iterations"] = 1
        atomic_json(out / "RUN_CONFIG.json", value)
    elif mutation == "source":
        monkeypatch.setattr(common, "source_identity", lambda: {"source.py": "changed"})
    else:
        (out / "api_calls.jsonl").write_text('{"estimated_usd": 1}\n')
    monkeypatch.setattr(_runner, "_workbench_path", lambda: pytest.fail("must reject before gold"))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        single.score(path, out)


@pytest.mark.parametrize("runner", [single, evoagent])
def test_official_workbench_scoring_keeps_task_failure_denominator(tmp_path, monkeypatch, runner):
    _runner._workbench_path()
    import loader
    tasks, gold = loader.load_tasks()
    first = next(task for task in tasks if gold[task.id])
    second = next(task for task in tasks if task.id != first.id)
    path = manifest_file(tmp_path, rows=[{"id": task.id, "task": task.task} for task in (first, second)])

    def execute(row):
        if row["id"] == second.id:
            raise RuntimeError("content_filter denied")
        return {"task_id": row["id"], "prediction": gold[row["id"]], "usage": {}}
    fake_runtime(monkeypatch, execute)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    monkeypatch.setattr(models, "model_settings", lambda *args, **kwargs: pytest.fail("scoring cannot need credentials"))
    report = runner.score(path, out)
    assert report["n"] == 2 and report["correct"] == 1 and report["completion"] == 0.5


@pytest.mark.parametrize("runner", [single, evoagent])
def test_hotpot_official_scoring_and_failure_denominator(tmp_path, monkeypatch, runner):
    from hotpot_fullwiki import loader
    from inherit_mas import release_config
    path = manifest_file(tmp_path, "hotpotqa", [{"id": "a"}, {"id": "b"}])

    def execute(row):
        if row["id"] == "b":
            raise RuntimeError("context_length exceeded")
        return {"task_id": row["id"], "prediction": {"answer": "Paris", "supporting_facts": [["France", 0]]}, "usage": {}}
    fake_runtime(monkeypatch, execute)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    monkeypatch.setattr(release_config, "ensure_hf_home", lambda: None)
    monkeypatch.setattr(loader, "open_snapshot", lambda: SimpleNamespace(digest="snapshot"))
    monkeypatch.setattr(loader, "load_examples", lambda rows, snapshot: [
        SimpleNamespace(id=row["id"], answer="Paris", supporting_facts=[("France", 0)]) for row in rows])
    monkeypatch.setattr(models, "model_settings", lambda *args, **kwargs: pytest.fail("scoring cannot need credentials"))
    report = runner.score(path, out)
    assert report["n"] == 2 and report["metrics"]["joint_f1"] == 0.5


def test_evoagent_nlp_preserves_quality_rejection_and_forced_fifth_accept():
    from baselines.evoagent.core import evolve_result
    calls = []
    result = evolve_result(task="task", initial_result="initial", iterations=3,
        propose_expert=lambda *args: "You are an expert",
        check_expert=lambda *args: "Discard",
        execute_expert=lambda *args: calls.append("execute") or "child",
        integrate=lambda *args: calls.append("integrate") or "integrated")
    assert calls == ["execute", "integrate"] * 3
    assert len(result.candidate_results) == 4
    assert all(len(row["role_attempts"]) == 5 and row["forced_accept"] for row in result.trace)


def test_evoagent_interactive_executes_only_integrated_action(monkeypatch):
    _runner._workbench_path()
    from baselines.evoagent import workbench_adapter as adapter
    actions = []
    monkeypatch.setattr(adapter.W, "reset_state", lambda: None)
    monkeypatch.setattr(adapter.W, "call_tool", lambda name, args: actions.append((name, args)) or "observed")
    monkeypatch.setattr(adapter.W, "render_action", lambda name, args: name)
    monkeypatch.setattr(adapter.W, "TOOL_BY_NAME", {"write": object()})

    class FakeModels:
        roles = []
        def chat(self, *, role, model, **kwargs):
            self.roles.append(role)
            assert model == "gpt-4o-mini"
            if ":integrator_action:0" in role:
                text = '{"action":"write","action_input":{}}'
            elif ":integrator_action:1" in role:
                text = '{"action":"Final Answer","action_input":"done"}'
            else:
                text = '{"action":"write","action_input":{"ignored":true}}'
            return ModelResult(text)
    fake = FakeModels()
    record = adapter.run(SimpleNamespace(id="t", task="task", source="", base_template=""), models=fake)
    assert actions == [("write", {})]
    assert len(fake.roles) == 8
    assert record["quality_check_called"] is False
    assert record["prediction"] == ["write"]


def test_evoagent_hotpot_four_independent_agent_budgets():
    from baselines.evoagent import hotpot_adapter
    calls = []

    class FakeModels:
        def research_agent(self, **kwargs):
            calls.append(kwargs)
            return ModelResult('{"answer":"Paris","supporting_facts":[],"confidence":0,"rationale":"test"}')
        def structured(self, **kwargs):
            return ModelResult("Retain" if "quality" in kwargs["role"] else "You are an expert")
        def chat(self, **kwargs):
            return ModelResult('{"answer":"Paris","supporting_facts":[],"confidence":0,"rationale":"test"}')
    example = SimpleNamespace(id="t", question="question")
    example.public = lambda: example
    record = hotpot_adapter.run(example, models=FakeModels(), retriever=object())
    assert len(calls) == 4
    assert all(call["max_search_calls"] == 4 and call["max_turns"] == 5
               and call["max_tokens_per_turn"] == 900 for call in calls)
    assert record["selected_candidate"] == 3


@pytest.mark.parametrize("backbone,attempts", [("gpt-4o-mini", 3), ("qwen3-32b", 1)])
def test_single_workbench_original_transient_retry_policy(monkeypatch, backbone, attempts):
    import httpx
    import openai
    _runner._workbench_path()
    from baselines.single_react import workbench_adapter
    monkeypatch.setattr(workbench_adapter.time, "sleep", lambda seconds: None)
    calls = []

    class FailingModels:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://localhost"))
    with pytest.raises(openai.APIConnectionError):
        workbench_adapter.run(SimpleNamespace(id="t", task="task"), models=FailingModels(), backbone=backbone)
    assert len(calls) == attempts
