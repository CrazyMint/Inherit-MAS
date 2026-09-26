"""Offline regression checks for the two distinct TacoMAS integrations."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.tacomas import hotpot_controller as taco
from baselines.tacomas import workbench
from baselines.tacomas.native_child import native_arguments
from baselines.tacomas.native_transport import (Budget, BudgetStop, RoleViolation, install_transport,
                                                logical_route, read_calls, role_scope, usage_summary)
from hotpot_fullwiki.baselines.execution import allocated_search_limits, execute_graph
from hotpot_fullwiki.baselines.graph import clone, tacomas_initial_graph
from hotpot_fullwiki.baselines.runner import _units, ledger_totals
from hotpot_fullwiki.common import digest


def test_native_settings_and_role_decoding():
    args = native_arguments(3, "gpt-4o-mini", "http://worker/v1", "http://meta/v1")
    expected = {"--max_fast_rounds": "10", "--fast_steps_per_window": "1",
                "--bd_check_interval": "2", "--graph_rewire_interval": "2",
                "--init_n_min": "5", "--init_n_max": "5", "--pop_n_min": "2",
                "--pop_n_max": "20", "--max_birth_death_pairs": "2", "--max_edge_edits": "8",
                "--max_iterations_per_agent": "20", "--instance_retries": "1",
                "--agent_temperature": "0", "--meta_temperature": "0.3"}
    assert all(args[args.index(flag) + 1] == value for flag, value in expected.items())
    assert "--skip_meta_init" in args
    assert "AgentState" in workbench.NATIVE_SETTINGS["controller"]
    assert not workbench.NATIVE_SETTINGS["typed_graph_projection"]
    assert workbench.NATIVE_SETTINGS["compiler_tokens"] == 1200
    assert workbench.NATIVE_SETTINGS["worker_tool_calls_per_round"] == 6


@pytest.mark.parametrize("role", ["worker_synthesis_memory", "worker_synthesis_top_k"])
def test_historical_and_qwen_synthesis_routes(role):
    assert logical_route("openai/gpt-5.4-mini", role, "gpt-4o-mini") == "gpt-5.4-mini"
    assert logical_route("openai/gpt-5.4-mini", role, "qwen3-32b") == "qwen3-32b"


def test_no_unscoped_or_wrong_role_model_fallback():
    for role in ("meta_controller", "online_judge", "contribution_judge", "meta_initialization"):
        assert logical_route("openai/gpt-5.4-mini", role, "qwen3-32b") == "gpt-5.4-mini"
    assert logical_route("openai/gpt-4o-mini", "worker", "qwen3-32b") == "qwen3-32b"
    with pytest.raises(RoleViolation):
        logical_route("openai/gpt-5.4-mini", "worker", "qwen3-32b")
    with pytest.raises(RoleViolation):
        logical_route("openai/gpt-4o-mini", "online_judge", "qwen3-32b")
    with pytest.raises(RoleViolation):
        logical_route("openai/other-model", "worker", "gpt-4o-mini")


def test_public_stage_does_not_leak_gold(tmp_path):
    row = {"id": "x", "task": "Public task", "gold": ["secret"], "outcome": "secret"}
    payload = workbench.stage_public_dataset([row], tmp_path / "dataset.json")
    assert payload["instances"][0]["expected_output"] is None
    assert "secret" not in json.dumps(payload)
    assert set(payload["instances"][0]) == {"id", "question", "source", "base_template", "expected_output"}


def test_hotpot_population_and_shared_retrieval_budget():
    graph = tacomas_initial_graph()
    assert len(graph["nodes"]) == 5
    limits = allocated_search_limits(graph)
    assert sum(limits.values()) == 4
    assert limits["searcher_a"] == limits["searcher_b"] == 2
    assert taco.N_MIN == 5 and taco.N_MAX == 20
    assert taco.MAX_BIRTH_DEATH_PAIRS == 2 and taco.MAX_EDGE_EDITS == 8


def test_capability_update_retains_other_agents_and_bounds_memory():
    graph = tacomas_initial_graph()
    update = taco._validate_capability({"updates": [{"agent_id": "planner", "contribution_score": 50,
        "score_reason": "Missing bridge", "prompt_delta": "Check the bridge entity."}],
        "continue_evolution": True, "rationale": "Improve plan"}, graph)
    child = taco._apply_capability(graph, update, 1)
    assert child["nodes"][1:] == graph["nodes"][1:]
    assert "Check the bridge entity" in child["nodes"][0]["system_prompt"]
    assert "Check the bridge entity" not in graph["nodes"][0]["system_prompt"]
    bad = clone(update)
    bad["updates"][0]["contribution_score"] = True
    with pytest.raises(ValueError):
        taco._validate_capability(bad, graph)


def test_fast_and_slow_loops_preserved(monkeypatch, tmp_path):
    calls = {"execute": 0, "capability": 0, "topology": 0}
    result = SimpleNamespace(prediction={"valid": True, "answer": "answer", "supporting_facts": []},
                             trace={"nodes": {}, "live_tokens": 1})
    def execute(*args, **kwargs):
        calls["execute"] += 1
        return result
    def structured(models, *, role, validator, **kwargs):
        if role == "tacomas:capability":
            calls["capability"] += 1
            value = {"updates": [], "continue_evolution": True, "rationale": "Continue"}
        else:
            calls["topology"] += 1
            value = {"graph": tacomas_initial_graph(), "continue_evolution": True, "rationale": "Retain"}
        return validator(value), []
    monkeypatch.setattr(taco, "execute_graph", execute)
    monkeypatch.setattr(taco, "structured_call", structured)
    monkeypatch.setattr(taco, "judge", lambda *a, **k: ({"quality_score": calls["execute"]}, []))
    record = taco.run(SimpleNamespace(id="x", question="Q"), models=None, retriever=None, run_dir=tmp_path)
    assert calls == {"execute": 10, "capability": 10, "topology": 4}
    assert record["selected_candidate"] == 9
    assert len(record["slow_updates"]) == 4
    assert record["settings"]["max_fast_rounds"] == 10


def test_cost_reservations_shared_and_unknown_usage_charged(tmp_path):
    a, b = Budget(tmp_path, 1), Budget(tmp_path, 1)
    token = a.reserve(0.6)
    with pytest.raises(BudgetStop):
        b.reserve(0.5)
    a.settle(token, None)
    assert json.loads((tmp_path / "BUDGET.json").read_text())["spent_usd"] == 0.6


def test_hotpot_rejects_stale_or_missing_candidate_selection(tmp_path):
    manifest = {"rows": [{"id": "x"}], "manifest_digest": "m"}
    config = {"method": "tacomas_adapted", "config_digest": "c"}
    root = tmp_path / "units" / "tacomas_adapted"
    root.mkdir(parents=True)
    unit = {"task_id": "x", "condition": "tacomas_adapted", "manifest_digest": "m",
            "config_digest": "c", "status": "complete", "record": {"task_id": "x",
            "candidates": [{"candidate_index": 0, "status": "complete"}], "selected_candidate": 1}}
    (root / "x.json").write_text(json.dumps(unit))
    with pytest.raises(RuntimeError, match="selected"):
        _units(tmp_path, manifest, config)
    unit["record"]["selected_candidate"] = 0
    unit["config_digest"] = "stale"
    (root / "x.json").write_text(json.dumps(unit))
    with pytest.raises(RuntimeError, match="identity"):
        _units(tmp_path, manifest, config)


@pytest.mark.parametrize("cap,workers", [(0, 1), (float("nan"), 1), (1, 0)])
def test_limits_fail_before_credentials_or_upstream(tmp_path, cap, workers):
    with pytest.raises(ValueError):
        workbench.run(tmp_path / "absent.json", tmp_path, usd_cap=cap, workers=workers)


def test_graph_executor_never_reuses_identical_nodes():
    from hotpot_fullwiki.models import ModelResult
    called = []
    class FakeModels:
        def chat(self, **kwargs):
            called.append(kwargs["role"])
            return ModelResult('{"answer":"A","supporting_facts":[],"confidence":0.5,"rationale":"R"}')
        def research_agent(self, **kwargs):
            return self.chat(**kwargs)
    example = SimpleNamespace(public=lambda: SimpleNamespace(question="Question"))
    graph = tacomas_initial_graph()
    first = execute_graph(graph, example, models=FakeModels(), retriever=None)
    second = execute_graph(graph, example, models=FakeModels(), retriever=None)
    assert len(called) == 10
    assert first.trace["reused_calls"] == second.trace["reused_tokens"] == 0
    assert not any(s["reused"] for s in second.snapshots.values())


def test_qwen_native_transport_forces_synthesis_to_worker_without_thinking(monkeypatch, tmp_path):
    import sys
    from inherit_mas import release_config
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="qwen3-32b", usage={"prompt_tokens": 12, "completion_tokens": 4})
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr(release_config, "credentials", lambda: {})
    transport = install_transport({"out": str(tmp_path), "backbone": "qwen3-32b", "usd_cap": 1,
        "tokenizer_path": "unused", "context_tokens": 32768, "qwen_url": "http://127.0.0.1:8000/v1"},
        completion_fn=completion, token_counter=lambda messages, tools: 100)
    scoped = role_scope(transport, "worker_synthesis_memory")
    scoped(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "Synthesize"}],
           temperature=0.3, tools=[])
    request = calls[0]
    assert request["model"] == "openai/qwen3-32b"
    assert request["max_tokens"] == 32768 - 100 - 64
    assert "tools" not in request
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert request["temperature"] == 0.3
    usage = usage_summary(read_calls(tmp_path))
    assert usage["total_tokens"] == 16 and usage["estimated_usd"] == 0


@pytest.mark.parametrize("logical,role", [("gpt-4o-mini", "worker"), ("gpt-5.4-mini", "online_judge")])
def test_native_transport_supports_custom_model_and_generic_endpoint(monkeypatch, tmp_path, logical, role):
    import sys
    from inherit_mas import release_config
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="my-meta", usage={"prompt_tokens": 12, "completion_tokens": 4})
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr(release_config, "credentials", lambda: {"api": {
        "base_url": "https://api.example.com/v1", "models": {logical: {
            "model": "my-meta", "api_key_env": "TEST_API_KEY"}}}})
    monkeypatch.setenv("TEST_API_KEY", "dummy-test-key")
    transport = install_transport({"out": str(tmp_path), "backbone": "gpt-4o-mini", "usd_cap": 1},
                                  completion_fn=completion)
    role_scope(transport, role)(model=f"openai/{logical}",
        messages=[{"role": "user", "content": "Judge"}], max_tokens=300, temperature=0)
    assert calls[0]["model"] == "openai/my-meta"
    assert calls[0]["api_base"] == "https://api.example.com/v1"
    assert calls[0]["api_key"] == "dummy-test-key"
    assert "api_version" not in calls[0]
    assert "dummy-test-key" not in (tmp_path / "api_calls.jsonl").read_text()


def test_usage_reports_stored_costs_and_rejects_malformed_ledger(tmp_path):
    path = tmp_path / "api_calls.jsonl"
    path.write_text(json.dumps({"model": "qwen3-32b", "total_tokens": 123, "estimated_usd": 0}) + "\n")
    assert ledger_totals(tmp_path)["by_model"]["qwen3-32b"]["total_tokens"] == 123
    path.write_text(json.dumps({"model": "gpt-5.4-mini", "total_tokens": -1, "estimated_usd": 1}) + "\n")
    with pytest.raises(RuntimeError, match="usage"):
        ledger_totals(tmp_path)


def test_native_score_is_offline_and_uses_stored_ledger(monkeypatch, tmp_path):
    import sys
    manifest = {"tasks": [{"id": "x", "task": "Task"}], "n": 1, "dataset_pin": "pin"}
    manifest["manifest_digest"] = digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {"settings": workbench.NATIVE_SETTINGS, "upstream_commit": workbench.COMMIT,
              "manifest_digest": manifest["manifest_digest"], "backbone": "qwen3-32b",
              "source_sha256": workbench._sources()}
    config["config_digest"] = digest(config)
    (tmp_path / "RUN_CONFIG.json").write_text(json.dumps(config))
    (tmp_path / "SEALED.json").write_text(json.dumps({"config_digest": config["config_digest"],
        "manifest_digest": manifest["manifest_digest"], "expected": 1}))
    (tmp_path / "RUN_STATUS.json").write_text(json.dumps({"status": "sealed"}))
    (tmp_path / "units").mkdir()
    unit = {"task_id": "x", "status": "complete", "failure": "", "config_digest": config["config_digest"],
            "manifest_digest": manifest["manifest_digest"],
            "successful_calls": 1,
            "record": {"task_id": "x", "prediction": [], "fast_rounds": 10}}
    workbench._unit_path(tmp_path, "x").write_text(json.dumps(unit))
    (tmp_path / "api_calls.jsonl").write_text(json.dumps({"model": "qwen3-32b", "status": "success",
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "estimated_cost_usd": 0}) + "\n")
    monkeypatch.setattr(workbench, "_workbench_path", lambda: None)
    monkeypatch.setitem(sys.modules, "loader", SimpleNamespace(dataset_pin=lambda: "pin",
        load_tasks=lambda: ([SimpleNamespace(id="x", task="Task")], {"x": []})))
    monkeypatch.setitem(sys.modules, "wb_env", SimpleNamespace(
        score_prediction=lambda *args: {"correct": True, "unwanted_side_effect": False}))
    result = workbench.score(manifest_path, tmp_path)
    assert result["completion"] == 1
    assert result["usage"]["total_tokens"] == 12


def test_full_workbench_public_task_staging(tmp_path):
    from public_runner.common import read_manifest
    root = Path(__file__).resolve().parents[1]
    manifest = read_manifest(root / "data/workbench.json", "workbench")
    staged = workbench.stage_public_dataset(manifest["tasks"], tmp_path / "staged.json")
    assert len(staged["instances"]) == manifest["n"] == 130
    assert [(r["id"], r["question"]) for r in staged["instances"]] == [
        (r["id"], r["task"]) for r in manifest["tasks"]]
    assert all(row["expected_output"] is None for row in staged["instances"])


def test_hotpot_public_run_resume_and_offline_score(monkeypatch, tmp_path):
    from hotpot_fullwiki.baselines import runner
    from hotpot_fullwiki import loader, retrieval, analyze
    from public_runner import models
    manifest = {"rows": [{"id": "x"}, {"id": "y"}], "n": 2, "dataset_revision": "snapshot"}
    manifest["manifest_digest"] = digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    examples = [SimpleNamespace(id=i, runtime=lambda i=i: SimpleNamespace(id=i, question="Q")) for i in ("x", "y")]
    monkeypatch.setattr(loader, "open_snapshot", lambda: SimpleNamespace(digest="snapshot"))
    monkeypatch.setattr(loader, "load_examples", lambda *args: examples)
    monkeypatch.setattr(models, "model_settings", lambda *args: {"worker_model": "qwen3-32b"})
    monkeypatch.setattr(models, "make_models", lambda *args: object())
    monkeypatch.setattr(retrieval, "BM25Retriever", lambda **kwargs: SimpleNamespace(preflight=lambda *args: None))
    called = []
    def execute(example, **kwargs):
        called.append(example.id)
        return {"task_id": example.id, "candidates": [{"candidate_index": 0, "status": "complete",
                "prediction": {"answer": "A", "supporting_facts": []}}], "selected_candidate": 0}
    controller = SimpleNamespace(__file__=taco.__file__, UPSTREAM_COMMIT=taco.UPSTREAM_COMMIT, run=execute)
    out = tmp_path / "out"
    report = runner.run(path, out, usd_cap=1, workers=2, backbone="qwen3-32b",
                        method="tacomas_adapted", controller=controller)
    assert report["status"] == "sealed" and sorted(called) == ["x", "y"]
    runner.run(path, out, usd_cap=1, workers=2, backbone="qwen3-32b", method="tacomas_adapted", controller=controller)
    assert len(called) == 2
    with pytest.raises(RuntimeError, match="configuration"):
        runner.run(path, out, usd_cap=2, workers=2, backbone="qwen3-32b", method="tacomas_adapted", controller=controller)
    monkeypatch.setattr(models, "model_settings", lambda *a: pytest.fail("scoring must not load credentials"))
    names = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")
    monkeypatch.setattr(analyze, "score_prediction", lambda *args: {n: 1.0 for n in names})
    scored = runner.score(path, out, method="tacomas_adapted")
    assert scored["metrics"]["joint_f1"] == 1.0
