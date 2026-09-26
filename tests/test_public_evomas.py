"""Synthetic tests: no upstream model calls, downloads, or historical outputs."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from baselines.evomas import native, runner
from hotpot_fullwiki.common import atomic_json, digest


def test_benchmark_settings_and_native_qwen_boundary():
    wb = native.settings("workbench", "gpt-4o-mini")
    hp = native.settings("hotpotqa", "gpt-4o-mini")
    assert (wb["num_parents"], wb["evolution_steps"], wb["agent_max_steps"], wb["agent_max_tokens"]) == (2, 3, 8, 1024)
    assert (hp["num_parents"], hp["evolution_steps"], hp["retrieval_calls_per_agent"]) == (2, 2, 4)
    assert wb["meta_temperature"] == 0.7 and wb["judge_temperature"] == 0
    assert wb["memory_evolution"] and wb["workers"] == 1
    qwen = native.settings("workbench", "qwen3-32b")
    assert qwen["worker_model"] == "openai:qwen3-32b" and qwen["worker_temperature"] == 0
    with pytest.raises(ValueError, match="graph adapter"):
        native.settings("hotpotqa", "qwen3-32b")


def test_worker_policy_rejects_meta_escalation_and_mixed_palette():
    path = Path(native.__file__).parent / "overlay/src/models/worker_policy.py"
    spec = importlib.util.spec_from_file_location("test_evomas_policy", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    with pytest.raises(ValueError):
        policy.validate_worker_palette(["azure:gpt-4o-mini", "azure:gpt-5.4-mini"])
    with pytest.raises(ValueError):
        policy.validate_worker_palette(["azure:gpt-4o-mini", "openai:qwen3-32b"])
    with pytest.raises(ValueError):
        policy.validate_model_role("azure:gpt-4o-mini", resolved_model="gpt-5.4-mini")
    policy.validate_model_role("azure:gpt-5.4-mini", role="judge")


@pytest.fixture
def native_stub(tmp_path, monkeypatch):
    import yaml
    from baselines.evomas import model_bridge
    import public_runner.models
    upstream = tmp_path / "upstream"
    for domain in native.DOMAINS:
        pool = upstream / "mas_pools/workbench" / domain
        pool.mkdir(parents=True)
        (pool / "seed.yaml").write_text(yaml.safe_dump({"agents": {"worker": {"model_id": "old"}}}))
        atomic_json(upstream / "dataset/workbench" / domain / "test.json", [
            {"query": "first public task", "gt": "private reference"},
            {"query": "second public task", "gt": "private reference"}])
    rows = [{"id": "email#0", "task": "first public task"}, {"id": "email#1", "task": "second public task"}]
    manifest = {"n": len(rows), "tasks": rows}
    manifest["manifest_digest"] = digest(manifest)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    monkeypatch.setattr(native, "UPSTREAM", upstream)
    monkeypatch.setattr(native, "verify", lambda root: {"commit": "test"})
    monkeypatch.setattr(public_runner.models, "model_settings", lambda backbone: {"worker": backbone})
    monkeypatch.setattr(model_bridge, "install", lambda backbone: None)
    model_bridge.provider_failures.clear()
    loader = types.ModuleType("src.dataset.load_dataset")
    loader.Dataset = type("Dataset", (), {})
    monkeypatch.setitem(sys.modules, "src.dataset.load_dataset", loader)
    main = types.ModuleType("main")
    calls = []
    def pipeline(**kwargs):
        calls.append(kwargs)
        memory = Path(kwargs["memory_path"])
        prior = json.loads(memory.read_text()) if memory.exists() else []
        prior.append(kwargs["task_ids"][0])
        atomic_json(memory, prior)
        unit = Path(kwargs["output_dir"])
        atomic_json(unit / "trajectories" / "task__final_selected.json", {
            "trace": [{"metadata": {"model_id": kwargs["model_list"][0]}}], "final_result": "[]"})
        return {"finished": True}
    main.run_evolution_pipeline = pipeline
    monkeypatch.setitem(sys.modules, "main", main)
    # The production subprocess changes cwd; restore it in this in-process fake.
    monkeypatch.chdir(tmp_path)
    return path, tmp_path / "out", calls, main


def test_serial_native_loop_seals_and_resumes_without_reexecution(native_stub):
    path, out, calls, _ = native_stub
    status = native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    assert status["status"] == "sealed"
    assert [call["task_ids"] for call in calls] == [[0], [1]]
    assert all(call["max_steps"] == 3 and call["memory_evolution"] and call["workers"] == 1 for call in calls)
    assert json.loads((out / "state/memory.json").read_text()) == [0, 1]
    staged = json.loads((out / "dataset/workbench/email/test.json").read_text())
    assert all(row["gt"] == "__SEALED_OFFLINE__" for row in staged)
    native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    assert len(calls) == 2
    runner._verify_seal(path, out, "workbench")


def test_failed_native_task_rolls_back_state_and_stops(native_stub):
    path, out, calls, main = native_stub
    original = main.run_evolution_pipeline
    def failing(**kwargs):
        result = original(**kwargs)
        if kwargs["task_ids"] == [1]:
            raise RuntimeError("synthetic provider failure")
        return result
    main.run_evolution_pipeline = failing
    status = native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    assert status["status"] == "incomplete" and not (out / "SEALED.json").exists()
    assert json.loads((out / "state/memory.json").read_text()) == [0]
    assert native.verify_resume(out, [{"id": "email#0"}, {"id": "email#1"}]) == 1


def test_cap_stops_between_serial_tasks(native_stub):
    path, out, calls, main = native_stub
    original = main.run_evolution_pipeline
    def charged(**kwargs):
        result = original(**kwargs)
        (Path(kwargs["output_dir"]) / "live_calls.jsonl").write_text(json.dumps({"cost_usd": 2}) + "\n")
        return result
    main.run_evolution_pipeline = charged
    status = native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    assert status["status"] == "cap_stop" and len(calls) == 1


def test_sparse_workbench_ids_preserve_native_positional_lookup(native_stub):
    path, out, calls, _ = native_stub
    manifest = {"n": 1, "tasks": [{"id": "email#1", "task": "second public task"}]}
    manifest["manifest_digest"] = digest(manifest)
    atomic_json(path, manifest)
    native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    staged = json.loads((out / "dataset/workbench/email/test.json").read_text())
    assert len(staged) == 2 and staged[1]["query"] == "second public task"
    assert calls[0]["task_ids"] == [1]


def test_sealed_artifact_tampering_rejected(native_stub):
    path, out, _, _ = native_stub
    native.run(path, out, benchmark="workbench", backbone="gpt-4o-mini", usd_cap=1)
    target = out / "units/email__0/trajectories/task__final_selected.json"
    target.write_text("{}")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        runner._verify_seal(path, out, "workbench")


def test_no_parallel_native_tasks(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"tasks": []})
    with pytest.raises(ValueError, match="serial"):
        runner.run(path, tmp_path / "out", workers=2, usd_cap=1)


def test_qwen_hotpot_dispatch_uses_evaluated_graph_adapter(tmp_path, monkeypatch):
    from hotpot_fullwiki.baselines import evomas, runner as shared
    path = tmp_path / "manifest.json"
    atomic_json(path, {"rows": []})
    observed = {}
    def fake(*args, **kwargs):
        observed.update(kwargs)
        return {"status": "sealed"}
    monkeypatch.setattr(shared, "run", fake)
    assert runner.run(path, tmp_path / "out", workers=2, usd_cap=1, backbone="qwen3-32b")["status"] == "sealed"
    assert observed["controller"] is evomas and observed["method"] == "evomas_adapted"
    assert evomas.NUM_PARENTS == 2 and evomas.MAX_STEPS == 2


def test_exact_python_dictionary_prediction_supported():
    from baselines.evomas.hotpot import parse_prediction
    assert parse_prediction("{'answer': 'x', 'supporting_facts': [('Title', 0)]}") == {
        "answer": "x", "supporting_facts": [["Title", 0]]}


def test_model_bridge_overrides_package_alias_and_rejects_worker_escalation(monkeypatch):
    from baselines.evomas import model_bridge
    from inherit_mas import release_config
    path = Path(native.__file__).parent / "overlay/src/models/worker_policy.py"
    spec = importlib.util.spec_from_file_location("src.models.worker_policy", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    package = types.ModuleType("src.models")
    module = types.ModuleType("src.models.model")
    class Base:
        def reset_token_usage(self):
            self._cumulative_input_tokens = self._cumulative_output_tokens = 0
        def _create_mock_smolagents_model(self):
            return types.SimpleNamespace(model_id=self.model)
    module.OpenAIModel = Base
    package.model = module
    package.get_model = object()
    root = types.ModuleType("src")
    root.models = package
    monkeypatch.setitem(sys.modules, "src", root)
    monkeypatch.setitem(sys.modules, "src.models", package)
    monkeypatch.setitem(sys.modules, "src.models.model", module)
    monkeypatch.setitem(sys.modules, "src.models.worker_policy", policy)
    monkeypatch.setattr(release_config, "credentials", lambda: {})
    logical = []
    monkeypatch.setattr(release_config, "deployment", lambda doc, name: logical.append(name) or {
        "deployment": "custom-deployment", "api_key_env": "TEST_KEY"})
    monkeypatch.setattr(release_config, "model_endpoint", lambda *a: "https://example.invalid/v1")
    monkeypatch.setattr(release_config, "api_key", lambda *a: "fake")
    monkeypatch.setattr(release_config, "chat_client", lambda **k: object())
    model_bridge.install("gpt-4o-mini")
    assert package.get_model is module.get_model
    worker = package.get_model("azure:gpt-4o-mini")
    assert worker.provider_model == "custom-deployment"
    assert worker.model == "gpt-4o-mini"
    package.get_model("azure:gpt-5.4-mini", role="meta")
    assert logical == ["gpt-4o-mini", "5.4-mini"]
    with pytest.raises(ValueError):
        package.get_model("azure:gpt-5.4-mini", role="CodeAgent")
    with pytest.raises(ValueError):
        package.get_model("openai:qwen3-32b", role="CodeAgent")


def test_setup_derives_public_workbench_tables_without_reference_actions(tmp_path):
    from baselines.evomas.prepare import DATA, ENVIRONMENT, prepare_workbench
    prepare_workbench(tmp_path)
    for path in (tmp_path / "dataset/workbench").glob("*/test.json"):
        rows = json.loads(path.read_text())
        assert rows and [row["id"] for row in rows] == list(range(len(rows)))
        assert all(row["gt"] == "__SEALED_OFFLINE__" and set(row) == {"id", "query", "gt", "tag", "source"} for row in rows)
    for domain, filename in ENVIRONMENT.items():
        assert (tmp_path / "dataset/workbench" / domain / "data.csv").read_bytes() == (DATA / filename).read_bytes()


def test_native_subprocess_interpreter_is_configurable(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"tasks": []})
    monkeypatch.setenv("INHERIT_EVOMAS_PYTHON", "/custom/venv/bin/python")
    command = runner.build_run_command(path, tmp_path / "out", usd_cap=1)
    assert command[0] == "/custom/venv/bin/python"
    assert command[1:3] == ["-m", "baselines.evomas.native"]


def test_runtime_lock_detects_changed_upstream_source(tmp_path):
    from baselines.evomas.runtime_lock import verify_lock, write_lock
    (tmp_path / "main.py").write_text("# synthetic upstream entry\n")
    directory = tmp_path / "src"
    directory.mkdir()
    for index in range(10):
        (directory / f"module_{index}.py").write_text("# synthetic source\n")
    write_lock(tmp_path)
    assert verify_lock(tmp_path)["files"]
    (directory / "module_0.py").write_text("# changed synthetic source\n")
    with pytest.raises(RuntimeError, match="source lock mismatch"):
        verify_lock(tmp_path)
