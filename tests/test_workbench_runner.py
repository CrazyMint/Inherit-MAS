from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hotpot_fullwiki.common import atomic_json, digest
from public_runner import workbench as runner


@pytest.fixture(autouse=True)
def api_settings(tmp_path, monkeypatch, isolate_credentials):
    path = tmp_path / "test_credentials.json"
    atomic_json(path, {"api": {"base_url": "https://example.com/v1", "models": {
        "gpt-4o-mini": {"model": "worker-deployment"},
        "gpt-5.4-mini": {"model": "controller-deployment"},
    }}})
    monkeypatch.setenv("INHERIT_MAS_CREDENTIALS", str(path))
    return path


def manifest_file(tmp_path, tasks=None):
    tasks = tasks or [{"id": "task#0", "task": "Public task", "gold": ["must not leak"]}]
    value = {"n": len(tasks), "tasks": tasks}
    value["manifest_digest"] = digest(value)
    path = tmp_path / "manifest.json"
    atomic_json(path, value)
    return path


def record(task_id, prediction=None, usd=0.2):
    return {"task_id": task_id, "selected_candidate": 1, "candidates": [
        {"candidate_index": 0, "status": "invalid_proposal", "proposal_attempts": [
            {"usage": [{"total_tokens": 3, "estimated_usd": 0.01}]}]},
        {"candidate_index": 1, "status": "complete", "prediction": prediction or [],
         "execution": {"live_tokens": 10, "reused_tokens": 20,
                       "nodes": {"node": {"usage": {"estimated_usd": usd}}}},
         "judge_attempts": [{"usage": [{"total_tokens": 5, "estimated_usd": 0.02}]}]},
    ]}


def fake_runtime(monkeypatch, execute=None):
    calls = []

    def run_condition(task, *, models, run_dir):
        assert vars(task).keys() == {"id", "task"}
        calls.append(task.id)
        return execute(task) if execute else record(task.id)

    monkeypatch.setattr(runner, "_runtime", lambda: (
        lambda id, task: SimpleNamespace(id=id, task=task), run_condition, object))
    return calls


def test_run_resume_and_usage_without_gold_leakage(tmp_path, monkeypatch):
    manifest = manifest_file(tmp_path)
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    result = runner.run(manifest, out, usd_cap=1, workers=1)
    assert result["status"] == "sealed"
    assert calls == ["task#0"]
    assert result["usage"] == {
        "worker_live_tokens": 10, "worker_reused_tokens": 20,
        "meta_judge_tokens": 8, "estimated_usd": pytest.approx(0.23),
        "unknown_cost_units": 0,
    }
    monkeypatch.setattr(runner, "_runtime", lambda: pytest.fail("resume must not create clients"))
    assert runner.run(manifest, out, usd_cap=1, workers=1)["status"] == "sealed"
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.run(manifest, out, usd_cap=2, workers=1)


@pytest.mark.parametrize("usd_cap,workers", [(0, 1), (-1, 1), (float("nan"), 1),
    (float("inf"), 1), (True, 1), (1, 0), (1, -1), (1, 1.5), (1, True)])
def test_invalid_limits_before_runtime(tmp_path, monkeypatch, usd_cap, workers):
    monkeypatch.setattr(runner, "_runtime", lambda: pytest.fail("must validate before runtime"))
    with pytest.raises(ValueError):
        runner.run(manifest_file(tmp_path), tmp_path / "run", usd_cap=usd_cap, workers=workers)


@pytest.mark.parametrize("change", ["digest", "count", "duplicate"])
def test_invalid_manifest(tmp_path, change):
    path = manifest_file(tmp_path)
    value = json.loads(path.read_text())
    if change == "digest":
        value["manifest_digest"] = "bad"
    else:
        value["n"] = 2
        if change == "duplicate":
            value["tasks"] *= 2
        value["manifest_digest"] = digest({key: row for key, row in value.items()
                                          if key != "manifest_digest"})
    atomic_json(path, value)
    with pytest.raises(ValueError):
        runner.run(path, tmp_path / "run", usd_cap=1, workers=1)


def test_cost_cap_stops_after_bounded_wave(tmp_path, monkeypatch):
    path = manifest_file(tmp_path, [{"id": str(i), "task": "task"} for i in range(5)])
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    result = runner.run(path, out, usd_cap=0.1, workers=2)
    assert result["status"] == "cap_stop"
    assert set(calls) == {"0", "1"}
    assert result["recorded"] == 2
    assert not (out / "SEALED.json").exists()
    monkeypatch.setattr(runner, "_runtime", lambda: pytest.fail("capped resume must not create clients"))
    assert runner.run(path, out, usd_cap=0.1, workers=2)["status"] == "cap_stop"


@pytest.mark.parametrize("message,status", [
    ("content_filter denied", "sealed"), ("context_length exceeded", "sealed"),
    ("connection failed", "incomplete"),
])
def test_failure_classification_and_sealing(tmp_path, monkeypatch, message, status):
    def fail(task):
        raise RuntimeError(message)
    fake_runtime(monkeypatch, fail)
    out = tmp_path / "run"
    result = runner.run(manifest_file(tmp_path), out, usd_cap=1, workers=1)
    assert result["status"] == status
    assert (out / "SEALED.json").exists() == (status == "sealed")
    assert result["usage"]["unknown_cost_units"] == 1


@pytest.mark.parametrize("mutation", ["config", "manifest", "task_id", "record_id", "status",
    "selection", "duplicate", "missing", "infra"])
def test_score_rejects_bad_units_before_gold(tmp_path, monkeypatch, mutation):
    fake_runtime(monkeypatch)
    path = manifest_file(tmp_path)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    unit_path = next((out / "units").glob("*.json"))
    value = json.loads(unit_path.read_text())
    if mutation == "config":
        value["config_digest"] = "bad"
    elif mutation == "manifest":
        value["manifest_digest"] = "bad"
    elif mutation == "task_id":
        value["task_id"] = "unexpected"
    elif mutation == "record_id":
        value["record"]["task_id"] = "unexpected"
    elif mutation == "status":
        value["status"] = "unknown"
    elif mutation == "selection":
        value["record"]["selected_candidate"] = 999
    elif mutation == "infra":
        value.update(status="infra_failure", record=None, infra_failure="network failed")
    atomic_json(unit_path, value)
    if mutation == "duplicate":
        atomic_json(out / "units" / "duplicate.json", value)
    elif mutation == "missing":
        unit_path.unlink()
    monkeypatch.setattr(runner, "_workbench_path", lambda: pytest.fail("must reject before gold"))
    with pytest.raises(RuntimeError):
        runner.score(path, out)


def test_resume_rejects_foreign_units(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    unit_path = next((out / "units").glob("*.json"))
    value = json.loads(unit_path.read_text())
    value["config_digest"] = "foreign"
    atomic_json(unit_path, value)
    with pytest.raises(RuntimeError, match="identity"):
        runner.run(path, out, usd_cap=1, workers=1)
    assert calls == ["task#0"]


@pytest.mark.parametrize("target,field", [("RUN_CONFIG.json", "worker_model"),
    ("SEALED.json", "manifest_digest"), ("SEALED.json", "config_digest"),
    ("SEALED.json", "expected")])
def test_score_rejects_config_and_seal_mismatch(tmp_path, monkeypatch, target, field):
    path = manifest_file(tmp_path)
    fake_runtime(monkeypatch)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    value = json.loads((out / target).read_text())
    value[field] = "bad"
    atomic_json(out / target, value)
    monkeypatch.setattr(runner, "_workbench_path", lambda: pytest.fail("must reject before gold"))
    with pytest.raises(RuntimeError, match="mismatch"):
        runner.score(path, out)


def test_official_offline_score_selected_candidate_and_failure_denominator(tmp_path, monkeypatch):
    runner._workbench_path()
    import loader
    import wb_env
    tasks, gold = loader.load_tasks()
    task = next(task for task in tasks if gold[task.id])
    failure_task = next(other for other in tasks if other.id != task.id)
    path = manifest_file(tmp_path, [{"id": item.id, "task": item.task}
                                    for item in (task, failure_task)])

    def execute(item):
        if item.id == failure_task.id:
            raise RuntimeError("content_filter denied")
        result = record(item.id, gold[item.id])
        result["candidates"][0].update(status="complete", prediction=[])
        return result

    fake_runtime(monkeypatch, execute)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    official = wb_env.score_prediction
    predictions = []

    def tracked(prediction, gold_actions, error=""):
        predictions.append(prediction)
        return official(prediction, gold_actions, error)

    monkeypatch.setattr(wb_env, "score_prediction", tracked)
    result = runner.score(path, out)
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["completion"] == 0.5
    assert predictions == [gold[task.id], []]
    assert result["per_task"][0]["selected_candidate"] == 1
    assert result["per_task"][1]["correct"] is False
    assert result["usage"]["unknown_cost_units"] == 1
    assert json.loads((out / "SCORE.json").read_text()) == result


@pytest.mark.parametrize("change", ["worker", "controller", "endpoint"])
def test_api_resume_rejects_changed_deployment_or_endpoint(tmp_path, monkeypatch, api_settings, change):
    path = manifest_file(tmp_path)
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    document = json.loads(api_settings.read_text())
    if change == "endpoint":
        document["api"]["base_url"] = "https://different.example.com/v1"
    else:
        model = "gpt-4o-mini" if change == "worker" else "gpt-5.4-mini"
        document["api"]["models"][model]["model"] = "other-deployment"
    atomic_json(api_settings, document)
    with pytest.raises(RuntimeError, match="different configuration"):
        runner.run(path, out, usd_cap=1, workers=1)
    assert calls == ["task#0"]


def test_api_key_rotation_does_not_change_resume_identity(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    calls = fake_runtime(monkeypatch)
    out = tmp_path / "run"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-first-key")
    runner.run(path, out, usd_cap=1, workers=1)
    before = (out / "RUN_CONFIG.json").read_text()
    monkeypatch.setenv("OPENAI_API_KEY", "fake-rotated-key")
    assert runner.run(path, out, usd_cap=1, workers=1)["status"] == "sealed"
    assert (out / "RUN_CONFIG.json").read_text() == before
    assert "fake-first-key" not in before and "fake-rotated-key" not in before
    assert "https://example.com" not in before
    assert calls == ["task#0"]


def test_resume_retries_only_infrastructure_failures_and_retains_history(tmp_path, monkeypatch):
    path = manifest_file(tmp_path, [{"id": task_id, "task": "task"}
                                   for task_id in ("broken", "finished", "filtered")])
    recovered = False

    def execute(task):
        if task.id == "broken" and not recovered:
            raise RuntimeError("connection failed")
        if task.id == "filtered":
            raise RuntimeError("content_filter denied")
        return record(task.id)

    calls = fake_runtime(monkeypatch, execute)
    out = tmp_path / "run"
    assert runner.run(path, out, usd_cap=2, workers=1)["status"] == "incomplete"
    failed_path = runner._unit_path(out, "broken")
    previous = json.loads(failed_path.read_text())
    previous["cost"]["estimated_usd"] = 0.12
    atomic_json(failed_path, previous)
    recovered = True
    result = runner.run(path, out, usd_cap=2, workers=1)
    assert result["status"] == "sealed"
    assert calls == ["broken", "finished", "filtered", "broken"]
    assert result["usage"]["estimated_usd"] == pytest.approx(0.58)
    assert result["usage"]["unknown_cost_units"] == 2
    recovered_unit = json.loads(failed_path.read_text())
    assert recovered_unit["attempt_history"] == [previous]
    assert recovered_unit["status"] == "complete"
    assert runner.run(path, out, usd_cap=2, workers=1)["usage"] == result["usage"]
    assert len(calls) == 4


def test_multiple_retries_flatten_history_without_duplicate_costs(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    recovered = False

    def execute(task):
        if not recovered:
            raise RuntimeError("temporary network failure")
        return record(task.id)

    calls = fake_runtime(monkeypatch, execute)
    out = tmp_path / "run"
    for _ in range(2):
        assert runner.run(path, out, usd_cap=1, workers=1)["status"] == "incomplete"
    recovered = True
    result = runner.run(path, out, usd_cap=1, workers=1)
    unit = json.loads(runner._unit_path(out, "task#0").read_text())
    assert result["status"] == "sealed"
    assert len(unit["attempt_history"]) == 2
    assert all("attempt_history" not in item for item in unit["attempt_history"])
    assert result["usage"]["estimated_usd"] == pytest.approx(0.23)
    assert result["usage"]["unknown_cost_units"] == 1
    assert len(calls) == 3


def test_failed_attempt_cost_still_limits_retry(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)

    def fail(task):
        raise RuntimeError("network failed")

    calls = fake_runtime(monkeypatch, fail)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    unit_path = runner._unit_path(out, "task#0")
    unit = json.loads(unit_path.read_text())
    unit["cost"]["estimated_usd"] = 1
    atomic_json(unit_path, unit)
    assert runner.run(path, out, usd_cap=1, workers=1)["status"] == "cap_stop"
    assert len(calls) == 1


def test_interrupted_replacement_preserves_previous_failure(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)

    def fail(task):
        raise RuntimeError("network failed")

    fake_runtime(monkeypatch, fail)
    out = tmp_path / "run"
    runner.run(path, out, usd_cap=1, workers=1)
    unit_path = runner._unit_path(out, "task#0")
    previous = unit_path.read_text()
    fake_runtime(monkeypatch)
    original = runner.atomic_json

    def interrupted(path, value):
        if path == unit_path:
            raise OSError("simulated interrupted commit")
        return original(path, value)

    monkeypatch.setattr(runner, "atomic_json", interrupted)
    with pytest.raises(OSError, match="interrupted commit"):
        runner.run(path, out, usd_cap=1, workers=1)
    assert unit_path.read_text() == previous
    monkeypatch.setattr(runner, "atomic_json", original)
    assert runner.run(path, out, usd_cap=1, workers=1)["status"] == "sealed"
    assert len(json.loads(unit_path.read_text())["attempt_history"]) == 1
