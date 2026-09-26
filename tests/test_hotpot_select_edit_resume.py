"""Offline resume identity, checkpoint, and cost-accounting regressions."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from hotpot_fullwiki import run_select_edit as runner
from hotpot_fullwiki.common import digest
from public_runner import models as public_models


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state = SimpleNamespace(
        out=tmp_path / "run", calls=Counter(), outcomes={},
        manifest={"dataset_revision": "snapshot", "manifest_digest": "manifest",
                  "rows": [{"id": "ok"}]},
        document={"api": {"base_url": "https://worker.example.invalid/v1", "models": {
            public_models.WORKER: {"model": "worker-deployment", "api_key_env": "WORKER_KEY"},
            public_models.META: {"model": "meta-deployment", "api_key_env": "META_KEY",
                                 "base_url": "https://meta.example.invalid/v1"}}}})
    monkeypatch.setattr(public_models.release_config, "credentials", lambda: state.document)
    monkeypatch.setenv("WORKER_KEY", "first-worker-secret")
    monkeypatch.setenv("META_KEY", "first-meta-secret")
    monkeypatch.setattr(runner.manifests, "load", lambda path: state.manifest)
    monkeypatch.setattr(runner, "open_snapshot", lambda: SimpleNamespace(digest="snapshot"))
    monkeypatch.setattr(runner, "load_examples", lambda rows, snapshot: [
        SimpleNamespace(id=row["id"], runtime=lambda task_id=row["id"]:
                        SimpleNamespace(id=task_id)) for row in rows])
    monkeypatch.setattr(runner, "BM25Retriever", lambda **kwargs:
                        SimpleNamespace(fingerprint={"retriever": "offline-fixture"}))
    monkeypatch.setattr(runner.Models, "api", lambda **kwargs:
                        SimpleNamespace(ledger=kwargs["ledger"]))

    def forbid_client(**kwargs):
        raise AssertionError("resume tests must not construct network clients")

    monkeypatch.setattr(public_models.release_config, "chat_client", forbid_client)

    def execute(example, *, models, retriever, run_dir):
        state.calls[example.id] += 1
        reservation = models.ledger.reserve(public_models.WORKER, [], [], 8)
        models.ledger.append({"task_id": example.id, "attempt": state.calls[example.id],
                              "estimated_usd": 0.25, "total_tokens": 10},
                             reservation=reservation)
        outcomes = state.outcomes.get(example.id, [])
        result = outcomes.pop(0) if outcomes else {"prediction": example.id, "success": True}
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(runner, "run_condition", execute)
    state.run = lambda **kwargs: runner.run(tmp_path / "manifest.json", state.out,
                                           usd_cap=10, workers=1, **kwargs)
    state.unit_path = lambda task_id: state.out / "units" / runner.CONDITION / f"{task_id}.json"
    return state


@pytest.mark.parametrize("role", [public_models.WORKER, public_models.META])
@pytest.mark.parametrize("field", ["model", "base_url"])
def test_api_identity_change_rejects_stale_run(harness, role, field):
    assert harness.run()["status"] == "sealed"
    before = harness.unit_path("ok").read_bytes()
    config = json.loads((harness.out / "RUN_CONFIG.json").read_text())
    assert config["model_settings"]["api"] == {
        public_models.WORKER: {"deployment": "worker-deployment",
                              "endpoint_digest": digest("https://worker.example.invalid/v1")},
        public_models.META: {"deployment": "meta-deployment",
                            "endpoint_digest": digest("https://meta.example.invalid/v1")}}
    harness.document["api"]["models"][role][field] = (
        "different-deployment" if field == "model" else "https://changed.example.invalid/v1")
    with pytest.raises(RuntimeError, match="different configuration"):
        harness.run()
    assert harness.calls == {"ok": 1}
    assert harness.unit_path("ok").read_bytes() == before


def test_api_key_rotation_does_not_change_identity(harness, monkeypatch):
    harness.run()
    config_path = harness.out / "RUN_CONFIG.json"
    before = config_path.read_bytes()
    monkeypatch.setenv("WORKER_KEY", "rotated-worker-secret")
    monkeypatch.setenv("META_KEY", "rotated-meta-secret")
    assert harness.run()["status"] == "sealed"
    assert config_path.read_bytes() == before
    assert harness.calls == {"ok": 1}
    for marker in ("secret", "WORKER_KEY", "META_KEY", "https://"):
        assert marker not in before.decode()


def test_only_infrastructure_failures_retry_and_history_costs_survive(harness):
    harness.manifest["rows"] = [{"id": task_id} for task_id in ("ok", "retry", "filter", "wrong")]
    harness.outcomes = {
        "retry": [ConnectionError("first transient outage"), TimeoutError("second transient outage")],
        "filter": [RuntimeError("content_filter rejected the request")],
        "wrong": [{"prediction": "incorrect", "success": False}],
    }
    first = harness.run()
    assert first["status"] == "incomplete"
    assert first["scorable_units"] == 3
    assert first["infra_failures"] == 1
    assert first["spent_usd"] == 1.0
    assert not (harness.out / "SEALED.json").exists()
    first_attempt = json.loads(harness.unit_path("retry").read_text())
    completed = {task_id: harness.unit_path(task_id).read_bytes()
                 for task_id in ("ok", "filter", "wrong")}
    ledger_path = harness.out / "api_calls.jsonl"
    initial_ledger = ledger_path.read_bytes()

    second = harness.run()
    assert second["status"] == "incomplete"
    assert second["recorded_units"] == 4
    assert second["spent_usd"] == 1.25
    second_attempt = json.loads(harness.unit_path("retry").read_text())
    assert second_attempt["attempt_history"] == [first_attempt]

    third = harness.run()
    assert third["status"] == "sealed"
    assert third["scorable_units"] == 4
    assert third["infra_failures"] == 0
    assert third["spent_usd"] == 1.5
    recovered = json.loads(harness.unit_path("retry").read_text())
    assert recovered["status"] == "complete"
    assert recovered["attempt_history"] == [first_attempt,
        {key: value for key, value in second_attempt.items() if key != "attempt_history"}]
    assert harness.calls == {"ok": 1, "retry": 3, "filter": 1, "wrong": 1}
    assert ledger_path.read_bytes().startswith(initial_ledger)
    assert len(ledger_path.read_text().splitlines()) == 6
    for task_id, value in completed.items():
        assert harness.unit_path(task_id).read_bytes() == value

    checkpoints = {path: path.read_bytes() for path in harness.unit_path("ok").parent.glob("*.json")}
    ledger_before = ledger_path.read_bytes()
    assert harness.run()["spent_usd"] == 1.5
    assert ledger_path.read_bytes() == ledger_before
    assert all(path.read_bytes() == value for path, value in checkpoints.items())


@pytest.mark.parametrize("failure", ["content_filter", "content filter", "context_length"])
def test_task_failures_are_terminal(harness, failure):
    harness.outcomes["ok"] = [RuntimeError(failure)]
    assert harness.run()["status"] == "sealed"
    before = harness.unit_path("ok").read_bytes()
    assert json.loads(before)["status"] == "task_failure"
    assert harness.run()["status"] == "sealed"
    assert harness.calls == {"ok": 1}
    assert harness.unit_path("ok").read_bytes() == before


def test_interruption_before_checkpoint_replace_preserves_failed_attempt(harness, monkeypatch):
    harness.outcomes["ok"] = [ConnectionError("transient outage")]
    harness.run()
    path = harness.unit_path("ok")
    before = path.read_bytes()
    original_atomic = runner.atomic_json

    def interrupted_atomic(target, value):
        if target == path:
            assert value["attempt_history"] == [json.loads(before)]
            raise OSError("simulated checkpoint interruption")
        original_atomic(target, value)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "atomic_json", interrupted_atomic)
        with pytest.raises(OSError, match="checkpoint interruption"):
            harness.run()
    assert path.read_bytes() == before
    report = harness.run()
    assert report["status"] == "sealed"
    assert report["spent_usd"] == 0.75
    assert harness.calls == {"ok": 3}
    assert json.loads(path.read_text())["attempt_history"] == [json.loads(before)]
    assert len((harness.out / "api_calls.jsonl").read_text().splitlines()) == 3


def test_budget_stop_keeps_prior_failure_and_ledger(harness, monkeypatch):
    harness.outcomes["ok"] = [ConnectionError("transient outage")]
    harness.run()
    path = harness.unit_path("ok")
    before = path.read_bytes()
    ledger_path = harness.out / "api_calls.jsonl"
    ledger_before = ledger_path.read_bytes()

    def cap_stop(*args, **kwargs):
        raise runner.BudgetExhausted("test cap reached")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "run_condition", cap_stop)
        report = harness.run()
    assert report["status"] == "cap_stop"
    assert report["recorded_units"] == 1
    assert report["infra_failures"] == 1
    assert report["spent_usd"] == 0.25
    assert path.read_bytes() == before
    assert ledger_path.read_bytes() == ledger_before
    assert harness.run()["status"] == "sealed"
    assert json.loads(path.read_text())["attempt_history"] == [json.loads(before)]


def test_qwen_resume_keeps_resolved_settings(harness, monkeypatch):
    settings = {"worker_model": public_models.QWEN, "meta_model": public_models.META,
                "qwen": {"provenance_digest": "offline-provenance"},
                "api": {public_models.META: {"deployment": "meta-deployment",
                                            "endpoint_digest": "offline-endpoint"}}}
    monkeypatch.setattr(public_models, "make_models", lambda ledger, backbone:
                        SimpleNamespace(ledger=ledger, settings=settings))
    assert harness.run(backbone=public_models.QWEN)["status"] == "sealed"
    payload = json.loads((harness.out / "RUN_CONFIG.json").read_text())
    assert payload["model_settings"] == settings
    settings["api"][public_models.META]["deployment"] = "changed-meta"
    with pytest.raises(RuntimeError, match="different configuration"):
        harness.run(backbone=public_models.QWEN)
    assert harness.calls == {"ok": 1}
