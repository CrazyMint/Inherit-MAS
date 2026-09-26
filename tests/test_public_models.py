"""Offline backbone routing and accounting checks."""
import json
from types import SimpleNamespace

import pytest

from hotpot_fullwiki.models import CallLedger
from public_runner import models


class FakeClient:
    def __init__(self, model):
        self.model = model
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(model=self.model, id="fake",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None))


def settings(backbone, need_meta=True):
    result = {"worker_model": backbone, "meta_model": models.META if need_meta else backbone,
              "qwen": {"provenance_digest": "test", "urls": ["http://localhost:8000/v1"]},
              "api": {}}
    for model in (models.WORKER, models.META):
        result["api"][model] = {"deployment": model, "endpoint_digest": "example"}
    return result


@pytest.mark.parametrize("backbone", [models.WORKER, models.QWEN])
@pytest.mark.parametrize("need_meta", [False, True])
def test_worker_and_meta_routes(tmp_path, monkeypatch, backbone, need_meta):
    monkeypatch.setattr(models, "model_settings", settings)
    worker, meta = FakeClient(backbone), FakeClient(models.META)
    ledger = CallLedger(tmp_path / "calls.jsonl", 1)
    client = models.RoutedModels(ledger, backbone, need_meta,
        clients={backbone: (worker,), models.META: (meta,)})
    result = client.chat(model=models.WORKER, role="worker", system="system", user="task", max_tokens=90)
    client.structured(role="judge", system="judge", user="trace", max_tokens=60)
    assert result.usage[0].model == backbone
    assert len(worker.requests) == (1 if need_meta else 2)
    assert len(meta.requests) == (1 if need_meta else 0)
    assert worker.requests[0]["model"] == backbone
    assert worker.requests[0]["messages"] == [{"role": "system", "content": "system"},
                                               {"role": "user", "content": "task"}]
    assert worker.requests[0]["max_tokens"] == 90
    if backbone == models.QWEN:
        assert worker.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert result.usage[0].estimated_usd == 0
    if need_meta:
        assert meta.requests[0]["max_completion_tokens"] == 60
    assert sum(x["total_tokens"] for x in ledger.records()) == 24


def test_qwen_never_falls_back_to_api(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "model_settings", settings)
    bad = FakeClient("unexpected-model")
    ledger = CallLedger(tmp_path / "calls.jsonl", 1)
    client = models.RoutedModels(ledger, models.QWEN, False, clients={models.QWEN: (bad,)})
    with pytest.raises(RuntimeError, match="unexpected served model"):
        client.chat(model=models.WORKER, role="worker", system="", user="", max_tokens=2)
    assert len(bad.requests) == 1
    assert ledger.records()[0]["unknown_usage"] is True


def test_local_only_settings_do_not_load_api_credentials(tmp_path, monkeypatch):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"model": "qwen3-32b", "revision": "example"}))
    monkeypatch.setenv("INHERIT_QWEN_PROVENANCE", str(path))
    monkeypatch.setenv("INHERIT_QWEN_BASE_URLS", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(models.release_config, "load_env", lambda: None)
    def forbidden():
        raise AssertionError("local-only baseline cannot load API credentials")
    monkeypatch.setattr(models.release_config, "credentials", forbidden)
    config = models.model_settings(models.QWEN, need_meta=False)
    assert "api" not in config
    assert config["worker_model"] == config["meta_model"] == models.QWEN


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://localhost:8000", "http://key@localhost:8000/v1"])
def test_qwen_configuration_rejects_nonlocal_or_credential_urls(monkeypatch, url):
    monkeypatch.setattr(models.release_config, "load_env", lambda: None)
    monkeypatch.setenv("INHERIT_QWEN_BASE_URLS", url)
    with pytest.raises(ValueError, match="loopback"):
        models.qwen_settings()


@pytest.mark.parametrize("need_meta", [False, True])
def test_generic_credentials_reach_model_clients(tmp_path, monkeypatch, need_meta):
    document = {"api": {"base_url": "https://api.example.com/v1", "models": {
        models.WORKER: {"model": "my-worker", "api_key_env": "OPENAI_API_KEY"},
        models.META: {"model": "my-controller", "api_key_env": "CONTROLLER_API_KEY",
                      "base_url": "https://controller.example.com/v1"}}}}
    credentials = tmp_path / "credentials.json"
    credentials.write_text(json.dumps(document))
    monkeypatch.setenv("INHERIT_MAS_CREDENTIALS", str(credentials))
    monkeypatch.setenv("OPENAI_API_KEY", "worker-test-secret")
    monkeypatch.setenv("CONTROLLER_API_KEY", "controller-test-secret")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(models.release_config, "load_env", lambda: None)
    clients, construction = {}, []

    def client_factory(**kwargs):
        construction.append(kwargs)
        model = "my-controller" if "controller" in kwargs["endpoint"] else "my-worker"
        client = clients[model] = FakeClient(model)
        return client

    monkeypatch.setattr(models.release_config, "chat_client", client_factory)
    ledger = CallLedger(tmp_path / "calls.jsonl", 1)
    routed = models.make_models(ledger, models.WORKER, need_meta=need_meta)
    routed.chat(model=models.WORKER, role="worker", system="", user="task", max_tokens=70)
    routed.structured(role="judge", system="", user="trace", max_tokens=80)
    assert construction[0] == {"endpoint": "https://api.example.com/v1",
                               "key": "worker-test-secret", "timeout": 180}
    assert clients["my-worker"].requests[0]["model"] == "my-worker"
    if need_meta:
        assert construction[1] == {"endpoint": "https://controller.example.com/v1",
                                   "key": "controller-test-secret", "timeout": 240}
        assert clients["my-controller"].requests[0]["model"] == "my-controller"
    else:
        assert len(construction) == 1
        assert len(clients["my-worker"].requests) == 2
    assert routed.fingerprint()["provider"] == "openai-compatible"
    assert "azure" not in routed.settings
    assert "test-secret" not in (tmp_path / "calls.jsonl").read_text()
