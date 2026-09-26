from __future__ import annotations

import json
import os

import pytest
from openai import AzureOpenAI, OpenAI

from . import release_config as rc


EXAMPLE_CREDENTIALS = rc.ROOT / "credentials.example.json"
EXAMPLE_ENV = rc.ROOT / ".env.example"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("INHERIT_MAS_ENV_FILE", str(tmp_path / "absent.env"))


def test_generic_clients_and_legacy_endpoint_compatibility():
    assert not rc.is_openai_compatible("https://res.openai.azure.com")
    assert rc.is_openai_compatible("https://res.openai.azure.com/openai/v1")
    assert rc.is_openai_compatible("https://api.openai.com/v1")
    assert rc.is_openai_compatible("http://localhost:8000/v1/")
    assert rc.is_openai_compatible("https://gateway.example/api")

    azure = rc.chat_client(endpoint="https://res.openai.azure.com", key="k", timeout=1)
    assert isinstance(azure, AzureOpenAI)
    assert str(azure.base_url) == "https://res.openai.azure.com/openai/"

    compatible = rc.chat_client(endpoint="https://api.openai.com/v1", key="k", timeout=1)
    assert isinstance(compatible, OpenAI) and not isinstance(compatible, AzureOpenAI)
    assert str(compatible.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_endpoint_precedence_is_model_then_environment_then_credentials(monkeypatch):
    credentials = {"api": {"base_url": "https://from-credentials/v1"}}
    monkeypatch.delenv(rc.ENDPOINT_ENV, raising=False)
    assert rc.model_endpoint(credentials, {}) == "https://from-credentials/v1"
    monkeypatch.setenv(rc.ENDPOINT_ENV, "https://from-env/v1/")
    assert rc.model_endpoint(credentials, {}) == "https://from-env/v1"
    assert rc.model_endpoint(credentials, {"base_url": "https://from-model/v1"}) \
        == "https://from-model/v1"


def test_generic_config_is_not_overridden_by_legacy_provider_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://legacy.invalid")
    assert rc.model_endpoint({"api": {"base_url": "https://current.invalid/v1"}}) == "https://current.invalid/v1"


def test_model_names_keys_and_aliases():
    document = {"api": {"models": {"gpt-5.4-mini": {
        "model": "my-controller", "base_url": "https://meta.invalid/v1", "api_key_env": "META_KEY"}}}}
    entry = rc.deployment(document, "5.4-mini")
    assert entry == rc.deployment(document, "gpt-5.4-mini")
    assert entry["deployment"] == "my-controller"
    assert entry["api_key_env"] == "META_KEY"
    assert rc.model_endpoint(document, entry) == "https://meta.invalid/v1"


def test_missing_configuration_names_the_variable_or_key_that_is_missing(monkeypatch):
    monkeypatch.delenv(rc.ENDPOINT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=rc.ENDPOINT_ENV):
        rc.model_endpoint({}, {})
    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOME_MISSING_KEY"):
        rc.api_key("SOME_MISSING_KEY")
    with pytest.raises(RuntimeError, match=r"api.models\['gpt-5.4-mini'\]"):
        rc.deployment({"api": {"models": {}}}, "5.4-mini")


def test_example_credentials_and_env_stay_consistent_with_the_loader(monkeypatch):
    monkeypatch.delenv(rc.ENDPOINT_ENV, raising=False)
    credentials = json.loads(EXAMPLE_CREDENTIALS.read_text())
    env_text = EXAMPLE_ENV.read_text()
    for logical in ("gpt-4o-mini", "5.4-mini"):
        entry = rc.deployment(credentials, logical)
        assert entry["deployment"]
        assert f"{entry['api_key_env']}=" in env_text
        assert rc.model_endpoint(credentials, entry).startswith("https://")
    assert f"{rc.ENDPOINT_ENV}=" in env_text
    assert "azure" not in EXAMPLE_CREDENTIALS.read_text().lower()
    assert "azure" not in env_text.lower()


@pytest.mark.parametrize("url", ["not-a-url", "ftp://example.invalid", "https://user:secret@example.invalid/v1", "https://example.invalid/v1?key=secret"])
def test_invalid_api_urls_fail_before_client_creation(url):
    with pytest.raises(RuntimeError, match="base URL"):
        rc.model_endpoint({"api": {"base_url": url}})


def test_explicit_missing_env_does_not_load_another_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("SHOULD_NOT_LOAD=1\n")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setenv("INHERIT_MAS_ENV_FILE", str(tmp_path / "missing.env"))
    assert rc.env_file() is None


def test_generic_env_file_loads_without_replacing_exported_values(monkeypatch, tmp_path):
    path = tmp_path / "configured.env"
    path.write_text("OPENAI_BASE_URL=https://from-file.invalid/v1\nOPENAI_API_KEY=local-test-value\n")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    monkeypatch.setenv("INHERIT_MAS_ENV_FILE", str(path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://exported.invalid/v1")
    rc.load_env()
    assert os.environ["OPENAI_BASE_URL"] == "https://exported.invalid/v1"
    assert rc.api_key("OPENAI_API_KEY") == "local-test-value"
