"""Offline checks must not load the operator's credentials."""
import os

import pytest


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("AZURE_", "OPENAI_", "ANTHROPIC_", "OPENROUTER_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("INHERIT_MAS_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("INHERIT_MAS_CREDENTIALS", str(tmp_path / "absent.json"))
