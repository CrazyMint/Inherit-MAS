"""Shared model configuration for OpenAI-compatible Chat Completions APIs.

Use credentials.json for model names and optional per-model settings. Set
OPENAI_BASE_URL and OPENAI_API_KEY in the checkout's .env or environment.
INHERIT_MAS_CREDENTIALS and INHERIT_MAS_ENV_FILE override the file locations.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]

ENDPOINT_ENV = "OPENAI_BASE_URL"
API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
DEFAULT_API_VERSION = "2024-12-01-preview"


def env_file() -> Path | None:
    configured = os.environ.get("INHERIT_MAS_ENV_FILE")
    target = Path(configured).expanduser() if configured else ROOT / ".env"
    return target if target.is_file() else None


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if os.environ.get("PYTHON_DOTENV_DISABLED", "").lower() in {"1", "true", "yes"}:
        return
    target = Path(path).expanduser() if path is not None else env_file()
    if target is None or not target.is_file():
        return
    for raw in target.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def credentials_path(explicit: str | os.PathLike | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    configured = os.environ.get("INHERIT_MAS_CREDENTIALS")
    path = Path(configured).expanduser() if configured else ROOT / "credentials.json"
    if not path.is_file():
        raise RuntimeError(
            f"credentials file not found at {path}. Copy credentials.example.json to "
            "credentials.json (or set INHERIT_MAS_CREDENTIALS) and configure your "
            "API model names; see README.md 'Configure'.")
    return path


def credentials(explicit: str | os.PathLike | None = None) -> dict:
    """Load .env, then return the parsed credentials document."""
    load_env()
    value = json.loads(credentials_path(explicit).read_text())
    if not isinstance(value, dict):
        raise RuntimeError("credentials.json must contain a JSON object")
    return value


def deployment(credentials_doc: dict, logical_model: str) -> dict:
    """Normalize a logical model's API settings for all existing adapters."""
    generic = "api" in credentials_doc
    section = credentials_doc.get("api" if generic else "azure_openai", {})
    entries = section.get("models" if generic else "deployments", {})
    canonical = "gpt-5.4-mini" if logical_model == "5.4-mini" else logical_model
    aliases = (canonical, "5.4-mini") if canonical == "gpt-5.4-mini" else (canonical,)
    entry = next((entries[name] for name in aliases if name in entries), None)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"credentials.json has no api.models['{canonical}']; "
            "copy credentials.example.json and configure the model.")
    name = entry.get("model") if generic else entry.get("deployment")
    key_env = entry.get("api_key_env", section.get("api_key_env", "OPENAI_API_KEY"))
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"missing model name for {canonical}")
    if not isinstance(key_env, str) or not key_env.strip():
        raise RuntimeError(f"missing api_key_env for {canonical}")
    result = {**entry, "deployment": name.strip(), "api_key_env": key_env.strip()}
    if entry.get("base_url"):
        result["endpoint"] = entry["base_url"]
    return result


def model_endpoint(credentials_doc: dict | None = None,
                   entry: dict | None = None) -> str:
    """Resolve the base URL: per-model, environment, then shared configuration."""
    document = credentials_doc or {}
    generic = "api" in document
    shared = (document.get("api", {}).get("base_url") if generic
              else document.get("azure_openai", {}).get("endpoint"))
    legacy = None if generic else os.environ.get("AZURE_OPENAI_ENDPOINT")
    value = ((entry or {}).get("base_url") or (entry or {}).get("endpoint")
             or os.environ.get(ENDPOINT_ENV) or legacy or shared)
    if not value:
        raise RuntimeError(
            f"missing {ENDPOINT_ENV}: set it in .env or api.base_url in "
            "credentials.json to your OpenAI-compatible API base URL.")
    parsed = urlsplit(str(value))
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise RuntimeError("API base URL must be HTTP(S) without credentials, query, or fragment")
    return str(value).rstrip("/")


def is_openai_compatible(endpoint: str) -> bool:
    """Use the standard client except for recognized legacy resource endpoints."""
    parsed = urlsplit(endpoint)
    legacy_host = (parsed.hostname or "").endswith((".openai.azure.com", ".cognitiveservices.azure.com"))
    return not legacy_host or parsed.path.rstrip("/").endswith("/v1")


def api_key(env_name: str) -> str:
    """Return the key held by `env_name`, naming that variable when it is unset."""
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(
            f"missing {env_name}: the credentials file names it as api_key_env, so put "
            f"{env_name}=<key> in .env (or export it); see .env.example.")
    return value


def chat_client(*, endpoint: str, key: str, timeout: float,
                max_retries: int = 0) -> Any:
    """Build the client the endpoint shape requires. No retries, no fallbacks."""
    from openai import AzureOpenAI, OpenAI  # noqa: PLC0415
    if is_openai_compatible(endpoint):
        return OpenAI(base_url=endpoint, api_key=key, max_retries=max_retries,
                      timeout=timeout)
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                       api_version=os.environ.get(API_VERSION_ENV, DEFAULT_API_VERSION),
                       max_retries=max_retries, timeout=timeout)


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser()


def ensure_hf_home() -> Path:
    path = hf_home()
    os.environ.setdefault("HF_HOME", str(path))
    return path


def pyserini_cache() -> Path:
    return Path(os.environ.get("PYSERINI_CACHE") or Path.home() / ".cache" / "pyserini").expanduser()


def ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    java = shutil.which("java")
    if java:
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parents[1])
