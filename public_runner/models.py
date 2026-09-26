"""Portable API and local-Qwen routing with the existing agent tool loops."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from hotpot_fullwiki.common import digest, file_sha256
from hotpot_fullwiki.models import Models, ModelResult, Usage
from inherit_mas import release_config

QWEN = "qwen3-32b"
WORKER = "gpt-4o-mini"
META = "gpt-5.4-mini"


def qwen_settings() -> dict:
    release_config.load_env()
    urls = tuple(x.strip().rstrip("/") for x in os.environ.get(
        "INHERIT_QWEN_BASE_URLS", "http://127.0.0.1:8000/v1").split(",") if x.strip())
    if not urls:
        raise RuntimeError("INHERIT_QWEN_BASE_URLS must name at least one Qwen server")
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                "localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("Qwen endpoints must be loopback URLs ending in /v1")
    raw = os.environ.get("INHERIT_QWEN_PROVENANCE")
    if not raw or not Path(raw).expanduser().is_file():
        raise RuntimeError("set INHERIT_QWEN_PROVENANCE to the Qwen server provenance JSON file")
    path = Path(raw).expanduser()
    if not isinstance(json.loads(path.read_text()), dict):
        raise ValueError("Qwen provenance must be a JSON object")
    return {"model": QWEN, "urls": list(urls), "provenance_digest": file_sha256(path),
            "temperature": 0, "enable_thinking": False, "max_retries": 0, "timeout_s": 240}


def _model_entry(document: dict, model: str) -> dict:
    entry = release_config.deployment(document, "5.4-mini" if model == META else model)
    return {**entry, "endpoint": release_config.model_endpoint(document, entry)}


def model_settings(backbone: str, need_meta: bool = True, request_timeout_s: int | None = None) -> dict:
    if backbone not in {WORKER, QWEN}:
        raise ValueError(f"unsupported backbone: {backbone}")
    settings = {"worker_model": backbone, "meta_model": META if need_meta else backbone,
                "temperature": 0, "max_retries": 0}
    if request_timeout_s is not None:
        if type(request_timeout_s) is not int or request_timeout_s <= 0:
            raise ValueError("request timeout must be a positive integer")
        settings["request_timeout_s"] = request_timeout_s
    if backbone == QWEN:
        settings["qwen"] = qwen_settings()
    if backbone == WORKER or need_meta:
        document = release_config.credentials()
        selected = ([WORKER] if backbone == WORKER else []) + ([META] if need_meta else [])
        settings["api"] = {}
        for model in selected:
            entry = _model_entry(document, model)
            settings["api"][model] = {"deployment": entry["deployment"],
                                        "endpoint_digest": digest(entry["endpoint"])}
    return settings


class RoutedModels(Models):
    def __init__(self, ledger, backbone: str, need_meta: bool = True, *, clients=None,
                 request_timeout_s: int | None = None):
        self.settings = (model_settings(backbone, need_meta) if request_timeout_s is None
                         else model_settings(backbone, need_meta, request_timeout_s))
        self.worker_model = backbone
        self.meta_model = META if need_meta else backbone
        self.ledger = ledger
        self.provider = "role-routed"
        self._lock = threading.Lock()
        self._index = 0
        self._clients = {} if clients is None else dict(clients)
        if clients is None:
            if backbone == QWEN:
                from openai import OpenAI
                self._clients[QWEN] = tuple(OpenAI(base_url=url, api_key="EMPTY",
                    max_retries=0, timeout=request_timeout_s or 240) for url in self.settings["qwen"]["urls"])
            if backbone == WORKER or need_meta:
                document = release_config.credentials()
                for model in self.settings["api"]:
                    entry = _model_entry(document, model)
                    self._clients[model] = (release_config.chat_client(endpoint=entry["endpoint"],
                        key=release_config.api_key(entry["api_key_env"]),
                        timeout=request_timeout_s or (240 if model == META else 180)),)
        self.worker_deployment = self._deployment(backbone)
        self.meta_deployment = self._deployment(self.meta_model)

    def _logical(self, requested: str) -> str:
        if requested in {WORKER, QWEN}:
            if requested == QWEN and self.worker_model != QWEN:
                raise ValueError("Qwen requested in an API-worker configuration")
            return self.worker_model
        if requested == META:
            return self.meta_model
        raise ValueError(f"unsupported logical model: {requested}")

    def _deployment(self, model: str) -> str:
        return QWEN if model == QWEN else self.settings["api"][model]["deployment"]

    def fingerprint(self, model: str = WORKER) -> dict:
        logical = self._logical(model)
        if logical == QWEN:
            return {"provider": "local-vllm", "model": logical, **self.settings["qwen"]}
        return {"provider": "openai-compatible", "logical_model": logical,
                **self.settings["api"][logical], "temperature": 0}

    def _create(self, *, model, role, messages, max_tokens, tools=None, tool_choice=None):
        logical = self._logical(model)
        tools = tools or []
        reservation = self.ledger.reserve(logical, messages, tools, max_tokens)
        kwargs = {"model": self._deployment(logical), "messages": messages, "temperature": 0}
        kwargs["max_completion_tokens" if logical == META else "max_tokens"] = int(max_tokens)
        if logical == QWEN:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        with self._lock:
            pool = self._clients[logical]
            client = pool[self._index % len(pool)]
            self._index += 1
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                raise RuntimeError(f"{logical} returned no choices")
            if logical == QWEN and getattr(response, "model", None) != QWEN:
                raise RuntimeError("Qwen endpoint returned an unexpected served model")
            raw = response.usage
            prompt, completion = getattr(raw, "prompt_tokens", None), getattr(raw, "completion_tokens", None)
            if any(type(x) is not int or x < 0 for x in (prompt, completion)):
                raise RuntimeError("model response lacks exact token usage")
            details = getattr(raw, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            usage = Usage(logical, role, prompt, completion, cached, time.perf_counter() - started)
        except Exception as exc:
            self.ledger.append({"status": "error", "model": logical, "role": role,
                "usage_known": False, "error_type": type(exc).__name__,
                "wall_s": time.perf_counter() - started, "estimated_usd": 0.0,
                "total_tokens": 0, "unknown_usage": True}, reservation=reservation)
            raise
        message = response.choices[0].message
        self.ledger.append({"status": "ok", "usage_known": True,
            "request_digest": digest({"fingerprint": self.fingerprint(model), "request": kwargs}),
            "response_id": str(getattr(response, "id", "")), "response_text": message.content or "",
            "response_tool_calls": [{"id": call.id, "name": call.function.name,
                "arguments": call.function.arguments} for call in (message.tool_calls or [])],
            **usage.to_dict()}, reservation=reservation)
        return response, usage

    def readonly_agent(self, *, role, system, user, tool_names, max_turns=8,
                       max_tokens_per_turn=1024, max_tool_calls=20):
        import wb_env as W
        from workbench.task_evolution.models import sanitized_tool_schemas
        schemas, original = sanitized_tool_schemas([W.TOOL_BY_NAME[name] for name in tool_names])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        usages, events, final = [], [], ""
        for _ in range(max_turns):
            response, usage = self._create(model=self.worker_model, role=role, messages=messages,
                                           max_tokens=max_tokens_per_turn, tools=schemas)
            usages.append(usage)
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            assistant = {"role": "assistant", "content": message.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                    "name": call.function.name, "arguments": call.function.arguments}} for call in calls]
            messages.append(assistant)
            if not calls:
                final = message.content or ""
                break
            for call in calls:
                if len(events) >= max_tool_calls:
                    return ModelResult(final, usages, messages, events, "tool_call_budget_exhausted")
                name = original.get(call.function.name, call.function.name)
                try:
                    args = json.loads(call.function.arguments)
                    if name not in tool_names:
                        raise ValueError("tool is outside this node's read-only assignment")
                    observation, error = W.call_tool(name, args), ""
                except Exception as exc:
                    args = {}
                    observation = error = f"Tool call error: {type(exc).__name__}: {exc}"
                events.append({"tool": name, "args": args, "observation": observation, "error": error})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
        else:
            return ModelResult(final, usages, messages, events, "agent_turn_budget_exhausted")
        return ModelResult(final, usages, messages, events)


def make_models(ledger, backbone: str = WORKER, *, need_meta: bool = True,
                request_timeout_s: int | None = None):
    return RoutedModels(ledger, backbone, need_meta, request_timeout_s=request_timeout_s)
