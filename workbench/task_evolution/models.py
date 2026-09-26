"""Injected model backends for task-time Inherit-MAS.

GPT-5.4-mini is used only for synthesis/refinement/judging. GPT-4o-mini is used
for workflow nodes. Exact usage is recorded per call; public list-price USD is
an estimate and is kept separate from quality scores.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import wb_env as W

LIST_PRICE_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "nemotron-3-ultra-free": {"input": 0.0, "output": 0.0},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}
NEMOTRON_OPENROUTER_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
GEMINI_OPENROUTER_ID = "google/gemini-3-flash-preview"
OPENROUTER_MODELS = {
    "nemotron-3-ultra-free": NEMOTRON_OPENROUTER_ID,
    "gemini-3-flash-preview": GEMINI_OPENROUTER_ID,
}


@dataclass
class Usage:
    model: str
    role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    wall_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_usd(self) -> float:
        price = LIST_PRICE_PER_MILLION[self.model]
        return (self.prompt_tokens * price["input"] + self.completion_tokens * price["output"]) / 1_000_000

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "role": self.role,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": self.estimated_usd,
            "wall_s": self.wall_s,
        }


@dataclass
class ModelResult:
    text: str
    usage: list[Usage] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    error: str = ""


class APIModels:
    """OpenAI-compatible clients with no hidden retries."""

    def __init__(self, credentials_path: str | None = None):
        from inherit_mas import release_config
        from openai import OpenAI

        credentials = release_config.credentials(credentials_path)
        meta = release_config.deployment(credentials, "gpt-5.4-mini")
        self.meta = release_config.chat_client(
            endpoint=release_config.model_endpoint(credentials, meta),
            key=release_config.api_key(meta["api_key_env"]), timeout=240)
        self.meta_deployment = meta["deployment"]

        worker = release_config.deployment(credentials, "gpt-4o-mini")
        self.worker = release_config.chat_client(
            endpoint=release_config.model_endpoint(credentials, worker),
            key=release_config.api_key(worker["api_key_env"]), timeout=180)
        self.worker_deployment = worker["deployment"]
        openrouter_key = os.environ.get("OPENROUTER_API_KEY_MAS") or os.environ.get("OPENROUTER_API_KEY_DEFAULT")
        self.openrouter = (OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key,
                                  max_retries=0, timeout=240) if openrouter_key else None)

    def fingerprint(self, model: str) -> dict:
        if model in OPENROUTER_MODELS:
            return {"provider": "openrouter", "model": model, "deployment": OPENROUTER_MODELS[model],
                    "reasoning_effort": "low", "reasoning_excluded": True}
        deployment = self.meta_deployment if model == "gpt-5.4-mini" else self.worker_deployment
        return {"provider": "openai-compatible", "model": model, "deployment": deployment}

    @staticmethod
    def _usage(resp, model: str, role: str, wall: float) -> Usage:
        usage = getattr(resp, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        return Usage(model, role, prompt, completion, cached, wall)

    def chat(self, *, model: str, role: str, system: str, user: str, max_tokens: int) -> ModelResult:
        if model in OPENROUTER_MODELS:
            if self.openrouter is None:
                raise RuntimeError("missing OPENROUTER_API_KEY_MAS/DEFAULT")
            client, deployment = self.openrouter, OPENROUTER_MODELS[model]
        else:
            client = self.meta if model == "gpt-5.4-mini" else self.worker
            deployment = self.meta_deployment if model == "gpt-5.4-mini" else self.worker_deployment
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        if model == "gpt-5.4-mini":
            kwargs["max_completion_tokens"] = int(max_tokens)
        else:
            kwargs["max_tokens"] = int(max_tokens)
        if model in OPENROUTER_MODELS:
            kwargs["extra_body"] = {"reasoning": {"effort": "low", "exclude": True}}
        started = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        wall = time.perf_counter() - started
        if not getattr(resp, "choices", None):
            raise RuntimeError(f"{model} returned no choices: {resp.model_dump_json()[:2000]}")
        return ModelResult(
            text=resp.choices[0].message.content or "",
            usage=[self._usage(resp, model, role, wall)],
            messages=kwargs["messages"],
        )

    def readonly_agent(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tool_names: list[str],
        max_turns: int = 8,
        max_tokens_per_turn: int = 1024,
        max_tool_calls: int = 20,
    ) -> ModelResult:
        tools = [W.TOOL_BY_NAME[name] for name in tool_names]
        schemas, original_by_sanitized = sanitized_tool_schemas(tools)
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        usages: list[Usage] = []
        events: list[dict] = []
        final = ""
        for _ in range(max_turns):
            started = time.perf_counter()
            kwargs = {"model": self.worker_deployment, "messages": messages, "temperature": 0,
                      "max_tokens": max_tokens_per_turn}
            if schemas:
                kwargs.update({"tools": schemas, "tool_choice": "auto"})
            resp = self.worker.chat.completions.create(**kwargs)
            usages.append(self._usage(resp, "gpt-4o-mini", role, time.perf_counter() - started))
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = []
                for call in calls:
                    assistant["tool_calls"].append({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    })
            messages.append(assistant)
            if not calls:
                final = msg.content or ""
                break
            for call in calls:
                if len(events) >= max_tool_calls:
                    return ModelResult(final, usages, messages, events, "tool_call_budget_exhausted")
                wire_name = call.function.name
                name = original_by_sanitized.get(wire_name, wire_name)
                try:
                    args = json.loads(call.function.arguments)
                    if name not in tool_names:
                        raise ValueError("tool is outside this node's read-only assignment")
                    observation = W.call_tool(name, args)
                    error = ""
                except Exception as exc:  # model error becomes an observation, never a write
                    args = {}
                    observation = f"Tool call error: {type(exc).__name__}: {exc}"
                    error = observation
                event = {"tool": name, "args": args, "observation": observation, "error": error}
                events.append(event)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
        else:
            return ModelResult(final, usages, messages, events, "agent_turn_budget_exhausted")
        return ModelResult(final, usages, messages, events)


def sanitized_tool_schemas(tools) -> tuple[list[dict], dict[str, str]]:
    """OpenAI-compatible function names cannot contain WorkBench's dots."""
    schemas = []
    original = {}
    for tool in tools:
        wire_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool.name)
        if wire_name in original and original[wire_name] != tool.name:
            raise ValueError(f"tool-name collision after sanitization: {tool.name}")
        original[wire_name] = tool.name
        schema = W.tool_to_openai_schema(tool)
        schema["function"]["name"] = wire_name
        schemas.append(schema)
    return schemas, original
