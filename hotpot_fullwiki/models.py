"""Shared local-Qwen and capped API clients for FullWiki nodes/meta/judge."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from . import config
from .common import canonical, digest

PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "qwen3-14b": {"input": 0.0, "output": 0.0},
    "qwen3-32b": {"input": 0.0, "output": 0.0},
}


class BudgetExhausted(RuntimeError):
    pass


class _Reservation(float):
    """USD reservation carrying a conservative provider-token reservation."""

    def __new__(cls, usd: float, tokens: int):
        value = float.__new__(cls, usd)
        value.tokens = int(tokens)
        return value


def load_dotenv(path: str | None = None) -> None:
    from inherit_mas import release_config
    target = Path(path).expanduser() if path is not None else release_config.env_file()
    if target is None or not target.exists():
        return
    for raw in target.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value.strip().strip("\"'"))


@dataclass(frozen=True)
class Usage:
    model: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    wall_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_usd(self) -> float:
        price = PRICES[self.model]
        return (self.prompt_tokens * price["input"] + self.completion_tokens * price["output"]) / 1_000_000

    def to_dict(self) -> dict:
        return {"model": self.model, "role": self.role,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens, "total_tokens": self.total_tokens,
                "wall_s": self.wall_s, "estimated_usd": self.estimated_usd}


@dataclass
class ModelResult:
    text: str
    usage: list[Usage] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    error: str = ""


class CallLedger:
    def __init__(self, path: str | Path, cap_usd: float, cap_tokens: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_usd = float(cap_usd)
        self.cap_tokens = int(cap_tokens)
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._reserved_tokens = 0

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @property
    def spent_usd(self) -> float:
        return sum(float(row.get("estimated_usd", 0.0)) for row in self.records())

    def reserve(self, model: str, messages: list[dict], tools: list[dict], max_tokens: int) -> float:
        chars = len(canonical({"messages": messages, "tools": tools}))
        price = PRICES[model]
        prompt_estimate = chars // 3 + 256
        token_estimate = prompt_estimate + int(max_tokens)
        estimate = (prompt_estimate * price["input"] + max_tokens * price["output"]) / 1_000_000
        with self._lock:
            records = self.records()
            spent = sum(float(row.get("estimated_usd", 0.0)) for row in records)
            if spent + self._reserved_usd + estimate > self.cap_usd:
                raise BudgetExhausted(
                    f"API cap exceeded: spent={spent:.4f}, pending={self._reserved_usd:.4f}, "
                    f"reserve={estimate:.4f}, cap={self.cap_usd:.4f}")
            spent_tokens = sum(int(row.get("total_tokens", 0)) for row in records)
            if (self.cap_tokens and
                    spent_tokens + self._reserved_tokens + token_estimate > self.cap_tokens):
                raise BudgetExhausted(
                    f"token cap exceeded: spent={spent_tokens}, pending={self._reserved_tokens}, "
                    f"reserve={token_estimate}, cap={self.cap_tokens}")
            self._reserved_usd += estimate
            self._reserved_tokens += token_estimate
        return _Reservation(estimate, token_estimate)

    def release(self, reservation: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - float(reservation))
            self._reserved_tokens = max(
                0, self._reserved_tokens - int(getattr(reservation, "tokens", 0)))

    def append(self, row: dict, *, reservation: float = 0.0) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - float(reservation))
            self._reserved_tokens = max(
                0, self._reserved_tokens - int(getattr(reservation, "tokens", 0)))
            with self.path.open("a") as handle:
                handle.write(canonical(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class Models:
    """A thin OpenAI-compatible client with one common tool loop."""

    def __init__(self, *, worker_client, worker_deployment: str, meta_client,
                 meta_deployment: str, provider: str, ledger: CallLedger | None):
        self.worker = worker_client
        self.worker_deployment = worker_deployment
        self.meta = meta_client
        self.meta_deployment = meta_deployment
        self.provider = provider
        self.ledger = ledger

    @classmethod
    def api(cls, *, ledger: CallLedger,
            credentials_path: str | None = None) -> "Models":
        from inherit_mas import release_config
        credentials = release_config.credentials(credentials_path)
        meta = release_config.deployment(credentials, "gpt-5.4-mini")
        worker = release_config.deployment(credentials, "gpt-4o-mini")
        meta_client = release_config.chat_client(
            endpoint=release_config.model_endpoint(credentials, meta),
            key=release_config.api_key(meta["api_key_env"]), timeout=240)
        worker_client = release_config.chat_client(
            endpoint=release_config.model_endpoint(credentials, worker),
            key=release_config.api_key(worker["api_key_env"]), timeout=180)
        return cls(worker_client=worker_client, worker_deployment=worker["deployment"],
                   meta_client=meta_client, meta_deployment=meta["deployment"],
                   provider="openai-compatible", ledger=ledger)

    @classmethod
    def local_qwen(cls, base_url: str) -> "Models":
        client = OpenAI(base_url=base_url.rstrip("/"), api_key="EMPTY",
                        max_retries=0, timeout=180)
        return cls(worker_client=client, worker_deployment=config.LOCAL_PREFLIGHT_MODEL,
                   meta_client=client, meta_deployment=config.LOCAL_PREFLIGHT_MODEL,
                   provider="local-vllm", ledger=None)

    def fingerprint(self, model: str) -> dict:
        if self.provider == "local-vllm":
            return {"provider": self.provider, "model": config.LOCAL_PREFLIGHT_MODEL,
                    "temperature": 0, "enable_thinking": False}
        deployment = self.meta_deployment if model == config.META_MODEL else self.worker_deployment
        return {"provider": self.provider, "logical_model": model,
                "deployment": deployment, "temperature": 0}

    def _logical(self, requested: str) -> str:
        return config.LOCAL_PREFLIGHT_MODEL if self.provider == "local-vllm" else requested

    def _create(self, *, model: str, role: str, messages: list[dict], max_tokens: int,
                tools: list[dict] | None = None, tool_choice: Any = None):
        logical = self._logical(model)
        tools = tools or []
        reservation = self.ledger.reserve(logical, messages, tools, max_tokens) if self.ledger else 0.0
        client = self.meta if model == config.META_MODEL else self.worker
        deployment = self.meta_deployment if model == config.META_MODEL else self.worker_deployment
        kwargs: dict[str, Any] = {"model": deployment, "messages": messages, "temperature": 0}
        if self.provider == "local-vllm":
            kwargs["max_tokens"] = int(max_tokens)
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif model == config.META_MODEL:
            kwargs["max_completion_tokens"] = int(max_tokens)
        else:
            kwargs["max_tokens"] = int(max_tokens)
        if tools:
            kwargs.update({"tools": tools, "tool_choice": tool_choice or "auto"})
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            if self.ledger:
                self.ledger.release(reservation)
            raise
        wall_s = time.perf_counter() - started
        if not response.choices:
            raise RuntimeError(f"{logical} returned no choices")
        raw_usage = response.usage
        details = getattr(raw_usage, "prompt_tokens_details", None)
        usage = Usage(logical, role,
                      int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                      int(getattr(raw_usage, "completion_tokens", 0) or 0),
                      int(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                      wall_s)
        if self.ledger:
            message = response.choices[0].message
            self.ledger.append({
                "schema": "hotpot_fullwiki_api_call/1",
                "request_digest": digest({"model": self.fingerprint(model),
                                          "messages": messages, "tools": tools}),
                "response_id": str(getattr(response, "id", "")),
                "response_text": message.content or "",
                "response_tool_calls": [
                    {"id": call.id, "name": call.function.name,
                     "arguments": call.function.arguments}
                    for call in (message.tool_calls or [])],
                **usage.to_dict(),
            }, reservation=reservation)
        return response, usage

    def chat(self, *, model: str, role: str, system: str, user: str,
             max_tokens: int) -> ModelResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response, usage = self._create(model=model, role=role, messages=messages,
                                       max_tokens=max_tokens)
        return ModelResult(response.choices[0].message.content or "", [usage], messages)

    def structured(self, *, role: str, system: str, user: str, max_tokens: int) -> ModelResult:
        return self.chat(model=config.META_MODEL, role=role, system=system, user=user,
                         max_tokens=max_tokens)

    def research_agent(self, *, role: str, system: str, user: str,
                       search_fn: Callable[[str], dict], max_turns: int,
                       max_tokens_per_turn: int, max_search_calls: int) -> ModelResult:
        tool = {"type": "function", "function": {
            "name": "search_fullwiki",
            "description": "BM25 search over the fixed FullWiki corpus. Returns titles and indexed sentences.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"], "additionalProperties": False}}}
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": user}]
        usages: list[Usage] = []
        events: list[dict] = []

        def finalize(reason: str) -> ModelResult:
            messages.append({"role": "user", "content":
                "Retrieval is closed. Using only prior observations, return the requested report or answer. "
                "Do not request another tool call and do not invent evidence."})
            response, usage = self._create(model=config.WORKER_MODEL,
                role=f"{role}:finalize", messages=messages, max_tokens=max_tokens_per_turn)
            usages.append(usage)
            text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": text})
            return ModelResult(text, usages, messages, events, reason)

        for _ in range(max_turns):
            response, usage = self._create(model=config.WORKER_MODEL, role=role,
                messages=messages, max_tokens=max_tokens_per_turn, tools=[tool])
            usages.append(usage)
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": call.id, "type": "function",
                    "function": {"name": call.function.name,
                                 "arguments": call.function.arguments}} for call in calls]
            messages.append(assistant)
            if not calls:
                return ModelResult(message.content or "", usages, messages, events)
            exhausted = False
            for call in calls:
                if len(events) >= max_search_calls:
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": "Retrieval budget exhausted."})
                    exhausted = True
                    continue
                try:
                    args = json.loads(call.function.arguments)
                    if call.function.name != "search_fullwiki" or set(args) != {"query"}:
                        raise ValueError("invalid search_fullwiki call")
                    result = search_fn(str(args["query"]))
                    observation = json.dumps(result, ensure_ascii=True)
                    event = {"query": str(args["query"]),
                             "titles": [row["title"] for row in result.get("results", [])],
                             "result": result, "error": result.get("error", "")}
                except Exception as exc:
                    from .retrieval import RetrievalError
                    if isinstance(exc, RetrievalError):
                        raise
                    observation = f"Retrieval error: {type(exc).__name__}: {exc}"
                    event = {"query": "", "titles": [], "result": {}, "error": observation}
                events.append(event)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": observation})
            if exhausted:
                return finalize("retrieval_budget_exhausted")
        return finalize("agent_turn_budget_exhausted")


def usage_totals(usages: list[Usage]) -> dict:
    return {"prompt_tokens": sum(row.prompt_tokens for row in usages),
            "completion_tokens": sum(row.completion_tokens for row in usages),
            "total_tokens": sum(row.total_tokens for row in usages),
            "cached_tokens": sum(row.cached_tokens for row in usages),
            "estimated_usd": sum(row.estimated_usd for row in usages),
            "calls": len(usages), "wall_s": sum(row.wall_s for row in usages),
            "records": [row.to_dict() for row in usages]}
