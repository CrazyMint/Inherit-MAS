"""Additive, fail-loud usage accounting for the paired TacoMAS run."""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


def _messages_text(kwargs: dict[str, Any]) -> str:
    messages = kwargs.get("messages") or []
    return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def _phase(model: str, prompt: str) -> str:
    low_model = model.lower()
    low = prompt.lower()
    if "gpt-4o-mini" in low_model:
        return "worker"
    if "expert grader. evaluate a candidate answer using the rubric" in low:
        return "rubric_judge"
    if "evaluator for multi-agent collaboration" in low:
        return "contribution_judge"
    if "meta-level evolutionary controller" in low or "meta-controller of a multi-agent" in low:
        return "meta_controller"
    if "classify the task into a general workflow profile" in low or "choose the best generic task schema" in low:
        return "meta_scaffolding"
    return "controller_other"


def _elapsed(start: Any, end: Any) -> float:
    if isinstance(start, dt.datetime) and isinstance(end, dt.datetime):
        return max(0.0, (end - start).total_seconds())
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    low = model.lower()
    if "gpt-4o-mini" in low:
        return prompt_tokens * 0.15e-6 + completion_tokens * 0.60e-6
    if "gpt-5.4-mini" in low or "gpt-5-4-mini" in low:
        return prompt_tokens * 0.75e-6 + completion_tokens * 4.50e-6
    return 0.0


class UsageLedger(CustomLogger):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, sort_keys=True, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = getattr(response_obj, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        usage = usage if isinstance(usage, dict) else {}
        model = str(getattr(response_obj, "model", "") or kwargs.get("model", ""))
        prompt = _messages_text(kwargs)
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        hidden = getattr(response_obj, "_hidden_params", {}) or {}
        reported_cost = hidden.get("response_cost")
        self._append({
            "status": "success",
            "instance_idx": int(os.environ.get("TACOMAS_INSTANCE_IDX", "-1")),
            "model": model,
            "phase": _phase(model, prompt),
            "input_tokens": pt,
            "output_tokens": ct,
            "total_tokens": tt,
            "cost_usd": float(reported_cost) if reported_cost is not None else _price(model, pt, ct),
            "estimated_cost_usd": _price(model, pt, ct),
            "wall_s": _elapsed(start_time, end_time),
            "prompt_chars": len(prompt),
            "cache_hit": bool(hidden.get("cache_hit", False)),
        })

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        model = str(kwargs.get("model", ""))
        prompt = _messages_text(kwargs)
        self._append({
            "status": "failure",
            "instance_idx": int(os.environ.get("TACOMAS_INSTANCE_IDX", "-1")),
            "model": model,
            "phase": _phase(model, prompt),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "estimated_cost_usd": 0.0,
            "wall_s": _elapsed(start_time, end_time),
            "prompt_chars": len(prompt),
            "error": str(response_obj)[:500],
        })


def install_usage_ledger(path: str | Path) -> UsageLedger:
    resolved = Path(path).resolve()
    for callback in litellm.callbacks:
        if isinstance(callback, UsageLedger) and callback.path.resolve() == resolved:
            return callback
    callback = UsageLedger(resolved)
    litellm.callbacks.append(callback)
    return callback
