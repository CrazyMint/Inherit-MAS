"""Observational cost and trajectory instrumentation for native EvoMAS."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fixed accounting tariffs for the evaluated configurations, not live quotes.
_PRICING: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-5.4-mini": (0.75e-6, 4.50e-6),
    "qwen3-32b": (0.0, 0.0),
}


def price_for(model_id: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD cost from token counts, or None if the model_id isn't priced."""
    if not model_id:
        return None
    key = model_id.split(":")[-1].split("/")[-1]
    best = None
    for name, rates in _PRICING.items():
        if name in model_id or name in key:
            if best is None or len(name) > len(best[0]):
                best = (name, rates)
    if best is None:
        return None
    ip, op = best[1]
    return input_tokens * ip + output_tokens * op


class CostLedger:
    """Thread-safe, append-only ledger of LLM interactions."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._local = threading.local()          # per-thread context
        self._global_ctx: Dict[str, Any] = {}    # fallback (last set)

    # -- context (step / task_id / config); best-effort tagging --
    def set_context(self, **kwargs) -> None:
        ctx = getattr(self._local, "ctx", None)
        if ctx is None:
            ctx = {}
            self._local.ctx = ctx
        ctx.update({k: v for k, v in kwargs.items() if v is not None})
        self._global_ctx.update({k: v for k, v in kwargs.items() if v is not None})

    def clear_context(self, *keys) -> None:
        ctx = getattr(self._local, "ctx", None)
        if ctx is None:
            return
        for k in keys or list(ctx.keys()):
            ctx.pop(k, None)

    def _ctx(self) -> Dict[str, Any]:
        merged = dict(self._global_ctx)
        merged.update(getattr(self._local, "ctx", {}) or {})
        return merged

    def context(self) -> Dict[str, Any]:
        """Public read of the current best-effort (step/task/config/...) context."""
        return self._ctx()

    # -- the one recording entrypoint --
    def record(
        self,
        phase: str,
        role: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        agent_id: Optional[str] = None,
        cost_usd: Optional[float] = None,
        seconds: float = 0.0,
        api_calls: int = 1,
        **extra,
    ) -> None:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        if cost_usd is None:
            cost_usd = price_for(model_id, input_tokens, output_tokens)
        ctx = self._ctx()
        rec = {
            "step": ctx.get("step"),
            "task_id": ctx.get("task_id"),
            "config": ctx.get("config"),
            "phase": phase,
            "role": role,
            "agent_id": agent_id,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "seconds": round(float(seconds or 0.0), 4),
            "api_calls": int(api_calls or 0),
        }
        if extra:
            rec.update(extra)
        with self._lock:
            self._records.append(rec)
            live_path = os.environ.get("EVOMAS_LIVE_LEDGER")
            if live_path:
                path = Path(live_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    def reset(self) -> None:
        with self._lock:
            self._records = []
        self._global_ctx = {}

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records)

    # -- rollups + persistence --
    def _rollup(self, key: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            k = str(r.get(key))
            b = out.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                   "total_tokens": 0, "cost_usd": 0.0, "seconds": 0.0})
            b["calls"] += 1
            b["input_tokens"] += r["input_tokens"]
            b["output_tokens"] += r["output_tokens"]
            b["total_tokens"] += r["total_tokens"]
            b["cost_usd"] += r["cost_usd"] or 0.0
            b["seconds"] += r["seconds"]
        for b in out.values():
            b["cost_usd"] = round(b["cost_usd"], 6)
            b["seconds"] = round(b["seconds"], 3)
        return out

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            tot_in = sum(r["input_tokens"] for r in self._records)
            tot_out = sum(r["output_tokens"] for r in self._records)
            tot_cost = sum((r["cost_usd"] or 0.0) for r in self._records)
            return {
                "totals": {
                    "calls": len(self._records),
                    "input_tokens": tot_in,
                    "output_tokens": tot_out,
                    "total_tokens": tot_in + tot_out,
                    "cost_usd": round(tot_cost, 6),
                },
                "by_phase": self._rollup("phase"),
                "by_role": self._rollup("role"),
                "by_model": self._rollup("model_id"),
                "by_agent": self._rollup("agent_id"),
                "by_step": self._rollup("step"),
                "by_task": self._rollup("task_id"),
            }

    def dump(self, path) -> Optional[str]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"summary": None, "records": list(self._records)}
        payload["summary"] = self.summary()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return str(path)


# Process-wide singleton used by the hooks.
LEDGER = CostLedger()


def record_worker_agents(mas_spec, context) -> None:
    """Record per-agent worker LLM cost from a completed MAS `context.trace`.

    Handles both EvoMAS-model agents (input/output_tokens in metadata) and
    SWE-agent agents (litellm `model_stats` with tokens_sent/received +
    instance_cost). Role/model are looked up from the MAS spec by agent_id.
    """
    try:
        agents = getattr(mas_spec, "agents", {}) or {}
        for entry in getattr(context, "trace", []) or []:
            aid = entry.get("agent_id")
            md = entry.get("metadata") or {}
            if not md:
                continue
            spec = agents.get(aid) if isinstance(agents, dict) else None
            role = getattr(spec, "role", None) or "worker"
            model_id = getattr(spec, "model_id", None) or ""
            in_tok = md.get("input_tokens", md.get("prompt_tokens", 0)) or 0
            out_tok = md.get("output_tokens", md.get("completion_tokens", 0)) or 0
            cost = None
            api_calls = 1
            ms = md.get("model_stats")
            if ms:  # SWE-agent (litellm) style
                in_tok = in_tok or ms.get("tokens_sent", 0) or 0
                out_tok = out_tok or ms.get("tokens_received", 0) or 0
                api_calls = ms.get("api_calls", api_calls) or api_calls
                if ms.get("instance_cost"):
                    cost = ms.get("instance_cost")
            if not in_tok and not out_tok and cost is None:
                continue  # error/empty entry -> nothing to attribute
            LEDGER.record(phase="worker", role=role, model_id=model_id,
                          input_tokens=in_tok, output_tokens=out_tok,
                          agent_id=aid, cost_usd=cost, api_calls=api_calls)
    except Exception:
        pass


def save_trajectory(output_dir, task_id: str, config: str, trace: List[Dict[str, Any]],
                    reports: Optional[Dict[str, str]] = None, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Persist a MAS execution trajectory (per-agent trace + reports) as JSON."""
    try:
        traj_dir = Path(output_dir) / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{task_id}__{config}")[:180]
        p = traj_dir / f"{safe}.json"
        payload = {"task_id": task_id, "config": config, "trace": trace}
        if reports is not None:
            payload["reports"] = reports
        if extra:
            payload.update(extra)
        with open(p, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return str(p)
    except Exception:
        return None
