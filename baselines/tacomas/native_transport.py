"""Role-routed completion transport; native population/evolution is untouched."""
from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

from hotpot_fullwiki.common import atomic_json
from inherit_mas import release_config

WORKER_PHASES = {"worker", "final_action_compiler", "worker_synthesis_top_k", "worker_synthesis_memory"}
META_PHASES = {"meta_scaffolding", "meta_controller", "meta_initialization", "online_judge", "contribution_judge"}
SYNTHESIS_PHASES = {"worker_synthesis_top_k", "worker_synthesis_memory"}
_LOCAL = threading.local()


class BudgetStop(RuntimeError):
    pass


class ContextLimit(ValueError):
    pass


class RoleViolation(RuntimeError):
    pass


class Budget:
    def __init__(self, root, cap):
        self.root, self.cap = Path(root), cap

    @contextlib.contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "BUDGET.json"
            value = json.loads(path.read_text()) if path.exists() else {
                "cap_usd": self.cap, "spent_usd": 0.0, "reserved": {}}
            if value["cap_usd"] != self.cap:
                raise RuntimeError("budget cap changed")
            try:
                yield value
            finally:
                atomic_json(path, value)

    def reserve(self, amount):
        token = uuid.uuid4().hex
        with self.locked() as value:
            if value["spent_usd"] + sum(value["reserved"].values()) + amount > self.cap:
                atomic_json(self.root / "CAP_STOP.json", {"reason": "cost"})
                raise BudgetStop("provider cost reservation cap reached")
            value["reserved"][token] = amount
        return token

    def settle(self, token, cost):
        with self.locked() as value:
            bound = value["reserved"].pop(token)
            value["spent_usd"] += bound if cost is None else cost


def append_call(root, event):
    with (Path(root) / "api_calls.jsonl").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_calls(root):
    path = Path(root) / "api_calls.jsonl"
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        calls = [json.loads(line) for line in stream if line.strip()]
    for call in calls:
        if not isinstance(call, dict) or call.get("status") not in {"success", "failure"}:
            raise RuntimeError("malformed native provider ledger")
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if type(call.get(key)) is not int or call[key] < 0:
                raise RuntimeError("malformed native token usage")
        cost = call.get("estimated_cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise RuntimeError("malformed native cost usage")
        if call["total_tokens"] != call["input_tokens"] + call["output_tokens"]:
            raise RuntimeError("inconsistent native token usage")
    return calls


def usage_summary(calls):
    totals = {"calls": len(calls), "failed_calls": sum(c["status"] == "failure" for c in calls),
              "input_tokens": sum(c["input_tokens"] for c in calls),
              "output_tokens": sum(c["output_tokens"] for c in calls),
              "total_tokens": sum(c["total_tokens"] for c in calls),
              "estimated_usd": sum(c["estimated_cost_usd"] for c in calls), "by_model": {}}
    for model in sorted({str(c.get("model", "unknown")) for c in calls}):
        subset = [c for c in calls if c.get("model", "unknown") == model]
        totals["by_model"][model] = {"calls": len(subset),
            "total_tokens": sum(c["total_tokens"] for c in subset),
            "estimated_usd": sum(c["estimated_cost_usd"] for c in subset)}
    return totals


def role_scope(function, role):
    @functools.wraps(function)
    def scoped(*args, **kwargs):
        previous = getattr(_LOCAL, "phase", None)
        _LOCAL.phase = role
        try:
            return function(*args, **kwargs)
        finally:
            _LOCAL.phase = previous
    return scoped


def install_role_overlays():
    from tacomas.meta_evolution.mas_runtime import MetaEvolutionMASRuntime
    from tacomas.meta_evolution.meta_llm import MetaLLMInterface
    from tacomas.meta_evolution.orchestration_adapter import InitializationMetaLLM
    from tacomas.datasets.workbench_official import WorkBenchOfficialDataset
    for name, role in {"_infer_task_profile": "meta_scaffolding", "_infer_task_schema": "meta_scaffolding",
                       "_score_agent_contribution_llm": "contribution_judge",
                       "_synthesize_top_k_with_llm": "worker_synthesis_top_k",
                       "_synthesize_from_memory": "worker_synthesis_memory"}.items():
        setattr(MetaEvolutionMASRuntime, name, role_scope(getattr(MetaEvolutionMASRuntime, name), role))
    MetaLLMInterface.call_meta_llm = role_scope(MetaLLMInterface.call_meta_llm, "meta_controller")
    InitializationMetaLLM.initialize = role_scope(InitializationMetaLLM.initialize, "meta_initialization")
    WorkBenchOfficialDataset._judge_plan = staticmethod(role_scope(WorkBenchOfficialDataset._judge_plan, "online_judge"))


class QwenTokenCounter:
    def __init__(self, model_path):
        from jinja2 import Environment
        from tokenizers import Tokenizer
        root = Path(model_path)
        config = json.loads((root / "tokenizer_config.json").read_text())
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda obj: json.dumps(obj, ensure_ascii=False)
        self.template = env.from_string(config["chat_template"])
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

    def __call__(self, messages, tools):
        rendered = self.template.render(messages=messages, tools=tools or None,
                                        add_generation_prompt=True, enable_thinking=False)
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)


def logical_route(requested, role, backbone):
    requested = requested.removeprefix("openai/")
    if requested not in {"gpt-4o-mini", "gpt-5.4-mini", "qwen3-32b"}:
        raise RoleViolation(f"unapproved native model: {requested}")
    if backbone == "qwen3-32b" and role in SYNTHESIS_PHASES:
        return backbone
    if requested in {"gpt-4o-mini", "qwen3-32b"}:
        if role not in WORKER_PHASES:
            raise RoleViolation(f"worker requested in {role} scope")
        return backbone
    allowed = META_PHASES | (SYNTHESIS_PHASES if backbone == "gpt-4o-mini" else set())
    if role not in allowed:
        raise RoleViolation(f"unscoped meta call refused: {role}")
    return requested


def install_transport(config, *, completion_fn=None, token_counter=None):
    import litellm
    original = completion_fn or litellm.completion
    root, backbone = Path(config["out"]), config["backbone"]
    budget = Budget(root, config["usd_cap"])
    counter = (token_counter or QwenTokenCounter(config["tokenizer_path"])) if backbone == "qwen3-32b" else None
    document = release_config.credentials()

    def completion(*args, **kwargs):
        if args:
            raise RoleViolation("completion calls must name their model explicitly")
        if (root / "CAP_STOP.json").exists():
            raise BudgetStop("run cap is already reached")
        if (root / "INFRA_STOP.json").exists():
            raise RuntimeError("provider calls stopped after infrastructure failure")
        requested = str(kwargs.get("model", ""))
        role = getattr(_LOCAL, "phase", None) or ("worker" if requested != "openai/gpt-5.4-mini" else "meta_other")
        event = {"call_id": uuid.uuid4().hex, "phase": role, "native_requested_model": requested,
                 "instance_idx": int(os.getenv("TACOMAS_INSTANCE_IDX", "-1")), "started_at": time.time(),
                 "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
        reservation = None
        try:
            model = logical_route(requested, role, backbone)
            event["model"] = model
            messages, tools = kwargs.get("messages", []), kwargs.get("tools") or []
            for name in ("openai_api_key", "openai_api_base", "api_key", "api_base"):
                kwargs.pop(name, None)
            if kwargs.get("max_completion_tokens") is not None:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            kwargs.update(num_retries=0, timeout=240)
            if model == "qwen3-32b":
                if not tools:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                count = counter(messages, tools)
                available = config["context_tokens"] - count - 64
                requested_tokens = kwargs.get("max_tokens")
                if available < 128 or (requested_tokens is not None and int(requested_tokens) > available):
                    raise ContextLimit("Qwen explicit output budget does not fit its context")
                kwargs.update(model="openai/qwen3-32b", api_base=config["qwen_url"], api_key="EMPTY",
                              max_tokens=int(requested_tokens) if requested_tokens is not None else available)
                kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}),
                                         "chat_template_kwargs": {"enable_thinking": False}}
                event.update(prompt_tokens_preflight=count, max_tokens=kwargs["max_tokens"], enable_thinking=False)
                input_price = output_price = 0.0
            else:
                entry = release_config.deployment(document, "5.4-mini" if model == "gpt-5.4-mini" else model)
                endpoint = release_config.model_endpoint(document, entry)
                compatible = release_config.is_openai_compatible(endpoint)
                kwargs.update(model=("openai/" if compatible else "azure/") + entry["deployment"],
                              api_base=endpoint, api_key=release_config.api_key(entry["api_key_env"]))
                if not compatible:
                    kwargs["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", release_config.DEFAULT_API_VERSION)
                input_price, output_price = (0.75, 4.50) if model == "gpt-5.4-mini" else (0.15, 0.60)
                prompt_upper = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 4096
                output_upper = int(kwargs.get("max_tokens") or 128000)
                reservation = budget.reserve((prompt_upper * input_price + output_upper * output_price) / 1e6)
            response = original(**kwargs)
            usage = getattr(response, "usage", None)
            usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
            if not isinstance(usage, dict) or any(type(usage.get(k)) is not int for k in ("prompt_tokens", "completion_tokens")):
                raise RuntimeError("model response is missing token usage")
            pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
            if model == "qwen3-32b" and str(getattr(response, "model", "")).lower() not in {
                    "qwen3-32b", "qwen/qwen3-32b", "openai/qwen3-32b"}:
                raise RoleViolation("worker response model mismatch")
            cost = (pt * input_price + ct * output_price) / 1e6
            event.update(status="success", input_tokens=pt, output_tokens=ct,
                         total_tokens=pt + ct, estimated_cost_usd=cost)
            if reservation:
                budget.settle(reservation, cost)
                reservation = None
            return response
        except BaseException as exc:
            event.update(status="failure", error_type=type(exc).__name__)
            task_error = isinstance(exc, ContextLimit) or any(x in f"{type(exc).__name__}: {exc}".lower() for x in (
                "context_length", "maximum context length", "contextwindowexceeded", "content_filter", "content filter"))
            event["task_failure"] = task_error
            if not task_error and not isinstance(exc, BudgetStop):
                atomic_json(root / "INFRA_STOP.json", {"error_type": type(exc).__name__})
            if reservation:
                budget.settle(reservation, None)
            raise
        finally:
            event["wall_s"] = time.time() - event["started_at"]
            append_call(root, event)

    litellm.completion = completion
    return completion
