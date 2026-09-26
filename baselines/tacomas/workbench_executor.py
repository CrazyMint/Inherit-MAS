"""Single final WorkBench executor for the native TacoMAS adapter.

The LLM call is an action compiler, not another autonomous solver: it receives
the public task, TacoMAS's selected plan, and write-tool signatures only.  It
cannot query WorkBench state.  The resulting typed actions are validated and
then executed deterministically against one pristine official sandbox.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from litellm import completion

from tacomas.workbench_bridge import canonical as W


MODEL = "openai/gpt-4o-mini"
MAX_ACTIONS = 12


def compiler_prompt(task: str, evolved_plan: str) -> str:
    return f"""You are the single final action compiler for a workplace agent.
Translate the evolved plan into the minimal ordered list of official WRITE
actions that should actually be executed. Do not solve missing state lookups
yourself and do not invent IDs or values: the evolved plan must supply them.
Return JSON only with this schema:
{{"actions":[{{"tool":"official.dotted_tool_name","args":{{"arg":"value"}}}}],
"reason":"short note"}}
Use an empty actions list when the public task requires no state change.

Public task:
{task}

Evolved TacoMAS plan:
{evolved_plan}

Official write-tool signatures:
{W.write_tool_schema_block()}"""


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("executor compiler returned no JSON object")


def validate_actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("actions must be a list")
    if len(value) > MAX_ACTIONS:
        raise ValueError(f"actions exceeds maximum {MAX_ACTIONS}")
    validated: list[dict[str, Any]] = []
    for index, action in enumerate(value):
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            raise ValueError(f"action {index} must contain exactly tool and args")
        name = action["tool"]
        args = action["args"]
        if name not in W.SIDE_EFFECT_TOOL_NAMES:
            raise ValueError(f"action {index} is not an official write tool: {name!r}")
        if not isinstance(args, dict):
            raise ValueError(f"action {index} args must be an object")
        schema = W.TOOL_BY_NAME[name].args_schema
        unknown = set(args) - set(schema)
        missing = {
            field for field, entry in schema.items()
            if "default" not in entry and field not in args
        }
        if unknown or missing:
            raise ValueError(
                f"action {index} argument mismatch: unknown={sorted(unknown)} "
                f"missing={sorted(missing)}"
            )
        validated.append({"tool": name, "args": dict(args)})
    return validated


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is not None and hasattr(usage, "__dict__"):
        usage = vars(usage)
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("prompt_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": output,
        "total_tokens": int(usage.get("total_tokens") or prompt + output),
    }


def compile_actions(
    task: str,
    evolved_plan: str,
    *,
    completion_fn: Callable[..., Any] = completion,
) -> dict[str, Any]:
    api_key = (
        os.getenv("TACOMAS_AGENT_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    api_base = (
        os.getenv("TACOMAS_AGENT_API_BASE", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    )
    if not api_key and completion_fn is completion:
        raise RuntimeError("WorkBench executor requires a GPT-4o-mini API key")
    response = completion_fn(
        model=MODEL,
        messages=[{"role": "user", "content": compiler_prompt(task, evolved_plan)}],
        api_key=api_key or None,
        api_base=api_base or None,
        temperature=0.0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        num_retries=0,
    )
    text = str(response.choices[0].message.content or "")
    payload = _json_object(text)
    actions = validate_actions(payload.get("actions"))
    return {
        "actions": actions,
        "reason": str(payload.get("reason", ""))[:500],
        "raw_response": text,
        "usage": _usage(response),
    }


def execute_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute one validated action sequence and return its official call trace."""
    actions = validate_actions(actions)
    W.reset_state()
    trace: list[dict[str, Any]] = []
    error = ""
    rendered: list[str] = []
    try:
        for action in actions:
            name, args = action["tool"], action["args"]
            call = W.render_action(name, args)
            rendered.append(call)
            try:
                observation = W.call_tool(name, args)
                trace.append({
                    "tool": name,
                    "args": args,
                    "rendered_action": call,
                    "observation": observation,
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.append({
                    "tool": name,
                    "args": args,
                    "rendered_action": call,
                    "error": error,
                })
                break
    finally:
        W.reset_state()
    return {"prediction": rendered, "trace": trace, "execution_error": error}
