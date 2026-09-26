"""Code-faithful interactive EvoAgent bridge using GPT-4o-mini everywhere.

This follows the released ScienceWorld `collaboration_func`: generate exactly
one specialist, execute it from the same state, integrate its action with the
base action, and execute only the integrated action.  Unlike Algorithm 1's NLP
path, the released interactive path does not invoke LLMQuality.
"""
from __future__ import annotations

import time
from typing import Any

import wb_env as W
from hotpot_fullwiki import config
from hotpot_fullwiki.models import usage_totals
from baselines.single_react.official_agent import (
    AGENT_STOPPED_MESSAGE,
    FINAL_ANSWER,
    PARSE_ERROR,
    build_system_prompt,
    parse_action,
)

from . import prompts

MAX_ITERATIONS = 20
TASK_CEILING_S = 1200.0
MAX_TOKENS = 8192
ROLE_MAX_TOKENS = 1600
ACTION_CONTRACT = (
    'Return exactly one official ReAct action JSON object: '
    '{"action":"tool.name or Final Answer","action_input":{}}. '
    "Use an available tool name and its exact argument names."
)


def _chat(models, *, role: str, system: str, user: str, max_tokens: int, usages: list) -> str:
    result = models.chat(
        model=config.WORKER_MODEL,
        role=role,
        system=system,
        user=user,
        max_tokens=max_tokens,
    )
    usages.extend(result.usage)
    return result.text


def run(task, *, models) -> dict[str, Any]:
    started = time.perf_counter()
    W.reset_state()
    base_system = build_system_prompt(
        list(W.ALL_TOOLS), W.DATETIME_PREFIX, act_without_confirmation=True)
    scratchpad = ""
    actions: list[str] = []
    trace: list[dict] = []
    usages = []
    task_failure = ""
    descriptions: list[str] = []

    for step in range(MAX_ITERATIONS):
        if time.perf_counter() - started > TASK_CEILING_S:
            task_failure = "task_ceiling_exceeded"
            break
        state_text = task.task + "\n\nThought:" + scratchpad
        base_action = _chat(
            models, role=f"evoagent_same4o:base_action:{step}",
            system=base_system, user=state_text, max_tokens=MAX_TOKENS,
            usages=usages)

        role_system, role_user = prompts.propose_expert(
            state_text, base_action, tuple(), ACTION_CONTRACT)
        description = _chat(
            models, role=f"evoagent_same4o:meta_action:{step}",
            system=role_system, user=role_user, max_tokens=ROLE_MAX_TOKENS,
            usages=usages).strip()
        descriptions.append(description)

        specialist = _chat(
            models, role=f"evoagent_same4o:specialist_action:{step}",
            system=description + "\n\n" + base_system,
            user=state_text, max_tokens=MAX_TOKENS, usages=usages)

        integrate_system, integrate_user = prompts.integrate(
            state_text, base_action, description, specialist, ACTION_CONTRACT)
        integrated = _chat(
            models, role=f"evoagent_same4o:integrator_action:{step}",
            system=base_system + "\n\n" + integrate_system,
            user=state_text + "\n\n" + integrate_user,
            max_tokens=MAX_TOKENS, usages=usages)

        action, action_input = parse_action(integrated)
        step_trace: dict[str, Any] = {
            "step": step,
            "base_action": base_action,
            "description": description,
            "specialist_action": specialist,
            "integrated_action": integrated,
            "parsed_action": action,
            "parsed_action_input": action_input,
            "quality_check_called": False,
            "state_changed_before_integration": False,
        }
        if action == FINAL_ANSWER:
            step_trace["observation"] = ""
            trace.append(step_trace)
            break
        if action == PARSE_ERROR:
            observation = str(action_input)
            step_trace["observation"] = observation
            trace.append(step_trace)
            scratchpad += integrated + f"\nObservation: {observation}\nThought:"
            continue
        if action not in W.TOOL_BY_NAME:
            observation = f"Tool '{action}' not found. Available tools: {', '.join(W.TOOL_BY_NAME)}"
            step_trace["observation"] = observation
            trace.append(step_trace)
            scratchpad += integrated + f"\nObservation: {observation}\nThought:"
            continue
        if not isinstance(action_input, dict):
            task_failure = "invalid_model_tool_call: action_input_not_object"
            step_trace["observation"] = task_failure
            trace.append(step_trace)
            break
        try:
            rendered = W.render_action(action, action_input)
            observation = W.call_tool(action, action_input)
        except (TypeError, ValueError) as exc:
            task_failure = f"invalid_model_tool_call: {type(exc).__name__}: {exc}"
            step_trace["observation"] = task_failure
            trace.append(step_trace)
            break
        actions.append(rendered)
        step_trace.update({
            "observation": observation,
            "executed_action": rendered,
            "write": action in W.SIDE_EFFECT_TOOL_NAMES,
        })
        trace.append(step_trace)
        scratchpad += integrated + f"\nObservation: {observation}\nThought:"
    else:
        task_failure = AGENT_STOPPED_MESSAGE

    return {
        "schema": "adapted_evoagent_workbench_same4o/1",
        "system": "Adapted-EvoAgent-GPT-4o-mini-backbone",
        "task_id": task.id,
        "source": task.source,
        "base_template": task.base_template,
        "prediction": actions,
        "n_pred": len(actions),
        "trace": trace,
        "expert_descriptions": descriptions,
        "task_failure": task_failure,
        "usage": usage_totals(usages),
        "tool_calls": len(actions),
        "write_tool_calls": sum(bool(row.get("write")) for row in trace),
        "elapsed_wall_s": time.perf_counter() - started,
        "gold_observed_online": False,
        "node_output_reuse": False,
        "backbone_model": "gpt-4o-mini",
        "separate_meta_model": False,
        "quality_check_called": False,
    }
