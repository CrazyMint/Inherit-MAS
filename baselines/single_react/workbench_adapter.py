"""Gold-free Single ReAct through the unchanged official text-action loop."""
from __future__ import annotations

import time
from dataclasses import asdict

import openai
import wb_env as W
from hotpot_fullwiki.models import usage_totals

from .official_agent import AGENT_STOPPED_MESSAGE, run_agent

MAX_ITERATIONS = 20
MAX_TOKENS = 8192
TASK_CEILING_S = 1200.0


def run(task, *, models, backbone="gpt-4o-mini"):
    W.reset_state()
    started = time.perf_counter()
    usages = []

    def call_llm(model_name, system, user, temperature):
        if temperature != 0:
            raise ValueError("Single ReAct requires temperature zero")
        attempts = 3 if backbone == "gpt-4o-mini" else 1
        for attempt in range(attempts):
            try:
                result = models.chat(model="gpt-4o-mini", role="single_agent", system=system,
                                     user=user, max_tokens=MAX_TOKENS)
                break
            except (openai.APITimeoutError, TimeoutError):
                raise
            except (openai.APIConnectionError, openai.RateLimitError):
                if attempt == attempts - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
            except openai.APIStatusError as exc:
                if exc.status_code < 500 or attempt == attempts - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        usages.extend(result.usage)
        return result.text

    try:
        result = run_agent("gpt-4o-mini", list(W.ALL_TOOLS), task.task, W.DATETIME_PREFIX,
                           max_iterations=MAX_ITERATIONS, max_execution_time=TASK_CEILING_S,
                           temperature=0, call_llm=call_llm)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_model_tool_call: {exc}") from exc
    elapsed = time.perf_counter() - started
    failure = "agent_stopped" if result.output == AGENT_STOPPED_MESSAGE else ""
    if elapsed > TASK_CEILING_S and not failure:
        failure = "task_ceiling_exceeded"
    return {"task_id": task.id, "prediction": [
        W.convert_intermediate_step_to_function_call(name, args)
        for name, args in result.intermediate_steps],
        "final_output": result.output, "trace": [asdict(step) for step in result.trace],
        "task_failure": failure, "usage": usage_totals(usages),
        "tool_calls": len(result.intermediate_steps),
        "write_tool_calls": sum(name in W.SIDE_EFFECT_TOOL_NAMES
                                for name, _ in result.intermediate_steps),
        "elapsed_wall_s": elapsed, "gold_observed_online": False,
        "node_output_reuse": False}
