"""WorkBench text-ReAct prompt/parser/loop, adapted from the pinned MIT source.

Provider routing and unused structured mode are omitted. A call callback replaces
the provider hook; prompt construction, parsing, and execution are unchanged.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from src.tools.tool import Tool, render_tool_description

PREFIX = "Respond to the human as helpfully and accurately as possible. You have access to the following tools:"

FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```"""

SUFFIX = "Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:."

ACT_WITHOUT_CONFIRMATION_SUFFIX = (
    " Do not ask for confirmation before executing actions."
    " Execute actions immediately and continue until the task is fully complete."
    " Do not stop after a search or lookup step."
)


def build_system_prompt(tools: list[Tool], datetime_prefix: str, act_without_confirmation: bool = False) -> str:
    tool_descriptions = "\n".join(render_tool_description(t) for t in tools)
    tool_names = ", ".join(f'"{t.name}"' for t in tools)
    format_block = FORMAT_INSTRUCTIONS.replace("{tool_names}", tool_names)
    suffix = SUFFIX + ACT_WITHOUT_CONFIRMATION_SUFFIX if act_without_confirmation else SUFFIX
    return datetime_prefix + "\n\n".join([PREFIX, tool_descriptions, format_block, suffix])


PARSE_ERROR = "__parse_error__"
FINAL_ANSWER = "Final Answer"
AGENT_STOPPED_MESSAGE = "Agent stopped due to iteration limit or time limit."


def parse_action(text: str) -> tuple[str, dict[str, str] | str]:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if not match:
        json_match = re.search(r'\{\s*"action"\s*:', text, re.DOTALL)
        if json_match:
            start = json_match.start()
            brace_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            raw = text[start:end]
        else:
            return FINAL_ANSWER, text.strip()
    else:
        raw = match.group(1).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return PARSE_ERROR, (
            f"Could not parse your action as JSON ({e}). "
            'Respond with a single JSON object with "action" and "action_input" keys.'
        )
    if not isinstance(parsed, dict):
        return PARSE_ERROR, 'Your action must be a JSON object with "action" and "action_input" keys.'
    action = parsed.get("action", FINAL_ANSWER)
    action_input = parsed.get("action_input", "")
    return action, action_input


@dataclass
class TraceStep:
    llm_input: str
    llm_output: str
    action: str
    action_input: dict[str, str] | str
    observation: str


@dataclass
class AgentResult:
    output: str
    intermediate_steps: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)


def run_agent(
    model_name: str,
    tools: list[Tool],
    task: str,
    datetime_prefix: str,
    max_iterations: int = 20,
    max_execution_time: float = 600,
    temperature: float = 0,
    act_without_confirmation: bool = False,
    *,
    call_llm,
) -> AgentResult:
    system_prompt = build_system_prompt(tools, datetime_prefix, act_without_confirmation)
    tool_map = {t.name: t for t in tools}
    scratchpad = ""
    steps: list[tuple[str, dict[str, str]]] = []
    trace: list[TraceStep] = []
    start_time = time.time()

    for _ in range(max_iterations):
        if time.time() - start_time > max_execution_time:
            return AgentResult(
                output=AGENT_STOPPED_MESSAGE,
                intermediate_steps=steps,
                trace=trace,
            )

        human_msg = task + "\n\nThought:" + scratchpad
        response_text = call_llm(model_name, system_prompt, human_msg, temperature)
        action, action_input = parse_action(response_text)

        if action == FINAL_ANSWER:
            output = action_input if isinstance(action_input, str) else json.dumps(action_input)
            trace.append(
                TraceStep(
                    llm_input=human_msg,
                    llm_output=response_text,
                    action=action,
                    action_input=action_input,
                    observation="",
                )
            )
            return AgentResult(output=output, intermediate_steps=steps, trace=trace)

        if action == PARSE_ERROR:
            observation = action_input if isinstance(action_input, str) else str(action_input)
            trace.append(
                TraceStep(
                    llm_input=human_msg,
                    llm_output=response_text,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )
            scratchpad += response_text + f"\nObservation: {observation}\nThought:"
            continue

        if action not in tool_map:
            observation = f"Tool '{action}' not found. Available tools: {', '.join(tool_map.keys())}"
        else:
            t = tool_map[action]
            if isinstance(action_input, dict):
                str_input = {k: str(v) for k, v in action_input.items()}
                observation = str(t(**str_input))
            else:
                observation = str(t(str(action_input)))

        trace.append(
            TraceStep(
                llm_input=human_msg,
                llm_output=response_text,
                action=action,
                action_input=action_input,
                observation=observation,
            )
        )
        steps.append((action, action_input if isinstance(action_input, dict) else {"input": str(action_input)}))
        scratchpad += response_text + f"\nObservation: {observation}\nThought:"

    return AgentResult(
        output=AGENT_STOPPED_MESSAGE,
        intermediate_steps=steps,
        trace=trace,
    )
