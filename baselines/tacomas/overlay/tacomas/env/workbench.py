"""Canonical read-only WorkBench environment for native TacoMAS evolution.

Intermediate evolutionary agents may inspect the pristine workplace state but
cannot mutate it.  A separate, single final executor applies the selected plan
after evolution.  This prevents exploratory fast rounds from contaminating the
state scored for the task.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from tacomas.env.base import AgentEnvironment
from tacomas.env.registry import register_env
from tacomas.workbench_bridge import canonical as W


ENV_NAME = "workbench-official"
DONE_TOOL = "done"


def alias_for(official_name: str) -> str:
    """Return an OpenAI-compatible function name for a dotted WorkBench tool."""
    return "wb__" + official_name.replace(".", "__")


READ_ALIAS_TO_OFFICIAL = {
    alias_for(tool.name): tool.name for tool in W.READ_ONLY_TOOLS
}
READ_TOOL_ALIASES = tuple(sorted(READ_ALIAS_TO_OFFICIAL))
PUBLIC_TOOL_NAMES = (*READ_TOOL_ALIASES, DONE_TOOL)

_JSON_TYPES: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _args_model(tool: Any) -> type[BaseModel]:
    fields: dict[str, tuple[type[Any], Any]] = {}
    for name, entry in tool.args_schema.items():
        value_type = _JSON_TYPES.get(str(entry.get("type", "string")), str)
        default = entry["default"] if "default" in entry else ...
        fields[name] = (value_type, Field(default=default))
    safe_name = tool.name.replace(".", "_")
    return create_model(f"WorkBench_{safe_name}_Args", **fields)


def _read_tool(official_tool: Any, *, task_bound: bool) -> StructuredTool:
    official_name = official_tool.name

    def invoke(**kwargs: Any) -> str:
        if not task_bound:
            raise RuntimeError("WorkBench tools require a task-bound TacoMAS worker")
        # Pydantic materializes omitted optional fields as None.  The canonical
        # WorkBench functions expect those fields to remain omitted so their
        # Python defaults apply; stringifying None produces invalid date text.
        supplied = {name: value for name, value in kwargs.items() if value is not None}
        return W.call_tool(official_name, supplied)

    return StructuredTool.from_function(
        func=invoke,
        name=alias_for(official_name),
        description=(
            f"Canonical read-only WorkBench tool `{official_name}`. "
            f"{official_tool.description}"
        ),
        args_schema=_args_model(official_tool),
    )


class _DoneArgs(BaseModel):
    answer: str
    confidence_score: int = Field(ge=0, le=100)


@register_env(ENV_NAME)
class WorkBenchOfficialEnvironment(AgentEnvironment):
    """Task-local read-only view of the canonical WorkBench sandbox."""

    def __init__(self, *args: Any, **kwargs: Any):
        instance = kwargs.get("dataset_instance")
        self.final_proposal: str | None = None
        if instance is not None:
            W.reset_state()

        additional = {
            alias_for(tool.name): _read_tool(tool, task_bound=instance is not None)
            for tool in W.READ_ONLY_TOOLS
        }

        def submit(answer: str, confidence_score: int) -> str:
            self.final_proposal = str(answer).strip()
            self.success = bool(self.final_proposal)
            return json.dumps(
                {"proposal": self.final_proposal, "confidence": int(confidence_score)},
                ensure_ascii=True,
                sort_keys=True,
            )

        additional[DONE_TOOL] = StructuredTool.from_function(
            func=submit,
            name=DONE_TOOL,
            description=(
                "Submit the best final WorkBench action plan. This records a proposal "
                "only; it does not execute any state-changing tool."
            ),
            args_schema=_DoneArgs,
        )
        super().__init__(*args, additional_env_tools=additional, **kwargs)

    def env_done(self) -> bool:
        return self.final_proposal is not None


def role_permissions() -> dict[str, list[str]]:
    """Native TacoMAS role permissions over the public read-only contract."""
    reads = list(READ_TOOL_ALIASES)
    return {
        "planner": [],
        "searcher": reads,
        "researcher": reads,
        "analyst": reads,
        "curator": reads,
        "verifier": reads,
        "schema_verifier": reads,
        "auditor": reads,
        "critic": reads,
        "calculator": reads,
        "forecaster": reads,
        "reflector": [],
        "summary": [DONE_TOOL],
        "synthesizer": [DONE_TOOL],
        "worker": reads,
        "subagent": reads,
    }
