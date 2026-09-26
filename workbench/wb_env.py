"""WorkBench runtime adapter for the pinned vendored tools and sandbox.

The upstream revision is recorded in ``vendor/workbench_upstream/.pinned_commit``.
Scoring uses the upstream outcome-based ``is_correct`` and ``has_side_effects``
functions.

Bootstrap imports upstream from the vendor directory, then makes its database
paths absolute so subsequent sandbox reads do not depend on the working directory.
"""
from __future__ import annotations

import os
import sys
import contextlib
from typing import Any, Callable

VENDOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "workbench_upstream")
_PINNED_FILE = os.path.join(VENDOR_ROOT, ".pinned_commit")
PINNED_COMMIT = open(_PINNED_FILE).read().strip() if os.path.exists(_PINNED_FILE) else "UNKNOWN"


@contextlib.contextmanager
def _chdir(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _bootstrap() -> None:
    """Make the vendored upstream importable and CWD-independent (idempotent)."""
    if VENDOR_ROOT not in sys.path:
        sys.path.insert(0, VENDOR_ROOT)
    # Import upstream with CWD pinned to the vendor root (covers any relative reads
    # at import time), then rewrite the CSV paths to absolute so RUNTIME reads never
    # depend on CWD again.
    with _chdir(VENDOR_ROOT):
        from src.tools import state as _state  # noqa: PLC0415

        if not getattr(_state, "_INHERIT_MAS_ABS_PATCHED", False):
            _state._CSV_PATHS = {k: os.path.join(VENDOR_ROOT, v) for k, v in _state._CSV_PATHS.items()}
            _state._pristine = None  # force reload from the absolute paths
            _state._INHERIT_MAS_ABS_PATCHED = True
        # Touch the pristine snapshot once so it is built while CWD is safe.
        _state._pristine_state()


_bootstrap()

# ---- upstream re-exports (single import surface for the whole port) -----------
from src.tools.state import ToolState, get_state, reset_state  # noqa: E402
from src.tools.tool import Tool, render_tool_description, tool_to_openai_schema  # noqa: E402
from src.tools.toolkits import all_tools, tools_with_side_effects  # noqa: E402
from src.evals.actions import (  # noqa: E402
    EXECUTED_STATE_FIELDS,
    convert_intermediate_step_to_function_call,
    execute_actions_and_reset_state,
)
from src.evals.evaluation import (  # noqa: E402
    SIDE_EFFECT_STATE_FIELDS,
    has_side_effects,
    is_correct,
    is_exact_match,
)

# ---- tool registries ----------------------------------------------------------
ALL_TOOLS: list[Tool] = list(all_tools)
SIDE_EFFECT_TOOLS: list[Tool] = list(tools_with_side_effects)
SIDE_EFFECT_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in SIDE_EFFECT_TOOLS)
READ_ONLY_TOOLS: list[Tool] = [t for t in ALL_TOOLS if t.name not in SIDE_EFFECT_TOOL_NAMES]
READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in READ_ONLY_TOOLS)
TOOL_BY_NAME: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

# The 5 recognised WorkBench domains (+ the crm alias used by some multi_domain tasks).
DOMAINS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
_DOMAIN_ALIASES = {"crm": "customer_relationship_manager"}


def normalize_domains(domains: list[str]) -> list[str]:
    """Normalize domain tags (notably ``crm`` -> ``customer_relationship_manager``),
    dedupe, and keep a stable order. See BENCHMARK_NOTES.md §2."""
    out: list[str] = []
    for d in domains:
        d = _DOMAIN_ALIASES.get(d, d)
        if d in DOMAINS and d not in out:
            out.append(d)
    return out


def tools_for_domains(domains: list[str], *, read_only: bool) -> list[Tool]:
    """Tools relevant to a task's domains. ``company_directory.find_email_address``
    is always included (needed to resolve names -> emails), matching upstream
    ``get_toolkits``. Domains are normalized first."""
    doms = normalize_domains(domains)
    pool = READ_ONLY_TOOLS if read_only else ALL_TOOLS
    picked = [t for t in pool if any(t.name.startswith(d + ".") for d in doms)]
    directory = [t for t in pool if t.name.startswith("company_directory.")]
    seen = {t.name for t in picked}
    return picked + [t for t in directory if t.name not in seen]


# The official fixed-clock prefix (must be present for date reasoning to match gold).
from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME  # noqa: E402

DATETIME_PREFIX = (
    f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
    f"and the current time is {HARDCODED_CURRENT_TIME.time()}. "
    f"Remember the current date and time when completing tasks. "
    f"Meetings must not start before 9am or end after 6pm."
)


# ---- sandbox helpers ----------------------------------------------------------
def call_tool(name: str, kwargs: dict[str, Any]) -> str:
    """Dispatch a single tool call against the current thread-local sandbox and
    return the observation string. Unknown tool -> a non-raising message (mirrors
    the agent loop). Callers own reset/isolation."""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return f"Tool '{name}' not found. Available tools: {', '.join(TOOL_BY_NAME)}"
    str_kwargs = {k: str(v) for k, v in kwargs.items()}
    return str(tool(**str_kwargs))


def render_action(name: str, kwargs: dict[str, Any]) -> str:
    """Render a (tool, kwargs) pair into the exact gold action-string format,
    e.g. ``email.delete_email.func(email_id="00000479")`` (upstream helper)."""
    return convert_intermediate_step_to_function_call(name, {k: str(v) for k, v in kwargs.items()})


def _tool_value_constraints() -> str:
    """The vendored tools' OWN declared argument-value constraints (each tool rejects
    other values at call time with a 'must be one of' error). Pulled verbatim from the
    upstream module constants — never hand-typed — so the block cannot drift from the
    API contract the executor actually enforces."""
    from src.tools import analytics as _ana, project_management as _pm  # noqa: PLC0415
    from src.tools import customer_relationship_manager as _crm  # noqa: PLC0415
    rows = [
        ("analytics.create_plot", "value_to_plot", _ana.VALID_VALUES_TO_PLOT),
        ("analytics.create_plot", "plot_type", _ana.VALID_PLOT_TYPES),
        ("project_management.*", "list_name", _pm.VALID_LISTS),
        ("project_management.*", "board", _pm.VALID_BOARDS),
        ("customer_relationship_manager.*", "status", _crm.VALID_STATUSES),
        ("customer_relationship_manager.*", "product_interest", _crm.VALID_PRODUCT_INTERESTS),
    ]
    return "\n".join(f"  {tool}: {arg} must be one of {vals}" for tool, arg, vals in rows)


def write_tool_schema_block() -> str:
    """Official argument schemas for the 14 state-changing tools, as their upstream
    signatures (exact arg names + types) plus the tools' own declared value
    constraints. Given to the integrator/verifier so the model can emit VALID
    executable actions — the arg names here are exactly what
    ``schemas._validate_action`` accepts."""
    sigs = "\n".join(f"- {t.signature_str}" for t in SIDE_EFFECT_TOOLS)
    return sigs + "\nArgument value constraints (enforced by the tools):\n" + _tool_value_constraints()


# ---- the official oracle (verbatim upstream semantics) ------------------------
def score_prediction(pred_actions: list[str], gold_actions: list[str], error: str = "") -> dict[str, Any]:
    """Score one task with the OFFICIAL outcome-centric evaluation.

    Returns the two headline outcomes plus the secondary exact-match diagnostic:
      correct               task completion (final-state match), the primary metric
      unwanted_side_effect  harmful/unintended state change (and not correct)
      exact_match           side-effect call-set == gold (SECONDARY diagnostic only)

    `error` is the run-level error string (e.g. "Context window exceeded" or an
    agent-stopped marker); a non-empty error forces correct=False, exactly as
    upstream `compute_metrics` does.
    """
    pred = [a.replace("\n", "\\n") for a in pred_actions]
    gold = [a.replace("\n", "\\n") for a in gold_actions]
    correct_state = is_correct(pred, gold, "")
    correct = bool(correct_state and not error)
    side = bool(has_side_effects(pred, correct_state))
    return {
        "correct": correct,
        "unwanted_side_effect": side,
        "exact_match": bool(is_exact_match(pred, gold)),
        "num_pred_actions": len(pred_actions),
        "num_gold_actions": len(gold_actions),
        "error": error,
    }
