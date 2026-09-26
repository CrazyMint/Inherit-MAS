"""WorkBench execution runner for typed role pipelines.

Executes a ``mas_family.SystemSpec`` against the WorkBench sandbox and returns the
scored prediction (a list of action strings) + a trace + a filled ``Ledger``.

The two model operations are injected and can be replaced with test doubles:

  complete_fn(role, system, user, ledger) -> str
      one chat completion (records tokens on the ledger); used by the LLM roles
      planner / coordinator / integrator / verifier.
  react_fn(role, tools, task_text, ledger) -> ReactResult
      a bounded ReAct tool-use loop restricted to ``tools`` (records tokens +
      tool calls); used by single_agent / executor_agent (WRITE tools) and by
      workers (READ-only tools). ``ReactResult.actions`` are gold-format action
      strings; ``.output`` is the final text.

Safety: workers are handed READ-ONLY tool sets; only the deterministic ``executor``
(or the S0/M1 write agent) ever emits state-changing actions — the
"only one executor may invoke state-changing tools" contract from mas_family.

Message modes transform worker evidence before integration or verification:
  normal    retain the worker evidence
  none      clear evidence items and summaries after workers finish
  shuffled  rotate facts and summaries between domain slots; leave a single
            worker's evidence unchanged
"""
from __future__ import annotations

from dataclasses import dataclass, field

import wb_env as W
import schemas as S
from ledger import Ledger
from mas_family import SystemSpec, SYSTEMS


@dataclass
class ReactResult:
    actions: list[str]          # gold-format tool-call strings the agent made
    output: str = ""            # final answer text (evidence JSON for a worker)
    error: str = ""


@dataclass
class RunOutput:
    system: str
    prediction: list[str]
    error: str = ""
    trace: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- prompts
def _plan_prompt(task: str) -> tuple[str, str]:
    sys = ("You are a planner. Given a workplace task, write a short numbered plan of the "
           "steps needed to complete it, naming the tools/records involved. Do not call tools.")
    return sys, task


def _coordinator_prompt(task_obj) -> tuple[str, str]:
    # Infer domains from task text without passing the dataset's domain labels.
    sys = ("You are a coordinator for a workplace assistant with five domains: email, calendar, "
           "analytics, project_management, customer_relationship_manager. Decide which domains the "
           "task needs and give a short goal per domain. If the task requires NO change to any "
           "record, set requires_writes=false and return empty subgoals. Reply with ONLY a JSON "
           'object: {"relevant_domains": [..], "subgoals": [{"domain": "..", "goal": ".."}], '
           '"requires_writes": true/false}.')
    user = f"{W.DATETIME_PREFIX}\n\nTask: {task_obj.task}"
    return sys, user


def _worker_prompt(domain: str, goal: str, task: str) -> str:
    return (f"You are the {domain} specialist. Using ONLY your read-only {domain} tools (and "
            f"company_directory to resolve names to emails), gather the exact facts needed for this "
            f"goal — record IDs, email addresses, dates, values. Do not change anything.\n"
            f"Goal: {goal}\nOverall task: {task}\n\n"
            f'When done, give your Final Answer as ONLY a JSON object: '
            f'{{"domain": "{domain}", "items": [{{"field": "..", "value": ".."}}], "summary": ".."}} '
            f'listing every resolved fact (e.g. the concrete record id/email/date) the writer will need.')


def _integrator_prompt(task_obj, evidence: list[dict]) -> tuple[str, str]:
    sys = ("You are the integrator. Using ONLY the workers' evidence, construct the ORDERED list of "
           "state-changing tool calls that complete the task. Use exact IDs/emails/dates from the "
           "evidence. Decide how many actions the task requires: (1) if the change is CONDITIONAL on a "
           "fact ('if/when/should X'), check the condition against the evidence — when it is false, "
           "return the empty list; (2) otherwise a direct request (plot/create/send/update/delete) "
           "requires at least one action — emit one action per requested item ('X and Y' means two "
           "actions). Never invent IDs. Reply with "
           'ONLY JSON: {"actions": [{"tool": "email.send_email", "args": {"recipient": "..", ...}}]}. '
           "Use ONLY these write tools, with EXACTLY these argument names:\n" + W.write_tool_schema_block())
    ev_lines = []
    for e in evidence:
        facts = "; ".join(f'{it["field"]}={it["value"]}' for it in e.get("items", []))
        ev_lines.append(f'[{e.get("domain")}] {facts} :: {e.get("summary", "")}')
    ev_block = "\n".join(ev_lines) if ev_lines else "(no evidence provided)"
    user = f"{W.DATETIME_PREFIX}\n\nTask: {task_obj.task}\n\nWorker evidence:\n{ev_block}"
    return sys, user


def _verifier_prompt(task_obj, proposed: list[dict], evidence: list[dict]) -> tuple[str, str]:
    sys = ("You are a safety verifier. Check the proposed state-changing actions against the task and "
           "evidence: correct recipients/IDs/dates/parameters, and REMOVE any action the task did not "
           "ask for (no unnecessary writes). Return the final vetted action list. Reply with ONLY JSON: "
           '{"verdict": "approve"|"revise", "actions": [{"tool": "..", "args": {..}}], "reasons": [".."]}.'
           "\nUse ONLY these write tools, with EXACTLY these argument names:\n" + W.write_tool_schema_block())
    prop = "; ".join(f'{a["tool"]}({a["args"]})' for a in proposed) or "(none)"
    ev_lines = [f'[{e.get("domain")}] ' + "; ".join(f'{it["field"]}={it["value"]}' for it in e.get("items", []))
                for e in evidence]
    ev_block = "\n".join(ev_lines) if ev_lines else "(no evidence)"
    user = f"{W.DATETIME_PREFIX}\n\nTask: {task_obj.task}\n\nProposed actions: {prop}\n\nEvidence:\n{ev_block}"
    return sys, user


# --------------------------------------------------------------------------- message modes
def _apply_message_mode(evs: list[dict], mode: str) -> list[dict]:
    if mode == "normal":
        return evs
    if mode == "none":
        return [{**e, "items": [], "summary": ""} for e in evs]
    if mode == "shuffled":
        if len(evs) < 2:
            return evs  # no other worker's evidence to rotate into this slot
        rot = evs[-1:] + evs[:-1]
        return [{**evs[i], "items": rot[i].get("items", []), "summary": rot[i].get("summary", "")}
                for i in range(len(evs))]
    raise ValueError(f"unknown message_mode {mode!r}")


# --------------------------------------------------------------------------- executor (det)
def _dispatch_writes(actions: list[dict], ledger: Ledger) -> list[str]:
    """Deterministically dispatch validated write actions against the live (pristine)
    sandbox and return their gold-format strings. Tolerant to per-action failure
    (mirrors upstream); scoring re-executes from pristine anyway."""
    rendered: list[str] = []
    for a in actions:
        rendered.append(W.render_action(a["tool"], a["args"]))
        try:
            W.call_tool(a["tool"], a["args"])
        except Exception:  # noqa: BLE001 — a bad call is contained; the string is still scored
            pass
        ledger.add_tool_call(write=True)
    return rendered


# --------------------------------------------------------------------------- the runner
def run_system(
    spec: SystemSpec | str,
    task,
    *,
    complete_fn,
    react_fn,
    ledger: Ledger | None = None,
    message_mode: str = "normal",
    strict: bool = False,
) -> RunOutput:
    """Run one MAS system on one task. Resets the sandbox to pristine first, so
    workers read pristine and the executor writes from pristine (matching scoring)."""
    if isinstance(spec, str):
        spec = SYSTEMS[spec]
    ledger = ledger or Ledger()
    W.reset_state()
    trace: dict = {"message_mode": message_mode, "invalids": []}

    if spec.name == "S0":
        with ledger.timed("solve"):
            res = react_fn("single_agent", W.ALL_TOOLS, task.task, ledger)
        return RunOutput("S0", res.actions, res.error, trace)

    if spec.name == "M1":
        with ledger.timed("plan"):
            plan = complete_fn("planner", *_plan_prompt(task.task), ledger)
        trace["plan"] = plan
        task_text = f"{task.task}\n\nA planner suggested this plan; use it as guidance:\n{plan}"
        with ledger.timed("act"):
            res = react_fn("executor_agent", W.ALL_TOOLS, task_text, ledger)
        return RunOutput("M1", res.actions, res.error, trace)

    # M2 / M3 -----------------------------------------------------------------
    with ledger.timed("coordinate"):
        c_sys, c_user = _coordinator_prompt(task)
        sub = S.parse_or_invalid("subgoals", complete_fn("coordinator", c_sys, c_user, ledger), strict=strict)
    trace["subgoals"] = sub
    if not sub.get("valid", False):
        trace["invalids"].append("subgoals")

    # worker fan-out: one read-only worker per relevant domain (0 => abstain)
    goal_by_domain = {s["domain"]: s["goal"] for s in sub.get("subgoals", [])}
    domains = sub.get("relevant_domains", []) or list(goal_by_domain)
    evidence: list[dict] = []
    with ledger.timed("workers"):
        for dom in domains:
            tools = W.tools_for_domains([dom], read_only=True)
            wtask = _worker_prompt(dom, goal_by_domain.get(dom, task.task), task.task)
            res = react_fn("worker", tools, wtask, ledger)
            ev = S.parse_or_invalid("evidence", res.output, strict=strict)
            ev["domain"] = dom  # pin the slot's domain regardless of what the LLM echoed
            if not ev.get("valid", False):
                trace["invalids"].append(f"evidence:{dom}")
                # OBSERVABILITY (non-behavioral): persist the raw worker output + the
                # parser error so invalid evidence can be diagnosed. The pipeline value
                # (the invalid marker in `evidence`) is unchanged. raw_output capped to
                # keep the artifact bounded.
                raw = res.output if len(res.output) <= 8000 else res.output[:8000] + "…[truncated]"
                trace.setdefault("evidence_debug", []).append({
                    "domain": dom, "task": task.id,
                    "parser_error": S.explain_invalid("evidence", res.output),
                    "raw_output": raw,
                })
            evidence.append(ev)
    trace["evidence"] = evidence
    trace["n_workers"] = len(evidence)

    ev_for_integ = _apply_message_mode(evidence, message_mode)
    with ledger.timed("integrate"):
        i_sys, i_user = _integrator_prompt(task, ev_for_integ)
        proposed = S.parse_or_invalid("proposed_actions", complete_fn("integrator", i_sys, i_user, ledger),
                                      strict=strict)
    trace["proposed_actions"] = proposed
    if not proposed.get("valid", False):
        trace["invalids"].append("proposed_actions")
    final_actions = proposed.get("actions", [])

    if spec.name == "M3":
        with ledger.timed("verify"):
            v_sys, v_user = _verifier_prompt(task, final_actions, _apply_message_mode(evidence, message_mode))
            decision = S.parse_or_invalid("verifier_decision", complete_fn("verifier", v_sys, v_user, ledger),
                                          strict=strict)
        trace["decision"] = decision
        if decision.get("valid", False):
            final_actions = decision["actions"]          # verifier's vetted list wins
        else:
            trace["invalids"].append("verifier_decision")  # invalid verifier -> keep integrator's plan

    with ledger.timed("execute"):
        prediction = _dispatch_writes(final_actions, ledger)
    return RunOutput(spec.name, prediction, "", trace)
