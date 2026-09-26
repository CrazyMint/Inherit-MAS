"""Typed inter-agent port schemas for WorkBench workflows.

Four strict-but-small artifacts flow on the edges of M2/M3:

  subgoals          (coordinator) -> which domains matter + per-domain subgoals
  evidence          (domain worker) -> resolved facts (IDs, emails, dates) from
                    READ-ONLY tools, plus a length-capped summary
  proposed_actions  (integrator) -> an ORDERED list of state-changing tool calls
  verifier_decision (verifier, M3) -> the vetted/edited final action list + reasons

Validation FAILS LOUD (SchemaError) on malformed structure. A tolerant
``parse_or_invalid`` is provided for the execution path: on malformed output it
returns an invalid-output marker (``{"valid": False, ...}``) so a bad LLM
message yields the invalid-output behaviour downstream (an empty/no-op plan),
never a silent pass or a crash.

Key safety invariant: ``proposed_actions``/``verifier_decision`` may reference ONLY
state-changing (side-effect) tools with argument names that exist on that tool's
signature. Read-only tools and unknown args are rejected here, so a fluent LLM
cannot smuggle an unrequested or malformed write into the executor.
"""
from __future__ import annotations

import json
import re
from typing import Any

import wb_env as W
import artifacts as AR
import agg_plan as AP

SUMMARY_CHAR_CAP = 800
GOAL_CHAR_CAP = 400
MAX_SUBGOALS = 10
MAX_EVIDENCE_ITEMS = 40
MAX_ACTIONS = 20
MAX_REASONS = 20


class SchemaError(ValueError):
    """A malformed or invalid structured inter-agent message (fail loud)."""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _s(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v)


# --------------------------------------------------------------------------- #
# subgoals (coordinator)
# --------------------------------------------------------------------------- #
def validate_subgoals(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise SchemaError("subgoals must be a JSON object")
    doms = obj.get("relevant_domains", [])
    if not isinstance(doms, list) or not all(isinstance(d, str) for d in doms):
        raise SchemaError("relevant_domains must be a list of strings")
    doms = W.normalize_domains(doms)  # drops unknown/aliased domains
    subs = obj.get("subgoals", [])
    if not isinstance(subs, list) or len(subs) > MAX_SUBGOALS:
        raise SchemaError(f"subgoals must be a list of <= {MAX_SUBGOALS}")
    canon_subs = []
    for s in subs:
        if not isinstance(s, dict) or "domain" not in s or "goal" not in s:
            raise SchemaError(f"bad subgoal (need domain+goal): {s!r}")
        d = W.normalize_domains([str(s["domain"])])
        if not d:
            raise SchemaError(f"subgoal domain not a WorkBench domain: {s['domain']!r}")
        canon_subs.append({"domain": d[0], "goal": _s(s["goal"])[:GOAL_CHAR_CAP]})
    requires_writes = bool(obj.get("requires_writes", True))
    return {"valid": True, "relevant_domains": doms, "subgoals": canon_subs,
            "requires_writes": requires_writes}


# --------------------------------------------------------------------------- #
# evidence (domain worker)
# --------------------------------------------------------------------------- #
def validate_evidence(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise SchemaError("evidence must be a JSON object")
    d = W.normalize_domains([str(obj.get("domain", ""))])
    if not d:
        raise SchemaError(f"evidence domain not a WorkBench domain: {obj.get('domain')!r}")
    items = obj.get("items", [])
    if not isinstance(items, list) or len(items) > MAX_EVIDENCE_ITEMS:
        raise SchemaError(f"items must be a list of <= {MAX_EVIDENCE_ITEMS}")
    canon = []
    for it in items:
        if not isinstance(it, dict) or "field" not in it or "value" not in it:
            raise SchemaError(f"bad evidence item (need field+value): {it!r}")
        canon.append({"field": _s(it["field"])[:120], "value": _s(it["value"])[:400]})
    summary = _s(obj.get("summary", ""))[:SUMMARY_CHAR_CAP]
    return {"valid": True, "domain": d[0], "items": canon, "summary": summary}


# --------------------------------------------------------------------------- #
# actions (shared by proposed_actions and verifier_decision)
# --------------------------------------------------------------------------- #
def _validate_action(a: Any) -> dict:
    if not isinstance(a, dict) or "tool" not in a:
        raise SchemaError(f"action needs a 'tool' key: {a!r}")
    tool = str(a["tool"])
    # allow both "email.send_email" and "email.send_email.func"
    tool = tool[: -len(".func")] if tool.endswith(".func") else tool
    if tool not in W.SIDE_EFFECT_TOOL_NAMES:
        raise SchemaError(f"action tool {tool!r} is not a state-changing WorkBench tool "
                          f"(only side-effect tools may be executed)")
    args = a.get("args", {})
    if not isinstance(args, dict):
        raise SchemaError(f"action args must be an object: {a!r}")
    allowed = set(W.TOOL_BY_NAME[tool].args_schema.keys())
    canon_args = {}
    for k, v in args.items():
        if k not in allowed:
            raise SchemaError(f"action {tool!r}: unknown argument {k!r} (allowed: {sorted(allowed)})")
        if AR.is_ref(v):                 # artifact reference -> keep (resolved before execution)
            canon_args[str(k)] = {"artifact_id": v.get("artifact_id") or v.get("$artifact"),
                                  "path": v.get("path") or v.get("$path")}
        else:
            canon_args[str(k)] = _s(v)
    return {"tool": tool, "args": canon_args}


def _validate_action_list(actions: Any) -> list[dict]:
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
        raise SchemaError(f"actions must be a list of <= {MAX_ACTIONS}")
    return [_validate_action(a) for a in actions]


def validate_proposed_actions(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise SchemaError("proposed_actions must be a JSON object")
    actions = _validate_action_list(obj.get("actions", []))
    return {"valid": True, "actions": actions}


def validate_verifier_decision(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise SchemaError("verifier_decision must be a JSON object")
    verdict = obj.get("verdict", "approve")
    if verdict not in ("approve", "revise"):
        raise SchemaError(f"verdict must be 'approve' or 'revise', got {verdict!r}")
    actions = _validate_action_list(obj.get("actions", []))
    reasons = obj.get("reasons", [])
    if not isinstance(reasons, list) or len(reasons) > MAX_REASONS:
        raise SchemaError(f"reasons must be a list of <= {MAX_REASONS}")
    return {"valid": True, "verdict": verdict, "actions": actions,
            "reasons": [_s(r)[:400] for r in reasons]}


# --------------------------------------------------------------------------- #
# tolerant parse (execution path): malformed -> invalid-output marker
# --------------------------------------------------------------------------- #
def validate_evidence_refs(obj: Any) -> dict:
    """Artifact-backed evidence: compact references rather than copied bulk values.
    {domain, refs:[{artifact_id, path?, fact, confidence}], summary}. Structure only —
    the runtime checks that each artifact_id actually exists in the task's store."""
    if not isinstance(obj, dict):
        raise SchemaError("evidence must be a JSON object")
    d = W.normalize_domains([str(obj.get("domain", ""))])
    if not d:
        raise SchemaError(f"evidence domain not a WorkBench domain: {obj.get('domain')!r}")
    refs = obj.get("refs", [])
    if not isinstance(refs, list) or len(refs) > MAX_EVIDENCE_ITEMS:
        raise SchemaError(f"refs must be a list of <= {MAX_EVIDENCE_ITEMS}")
    canon = []
    for r in refs:
        if not isinstance(r, dict) or "artifact_id" not in r or "fact" not in r:
            raise SchemaError(f"bad evidence ref (need artifact_id+fact): {r!r}")
        path = r.get("path")
        if path is not None and not isinstance(path, str):
            raise SchemaError("ref path must be a string or null")
        conf = r.get("confidence", 1.0)
        conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0
        canon.append({"artifact_id": str(r["artifact_id"]), "path": (str(path) if path else None),
                      "fact": _s(r["fact"])[:400], "confidence": max(0.0, min(1.0, conf))})
    return {"valid": True, "domain": d[0], "refs": canon,
            "summary": _s(obj.get("summary", ""))[:SUMMARY_CHAR_CAP]}


def _validate_agg_plan(obj: Any) -> dict:
    """Adapter: raise SchemaError (not AggPlanError) so parse_or_invalid handles it
    uniformly. Structure only; the reducer enforces field/type checks against data."""
    try:
        return AP.validate_agg_plan(obj)
    except AP.AggPlanError as e:
        raise SchemaError(str(e)) from e


_VALIDATORS = {
    "subgoals": validate_subgoals,
    "evidence": validate_evidence,
    "evidence_refs": validate_evidence_refs,
    "agg_plan": _validate_agg_plan,
    "proposed_actions": validate_proposed_actions,
    "verifier_decision": validate_verifier_decision,
}
_INVALID = {
    "subgoals": {"valid": False, "relevant_domains": [], "subgoals": [], "requires_writes": False},
    "evidence": {"valid": False, "domain": None, "items": [], "summary": ""},
    "evidence_refs": {"valid": False, "domain": None, "refs": [], "summary": ""},
    "agg_plan": {"valid": False, "sources": [], "steps": [], "output": None},
    "proposed_actions": {"valid": False, "actions": []},
    "verifier_decision": {"valid": False, "verdict": "revise", "actions": [], "reasons": []},
}


def parse_or_invalid(kind: str, text: str, *, strict: bool = False) -> dict:
    """Extract+validate a message of ``kind``. In strict mode raise on any problem;
    otherwise return the invalid-output marker for that kind (an empty/no-op
    artifact) so downstream stays safe and measurable."""
    if kind not in _VALIDATORS:
        raise SchemaError(f"unknown artifact kind {kind!r}")
    obj = _extract_json(text)
    if obj is None:
        if strict:
            raise SchemaError(f"{kind}: no JSON object found")
        return dict(_INVALID[kind])
    try:
        return _VALIDATORS[kind](obj)
    except SchemaError:
        if strict:
            raise
        return dict(_INVALID[kind])


def render_actions(actions: list[dict]) -> list[str]:
    """Turn validated [{tool,args}] into gold-format action strings."""
    return [W.render_action(a["tool"], a["args"]) for a in actions]


def explain_invalid(kind: str, text: str) -> str:
    """Observability only: return WHY ``parse_or_invalid(kind, text)`` is invalid (the
    parser error message), or "" if it actually parses. Re-runs the SAME validator in
    strict mode — it does NOT change any pipeline value or acceptance rule."""
    try:
        parse_or_invalid(kind, text, strict=True)
        return ""
    except SchemaError as e:
        return str(e)
