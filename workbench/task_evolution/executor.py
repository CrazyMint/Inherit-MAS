"""Generic typed-DAG executor with dirty scheduling and exact snapshot reuse."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import schemas as WB_SCHEMAS
import wb_env as W
from execution import _dispatch_writes
from ledger import Ledger
from inherit_mas.cache import resolved_request as resolved_cache_request

from .edits import dirty_nodes
from .models import sanitized_tool_schemas
from .schema import node_map, topological_order, validate_graph
from .snapshots import RoundArtifacts, SnapshotStore, digest

PRISTINE_ENV_DIGEST = digest({"workbench": W.PINNED_COMMIT, "state": "pristine-v1"})
WORKER_MAX_TURNS = 8
WORKER_MAX_TOKENS = 1024
WORKER_MAX_TOOL_CALLS = 20
ROLE_MAX_TOKENS = 1800


@dataclass
class ExecutionResult:
    prediction: list[str]
    snapshots: dict[str, dict]
    trace: dict


def _upstream_payload(snapshot: dict) -> dict:
    return {
        "node_id": snapshot["node_id"],
        "role": snapshot["role"],
        "output": snapshot["output"],
        "tool_events": snapshot.get("tool_events", []),
        "error": snapshot.get("error", ""),
    }


def _resolved_request(task, node: dict, model_fp: dict,
                      system: str, user: str) -> dict:
    role = node["role"]
    tool_schemas = sanitized_tool_schemas(
        [W.TOOL_BY_NAME[name] for name in node.get("tools", [])]
    )[0]
    if role == "worker":
        decode = {
            "temperature": 0,
            "max_turns": WORKER_MAX_TURNS,
            "max_tokens_per_turn": WORKER_MAX_TOKENS,
            "max_tool_calls": WORKER_MAX_TOOL_CALLS,
        }
    else:
        decode = {"temperature": 0, "max_tokens": ROLE_MAX_TOKENS}
    parser = {
        "integrator": "workbench.proposed_actions/1",
        "verifier": "workbench.verifier_decision/1",
    }.get(role, "text/1")
    return resolved_cache_request(
        benchmark="workbench-v1",
        task_id=str(task.id),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        model=model_fp,
        decode=decode,
        tools=tool_schemas,
        environment={"state_digest": PRISTINE_ENV_DIGEST,
                     "workbench_commit": W.PINNED_COMMIT},
        node_function={"runtime": "workbench-task-evolution/2",
                       "role": role, "output_parser": parser},
    )


def _role_prompt(task, node: dict, upstream: list[tuple[str, dict]]) -> tuple[str, str]:
    shared = (
        "You are one read-only node in a multi-agent workflow. Follow your assigned subtask. "
        "Do not claim access to hidden gold outcomes. Do not execute state-changing actions."
    )
    system = shared + "\n\n" + node.get("system_prompt", "")
    upstream_text = json.dumps(
        [{"port": port, **_upstream_payload(snap)} for port, snap in upstream],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    user = f"{W.DATETIME_PREFIX}\nTask: {task.task}\nAssigned subtask: {node.get('subtask','')}\nUpstream records: {upstream_text}"
    role = node["role"]
    if role == "worker":
        system += "\nUse assigned read-only tools when needed. Preserve concrete IDs, values, and complete findings in your final response."
    elif role == "integrator":
        system += (
            "\nProduce only JSON {\"actions\":[{\"tool\":\"...\",\"args\":{...}}]}. "
            "Use the fewest state-changing actions that fully satisfy the task. Exact schemas:\n"
            + W.write_tool_schema_block()
        )
    elif role == "verifier":
        system += (
            "\nAudit the proposed plan for task coverage, grounded arguments, and harmful extras. "
            "Return only JSON {\"verdict\":\"approve|revise\",\"actions\":[...],\"reasons\":[...]}.\n"
            + W.write_tool_schema_block()
        )
    else:
        system += "\nReturn a concise critique or reasoning artifact for downstream nodes."
    return system, user


def _usage_totals(usages: list) -> dict:
    return {
        "prompt_tokens": sum(u.prompt_tokens for u in usages),
        "completion_tokens": sum(u.completion_tokens for u in usages),
        "total_tokens": sum(u.total_tokens for u in usages),
        "estimated_usd": sum(u.estimated_usd for u in usages),
        "calls": len(usages),
        "wall_s": sum(u.wall_s for u in usages),
        "records": [u.to_dict() for u in usages],
    }


def execute_graph(
    graph: dict,
    task,
    *,
    models,
    cache: SnapshotStore,
    parent_graph: dict | None = None,
    parent_snapshots: dict[str, dict] | None = None,
) -> ExecutionResult:
    validate_graph(graph)
    W.reset_state()
    nodes = node_map(graph)
    parent_snapshots = parent_snapshots or {}
    dirty = set(nodes) if parent_graph is None else dirty_nodes(parent_graph, graph)
    snapshots: dict[str, dict] = {}
    round_artifacts = RoundArtifacts()
    trace = {
        "dirty_nodes": sorted(dirty),
        "nodes": {},
        "live_tokens": 0,
        "reused_tokens": 0,
        "live_calls": 0,
        "reused_calls": 0,
        "started_at": time.time(),
    }
    ledger = Ledger()
    prediction: list[str] = []

    for nid in topological_order(graph):
        node = nodes[nid]
        upstream = [(edge["port"], snapshots[edge["from"]]) for edge in node.get("inputs", [])]
        if node["role"] == "executor":
            source = upstream[0][1]
            actions = source.get("parsed_actions", [])
            with ledger.timed("write_executor"):
                prediction = _dispatch_writes(actions, ledger)
            snap = {
                "node_id": nid,
                "role": "executor",
                "request_digest": digest({"executor": actions, "task": task.id, "env": PRISTINE_ENV_DIGEST}),
                "output": json.dumps({"prediction": prediction}),
                "parsed_actions": actions,
                "tool_events": [],
                "artifacts": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                          "estimated_usd": 0.0, "calls": 0, "wall_s": 0.0, "records": []},
                "error": "",
                "reused": False,
                "reuse_reason": "write_executor_always_live",
            }
            snapshots[nid] = snap
            trace["nodes"][nid] = snap
            continue

        system, user = _role_prompt(task, node, upstream)
        request = _resolved_request(
            task, node, models.fingerprint("gpt-4o-mini"), system, user)
        key = cache.key(request)
        snap = None
        reason = ""
        parent_snap = parent_snapshots.get(nid)
        if nid not in dirty and parent_snap is not None:
            if parent_snap.get("request_digest") != key:
                raise RuntimeError(f"dirty classifier marked {nid} clean but resolved request changed")
            snap = parent_snap
            reason = "structurally_clean"
        if snap is None:
            snap = cache.get(key)
            if snap is not None:
                reason = "exact_request"
        if snap is not None:
            snap = json.loads(json.dumps(snap))
            snap["reused"] = True
            snap["reuse_reason"] = reason
            snap["artifacts"] = round_artifacts.ingest(nid, snap.get("tool_events", []))
            snapshots[nid] = snap
            usage = snap["usage"]
            trace["reused_tokens"] += usage["total_tokens"]
            trace["reused_calls"] += usage["calls"]
            trace["nodes"][nid] = snap
            continue

        if node["role"] == "worker":
            result = models.readonly_agent(
                role=f"worker:{nid}", system=system, user=user, tool_names=node.get("tools", []),
                max_turns=WORKER_MAX_TURNS, max_tokens_per_turn=WORKER_MAX_TOKENS,
                max_tool_calls=WORKER_MAX_TOOL_CALLS,
            )
        else:
            result = models.chat(
                model="gpt-4o-mini", role=f"{node['role']}:{nid}", system=system, user=user,
                max_tokens=ROLE_MAX_TOKENS,
            )
        parsed_actions = []
        parse_valid = True
        if node["role"] == "integrator":
            parsed = WB_SCHEMAS.parse_or_invalid("proposed_actions", result.text)
            parse_valid = bool(parsed.get("valid"))
            parsed_actions = parsed.get("actions", []) if parsed.get("valid") else []
        elif node["role"] == "verifier":
            parsed = WB_SCHEMAS.parse_or_invalid("verifier_decision", result.text)
            parse_valid = bool(parsed.get("valid"))
            parsed_actions = parsed.get("actions", []) if parsed.get("valid") else []
        usage = _usage_totals(result.usage)
        snap = {
            "node_id": nid,
            "role": node["role"],
            "request_digest": key,
            "output": result.text,
            "parsed_actions": parsed_actions,
            "messages": result.messages,
            "tool_events": result.tool_events,
            "artifacts": round_artifacts.ingest(nid, result.tool_events),
            "usage": usage,
            "error": result.error,
            "parse_valid": parse_valid,
            "reused": False,
            "reuse_reason": "",
        }
        cache.put(key, snap)
        snapshots[nid] = snap
        trace["live_tokens"] += usage["total_tokens"]
        trace["live_calls"] += usage["calls"]
        trace["nodes"][nid] = snap

    trace["artifact_manifest"] = round_artifacts.manifest()
    trace["write_ledger"] = ledger.to_dict()
    trace["wall_s"] = time.time() - trace["started_at"]
    return ExecutionResult(prediction, snapshots, trace)
