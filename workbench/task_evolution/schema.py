"""Typed task-specific MAS graph and gold-free judge contracts.

All LLM nodes are read-only. A single deterministic executor is the only node
allowed to apply state-changing WorkBench tools.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import wb_env as W

MAX_LLM_NODES = 8
MAX_EDGES = 12
MAX_PROMPT_CHARS = 6000
MAX_SUBTASK_CHARS = 1200
NODE_ROLES = {"worker", "integrator", "verifier", "critic", "executor"}
_ID = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class GraphError(ValueError):
    """A workflow, edit, or structured assessment violates the protocol."""


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_graph(graph: dict) -> dict:
    """Return the executable identity. Display metadata is intentionally excluded."""
    validate_graph(graph)
    nodes = []
    for node in sorted(graph["nodes"], key=lambda n: n["id"]):
        nodes.append({
            "id": node["id"],
            "role": node["role"],
            "subtask": node.get("subtask", ""),
            "system_prompt": node.get("system_prompt", ""),
            "tools": list(node.get("tools", [])),
            "inputs": [{"port": e["port"], "from": e["from"]} for e in node.get("inputs", [])],
        })
    return {"nodes": nodes, "sink": graph["sink"]}


def graph_digest(graph: dict) -> str:
    return hashlib.sha256(_canon(canonical_graph(graph)).encode()).hexdigest()


def node_map(graph: dict) -> dict[str, dict]:
    return {node["id"]: node for node in graph["nodes"]}


def topological_order(graph: dict) -> list[str]:
    validate_graph(graph, check_order=False)
    nodes = node_map(graph)
    indeg = {nid: 0 for nid in nodes}
    children = {nid: [] for nid in nodes}
    for node in nodes.values():
        for edge in node.get("inputs", []):
            indeg[node["id"]] += 1
            children[edge["from"]].append(node["id"])
    ready = sorted(nid for nid, degree in indeg.items() if degree == 0)
    out = []
    while ready:
        nid = ready.pop(0)
        out.append(nid)
        for child in sorted(children[nid]):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
                ready.sort()
    if len(out) != len(nodes):
        raise GraphError("graph contains a cycle")
    return out


def validate_graph(graph: Any, *, check_order: bool = True) -> dict:
    if not isinstance(graph, dict) or set(graph) - {"name", "nodes", "sink", "rationale"}:
        raise GraphError("graph must contain only name/nodes/sink/rationale")
    raw_nodes = graph.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise GraphError("nodes must be a non-empty list")
    if len(raw_nodes) > MAX_LLM_NODES + 1:
        raise GraphError(f"at most {MAX_LLM_NODES} LLM nodes plus one executor")
    ids = [n.get("id") if isinstance(n, dict) else None for n in raw_nodes]
    if len(ids) != len(set(ids)):
        raise GraphError("duplicate node ids")
    if any(not isinstance(nid, str) or not _ID.fullmatch(nid) for nid in ids):
        raise GraphError("invalid node id")
    nodes = node_map(graph)
    edge_count = 0
    for node in raw_nodes:
        allowed = {"id", "role", "subtask", "system_prompt", "tools", "inputs"}
        if set(node) - allowed:
            raise GraphError(f"{node['id']}: unknown fields {sorted(set(node) - allowed)}")
        role = node.get("role")
        if role not in NODE_ROLES:
            raise GraphError(f"{node['id']}: unsupported role {role!r}")
        subtask = node.get("subtask", "")
        prompt = node.get("system_prompt", "")
        tools = node.get("tools", [])
        inputs = node.get("inputs", [])
        if not isinstance(subtask, str) or len(subtask) > MAX_SUBTASK_CHARS:
            raise GraphError(f"{node['id']}: invalid subtask")
        if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
            raise GraphError(f"{node['id']}: invalid system_prompt")
        if not isinstance(tools, list) or len(tools) != len(set(tools)):
            raise GraphError(f"{node['id']}: tools must be a unique list")
        if any(t not in W.READ_ONLY_TOOL_NAMES for t in tools):
            raise GraphError(f"{node['id']}: LLM nodes may use read-only tools only")
        if role != "worker" and tools:
            raise GraphError(f"{node['id']}: only worker nodes may use tools")
        if role == "worker" and not subtask:
            raise GraphError(f"{node['id']}: worker needs a subtask")
        if not isinstance(inputs, list):
            raise GraphError(f"{node['id']}: inputs must be a list")
        ports = []
        for edge in inputs:
            if not isinstance(edge, dict) or set(edge) != {"port", "from"}:
                raise GraphError(f"{node['id']}: malformed input edge")
            if not isinstance(edge["port"], str) or not edge["port"]:
                raise GraphError(f"{node['id']}: invalid input port")
            if edge["from"] not in nodes or edge["from"] == node["id"]:
                raise GraphError(f"{node['id']}: dangling/self edge from {edge['from']!r}")
            ports.append(edge["port"])
        if len(ports) != len(set(ports)):
            raise GraphError(f"{node['id']}: duplicate input ports")
        source_roles = [nodes[e["from"]]["role"] for e in inputs]
        if "executor" in source_roles:
            raise GraphError(f"{node['id']}: no node may consume executor output")
        if role == "worker" and any(r not in {"worker", "critic"} for r in source_roles):
            raise GraphError(f"{node['id']}: worker inputs must be peer reports/critiques")
        if role == "critic" and (not inputs or any(r == "verifier" for r in source_roles)):
            raise GraphError(f"{node['id']}: critic needs upstream worker/integrator/critic input")
        if role == "integrator" and (not inputs or any(r not in {"worker", "critic"} for r in source_roles)):
            raise GraphError(f"{node['id']}: integrator needs worker/critic inputs")
        if role == "verifier":
            plan_edges = [e for e in inputs if e["port"] == "plan"]
            if len(plan_edges) != 1 or nodes[plan_edges[0]["from"]]["role"] != "integrator":
                raise GraphError(f"{node['id']}: verifier needs one plan input from an integrator")
            for edge in inputs:
                if edge["port"] != "plan" and nodes[edge["from"]]["role"] not in {"worker", "critic"}:
                    raise GraphError(f"{node['id']}: verifier context must come from workers/critics")
        edge_count += len(inputs)
    if edge_count > MAX_EDGES:
        raise GraphError(f"at most {MAX_EDGES} edges")

    executors = [n for n in raw_nodes if n["role"] == "executor"]
    if len(executors) != 1 or graph.get("sink") != executors[0]["id"]:
        raise GraphError("exactly one executor is required and it must be the sink")
    executor = executors[0]
    if executor.get("tools") or executor.get("system_prompt"):
        raise GraphError("executor is deterministic and has no model prompt/tools")
    if len(executor.get("inputs", [])) != 1 or executor["inputs"][0]["port"] != "plan":
        raise GraphError("executor requires exactly one plan input")
    plan_src = nodes[executor["inputs"][0]["from"]]
    if plan_src["role"] not in {"integrator", "verifier"}:
        raise GraphError("executor plan must come from an integrator or verifier")
    if not any(n["role"] == "integrator" for n in raw_nodes):
        raise GraphError("at least one integrator is required")
    if sum(n["role"] != "executor" for n in raw_nodes) > MAX_LLM_NODES:
        raise GraphError(f"at most {MAX_LLM_NODES} LLM nodes")

    if check_order:
        order = topological_order(graph)
        if order[-1] != graph["sink"]:
            raise GraphError("executor must be the sole terminal node")
        children = {nid: set() for nid in nodes}
        for n in raw_nodes:
            for edge in n.get("inputs", []):
                children[edge["from"]].add(n["id"])
        terminals = [nid for nid, cs in children.items() if not cs]
        if terminals != [graph["sink"]]:
            raise GraphError(f"all nodes must reach the sink; terminals={terminals}")
    return graph


def validate_judgment(value: Any) -> dict:
    if not isinstance(value, dict):
        raise GraphError("judgment must be an object")
    required_lists = ["correct", "preserve", "recommended_changes"]
    for key in required_lists:
        if not isinstance(value.get(key), list) or not all(isinstance(x, str) for x in value[key]):
            raise GraphError(f"judgment.{key} must be a string list")
    q = value.get("quality_score")
    safety = value.get("safety_score")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q <= 100:
        raise GraphError("quality_score must be in [0,100]")
    if isinstance(safety, bool) or not isinstance(safety, (int, float)) or not 0 <= safety <= 100:
        raise GraphError("safety_score must be in [0,100]")
    if not isinstance(value.get("satisfied"), bool):
        raise GraphError("satisfied must be boolean")
    checks = value.get("obligation_checks")
    if not isinstance(checks, list) or not checks:
        raise GraphError("judgment.obligation_checks must be a non-empty list")
    normalized_checks = []
    statuses = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != {"obligation", "status", "evidence"}:
            raise GraphError(f"judgment.obligation_checks[{index}] has invalid fields")
        if not isinstance(check["obligation"], str) or not check["obligation"].strip():
            raise GraphError(f"judgment.obligation_checks[{index}].obligation must be non-empty")
        if check["status"] not in {"verified", "wrong", "missing", "uncertain"}:
            raise GraphError(f"judgment.obligation_checks[{index}].status is invalid")
        if not isinstance(check["evidence"], str) or not check["evidence"].strip():
            raise GraphError(f"judgment.obligation_checks[{index}].evidence must be non-empty")
        statuses.add(check["status"])
        normalized_checks.append({key: check[key] for key in ("obligation", "status", "evidence")})
    # The typed checks are authoritative. Normalize the redundant summaries from
    # them rather than rejecting a sound audit over summary formatting.
    wrong = [f'{check["obligation"]}: {check["evidence"]}' for check in normalized_checks
             if check["status"] == "wrong"]
    missing = [f'{check["obligation"]}: {check["evidence"]}' for check in normalized_checks
               if check["status"] in {"missing", "uncertain"}]
    strict_q = float(q)
    if statuses & {"wrong", "missing"}:
        strict_q = min(strict_q, 70.0)
    elif "uncertain" in statuses:
        strict_q = min(strict_q, 80.0)
    strict_satisfied = bool(
        value["satisfied"]
        and statuses == {"verified"}
        and not wrong
        and not missing
        and not value["recommended_changes"]
        and q >= 95
        and safety >= 95
    )
    return {
        "quality_score": float(q),
        "strict_quality_score": strict_q,
        "safety_score": float(safety),
        "obligation_checks": normalized_checks,
        "correct": value["correct"],
        "wrong": wrong,
        "missing": missing,
        "preserve": value["preserve"],
        "recommended_changes": value["recommended_changes"],
        "reported_satisfied": value["satisfied"],
        "satisfied": strict_satisfied,
    }
