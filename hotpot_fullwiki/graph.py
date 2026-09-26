"""Strict typed DAG for task-time FullWiki workflow synthesis."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .common import digest

ALLOWED_ROLES = {"planner", "researcher", "verifier", "synthesizer"}
ALLOWED_TOOLS = {"search_fullwiki"}
EMITS = {"planner": "plan", "researcher": "evidence",
         "verifier": "critique", "synthesizer": "answer"}
ACCEPTS = {
    "planner": set(),
    "researcher": {"plan", "evidence"},
    "verifier": {"evidence", "critique"},
    "synthesizer": {"plan", "evidence", "critique"},
}
MIN_NODES = 2
MAX_NODES = 7


class GraphError(ValueError):
    pass


def node_map(graph: dict) -> dict[str, dict]:
    return {node["id"]: node for node in graph["nodes"]}


def canonical_node(node: dict) -> dict:
    return {
        "role": node["role"],
        "subtask": node["subtask"].strip(),
        "system_prompt": node["system_prompt"].strip(),
        "tools": list(node["tools"]),
        "inputs": [{"from": edge["from"], "port": edge["port"]}
                   for edge in node["inputs"]],
    }


def validate_graph(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"nodes", "output"}:
        raise GraphError("graph must contain exactly nodes and output")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or not MIN_NODES <= len(nodes) <= MAX_NODES:
        raise GraphError(f"graph must have {MIN_NODES}..{MAX_NODES} nodes")
    ids: list[str] = []
    required = {"id", "role", "subtask", "system_prompt", "tools", "inputs"}
    for node in nodes:
        if not isinstance(node, dict) or set(node) != required:
            raise GraphError("node fields must be id, role, subtask, system_prompt, tools, inputs")
        nid = node["id"]
        if not isinstance(nid, str) or not nid or not nid.replace("_", "").isalnum():
            raise GraphError("node id must be alphanumeric/underscore")
        if nid in ids:
            raise GraphError(f"duplicate node id {nid}")
        ids.append(nid)
        role = node["role"]
        if role not in ALLOWED_ROLES:
            raise GraphError(f"unsupported role {role!r}")
        if not isinstance(node["subtask"], str) or not node["subtask"].strip():
            raise GraphError(f"node {nid} has empty subtask")
        if not isinstance(node["system_prompt"], str) or not node["system_prompt"].strip():
            raise GraphError(f"node {nid} has empty system_prompt")
        tools = node["tools"]
        if not isinstance(tools, list) or len(tools) != len(set(tools)) or not set(tools) <= ALLOWED_TOOLS:
            raise GraphError(f"node {nid} has invalid tools")
        edges = node["inputs"]
        if not isinstance(edges, list):
            raise GraphError(f"node {nid} inputs must be a list")
        seen_edges = set()
        for edge in edges:
            if not isinstance(edge, dict) or set(edge) != {"from", "port"}:
                raise GraphError("each edge requires from and port")
            pair = (edge["from"], edge["port"])
            if pair in seen_edges:
                raise GraphError("duplicate ordered input edge")
            seen_edges.add(pair)

    if value["output"] not in ids:
        raise GraphError("output references missing node")
    by_id = node_map(value)
    if by_id[value["output"]]["role"] != "synthesizer":
        raise GraphError("output node must be synthesizer")
    if sum(node["role"] == "synthesizer" for node in nodes) != 1:
        raise GraphError("graph must have exactly one synthesizer")
    if not any(node["role"] == "researcher" for node in nodes):
        raise GraphError("graph must have at least one researcher")
    if not any("search_fullwiki" in node["tools"] for node in nodes):
        raise GraphError("FullWiki graph must retain at least one retrieval-capable node")

    children: dict[str, list[str]] = defaultdict(list)
    indegree = {nid: 0 for nid in ids}
    for node in nodes:
        for edge in node["inputs"]:
            source = edge["from"]
            if source not in by_id:
                raise GraphError(f"dangling edge from {source!r}")
            if source == node["id"]:
                raise GraphError("self edges are forbidden")
            emitted = EMITS[by_id[source]["role"]]
            if edge["port"] != emitted or emitted not in ACCEPTS[node["role"]]:
                raise GraphError(f"typed edge {source}->{node['id']} has invalid port {edge['port']!r}")
            children[source].append(node["id"])
            indegree[node["id"]] += 1
    roots = [by_id[nid] for nid in ids if indegree[nid] == 0]
    if any(node["role"] not in {"planner", "researcher"} for node in roots):
        raise GraphError("only planner/researcher nodes may be roots")
    queue = deque([nid for nid in ids if indegree[nid] == 0])
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        raise GraphError("graph contains a cycle")
    if children[value["output"]]:
        raise GraphError("output must be the only sink")
    reachable, stack = set(), [value["output"]]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(edge["from"] for edge in by_id[nid]["inputs"])
    if reachable != set(ids):
        raise GraphError("every node must reach output")
    return value


def topological_order(graph: dict) -> list[str]:
    validate_graph(graph)
    by_id = node_map(graph)
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {nid: 0 for nid in by_id}
    for node in graph["nodes"]:
        for edge in node["inputs"]:
            children[edge["from"]].append(node["id"])
            indegree[node["id"]] += 1
    queue = deque(node["id"] for node in graph["nodes"] if indegree[node["id"]] == 0)
    order = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order


def graph_digest(graph: dict) -> str:
    validate_graph(graph)
    return digest({"nodes": [{"id": node["id"], **canonical_node(node)}
                             for node in graph["nodes"]], "output": graph["output"]})


def default_graph(strategy: str = "bridge") -> dict:
    prompt = {
        "bridge": "Find the first bridge entity and the facts needed to answer the question.",
        "comparison": "Retrieve both candidate entities and the exact attributes being compared.",
        "verification": "Retrieve likely answer evidence and actively check for contradictions.",
        "broad": "Use focused multi-hop queries and return concise sentence-level evidence.",
        "independent": "Solve the question independently with grounded multi-hop evidence.",
    }.get(strategy, "Find grounded evidence needed to answer the question.")
    graph = {
        "nodes": [
            {"id": "researcher", "role": "researcher", "subtask": prompt,
             "system_prompt": prompt, "tools": ["search_fullwiki"], "inputs": []},
            {"id": "answerer", "role": "synthesizer", "subtask": "Answer from grounded evidence.",
             "system_prompt": "Return the short answer and official title/sentence citations.",
             "tools": [], "inputs": [{"from": "researcher", "port": "evidence"}]},
        ],
        "output": "answerer",
    }
    return validate_graph(graph)
