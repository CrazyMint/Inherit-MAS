"""A shared typed graph contract for adapted external baselines.

This contract is intentionally separate from Inherit-MAS's frozen graph module.
It supports TacoMAS's larger population while keeping the same HotpotQA role,
port, retrieval, and single-output requirements for every compared method.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from ..common import digest

ALLOWED_ROLES = {"planner", "researcher", "verifier", "synthesizer"}
EMITS = {"planner": "plan", "researcher": "evidence",
         "verifier": "critique", "synthesizer": "answer"}
ACCEPTS = {
    "planner": set(),
    "researcher": {"plan", "evidence"},
    "verifier": {"evidence", "critique"},
    "synthesizer": {"plan", "evidence", "critique"},
}


class BaselineGraphError(ValueError):
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


def validate_graph(value: Any, *, max_nodes: int = 20) -> dict:
    if not isinstance(value, dict) or set(value) != {"nodes", "output"}:
        raise BaselineGraphError("graph must contain exactly nodes and output")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= max_nodes:
        raise BaselineGraphError(f"graph must have 2..{max_nodes} nodes")
    required = {"id", "role", "subtask", "system_prompt", "tools", "inputs"}
    ids: list[str] = []
    for node in nodes:
        if not isinstance(node, dict) or set(node) != required:
            raise BaselineGraphError("invalid node fields")
        nid = node["id"]
        if (not isinstance(nid, str) or not nid or
                not nid.replace("_", "").isalnum() or nid in ids):
            raise BaselineGraphError(f"invalid or duplicate node id {nid!r}")
        ids.append(nid)
        if node["role"] not in ALLOWED_ROLES:
            raise BaselineGraphError(f"unsupported role {node['role']!r}")
        if not isinstance(node["subtask"], str) or not node["subtask"].strip():
            raise BaselineGraphError(f"node {nid} has empty subtask")
        if not isinstance(node["system_prompt"], str) or not node["system_prompt"].strip():
            raise BaselineGraphError(f"node {nid} has empty system_prompt")
        if (not isinstance(node["tools"], list) or
                len(node["tools"]) != len(set(node["tools"])) or
                not set(node["tools"]) <= {"search_fullwiki"}):
            raise BaselineGraphError(f"node {nid} has invalid tools")
        if not isinstance(node["inputs"], list):
            raise BaselineGraphError(f"node {nid} inputs must be a list")
        seen = set()
        for edge in node["inputs"]:
            if not isinstance(edge, dict) or set(edge) != {"from", "port"}:
                raise BaselineGraphError("each edge requires from and port")
            pair = (edge["from"], edge["port"])
            if pair in seen:
                raise BaselineGraphError("duplicate ordered input edge")
            seen.add(pair)

    if value["output"] not in ids:
        raise BaselineGraphError("output references missing node")
    by_id = node_map(value)
    if by_id[value["output"]]["role"] != "synthesizer":
        raise BaselineGraphError("output must be a synthesizer")
    if sum(node["role"] == "synthesizer" for node in nodes) != 1:
        raise BaselineGraphError("graph must have exactly one synthesizer")
    if not any(node["role"] == "researcher" for node in nodes):
        raise BaselineGraphError("graph must retain a researcher")
    if not any("search_fullwiki" in node["tools"] for node in nodes):
        raise BaselineGraphError("graph must retain retrieval")

    children: dict[str, list[str]] = defaultdict(list)
    indegree = {nid: 0 for nid in ids}
    for node in nodes:
        for edge in node["inputs"]:
            source = edge["from"]
            if source not in by_id:
                raise BaselineGraphError(f"dangling edge from {source!r}")
            if source == node["id"]:
                raise BaselineGraphError("self edge")
            port = EMITS[by_id[source]["role"]]
            if edge["port"] != port or port not in ACCEPTS[node["role"]]:
                raise BaselineGraphError(
                    f"invalid typed edge {source}->{node['id']} on {edge['port']}")
            children[source].append(node["id"])
            indegree[node["id"]] += 1
    roots = [by_id[nid] for nid in ids if indegree[nid] == 0]
    if any(node["role"] not in {"planner", "researcher"} for node in roots):
        raise BaselineGraphError("only planner/researcher nodes may be roots")
    queue = deque(nid for nid in ids if indegree[nid] == 0)
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        raise BaselineGraphError("graph contains a cycle")
    if children[value["output"]]:
        raise BaselineGraphError("output must be the only sink")
    reachable, stack = set(), [value["output"]]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(edge["from"] for edge in by_id[nid]["inputs"])
    if reachable != set(ids):
        raise BaselineGraphError("every node must reach output")
    return value


def topological_order(graph: dict) -> list[str]:
    validate_graph(graph)
    by_id = node_map(graph)
    children = {nid: [] for nid in by_id}
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


def clone(graph: dict) -> dict:
    return json.loads(json.dumps(graph))


def _node(nid: str, role: str, prompt: str, *, tools=(), inputs=()) -> dict:
    return {"id": nid, "role": role, "subtask": prompt,
            "system_prompt": prompt, "tools": list(tools),
            "inputs": [{"from": source, "port": port} for source, port in inputs]}


def evomas_seed_pool() -> dict[str, dict]:
    """Five benchmark-generic MAS seeds analogous to EvoMAS's public pool."""
    pool = {
        "single_researcher": {"nodes": [
            _node("researcher", "researcher", "Solve both hops with grounded retrieval.",
                  tools=("search_fullwiki",)),
            _node("answerer", "synthesizer", "Produce the grounded short answer.",
                  inputs=(("researcher", "evidence"),)),
        ], "output": "answerer"},
        "plan_then_research": {"nodes": [
            _node("planner", "planner", "Decompose the bridge or comparison question."),
            _node("researcher", "researcher", "Follow the plan and retrieve both hops.",
                  tools=("search_fullwiki",), inputs=(("planner", "plan"),)),
            _node("answerer", "synthesizer", "Synthesize only supported evidence.",
                  inputs=(("planner", "plan"), ("researcher", "evidence"))),
        ], "output": "answerer"},
        "parallel_research": {"nodes": [
            _node("researcher_a", "researcher", "Investigate the first hop.",
                  tools=("search_fullwiki",)),
            _node("researcher_b", "researcher", "Independently investigate the second hop.",
                  tools=("search_fullwiki",)),
            _node("answerer", "synthesizer", "Reconcile both evidence reports.",
                  inputs=(("researcher_a", "evidence"), ("researcher_b", "evidence"))),
        ], "output": "answerer"},
        "peer_review": {"nodes": [
            _node("researcher_a", "researcher", "Retrieve a complete grounded solution.",
                  tools=("search_fullwiki",)),
            _node("researcher_b", "researcher", "Retrieve an independent grounded solution.",
                  tools=("search_fullwiki",)),
            _node("verifier", "verifier", "Cross-check both reports and identify support gaps.",
                  inputs=(("researcher_a", "evidence"), ("researcher_b", "evidence"))),
            _node("answerer", "synthesizer", "Revise into a supported final answer.",
                  inputs=(("researcher_a", "evidence"), ("researcher_b", "evidence"),
                          ("verifier", "critique"))),
        ], "output": "answerer"},
        "planned_parallel_review": {"nodes": [
            _node("planner", "planner", "Plan complementary two-hop investigations."),
            _node("researcher_a", "researcher", "Investigate the first assigned subproblem.",
                  tools=("search_fullwiki",), inputs=(("planner", "plan"),)),
            _node("researcher_b", "researcher", "Investigate the second assigned subproblem.",
                  tools=("search_fullwiki",), inputs=(("planner", "plan"),)),
            _node("verifier", "verifier", "Audit agreement and citation support.",
                  inputs=(("researcher_a", "evidence"), ("researcher_b", "evidence"))),
            _node("answerer", "synthesizer", "Return the short grounded answer.",
                  inputs=(("planner", "plan"), ("researcher_a", "evidence"),
                          ("researcher_b", "evidence"), ("verifier", "critique"))),
        ], "output": "answerer"},
    }
    return {name: validate_graph(graph, max_nodes=7) for name, graph in pool.items()}


def tacomas_initial_graph() -> dict:
    """Official-size five-agent initial population specialized only by role."""
    graph = {"nodes": [
        _node("planner", "planner", "Decompose the task and assign complementary evidence goals."),
        _node("searcher_a", "researcher", "Retrieve the bridge entity and first-hop evidence.",
              tools=("search_fullwiki",), inputs=(("planner", "plan"),)),
        _node("searcher_b", "researcher", "Retrieve the answer-bearing second-hop evidence.",
              tools=("search_fullwiki",), inputs=(("planner", "plan"),)),
        _node("verifier", "verifier", "Cross-check evidence, citations, and contradictions.",
              inputs=(("searcher_a", "evidence"), ("searcher_b", "evidence"))),
        _node("synthesizer", "synthesizer", "Return the grounded answer and citations.",
              inputs=(("planner", "plan"), ("searcher_a", "evidence"),
                      ("searcher_b", "evidence"), ("verifier", "critique"))),
    ], "output": "synthesizer"}
    return validate_graph(graph, max_nodes=20)

