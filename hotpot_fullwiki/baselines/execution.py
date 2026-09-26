"""Full-evaluation executor shared by adapted EvoMAS and TacoMAS."""
from __future__ import annotations

import time

from .. import config
from ..executor import ExecutionResult, _parse_output, role_prompt
from ..models import usage_totals
from ..retrieval import BM25Retriever, RetrievalSession
from .graph import node_map, topological_order, validate_graph


def allocated_search_limits(graph: dict) -> dict[str, int]:
    nodes, order = node_map(graph), topological_order(graph)
    limits = {nid: 0 for nid in order}
    tool_nodes = [nid for nid in order if nodes[nid]["tools"]]
    if tool_nodes:
        quotient, remainder = divmod(config.RETRIEVAL_CALLS_PER_CANDIDATE, len(tool_nodes))
        for index, nid in enumerate(tool_nodes):
            limits[nid] = quotient + int(index < remainder)
    return limits


def execute_graph(graph: dict, example, *, models, retriever: BM25Retriever) -> ExecutionResult:
    """Execute every node; external baselines receive no MERIDIAN reuse."""
    validate_graph(graph)
    task = example.public()
    nodes = node_map(graph)
    limits = allocated_search_limits(graph)
    session = RetrievalSession(retriever, example)
    snapshots: dict[str, dict] = {}
    trace = {"nodes": {}, "live_tokens": 0, "reused_tokens": 0,
             "live_calls": 0, "reused_calls": 0, "search_calls": 0,
             "retrieval_cache_hits": 0,
             "retrieval_call_budget": config.RETRIEVAL_CALLS_PER_CANDIDATE,
             "node_search_limits": limits, "started_at": time.time()}
    for nid in topological_order(graph):
        node = nodes[nid]
        upstream = [{"port": edge["port"],
                     "role": snapshots[edge["from"]]["role"],
                     "output": snapshots[edge["from"]]["output"],
                     "error": snapshots[edge["from"]].get("error", "")}
                    for edge in node["inputs"]]
        system, user = role_prompt(task, node, upstream)
        if node["tools"]:
            result = models.research_agent(
                role=f"baseline:{node['role']}:{nid}", system=system, user=user,
                search_fn=session.search, max_turns=config.MAX_TOOL_TURNS,
                max_tokens_per_turn=config.WORKER_MAX_TOKENS,
                max_search_calls=limits[nid])
        else:
            result = models.chat(
                model=config.WORKER_MODEL, role=f"baseline:{node['role']}:{nid}",
                system=system, user=user, max_tokens=config.WORKER_MAX_TOKENS)
        usage = usage_totals(result.usage)
        snapshot = {"node_id": nid, "role": node["role"], "request_digest": "",
                    "output": result.text, "messages": result.messages,
                    "tool_events": result.tool_events, "usage": usage,
                    "error": result.error, "reused": False, "reuse_reason": ""}
        snapshots[nid] = snapshot
        trace["nodes"][nid] = snapshot
        trace["live_tokens"] += usage["total_tokens"]
        trace["live_calls"] += usage["calls"]
        trace["search_calls"] += len(result.tool_events)
        trace["retrieval_cache_hits"] += sum(
            bool(event.get("result", {}).get("cache_hit")) for event in result.tool_events)
    prediction = _parse_output(snapshots[graph["output"]]["output"], snapshots)
    trace["retrieved_titles"] = sorted({
        row["title"] for snapshot in snapshots.values()
        for event in snapshot.get("tool_events", [])
        for row in event.get("result", {}).get("results", [])})
    trace["wall_s"] = time.time() - trace["started_at"]
    return ExecutionResult(prediction, snapshots, trace)


def compact_trace(trace: dict) -> dict:
    return {"live_tokens": trace.get("live_tokens", 0),
            "wall_s": trace.get("wall_s", 0.0),
            "search_calls": trace.get("search_calls", 0),
            "retrieved_titles": trace.get("retrieved_titles", []),
            "nodes": {nid: {"role": row.get("role"), "output": row.get("output"),
                             "error": row.get("error")}
                      for nid, row in trace.get("nodes", {}).items()}}
