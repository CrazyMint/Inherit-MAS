"""Typed-DAG execution with per-task trajectory snapshot reuse."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from . import config
from .edits import dirty_nodes
from .graph import node_map, topological_order, validate_graph
from .loader import PublicQuestion, RuntimeExample
from .models import usage_totals
from .retrieval import BM25Retriever, RetrievalSession
from .schemas import SchemaError, invalid_prediction, prediction_from_text
from inherit_mas.cache import ExactSnapshotStore as SnapshotStore
from inherit_mas.cache import resolved_request as resolved_cache_request


@dataclass
class ExecutionResult:
    prediction: dict
    snapshots: dict[str, dict]
    trace: dict


def _upstream(snapshot: dict, port: str) -> dict:
    return {"port": port, "role": snapshot["role"], "output": snapshot["output"],
            "error": snapshot.get("error", "")}


def role_prompt(task: PublicQuestion, node: dict, upstream: list[dict]) -> tuple[str, str]:
    shared = (
        "You are one node in a HotpotQA FullWiki multi-agent DAG. The retrieval tool is deterministic BM25. "
        "Use only retrieved observations and ordered upstream reports. Retrieved sentences are addressed by exact "
        "[title, sentence_id] pairs. Never invent a title, sentence index, or fact."
    )
    if node["role"] == "planner":
        instruction = "Decompose the two-hop question into precise retrieval subgoals."
    elif node["role"] == "researcher":
        instruction = (
            "Retrieve and report concise bridge/comparison evidence. Include exact [title, sentence_id] citations "
            "for every useful fact and distinguish evidence from hypotheses."
        )
    elif node["role"] == "verifier":
        instruction = (
            "Audit upstream evidence, resolve contradictions, and search only if necessary. Return the supported "
            "answer candidate and exact [title, sentence_id] citations."
        )
    else:
        instruction = (
            "Return JSON only: {\"answer\":\"short answer\",\"supporting_facts\":[[\"exact title\",0]],"
            "\"confidence\":0.0,\"rationale\":\"brief\"}. Cite only observed indexed sentences."
        )
    system = f"{shared} {instruction}\n\nNode instruction:\n{node['system_prompt']}"
    user = (f"Question:\n{task.question}\n\nAssigned subtask:\n{node['subtask']}\n\n"
            "Ordered upstream reports:\n"
            f"{json.dumps(upstream, ensure_ascii=True, sort_keys=True, separators=(',', ':'))}")
    return system, user


def resolved_request(task: PublicQuestion, node: dict, upstream: list[dict],
                     model_fp: dict, retriever_digest: str, search_limit: int) -> dict:
    system, user = role_prompt(task, node, upstream)
    return resolved_cache_request(
        benchmark="hotpotqa-fullwiki",
        task_id=str(task.id),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        model=model_fp,
        decode={"temperature": 0, "max_turns": config.MAX_TOOL_TURNS,
                "max_tokens_per_turn": config.WORKER_MAX_TOKENS,
                "max_search_calls": search_limit},
        tools=list(node["tools"]),
        environment={"retriever_digest": retriever_digest},
        node_function={"runtime": "hotpot-fullwiki-executor/2",
                       "role": node["role"],
                       "output_parser": "hotpot-answer-supporting-facts/1"},
    )


def allocated_search_limits(graph: dict) -> dict[str, int]:
    nodes, order = node_map(graph), topological_order(graph)
    limits = {nid: 0 for nid in order}
    tool_nodes = [nid for nid in order if nodes[nid]["tools"]]
    if tool_nodes:
        quotient, remainder = divmod(config.RETRIEVAL_CALLS_PER_CANDIDATE, len(tool_nodes))
        for index, nid in enumerate(tool_nodes):
            limits[nid] = quotient + (index < remainder)
    return limits


def _propagate(graph: dict, dirty: set[str]) -> set[str]:
    children = {node["id"]: [] for node in graph["nodes"]}
    for node in graph["nodes"]:
        for edge in node["inputs"]:
            children[edge["from"]].append(node["id"])
    queue = list(dirty)
    while queue:
        source = queue.pop(0)
        for target in children[source]:
            if target not in dirty:
                dirty.add(target)
                queue.append(target)
    return dirty


def _available_citations(snapshots: dict[str, dict]) -> set[tuple[str, int]]:
    available: set[tuple[str, int]] = set()
    for snapshot in snapshots.values():
        for event in snapshot.get("tool_events", []):
            for row in event.get("result", {}).get("results", []):
                for sentence in row.get("sentences", []):
                    available.add((str(row["title"]), int(sentence["sentence_id"])))
    return available


def _parse_output(text: str, snapshots: dict[str, dict]) -> dict:
    try:
        prediction = prediction_from_text(text)
        prediction["valid"] = True
        prediction["error"] = ""
    except SchemaError as exc:
        return invalid_prediction(str(exc))
    available = _available_citations(snapshots)
    facts = [(str(title), int(index)) for title, index in prediction["supporting_facts"]]
    prediction["citation_count"] = len(facts)
    prediction["valid_citation_count"] = sum(fact in available for fact in facts)
    prediction["citation_valid"] = all(fact in available for fact in facts)
    if not prediction["citation_valid"]:
        prediction["valid"] = False
        prediction["error"] = "citation_not_retrieved"
    return prediction


def execute_graph(graph: dict, example: RuntimeExample, *, models, retriever: BM25Retriever,
                  cache: SnapshotStore, parent_graph: dict | None = None,
                  parent_snapshots: dict[str, dict] | None = None) -> ExecutionResult:
    validate_graph(graph)
    task = example.public()
    nodes = node_map(graph)
    parent_snapshots = parent_snapshots or {}
    dirty = set(nodes) if parent_graph is None else dirty_nodes(parent_graph, graph)
    limits = allocated_search_limits(graph)
    if parent_graph is not None:
        previous = allocated_search_limits(parent_graph)
        dirty.update(nid for nid, limit in limits.items() if previous.get(nid) != limit)
        dirty = _propagate(graph, dirty)
    session = RetrievalSession(retriever, example)
    snapshots: dict[str, dict] = {}
    trace = {"dirty_nodes": sorted(dirty), "nodes": {}, "live_tokens": 0,
             "reused_tokens": 0, "live_calls": 0, "reused_calls": 0,
             "search_calls": 0, "retrieval_cache_hits": 0,
             "retrieval_call_budget": config.RETRIEVAL_CALLS_PER_CANDIDATE,
             "node_search_limits": limits, "started_at": time.time()}

    for nid in topological_order(graph):
        node = nodes[nid]
        upstream = [_upstream(snapshots[edge["from"]], edge["port"])
                    for edge in node["inputs"]]
        request = resolved_request(task, node, upstream,
                                   models.fingerprint(config.WORKER_MODEL),
                                   retriever.fingerprint_digest, limits[nid])
        key = cache.key(request)
        snapshot, reason = None, ""
        if nid not in dirty and nid in parent_snapshots:
            if parent_snapshots[nid].get("request_digest") != key:
                raise RuntimeError(f"dirty classifier marked {nid} clean but request changed")
            snapshot, reason = parent_snapshots[nid], "structurally_clean"
        if snapshot is None:
            snapshot = cache.get(key)
            if snapshot is not None:
                reason = "exact_request"
        if snapshot is not None:
            snapshot = json.loads(json.dumps(snapshot))
            snapshot.update({"node_id": nid, "reused": True, "reuse_reason": reason})
            snapshots[nid] = snapshot
            trace["nodes"][nid] = snapshot
            trace["reused_tokens"] += snapshot["usage"]["total_tokens"]
            trace["reused_calls"] += snapshot["usage"]["calls"]
            continue

        system, user = role_prompt(task, node, upstream)
        if node["tools"]:
            result = models.research_agent(
                role=f"{node['role']}:{nid}", system=system, user=user,
                search_fn=session.search, max_turns=config.MAX_TOOL_TURNS,
                max_tokens_per_turn=config.WORKER_MAX_TOKENS,
                max_search_calls=limits[nid])
        else:
            result = models.chat(model=config.WORKER_MODEL, role=f"{node['role']}:{nid}",
                                 system=system, user=user, max_tokens=config.WORKER_MAX_TOKENS)
        usage = usage_totals(result.usage)
        snapshot = {"node_id": nid, "role": node["role"], "request_digest": key,
                    "output": result.text, "messages": result.messages,
                    "tool_events": result.tool_events, "usage": usage,
                    "error": result.error, "reused": False, "reuse_reason": ""}
        cache.put(key, snapshot)
        snapshots[nid] = snapshot
        trace["nodes"][nid] = snapshot
        trace["live_tokens"] += usage["total_tokens"]
        trace["live_calls"] += usage["calls"]
        trace["search_calls"] += len(result.tool_events)
        trace["retrieval_cache_hits"] += sum(
            bool(event.get("result", {}).get("cache_hit")) for event in result.tool_events)

    prediction = _parse_output(snapshots[graph["output"]]["output"], snapshots)
    retrieved_titles = {row["title"] for snapshot in snapshots.values()
                        for event in snapshot.get("tool_events", [])
                        for row in event.get("result", {}).get("results", [])}
    trace["retrieved_titles"] = sorted(retrieved_titles)
    trace["wall_s"] = time.time() - trace["started_at"]
    return ExecutionResult(prediction, snapshots, trace)


def execute_single(example: RuntimeExample, *, models, retriever: BM25Retriever,
                   strategy: str = "independent") -> ExecutionResult:
    task = example.public()
    session = RetrievalSession(retriever, example)
    system = (
        "You are a single HotpotQA FullWiki research agent. Use BM25 search iteratively to solve the two-hop "
        "question. Ground every claim in exact retrieved [title, sentence_id] pairs. Return JSON only as "
        "{\"answer\":\"short answer\",\"supporting_facts\":[[\"title\",0]],"
        "\"confidence\":0.0,\"rationale\":\"brief\"}. "
        f"Trajectory strategy: {strategy}."
    )
    result = models.research_agent(
        role=f"single:{strategy}", system=system, user=f"Question:\n{task.question}",
        search_fn=session.search, max_turns=config.MAX_TOOL_TURNS,
        max_tokens_per_turn=config.WORKER_MAX_TOKENS,
        max_search_calls=config.RETRIEVAL_CALLS_PER_CANDIDATE)
    usage = usage_totals(result.usage)
    snapshot = {"node_id": "single", "role": "synthesizer", "request_digest": "",
                "output": result.text, "messages": result.messages,
                "tool_events": result.tool_events, "usage": usage, "error": result.error,
                "reused": False, "reuse_reason": ""}
    prediction = _parse_output(result.text, {"single": snapshot})
    trace = {"nodes": {"single": snapshot}, "live_tokens": usage["total_tokens"],
             "reused_tokens": 0, "live_calls": usage["calls"], "reused_calls": 0,
             "search_calls": len(result.tool_events),
             "retrieval_cache_hits": sum(bool(event.get("result", {}).get("cache_hit"))
                                         for event in result.tool_events),
             "retrieved_titles": sorted(session.retrieved_titles),
             "wall_s": usage["wall_s"]}
    return ExecutionResult(prediction, {"single": snapshot}, trace)
