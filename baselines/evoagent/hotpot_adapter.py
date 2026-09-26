"""EvoAgent's NLP population mechanism on HotpotQA FullWiki."""
from __future__ import annotations

import json
import time
from typing import Any

from hotpot_fullwiki import config
from hotpot_fullwiki.executor import _parse_output
from hotpot_fullwiki.models import usage_totals
from hotpot_fullwiki.retrieval import RetrievalSession

from . import prompts
from .core import NLP_ITERATIONS, evolve_result

OUTPUT_CONTRACT = (
    'Return JSON only: {"answer":"short answer","supporting_facts":'
    '[["exact retrieved title",0]],"confidence":0.0,"rationale":"brief"}. '
    "Cite only exact [title, sentence_id] pairs observed through BM25 search."
)


def _snapshot(node_id: str, role: str, result) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "role": role,
        "request_digest": "",
        "output": result.text,
        "messages": result.messages,
        "tool_events": result.tool_events,
        "usage": usage_totals(result.usage),
        "error": result.error,
        "reused": False,
        "reuse_reason": "",
    }


def run(example, *, models, retriever) -> dict[str, Any]:
    """Execute `EvoAgent(1,3)` without access to scorer-only gold."""
    started = time.time()
    task = example.public()
    snapshots: dict[str, dict] = {}
    meta_usage = []
    integration_usage = []

    initial_session = RetrievalSession(retriever, example)
    initial = models.research_agent(
        role="evoagent:initial",
        system=(
            "You are the initial HotpotQA FullWiki research agent in an EvoAgent population. "
            "Solve the two-hop question using deterministic BM25 search. " + OUTPUT_CONTRACT
        ),
        user=f"Question:\n{task.question}",
        search_fn=initial_session.search,
        max_turns=config.MAX_TOOL_TURNS,
        max_tokens_per_turn=config.WORKER_MAX_TOKENS,
        max_search_calls=config.RETRIEVAL_CALLS_PER_CANDIDATE,
    )
    snapshots["initial"] = _snapshot("initial", "initial", initial)

    def propose(task_text, current, prior, iteration, proposal_index):
        system, user = prompts.propose_expert(task_text, current, prior, OUTPUT_CONTRACT)
        result = models.structured(
            role=f"evoagent:meta:{iteration}:{proposal_index}",
            system=system, user=user, max_tokens=config.META_MAX_TOKENS)
        meta_usage.extend(result.usage)
        return result.text

    def check(task_text, prior, description, iteration, proposal_index):
        system, user = prompts.check_expert(task_text, prior, description)
        result = models.structured(
            role=f"evoagent:quality:{iteration}:{proposal_index}",
            system=system, user=user, max_tokens=config.META_MAX_TOKENS)
        meta_usage.extend(result.usage)
        return result.text

    def execute(task_text, description, iteration):
        session = RetrievalSession(retriever, example)
        result = models.research_agent(
            role=f"evoagent:expert:{iteration}",
            system=prompts.expert_system(description, OUTPUT_CONTRACT),
            user=f"Question:\n{task.question}",
            search_fn=session.search,
            max_turns=config.MAX_TOOL_TURNS,
            max_tokens_per_turn=config.WORKER_MAX_TOKENS,
            max_search_calls=config.RETRIEVAL_CALLS_PER_CANDIDATE,
        )
        snapshots[f"expert_{iteration}"] = _snapshot(
            f"expert_{iteration}", "expert", result)
        return result.text

    def integrate(task_text, current, description, child, iteration):
        system, user = prompts.integrate(
            task_text, current, description, child, OUTPUT_CONTRACT)
        result = models.chat(
            model=config.WORKER_MODEL,
            role=f"evoagent:integrator:{iteration}",
            system=system, user=user, max_tokens=config.WORKER_MAX_TOKENS)
        integration_usage.extend(result.usage)
        snapshots[f"integrator_{iteration}"] = _snapshot(
            f"integrator_{iteration}", "integrator", result)
        return result.text

    evolved = evolve_result(
        task=task.question,
        initial_result=initial.text,
        iterations=NLP_ITERATIONS,
        propose_expert=propose,
        check_expert=check,
        execute_expert=execute,
        integrate=integrate,
    )

    candidate_rows = []
    candidate_texts = evolved.candidate_results
    for index, text in enumerate(candidate_texts):
        visible = {"initial": snapshots["initial"]}
        for j in range(index):
            visible[f"expert_{j}"] = snapshots[f"expert_{j}"]
            visible[f"integrator_{j}"] = snapshots[f"integrator_{j}"]
        prediction = _parse_output(text, visible)
        retrieved_titles = sorted({
            row["title"]
            for snapshot in visible.values()
            for event in snapshot.get("tool_events", [])
            for row in event.get("result", {}).get("results", [])
        })
        candidate_rows.append({
            "candidate_index": index,
            "status": "complete",
            "prediction": prediction,
            "execution": {"retrieved_titles": retrieved_titles},
        })

    # Snapshots retain already-aggregated usage. Meta/integration usages remain
    # separate so population overhead is not confused with retrieval-agent cost.
    per_node_usage = {key: value["usage"] for key, value in snapshots.items()}
    total = {
        field: sum(int(row.get(field, 0)) for row in per_node_usage.values())
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "calls")
    }
    meta_total = usage_totals(meta_usage)
    integration_total = usage_totals(integration_usage)
    # Integrator snapshots are already in per_node_usage; meta calls are not.
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "calls"):
        total[field] += int(meta_total[field])
    total["estimated_usd"] = (
        sum(float(row.get("estimated_usd", 0.0)) for row in per_node_usage.values())
        + float(meta_total["estimated_usd"])
    )

    return {
        "schema": "adapted_evoagent_hotpot/1",
        "condition": "evoagent_adapted",
        "task_id": example.id,
        "candidates": candidate_rows,
        "selected_candidate": NLP_ITERATIONS,
        "evolution_trace": list(evolved.trace),
        "expert_descriptions": list(evolved.expert_descriptions),
        "snapshots": snapshots,
        "usage": total,
        "meta_usage": meta_total,
        "integration_usage": integration_total,
        "search_calls": sum(
            len(snapshot.get("tool_events", [])) for snapshot in snapshots.values()),
        "wall_s": time.time() - started,
        "gold_observed_online": False,
        "node_output_reuse": False,
    }
