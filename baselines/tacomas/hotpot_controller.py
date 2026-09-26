"""HotpotQA adapter for TacoMAS fast capability and slow topology loops."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hotpot_fullwiki import config
from hotpot_fullwiki.controller import StructuredCallError
from hotpot_fullwiki.schemas import SchemaError
from hotpot_fullwiki.baselines.common import (candidate_record, judge, selected_by_reward,
                     structured_call)
from hotpot_fullwiki.baselines.execution import compact_trace, execute_graph
from hotpot_fullwiki.baselines.graph import clone, node_map, tacomas_initial_graph, validate_graph
from hotpot_fullwiki.baselines.prompts import taco_capability_prompt, taco_topology_prompt

UPSTREAM_COMMIT = "6f0d545f2493cf95d2eb6a325d1a6686acf658eb"
MAX_FAST_ROUNDS = 10
BD_CHECK_INTERVAL = 2
GRAPH_REWIRE_INTERVAL = 2
N_MIN = 5
N_MAX = 20
MAX_BIRTH_DEATH_PAIRS = 2
MAX_EDGE_EDITS = 8
DIRECT_STOP_QUALITY = 99.9


def _validate_capability(value: Any, graph: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {
            "updates", "continue_evolution", "rationale"}:
        raise SchemaError("capability update requires updates, continue_evolution, rationale")
    if not isinstance(value["continue_evolution"], bool):
        raise SchemaError("continue_evolution must be boolean")
    if not isinstance(value["rationale"], str):
        raise SchemaError("rationale must be a string")
    if not isinstance(value["updates"], list):
        raise SchemaError("updates must be a list")
    valid_ids, seen = set(node_map(graph)), set()
    updates = []
    required = {"agent_id", "contribution_score", "score_reason", "prompt_delta"}
    for update in value["updates"]:
        if not isinstance(update, dict) or set(update) != required:
            raise SchemaError("invalid capability-update fields")
        aid = update["agent_id"]
        if aid not in valid_ids or aid in seen:
            raise SchemaError("capability update references unknown/duplicate agent")
        seen.add(aid)
        score = update["contribution_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise SchemaError("contribution_score must be in [0,100]")
        if (not isinstance(update["score_reason"], str) or
                not isinstance(update["prompt_delta"], str) or
                len(update["prompt_delta"]) > 2000):
            raise SchemaError("score_reason/prompt_delta must be bounded strings")
        updates.append({**update, "contribution_score": float(score)})
    return {"updates": updates, "continue_evolution": value["continue_evolution"],
            "rationale": value["rationale"]}


def _edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {(edge["from"], node["id"], edge["port"])
            for node in graph["nodes"] for edge in node["inputs"]}


def _validate_topology(parent: dict, value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"rationale", "graph", "continue_evolution"}:
        raise SchemaError("topology update requires rationale, graph, continue_evolution")
    if not isinstance(value["rationale"], str) or not isinstance(value["continue_evolution"], bool):
        raise SchemaError("invalid topology rationale/control")
    child = validate_graph(value["graph"], max_nodes=N_MAX)
    if len(child["nodes"]) < N_MIN:
        raise SchemaError(f"TacoMAS population must retain at least {N_MIN} agents")
    before, after = set(node_map(parent)), set(node_map(child))
    born, died = after - before, before - after
    if len(born) > MAX_BIRTH_DEATH_PAIRS or len(died) > MAX_BIRTH_DEATH_PAIRS:
        raise SchemaError("birth-death update exceeds pair bound")
    edge_edits = len(_edge_set(parent) ^ _edge_set(child))
    if edge_edits > MAX_EDGE_EDITS:
        raise SchemaError("topology update exceeds edge-edit bound")
    return {"rationale": value["rationale"], "graph": child,
            "continue_evolution": value["continue_evolution"],
            "born": sorted(born), "died": sorted(died), "edge_edits": edge_edits}


def _apply_capability(graph: dict, update: dict, round_index: int) -> dict:
    child = clone(graph)
    by_id = node_map(child)
    for row in update["updates"]:
        patch = row["prompt_delta"].strip()
        if not patch:
            continue
        node = by_id[row["agent_id"]]
        memory = (f"\n\nTacoMAS capability update after round {round_index}:\n{patch}\n"
                  f"Contribution audit: {row['score_reason']}")
        node["system_prompt"] = (node["system_prompt"] + memory)[-6000:]
    return validate_graph(child, max_nodes=N_MAX)


def run(example, *, models, retriever, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    graph = tacomas_initial_graph()
    candidates: list[dict] = []
    history: list[dict] = []
    slow_updates: list[dict] = []
    stop_reason = "max_fast_rounds"

    for round_index in range(1, MAX_FAST_ROUNDS + 1):
        candidate_index = len(candidates)
        result = execute_graph(graph, example, models=models, retriever=retriever)
        judgment, judge_attempts = judge(models, example.question, graph,
                                         result.prediction, result.trace)
        summary = {"round": round_index, "quality_score": judgment["quality_score"],
                   "graph_nodes": len(graph["nodes"]),
                   "hard_failures": int(not result.prediction.get("valid", False)),
                   "execution": compact_trace(result.trace)}
        history.append(summary)

        system, user = taco_capability_prompt(
            example.question, graph, judgment, compact_trace(result.trace), round_index)
        capability_attempts: list[dict] = []
        try:
            capability, capability_attempts = structured_call(
                models, role="tacomas:capability", system=system, user=user,
                validator=lambda value: _validate_capability(value, graph),
                contract="updates + continue_evolution + rationale",
                max_tokens=config.META_MAX_TOKENS)
        except StructuredCallError as exc:
            capability_attempts = exc.attempts
            capability = {"updates": [], "continue_evolution": True,
                          "rationale": "invalid capability update; graph retained"}
        next_graph = _apply_capability(graph, capability, round_index)

        topology_attempts: list[dict] = []
        topology_record = None
        if round_index % BD_CHECK_INTERVAL == 0 and round_index < MAX_FAST_ROUNDS:
            system, user = taco_topology_prompt(
                example.question, next_graph, history, round_index)
            try:
                topology, topology_attempts = structured_call(
                    models, role="tacomas:topology", system=system, user=user,
                    validator=lambda value: _validate_topology(next_graph, value),
                    contract="bounded birth-death graph update",
                    max_tokens=config.META_MAX_TOKENS)
                next_graph = topology["graph"]
                topology_record = {key: topology[key] for key in
                                   ("rationale", "continue_evolution", "born", "died", "edge_edits")}
                if not topology["continue_evolution"]:
                    capability["continue_evolution"] = False
            except StructuredCallError as exc:
                topology_attempts = exc.attempts
                topology_record = {"invalid": True, "rationale": "projected to no change"}
            slow_updates.append({"round": round_index, "result": topology_record,
                                 "attempts": topology_attempts})

        proposal_attempts = capability_attempts + topology_attempts
        proposal = {"type": "fast_capability_round", "round": round_index,
                    "capability": capability, "slow_topology": topology_record}
        candidates.append(candidate_record(
            candidate_index, graph, result, judgment, judge_attempts,
            proposal, proposal_attempts, reward=float(judgment["quality_score"])))

        if float(judgment["quality_score"]) >= DIRECT_STOP_QUALITY:
            stop_reason = "quality_threshold"
            break
        if not capability["continue_evolution"]:
            stop_reason = "meta_time_control"
            break
        graph = next_graph

    return {"schema": "hotpot_fullwiki_tacomas_adapted/1",
            "condition": "tacomas_adapted", "task_id": example.id,
            "upstream_commit": UPSTREAM_COMMIT,
            "settings": {"max_fast_rounds": MAX_FAST_ROUNDS,
                         "bd_check_interval": BD_CHECK_INTERVAL,
                         "graph_rewire_interval": GRAPH_REWIRE_INTERVAL,
                         "n_min": N_MIN, "n_max": N_MAX,
                         "max_birth_death_pairs": MAX_BIRTH_DEATH_PAIRS,
                         "max_edge_edits": MAX_EDGE_EDITS,
                         "direct_stop_quality": DIRECT_STOP_QUALITY},
            "slow_updates": slow_updates, "stop_reason": stop_reason,
            "candidates": candidates,
            "selected_candidate": selected_by_reward(candidates)}
