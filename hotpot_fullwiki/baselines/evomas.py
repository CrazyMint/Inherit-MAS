"""HotpotQA adapter for the pinned EvoMAS evolutionary operators."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from .. import config
from ..controller import StructuredCallError
from ..schemas import SchemaError
from .common import (candidate_record, evomas_reward, graph_validator, invalid_record,
                     judge, selected_by_reward, structured_call, validate_selection)
from .execution import compact_trace, execute_graph
from .graph import evomas_seed_pool, node_map, validate_graph
from .prompts import (evomas_crossover_prompt, evomas_generate_prompt,
                      evomas_mutation_prompt, evomas_selection_prompt,
                      executable_signature, topology_signature)

UPSTREAM_COMMIT = "93fd9d6766b093f6bbdeeffc35896910cbece6e2"
NUM_PARENTS = 2
MAX_STEPS = 2  # Official benchmark-script default.
MUTATION_PROB = 0.8
MAX_NODES = 7


def _seed(question: str) -> int:
    return int(hashlib.sha256(f"evomas-hotpot\0{question}".encode()).hexdigest()[:16], 16)


def _validate_mutation(parent: dict, value) -> dict:
    if not isinstance(value, dict) or set(value) != {"component_type", "rationale", "graph"}:
        raise SchemaError("mutation requires component_type, rationale, graph")
    component = value["component_type"]
    if component not in {"prompts", "tools", "topology"}:
        raise SchemaError("component_type must be prompts, tools, or topology")
    if not isinstance(value["rationale"], str):
        raise SchemaError("rationale must be a string")
    child = validate_graph(value["graph"], max_nodes=MAX_NODES)
    parent_nodes, child_nodes = node_map(parent), node_map(child)
    if list(parent_nodes) != list(child_nodes):
        raise SchemaError("mutation cannot add, remove, reorder, or rename agents")
    changed = False
    for nid in parent_nodes:
        before, after = parent_nodes[nid], child_nodes[nid]
        if before["role"] != after["role"]:
            raise SchemaError("mutation cannot change roles")
        if component == "prompts":
            if before["tools"] != after["tools"] or before["inputs"] != after["inputs"]:
                raise SchemaError("prompt mutation changed tools or topology")
            changed |= (before["subtask"], before["system_prompt"]) != (
                after["subtask"], after["system_prompt"])
        elif component == "tools":
            if (before["subtask"], before["system_prompt"], before["inputs"]) != (
                    after["subtask"], after["system_prompt"], after["inputs"]):
                raise SchemaError("tool mutation changed prompts or topology")
            changed |= before["tools"] != after["tools"]
        else:
            if (before["subtask"], before["system_prompt"], before["tools"]) != (
                    after["subtask"], after["system_prompt"], after["tools"]):
                raise SchemaError("topology mutation changed prompts or tools")
            changed |= before["inputs"] != after["inputs"]
    if component == "topology":
        changed |= parent["output"] != child["output"]
    if not changed:
        raise SchemaError("mutation is a no-op")
    return {"component_type": component, "rationale": value["rationale"], "graph": child}


def _validate_crossover(parent_a: dict, parent_b: dict, value) -> dict:
    if not isinstance(value, dict) or set(value) != {"topology_parent", "rationale", "graph"}:
        raise SchemaError("crossover requires topology_parent, rationale, graph")
    inherited = value["topology_parent"]
    if inherited not in {"a", "b"}:
        raise SchemaError("topology_parent must be a or b")
    child = validate_graph(value["graph"], max_nodes=MAX_NODES)
    source = parent_a if inherited == "a" else parent_b
    if topology_signature(child) != topology_signature(source):
        raise SchemaError("child did not inherit one complete parent topology")
    if not isinstance(value["rationale"], str):
        raise SchemaError("rationale must be a string")
    if executable_signature(child) == executable_signature(source):
        raise SchemaError("crossover is a no-op relative to topology parent")
    return {"topology_parent": inherited, "rationale": value["rationale"], "graph": child}


def run(example, *, models, retriever, run_dir: str | Path) -> dict:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    pool = evomas_seed_pool()
    meta_operations: list[dict] = []
    selection_fallback = False
    system, user = evomas_selection_prompt(example.question, pool)
    try:
        selected, attempts = structured_call(
            models, role="evomas:select", system=system, user=user,
            validator=lambda value: validate_selection(value, set(pool), NUM_PARENTS),
            contract='{"selected":["pool_id_1","pool_id_2"],"rationale":"brief"}',
            max_tokens=config.META_MAX_TOKENS)
    except StructuredCallError as exc:
        selected = ["single_researcher", "planned_parallel_review"]
        attempts = exc.attempts
        selection_fallback = True
    meta_operations.append({"op": "select", "attempts": attempts,
                            "fallback": selection_fallback, "selected": selected})

    candidates: list[dict] = []
    for parent_id in selected:
        index = len(candidates)
        system, user = evomas_generate_prompt(example.question, parent_id, pool[parent_id])
        try:
            graph, attempts = structured_call(
                models, role="evomas:generate", system=system, user=user,
                validator=graph_validator(MAX_NODES), contract="full graph JSON",
                max_tokens=config.META_MAX_TOKENS)
            result = execute_graph(graph, example, models=models, retriever=retriever)
            judgment, judge_attempts = judge(models, example.question, graph,
                                             result.prediction, result.trace)
            reward = evomas_reward(judgment, result.trace)
            candidates.append(candidate_record(
                index, graph, result, judgment, judge_attempts,
                {"type": "generate", "seed_parent": parent_id}, attempts, reward=reward))
        except StructuredCallError as exc:
            candidates.append(invalid_record(index, "invalid_generation", exc.attempts,
                                             {"type": "generate", "seed_parent": parent_id}))

    rng = random.Random(_seed(example.question))
    for step in range(1, MAX_STEPS + 1):
        index = len(candidates)
        complete = [row for row in candidates if row.get("status") == "complete"]
        if not complete:
            candidates.append(invalid_record(index, "no_valid_parent", [],
                                             {"type": "evolution", "step": step}))
            continue
        ranked = sorted(complete, key=lambda row: row["reward"], reverse=True)
        use_mutation = rng.random() < MUTATION_PROB or len(ranked) < 2
        try:
            if use_mutation:
                parent = ranked[0]
                system, user = evomas_mutation_prompt(
                    example.question, parent["graph"], parent["judgment"],
                    compact_trace(parent["execution"]))
                evolved, attempts = structured_call(
                    models, role="evomas:mutate", system=system, user=user,
                    validator=lambda value: _validate_mutation(parent["graph"], value),
                    contract="component_type + rationale + complete child graph",
                    max_tokens=config.META_MAX_TOKENS)
                graph = evolved["graph"]
                proposal = {"type": "mutation", "step": step,
                            "parent": parent["candidate_index"],
                            "component_type": evolved["component_type"],
                            "rationale": evolved["rationale"]}
            else:
                first, second = ranked[:2]
                system, user = evomas_crossover_prompt(
                    example.question, first["graph"], second["graph"],
                    [{"candidate": row["candidate_index"], "reward": row["reward"],
                      "judge": row["judgment"]} for row in (first, second)])
                evolved, attempts = structured_call(
                    models, role="evomas:crossover", system=system, user=user,
                    validator=lambda value: _validate_crossover(
                        first["graph"], second["graph"], value),
                    contract="topology_parent + rationale + complete child graph",
                    max_tokens=config.META_MAX_TOKENS)
                graph = evolved["graph"]
                proposal = {"type": "crossover", "step": step,
                            "parents": [first["candidate_index"], second["candidate_index"]],
                            "topology_parent": evolved["topology_parent"],
                            "rationale": evolved["rationale"]}
            result = execute_graph(graph, example, models=models, retriever=retriever)
            judgment, judge_attempts = judge(models, example.question, graph,
                                             result.prediction, result.trace)
            reward = evomas_reward(judgment, result.trace)
            candidates.append(candidate_record(index, graph, result, judgment,
                                                judge_attempts, proposal, attempts,
                                                reward=reward))
        except StructuredCallError as exc:
            candidates.append(invalid_record(index, "invalid_evolution", exc.attempts,
                                             {"type": "mutation" if use_mutation else "crossover",
                                              "step": step}))

    return {"schema": "hotpot_fullwiki_evomas_adapted/1",
            "condition": "evomas_adapted", "task_id": example.id,
            "upstream_commit": UPSTREAM_COMMIT,
            "settings": {"num_parents": NUM_PARENTS, "max_steps": MAX_STEPS,
                         "mutation_probability": MUTATION_PROB,
                         "memory_evolution": False, "max_nodes": MAX_NODES},
            "meta_operations": meta_operations, "candidates": candidates,
            "selected_candidate": selected_by_reward(candidates)}
