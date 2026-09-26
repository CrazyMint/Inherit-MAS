"""Benchmark adaptation prompts derived from pinned external algorithms."""
from __future__ import annotations

import json

from .graph import canonical_node

GRAPH_CONTRACT = r'''{
  "nodes": [
    {"id":"researcher","role":"researcher","subtask":"...","system_prompt":"...",
     "tools":["search_fullwiki"],"inputs":[]},
    {"id":"answerer","role":"synthesizer","subtask":"...","system_prompt":"...",
     "tools":[],"inputs":[{"from":"researcher","port":"evidence"}]}
  ],
  "output":"answerer"
}'''


def _pool_description(pool: dict[str, dict]) -> list[dict]:
    return [{"id": name,
             "roles": [node["role"] for node in graph["nodes"]],
             "nodes": len(graph["nodes"]),
             "topology": [{"to": node["id"], "inputs": node["inputs"]}
                          for node in graph["nodes"]]}
            for name, graph in sorted(pool.items())]


def evomas_selection_prompt(question: str, pool: dict[str, dict]) -> tuple[str, str]:
    system = (
        "You are the EvoMAS selection operator. Select exactly two distinct parent MAS configurations "
        "whose roles and topology are relevant and structurally diverse for the given task. Do not solve "
        "the task and do not use gold information. Return JSON only."
    )
    user = json.dumps({"question": question, "pool": _pool_description(pool),
                       "contract": {"selected": ["pool_id_1", "pool_id_2"],
                                    "rationale": "brief"}}, ensure_ascii=True)
    return system, user


def evomas_generate_prompt(question: str, parent_id: str, parent: dict) -> tuple[str, str]:
    system = (
        "You are the EvoMAS generation operator. Adapt the selected MAS to this HotpotQA FullWiki "
        "question. You may modify roles, prompts, retrieval assignment, and topology, or add/remove "
        "agents, while retaining 2-7 nodes, exactly one synthesizer sink, at least one researcher, "
        "and at least one search_fullwiki tool. All agents use the fixed worker model. Return only the "
        "full graph JSON."
    )
    user = json.dumps({"question": question, "parent_id": parent_id,
                       "parent_graph": parent, "contract": GRAPH_CONTRACT}, ensure_ascii=True)
    return system, user


def evomas_mutation_prompt(question: str, graph: dict, judgment: dict,
                           trace: dict) -> tuple[str, str]:
    system = (
        "You are the EvoMAS mutation operator. Mutate exactly one component type across the MAS: "
        "prompts (subtask/system_prompt), tools (search assignment), or topology (ordered inputs). "
        "Do not add/remove/rename agents. Use the gold-free judge and execution observations. Return "
        "JSON with component_type, rationale, and the complete child graph."
    )
    user = json.dumps({"question": question, "parent_graph": graph, "judge": judgment,
                       "execution": trace,
                       "contract": {"component_type": "prompts|tools|topology",
                                    "rationale": "brief", "graph": GRAPH_CONTRACT}},
                      ensure_ascii=True)
    return system, user


def evomas_crossover_prompt(question: str, graph_a: dict, graph_b: dict,
                            evidence: list[dict]) -> tuple[str, str]:
    system = (
        "You are the EvoMAS crossover operator. Inherit the entire node-id set and topology from "
        "exactly one parent, then recombine or improve agent prompts and retrieval assignments using "
        "both parents' strengths. Do not add/remove/rename agents or invent a third topology. Return "
        "JSON with topology_parent, rationale, and the complete child graph."
    )
    user = json.dumps({"question": question, "parent_a": graph_a, "parent_b": graph_b,
                       "parent_diagnostics": evidence,
                       "contract": {"topology_parent": "a|b", "rationale": "brief",
                                    "graph": GRAPH_CONTRACT}}, ensure_ascii=True)
    return system, user


def taco_capability_prompt(question: str, graph: dict, judgment: dict,
                           trace: dict, round_index: int) -> tuple[str, str]:
    system = (
        "You are TacoMAS's fast capability-update meta-LLM. Using the gold-free judge and current "
        "agent outputs, score contributions and emit targeted prompt/memory deltas for zero or more "
        "existing agents. Do not change graph topology in this operation. Return JSON only."
    )
    user = json.dumps({"question": question, "round": round_index, "graph": graph,
                       "judge": judgment, "execution": trace,
                       "contract": {"updates": [{"agent_id": "existing id",
                                                  "contribution_score": 0,
                                                  "score_reason": "brief",
                                                  "prompt_delta": "specific next-round update"}],
                                    "continue_evolution": True,
                                    "rationale": "brief"}}, ensure_ascii=True)
    return system, user


def taco_topology_prompt(question: str, graph: dict, history: list[dict],
                         round_index: int) -> tuple[str, str]:
    system = (
        "You are TacoMAS's slow topology-update meta-LLM. Emit one bounded birth-death graph "
        "co-evolution update: at most two added agents, at most two removed agents, and at most "
        "eight directed edge edits. You may simultaneously realign surviving prompts. Preserve "
        "5-20 agents, typed acyclicity, exactly one synthesizer sink, and retrieval capability. "
        "A no-change graph is allowed when evidence does not justify an edit. Return JSON only."
    )
    user = json.dumps({"question": question, "round": round_index,
                       "current_graph": graph, "recent_rounds": history[-2:],
                       "contract": {"rationale": "brief", "graph": GRAPH_CONTRACT,
                                    "continue_evolution": True}}, ensure_ascii=True)
    return system, user


def topology_signature(graph: dict) -> tuple:
    return (tuple(node["id"] for node in graph["nodes"]), graph["output"],
            tuple((node["id"], tuple((edge["from"], edge["port"])
                                     for edge in node["inputs"]))
                  for node in graph["nodes"]))


def executable_signature(graph: dict) -> dict:
    return {node["id"]: canonical_node(node) for node in graph["nodes"]}

