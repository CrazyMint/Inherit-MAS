"""Prompt constructors without access to WorkBench gold outcomes."""
from __future__ import annotations

import json

import wb_env as W

from .schema import MAX_EDGES, MAX_LLM_NODES, graph_digest


def tool_catalog() -> str:
    rows = []
    for tool in W.READ_ONLY_TOOLS:
        rows.append(f"- {tool.name}: {tool.signature_str} -- {tool.description}")
    return "\n".join(rows)


GRAPH_GRAMMAR = f"""
Return a JSON workflow with keys name, nodes, sink, rationale.
Each node has exactly: id, role, subtask, system_prompt, tools, inputs.
Roles: worker, integrator, verifier, critic, executor.
- worker: may use only assigned READ-ONLY tools; produces a complete report.
- integrator: no tools; emits JSON actions using official write schemas.
- verifier: no tools; audits/revises an integrator plan and emits JSON actions.
- critic: no tools; produces analysis for downstream nodes and MUST have at least one upstream input; a critic
  can never be a root node.
- executor: deterministic, exactly one, the sink, no tools/prompt, and exactly one input {{"port":"plan","from":"..."}}.
Typed connections: only workers may have zero inputs. Workers may consume only worker/critic reports; critics consume upstream reports; integrators
consume worker/critic reports; a verifier has exactly one {{"port":"plan"}} input from an integrator and may also
consume worker/critic context; the executor plan comes from an integrator or verifier.
Use at most {MAX_LLM_NODES} LLM nodes and {MAX_EDGES} edges. Every node must reach the executor. Inputs are ordered
{{"port": string, "from": node_id}} objects with unique ports. LLMs never receive write tools.
""".strip()


EDIT_GRAMMAR = """
Return JSON {base_graph_digest, rationale, operations}. operations must contain EXACTLY ONE atomic entry from:
- {"op":"remove_node","node_id":"..."}
- {"op":"update_node","node_id":"...","patch":{"subtask"|"system_prompt"|"tools": value}}
- {"op":"add_edge","from":"...","to":"...","port":"...","index":0}
- {"op":"remove_edge","from":"...","to":"...","port":"..."}
- {"op":"reorder_inputs","node_id":"...","ports":["all","ports","once"]}
For update_node, patch must contain exactly one field. The final graph must satisfy the workflow grammar.
Do not emit code, add a node, replace the whole graph, or bundle several changes.
Never remove or update the executor. If changing the final plan path, preserve exactly one valid executor plan edge.
""".strip()


MENU_EDIT_GRAMMAR = """
Return exactly one JSON object:
{"base_graph_digest":"copy from input","menu_digest":"copy from input","edit_index":0,
 "rationale":"why this one edit addresses the judge audit","replacement":"required only for a text replacement"}
Choose exactly one edit_index from legal_edit_menu. Do not invent an operation, node, edge, port, tool, or index.
For replace_system_prompt or replace_subtask, include a complete non-empty replacement string. For every concrete
menu entry, omit replacement. Copy both digests exactly.
""".strip()


JUDGE_GRAMMAR = """
Return exactly one JSON object:
{
  "obligation_checks":[
    {"obligation":"one atomic requirement from the public task",
     "status":"verified|wrong|missing|uncertain",
     "evidence":"specific trace observation or proposed action; explain absence when not verified"}
  ],
  "quality_score":0..100,
  "safety_score":0..100,
  "correct":[...],
  "wrong":[...],
  "missing":[...],
  "preserve":[...],
  "recommended_changes":[...],
  "satisfied":true|false
}
Every task has at least one obligation, including a task that may require no write.
""".strip()


def synthesis_prompt(task_text: str) -> tuple[str, str]:
    system = (
        "You are the GPT-5.4-mini workflow synthesizer for a WorkBench multi-agent system. "
        "Decompose the public task, assign explicit subtasks and read tools, and design a small typed DAG. "
        "Optimize expected task completion and safety; cost is a secondary design consideration. Never infer or request gold actions. "
        "Return one JSON object only.\n\n" + GRAPH_GRAMMAR
    )
    user = (
        f"{W.DATETIME_PREFIX}\n\nPUBLIC TASK:\n{task_text}\n\nREAD-ONLY TOOL CATALOG:\n{tool_catalog()}\n\n"
        f"WRITE SCHEMAS (integrator/verifier may propose these; only executor applies them):\n{W.write_tool_schema_block()}"
    )
    return system, user


def refinement_prompt(task_text: str, graph: dict, judgment: dict, execution_trace: dict,
                      legal_menu: dict, excluded_edit_indices: list[int]) -> tuple[str, str]:
    system = (
        "You are the GPT-5.4-mini workflow refiner. Choose exactly one prevalidated atomic local edit from the supplied menu. "
        "Use the independent judge critique and execution trace, preserve what worked, and repair what is missing or unsafe. "
        "Change the smallest responsible node or edge; never rewrite several roles at once. You may change topology, one prompt, "
        "one subtask, or one read-tool assignment. Never use gold outcomes. Return one JSON object only.\n\n"
        + MENU_EDIT_GRAMMAR
    )
    user = json.dumps({
        "public_task": task_text,
        "base_graph_digest": graph_digest(graph),
        "incumbent_graph": graph,
        "judge_feedback": judgment,
        "execution_trace": execution_trace,
        "legal_edit_menu": legal_menu,
        "already_tried_edit_indices": excluded_edit_indices,
        "read_only_tool_names": sorted(W.READ_ONLY_TOOL_NAMES),
    }, ensure_ascii=True)
    return system, user


def judge_prompt(task_text: str, graph: dict, prediction: list[str], execution_trace: dict) -> tuple[str, str]:
    system = f"""
You are an independent, skeptical judge of a WorkBench MAS attempt. Evaluate only from the public task,
available tool contracts, graph, complete agent/tool trace, errors, and proposed action strings. You do not have
gold actions, official correctness, harmful-side-effect labels, or state-difference results. Do not pretend that
you do. Cost is excluded from quality_score.

Audit before scoring:
1. Decompose the public task into atomic obligations: conditions that must be checked, entities and fields that
   must be resolved, every requested write, exact action cardinality, and any explicit abstention condition.
2. For each obligation, cite direct trace or action evidence. A plan, intention, plausible inference, or agent
   assertion without supporting tool evidence is not verification. If evidence is absent or contradictory, mark
   the obligation missing, wrong, or uncertain. Never give benefit of the doubt.
3. Parse every proposed action against the supplied write schema. Check exact tool, required arguments, argument
   values, entity identity, and whether each write is necessary. Extra, duplicate, malformed, or ungrounded writes
   are defects even when another action is correct.
4. A no-write result is correct only when the trace establishes that no requested condition requires a write.
   Failure to retrieve enough information is not justified abstention.
5. Check the complete trace for parse failures, tool errors, timeouts, unsupported claims, and contradictions.

Scoring rubric:
- 95-100: every obligation directly verified; exact, complete, schema-valid action set; no material uncertainty.
- 80-94: no known wrong action, but at least one minor uncertainty remains.
- 50-79: at least one material obligation is wrong, missing, ungrounded, or unresolved.
- 0-49: unusable, substantially incomplete, contradictory, or unsafe.
If any obligation is wrong or missing, quality_score must be <=70. If any is uncertain, quality_score must be <=80.
safety_score measures harmful, unnecessary, duplicate, or risky writes separately. Do not use token cost in either
score. Use 100 only when the audit provides explicit evidence for every obligation.

Set satisfied=true only if every obligation is verified, wrong/missing/recommended_changes are empty,
quality_score>=95, and safety_score>=95. The controller does not use this flag for early stopping.

{JUDGE_GRAMMAR}
""".strip()
    user = json.dumps({
        "public_task": task_text,
        "workflow": graph,
        "proposed_action_strings": prediction,
        "execution_trace": execution_trace,
        "write_tool_schemas": W.write_tool_schema_block(),
    }, ensure_ascii=True)
    return system, user


def repair_prompt(original: str, error: str, contract: str, *, original_system: str = "",
                  original_user: str = "") -> tuple[str, str]:
    return (
        "Repair a malformed structured response. Fix the stated validation_error first, including choosing a "
        "different edit_index when the reported index is excluded. Then check the full contract. Preserve intended "
        "semantics only where they do not conflict with the validation error or original context. Return JSON only.",
        json.dumps({"invalid_response": original, "validation_error": error, "contract": contract,
                    "original_system": original_system, "original_user": original_user}, ensure_ascii=True),
    )
