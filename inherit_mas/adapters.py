"""Gold-free adapters for the first two Inherit-MAS benchmark families."""
from __future__ import annotations

import json
from typing import Any


HOTPOT_WORKFLOW_CONTRACT = r'''{
  "nodes": [
    {"id":"researcher_a","role":"researcher","subtask":"...","system_prompt":"...",
     "tools":["search_fullwiki"],"inputs":[]},
    {"id":"answerer","role":"synthesizer","subtask":"...","system_prompt":"...",
     "tools":[],"inputs":[{"from":"researcher_a","port":"evidence"}]}
  ],
  "output":"answerer"
}
Typed roles: planner emits plan and accepts no inputs; researcher emits evidence and accepts plan/evidence;
verifier emits critique and accepts evidence/critique; synthesizer emits answer and accepts plan/evidence/critique.
Each edge port MUST equal the source role's emitted type. Roots are planner/researcher. Exactly one synthesizer
is the output and only sink. At least one researcher has search_fullwiki. Use 2..7 nodes.'''

HOTPOT_INTERFACE_CONTRACT = (
    "The output is JSON with answer:string and supporting_facts:[[title:string,sentence_id:int],...]. "
    "Each supporting fact must name a sentence actually observed through search_fullwiki. The evaluator "
    "scores answer and supporting facts, but neither gold field is available during synthesis, execution, "
    "judging, refinement, or selection."
)

WORKBENCH_WORKFLOW_CONTRACT = r'''{
  "name":"short name",
  "nodes":[
    {"id":"worker_a","role":"worker","subtask":"...","system_prompt":"...",
     "tools":["declared read-only tool"],"inputs":[]},
    {"id":"integrator","role":"integrator","subtask":"combine evidence",
     "system_prompt":"...","tools":[],
     "inputs":[{"port":"evidence_a","from":"worker_a"}]},
    {"id":"executor","role":"executor","subtask":"execute approved plan",
     "system_prompt":"","tools":[],"inputs":[{"port":"plan","from":"integrator"}]}
  ],
  "sink":"executor",
  "rationale":"brief"
}
Roles are worker, integrator, verifier, critic, executor. Only workers may be roots or use declared read-only
tools. Critics consume worker/critic/integrator reports. Integrators consume worker/critic reports. A verifier
has exactly one plan input from an integrator and may also consume worker/critic evidence. Exactly one
deterministic executor is the sole sink and accepts exactly one plan input from an integrator or verifier.
Every node reaches the sink; the graph is acyclic; use at most eight LLM nodes and twelve ordered edges.'''

WORKBENCH_INTERFACE_CONTRACT = (
    "The output is an ordered list of state-changing tool calls using only declared tool names and argument "
    "schemas. Read-only observations may justify arguments but are not output actions. The official evaluator "
    "compares the resulting environment state and harmful side effects; gold actions and state diffs are hidden "
    "from synthesis, execution, judging, refinement, and selection."
)


def _observed_hotpot_sentences(trace: dict) -> dict[tuple[str, int], str]:
    observed: dict[tuple[str, int], str] = {}
    for node in trace.get("nodes", {}).values():
        for event in node.get("tool_events", []):
            for row in event.get("result", {}).get("results", []):
                title = str(row.get("title", ""))
                for sentence in row.get("sentences", []):
                    try:
                        sid = int(sentence.get("sentence_id"))
                    except (TypeError, ValueError):
                        continue
                    observed[(title, sid)] = str(sentence.get("text", ""))
    return observed


class HotpotQAAdapter:
    name = "hotpotqa-fullwiki"
    workflow_contract = HOTPOT_WORKFLOW_CONTRACT
    interface_contract = HOTPOT_INTERFACE_CONTRACT

    def task_id(self, task: Any) -> str:
        return str(task.id)

    def task_text(self, task: Any) -> str:
        return str(task.question)

    def audit(self, prediction: Any, trace: dict) -> dict:
        prediction = prediction if isinstance(prediction, dict) else {}
        observed = _observed_hotpot_sentences(trace)
        cited = []
        for raw in prediction.get("supporting_facts", []):
            if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
                continue
            try:
                key = (str(raw[0]), int(raw[1]))
            except (TypeError, ValueError):
                continue
            cited.append({"artifact_id": f"sentence:{key[0]}:{key[1]}",
                          "source": key[0], "locator": key[1],
                          "observed": key in observed,
                          "content": observed.get(key, "")[:700]})
        citation_valid = bool(cited) and all(row["observed"] for row in cited)
        return {"schema": "inherit_mas_artifact_audit/1",
                "output_valid": bool(prediction.get("valid", False)
                                     and prediction.get("citation_valid", citation_valid)),
                "artifacts": cited,
                "execution_errors": {nid: row.get("error", "")
                                     for nid, row in trace.get("nodes", {}).items()
                                     if row.get("error")},
                "interface_diagnostics": {"observed_citation_count": sum(r["observed"] for r in cited),
                                          "declared_citation_count": len(cited)}}

    def compact_trace(self, trace: dict) -> dict:
        return {"live_tokens": trace.get("live_tokens", 0),
                "reused_tokens": trace.get("reused_tokens", 0),
                "tool_calls": trace.get("search_calls", 0),
                "nodes": {nid: {"role": row.get("role"), "error": row.get("error", ""),
                                "reused": bool(row.get("reused")),
                                "output_excerpt": str(row.get("output", ""))[:240]}
                          for nid, row in trace.get("nodes", {}).items()}}


class WorkBenchAdapter:
    name = "workbench"
    workflow_contract = WORKBENCH_WORKFLOW_CONTRACT
    interface_contract = WORKBENCH_INTERFACE_CONTRACT

    def task_id(self, task: Any) -> str:
        return str(task.id)

    def task_text(self, task: Any) -> str:
        return str(task.task)

    def audit(self, prediction: Any, trace: dict) -> dict:
        actions = prediction if isinstance(prediction, list) else []
        artifacts = []
        for node_id, node in trace.get("nodes", {}).items():
            for index, event in enumerate(node.get("tool_events", [])):
                artifacts.append({
                    "artifact_id": f"{node_id}:{index}",
                    "source": str(event.get("tool", event.get("name", ""))),
                    "locator": None, "observed": True,
                    "content": json.dumps(event.get("result", event.get("observation", event)),
                                          ensure_ascii=True, default=str)[:1000],
                })
        invalids = list(trace.get("invalids", [])) + list(trace.get("ref_rejects", []))
        node_errors = {
            node_id: str(node.get("error", "") or "invalid_structured_output")
            for node_id, node in trace.get("nodes", {}).items()
            if node.get("error") or node.get("parse_valid") is False
        }
        return {"schema": "inherit_mas_artifact_audit/1",
                "output_valid": isinstance(prediction, list) and not invalids and not node_errors,
                "artifacts": artifacts,
                "execution_errors": node_errors,
                "interface_diagnostics": {"action_count": len(actions),
                                          "invalid_output_count": len(invalids),
                                          "actions": actions}}

    def compact_trace(self, trace: dict) -> dict:
        return {"live_tokens": trace.get("live_tokens", trace.get("total_tokens", 0)),
                "reused_tokens": trace.get("reused_tokens", 0),
                "tool_calls": trace.get("tool_calls", trace.get("worker_iters", 0)),
                "active_domains": trace.get("active_domains", []),
                "invalids": trace.get("invalids", []),
                "ref_rejects": trace.get("ref_rejects", []),
                "observed_artifact_count": sum(
                    len(node.get("tool_events", []))
                    for node in trace.get("nodes", {}).values()),
                "nodes": {node_id: {"role": node.get("role"),
                                      "error": node.get("error", ""),
                                      "parse_valid": node.get("parse_valid", True),
                                      "reused": bool(node.get("reused")),
                                      "output_excerpt": str(node.get("output", ""))[:240]}
                          for node_id, node in trace.get("nodes", {}).items()}}


def adapter_contract_digest_payload(adapter: Any) -> dict:
    """Stable adapter identity for configuration validation."""
    return {"name": adapter.name, "workflow_contract": adapter.workflow_contract,
            "interface_contract": adapter.interface_contract,
            "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}"}


__all__ = ["HotpotQAAdapter", "WorkBenchAdapter", "adapter_contract_digest_payload",
           "HOTPOT_INTERFACE_CONTRACT", "HOTPOT_WORKFLOW_CONTRACT",
           "WORKBENCH_INTERFACE_CONTRACT", "WORKBENCH_WORKFLOW_CONTRACT"]
