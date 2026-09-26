#!/usr/bin/env python
"""Run the whole Inherit-MAS loop on one toy task with a fake model client.

No credentials, no network, no dataset download, no API spend: the model client is
the deterministic stub the WorkBench test suite already uses
(``task_evolution.tests.test_executor.FakeModels``), extended with fixed replies
for the four controller roles (synthesizer, judge, selector, refiner). Everything
else -- synthesis, typed-graph validation, execution, the gold-free judge, node
selection, validated edits, execution inheritance, and the final ranking -- uses
the WorkBench runtime implementation.

Usage:  python scripts/smoke_test.py
Writes only into a temporary directory, and exits 0 when the loop discarded at
least one node, applied at least one validated edit, and inherited at least one
node instead of re-executing it.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workbench"))

from task_evolution.controller import PublicTask  # noqa: E402
from task_evolution.models import ModelResult, Usage  # noqa: E402
from task_evolution.schema import validate_graph  # noqa: E402
from task_evolution.select_edit import CONDITION, run_condition  # noqa: E402
from task_evolution.tests.test_executor import FakeModels  # noqa: E402

TASK = PublicTask("smoke#0000", "Find the requested record and send the update.")
# One score per judged candidate, so the final ranking is fixed.
JUDGE_SCORES = (70.0, 78.0, 74.0, 82.0, 86.0)


def toy_graph() -> dict:
    """Two workers feed the integrator, so Select has a node it may discard."""
    return validate_graph({
        "name": "smoke",
        "nodes": [
            {"id": "worker", "role": "worker", "subtask": "inspect the record",
             "system_prompt": "be exact", "tools": [], "inputs": []},
            {"id": "checker", "role": "worker", "subtask": "cross-check the record",
             "system_prompt": "be exact", "tools": [], "inputs": []},
            {"id": "integrator", "role": "integrator", "subtask": "plan",
             "system_prompt": "emit actions", "tools": [],
             "inputs": [{"port": "evidence", "from": "worker"},
                        {"port": "check", "from": "checker"}]},
            {"id": "executor", "role": "executor", "subtask": "execute",
             "system_prompt": "", "tools": [],
             "inputs": [{"port": "plan", "from": "integrator"}]},
        ],
        "sink": "executor", "rationale": "smoke",
    })


class SmokeModels(FakeModels):
    """FakeModels plus deterministic synthesizer / judge / selector / refiner replies."""

    def __init__(self):
        super().__init__()
        self.judged = 0
        self.selected = 0
        self.edited = 0

    def chat(self, *, model, role, system, user, max_tokens):
        if role == "inherit_mas_synthesizer":
            return ModelResult(json.dumps(toy_graph()), [Usage(model, role, 40, 20)])
        if role == "inherit_mas_judge":
            score = JUDGE_SCORES[min(self.judged, len(JUDGE_SCORES) - 1)]
            self.judged += 1
            return ModelResult(json.dumps({
                "quality_score": score, "completion_likelihood": score,
                "requirement_coverage": score, "artifact_grounding": score,
                "critique": "fixed smoke judgment: checker repeats worker",
                "keep": ["worker"], "fix": ["state the update explicitly"],
            }), [Usage(model, role, 30, 12)])
        if role == "inherit_mas_selector":
            # Discard the redundant checker once; keep every node afterwards.
            discardable = json.loads(user)["discardable_nodes"]
            discard = ["checker"] if self.selected == 0 and "checker" in discardable else []
            self.selected += 1
            return ModelResult(json.dumps({
                "discard": discard, "rationale": "smoke selection",
            }), [Usage(model, role, 25, 10)])
        if role == "inherit_mas_refiner":
            entry = next(row for row in json.loads(user)["available_edits"]
                         if row["op"] in {"edit_prompt", "edit_subtask"})
            self.edited += 1
            return ModelResult(json.dumps({
                "edit_index": entry["edit_index"], "rationale": "smoke edit",
                "replacement": f"{entry['current_value']} [smoke {self.edited}]",
            }), [Usage(model, role, 35, 15)])
        if role.startswith("inherit_mas_"):
            raise AssertionError(f"the smoke stub has no reply for controller role {role!r}")
        return super().chat(model=model, role=role, system=system, user=user,
                            max_tokens=max_tokens)


def report(record: dict) -> tuple[int, int, int]:
    """Print every candidate with its node-level inheritance; return (discards, edits, inherited)."""
    discards = edits = inherited = 0
    for candidate in record["candidates"]:
        index = candidate["candidate_index"]
        if candidate["status"] != "complete":
            print(f"candidate {index}: {candidate['status']} "
                  f"({candidate['proposal_attempts'][-1]['error']})")
            continue
        proposal = candidate["proposal"]
        kind = proposal.get("kind", proposal["type"])
        if proposal["type"] == "select_edit":
            edits += 1
            discards += len(proposal["discard"])
            kind = f"select(discard={proposal['discard']}) edit({kind})"
        trace = candidate["execution"]
        chosen = " <- returned" if index == record["selected_candidate"] else ""
        print(f"candidate {index}: {kind}"
              f"  quality={candidate['judgment']['quality_score']:.0f}"
              f"  valid_output={candidate['artifact_audit']['output_valid']}"
              f"  live={trace['live_tokens']}tok/{trace['live_calls']}calls"
              f"  inherited={trace['reused_tokens']}tok/{trace['reused_calls']}calls{chosen}")
        for node_id, node in trace["nodes"].items():
            if node["reused"]:
                inherited += 1
                print(f"    {node_id:12s} inherited stored result ({node['reuse_reason']})")
            else:
                print(f"    {node_id:12s} executed live"
                      f"{' (state-changing sink, never inherited)' if node['role'] == 'executor' else ''}")
    return discards, edits, inherited


def main() -> int:
    models = SmokeModels()
    with tempfile.TemporaryDirectory(prefix="inherit-mas-smoke-") as tmp:
        record = run_condition(TASK, models=models, run_dir=Path(tmp) / "run")
    print(f"condition={record['condition']} benchmark={record['benchmark']} "
          f"task={record['task_id']} candidates={len(record['candidates'])}")
    discards, edits, inherited = report(record)
    print(f"returned candidate: {record['selected_candidate']}  "
          f"prediction: {record['selected_prediction']}")
    if record["condition"] != CONDITION or record["selected_candidate"] is None:
        print("FAIL: the loop returned no candidate", file=sys.stderr)
        return 1
    if not discards:
        print("FAIL: Select discarded no node", file=sys.stderr)
        return 1
    if not edits:
        print("FAIL: no validated edit was applied", file=sys.stderr)
        return 1
    if not inherited:
        print("FAIL: no node inherited a stored result", file=sys.stderr)
        return 1
    print(f"OK: {discards} node(s) discarded, {edits} validated edit(s) applied, "
          f"{inherited} node execution(s) inherited, no model call left this process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
