from __future__ import annotations

from dataclasses import dataclass

import pytest

from .adapters import HotpotQAAdapter, WorkBenchAdapter
from .core import (GENERAL_JUDGMENT_CONTRACT, _available_menu, _judge_prompt,
                   _repair_prompt, _synthesis_prompt, initial_candidate_index,
                   selected_index, validate_judgment)


@dataclass
class HotTask:
    id: str = "h1"
    question: str = "Which result follows from the provided sources?"


@dataclass
class WBTask:
    id: str = "w1"
    task: str = "Send the requested update after checking the record."


def _candidate(index, quality, coverage, grounding, *, valid=True, hard=0):
    return {"status": "complete", "candidate_index": index, "hard_failures": hard,
            "artifact_audit": {"output_valid": valid},
            "judgment": {"quality_score": quality, "completion_likelihood": quality,
                         "requirement_coverage": coverage, "artifact_grounding": grounding,
                         "critique": "x", "keep": [], "fix": []}}


def test_generic_selector_has_no_source_count_gate():
    weak = _candidate(0, 60, 60, 95)
    strong = _candidate(1, 90, 90, 80)
    weak["artifact_audit"]["artifacts"] = [{"source": "a"}, {"source": "b"}]
    strong["artifact_audit"]["artifacts"] = [{"source": "a"}]
    assert selected_index([weak, strong]) == 1


def test_hard_execution_failures_dominate_subjective_confidence():
    clean = _candidate(0, 55, 55, 55)
    broken = _candidate(1, 99, 99, 99, hard=1)
    assert selected_index([clean, broken]) == 0


def test_judgment_contract_is_benchmark_neutral_and_strict():
    value = {"quality_score": 80, "completion_likelihood": 70,
             "requirement_coverage": 60, "artifact_grounding": 50,
             "critique": "missing one requirement", "keep": ["worker"], "fix": ["repair"]}
    assert validate_judgment(value)["quality_score"] == 80
    with pytest.raises(ValueError):
        validate_judgment({**value, "gold_score": 1})
    assert "title" not in GENERAL_JUDGMENT_CONTRACT.lower()
    assert "action" not in GENERAL_JUDGMENT_CONTRACT.lower()


def test_available_menu_hides_tried_edits_and_exposes_noop_guard():
    graph = {"nodes": [{"id": "a", "system_prompt": "old", "subtask": "task"}]}
    menu = [{"edit_index": 0, "op": "edit_prompt", "target": "a", "needs": "replacement"},
            {"edit_index": 1, "op": "prune_node", "target": "a", "needs": "none"}]
    shown = _available_menu(graph, menu, {1})
    assert [row["edit_index"] for row in shown] == [0]
    assert shown[0]["current_value"] == "old"
    assert "differ" in shown[0]["constraint"]


def test_same_core_prompts_accept_hotpot_and_workbench_adapters():
    hot, wb = HotpotQAAdapter(), WorkBenchAdapter()
    hs, hu = _synthesis_prompt(hot, HotTask(), escape=None)
    ws, wu = _synthesis_prompt(wb, WBTask(), escape=None)
    assert hs == ws
    assert "hotpotqa-fullwiki" in hu and "workbench" in wu
    hjs, _ = _judge_prompt(hot, HotTask(), {}, {"output_valid": True, "artifacts": []})
    wjs, _ = _judge_prompt(wb, WBTask(), [], {"output_valid": True, "artifacts": []})
    assert hjs == wjs
    assert "benchmark-specific number" in hjs


def test_adapter_audits_share_one_schema_without_gold():
    hot = HotpotQAAdapter().audit(
        {"valid": True, "citation_valid": True, "supporting_facts": [["Doc", 0]]},
        {"nodes": {"r": {"tool_events": [{"result": {"results": [
            {"title": "Doc", "sentences": [{"sentence_id": 0, "text": "fact"}]}
        ]}}]}}})
    wb = WorkBenchAdapter().audit(
        ['email.send_email.func(recipient="a@b.com")'],
        {"observed_artifacts": [{"artifact_id": "a1", "tool": "find_email",
                                  "content": "a@b.com"}], "invalids": [], "ref_rejects": []})
    assert hot["schema"] == wb["schema"] == "inherit_mas_artifact_audit/1"
    assert hot["output_valid"] and wb["output_valid"]
    for value in (hot, wb):
        text = repr(value).lower()
        assert "gold" not in text and "official" not in text


def test_repair_keeps_original_task_and_interface_context():
    system, user = _repair_prompt("bad", "unknown tool", "contract",
                                  original_system="synthesis system",
                                  original_user="task plus tool catalog")
    assert "unknown tool" in user
    assert "task plus tool catalog" in user
    assert "synthesis system" in user


def test_initial_candidate_is_first_executed_synthesis_not_first_attempt():
    candidates = [
        {"status": "invalid_proposal", "candidate_index": 0},
        {"status": "complete", "candidate_index": 1,
         "proposal": {"type": "initial_synthesis"}},
        {"status": "complete", "candidate_index": 2,
         "proposal": {"type": "atomic_edit"}},
    ]
    assert initial_candidate_index(candidates) == 1
