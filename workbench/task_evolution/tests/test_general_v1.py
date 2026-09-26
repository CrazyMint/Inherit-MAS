from task_evolution.general_v1 import ADAPTER, apply_generic_edit, generic_menu
from task_evolution.schema import validate_graph


def graph():
    return validate_graph({
        "name": "g",
        "nodes": [
            {"id": "worker", "role": "worker", "subtask": "inspect",
             "system_prompt": "be exact", "tools": [], "inputs": []},
            {"id": "integrator", "role": "integrator", "subtask": "plan",
             "system_prompt": "emit actions", "tools": [],
             "inputs": [{"port": "evidence", "from": "worker"}]},
            {"id": "executor", "role": "executor", "subtask": "execute",
             "system_prompt": "", "tools": [],
             "inputs": [{"port": "plan", "from": "integrator"}]},
        ],
        "sink": "executor", "rationale": "test",
    })


def test_native_menu_bridge_applies_one_atomic_edit():
    parent = graph()
    menu = generic_menu(parent)
    prompt_edit = next(row for row in menu if row["op"] == "edit_prompt")
    child, applied = apply_generic_edit(parent, menu, {
        "edit_index": prompt_edit["edit_index"], "rationale": "clarify",
        "replacement": "new exact instruction",
    })
    assert child != parent
    assert applied["kind"] == "replace_system_prompt"
    validate_graph(child)


def test_workbench_menu_uses_generic_edit_surface():
    menu = generic_menu(graph())
    assert all({"edit_index", "op", "target", "needs"} <= set(row) for row in menu)
    assert {row["op"] for row in menu} >= {"edit_prompt", "edit_subtask"}


def test_generic_adapter_exposes_exact_workbench_tool_contract():
    assert "email.search_emails" in ADAPTER.interface_contract
    assert "calendar.search_events" in ADAPTER.interface_contract
    assert "DECLARED WRITE ACTION SCHEMAS" in ADAPTER.interface_contract

