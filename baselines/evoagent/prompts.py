"""Task-neutral adaptations of the official EvoAgent collaboration prompts."""
from __future__ import annotations

import json


def propose_expert(task: str, result: str, prior: tuple[str, ...], contract: str) -> tuple[str, str]:
    system = (
        "You create one additional expert for an EvoAgent population. Describe the expert in the "
        "second person, beginning with 'You are'. Give it one distinct specialization or subtask. "
        "Return only the expert description. Do not solve the task."
    )
    user = (
        f"Task:\n{task}\n\nCurrent population result:\n{result}\n\n"
        f"Previously created experts:\n{json.dumps(prior, ensure_ascii=True)}\n\n"
        f"Result contract:\n{contract}\n\nCreate exactly one useful, non-duplicate expert."
    )
    return system, user


def check_expert(task: str, prior: tuple[str, ...], description: str) -> tuple[str, str]:
    system = (
        "You are EvoAgent's quality-selection step. Evaluate whether the proposed expert can help "
        "with the task and is distinct from earlier experts. Give a short reason, then end with "
        "exactly Retain or Discard. Do not solve the task."
    )
    user = (
        f"Task:\n{task}\n\nEarlier experts:\n{json.dumps(prior, ensure_ascii=True)}\n\n"
        f"Proposed expert:\n{description}"
    )
    return system, user


def expert_system(description: str, base_contract: str) -> str:
    return (
        f"{description}\n\nWork independently on the complete task using your specialization. "
        "You receive the same task information as the initial agent. "
        f"{base_contract}"
    )


def integrate(task: str, old_result: str, description: str, child_result: str,
              output_contract: str) -> tuple[str, str]:
    system = (
        "You integrate one new EvoAgent expert into the current population result. Critically decide "
        "which claims or proposed actions to keep; the new expert may be wrong. Return only a result "
        "that satisfies the exact output contract."
    )
    user = (
        f"Task:\n{task}\n\nCurrent result:\n{old_result}\n\n"
        f"New expert description:\n{description}\n\nNew expert result:\n{child_result}\n\n"
        f"Output contract:\n{output_contract}"
    )
    return system, user
