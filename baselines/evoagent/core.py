"""Benchmark-neutral EvoAgent population evolution.

The upstream repository implements the algorithm inside task scripts.  This
module factors out the common control flow without changing it: generate one
new expert, quality-filter it, execute it independently, and integrate its
result into the population result.  Population size is one per iteration in
the published NLP and ScienceWorld configurations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

UPSTREAM_COMMIT = "fc6d087b119df69466c2372cfcaf588c040aaba8"
POPULATION_PER_ITERATION = 1
NLP_ITERATIONS = 3
INTERACTIVE_ITERATIONS = 1
# Upstream accepts the fifth proposed role after four rejected checks.
MAX_ROLE_PROPOSALS = 5


@dataclass(frozen=True)
class EvolutionResult:
    initial_result: str
    final_result: str
    expert_descriptions: tuple[str, ...]
    trace: tuple[dict, ...]

    @property
    def candidate_results(self) -> tuple[str, ...]:
        return (self.initial_result,) + tuple(row["integrated_result"] for row in self.trace)


def evolve_result(
    *,
    task: str,
    initial_result: str,
    iterations: int,
    propose_expert: Callable[[str, str, tuple[str, ...], int, int], str],
    check_expert: Callable[[str, tuple[str, ...], str, int, int], str],
    execute_expert: Callable[[str, str, int], str],
    integrate: Callable[[str, str, str, str, int], str],
) -> EvolutionResult:
    """Run Algorithm 1's one-child-per-iteration specialization.

    Quality rejection is not resampled away: every proposal and verdict is
    retained in the trace.  The fifth proposal is accepted exactly as the
    upstream ``flag > 3`` escape hatch does, and is marked ``forced_accept``.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    current = str(initial_result)
    descriptions: list[str] = []
    trace: list[dict] = []
    for iteration in range(iterations):
        attempts: list[dict] = []
        description = ""
        forced = False
        for proposal_index in range(MAX_ROLE_PROPOSALS):
            description = str(propose_expert(
                task, current, tuple(descriptions), iteration, proposal_index)).strip()
            verdict = str(check_expert(
                task, tuple(descriptions), description, iteration, proposal_index)).strip()
            rejected = "discard" in verdict.lower()
            forced = rejected and proposal_index == MAX_ROLE_PROPOSALS - 1
            attempts.append({
                "proposal_index": proposal_index,
                "description": description,
                "verdict": verdict,
                "rejected": rejected,
                "forced_accept": forced,
            })
            if not rejected or forced:
                break
        descriptions.append(description)
        child = str(execute_expert(task, description, iteration))
        integrated = str(integrate(task, current, description, child, iteration))
        trace.append({
            "iteration": iteration,
            "prior_result": current,
            "role_attempts": attempts,
            "description": description,
            "child_result": child,
            "integrated_result": integrated,
            "forced_accept": forced,
        })
        current = integrated
    return EvolutionResult(initial_result, current, tuple(descriptions), tuple(trace))
