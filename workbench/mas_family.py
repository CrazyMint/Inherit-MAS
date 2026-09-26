"""WorkBench single-agent and multi-agent workflow specifications.

Typed role pipelines use the model callbacks supplied to the execution runner:
  single_agent   (react, WRITE)  official-style ReAct agent over ALL tools       [S0]
  planner        (llm)           decomposes the task into a short plan           [M1]
  executor_agent (react, WRITE)  ReAct agent over ALL tools, conditioned on plan [M1]
  coordinator    (llm)           -> subgoals: relevant domains + per-domain goals [M2,M3]
  worker         (react, READ)   read-only evidence gathering, one per domain     [M2,M3]
  integrator     (llm)           -> proposed_actions: an ordered write plan        [M2,M3]
  verifier       (llm)           -> verifier_decision: vetted/edited final actions [M3]
  executor       (det, WRITE)    dispatches ONLY the vetted state-changing actions [M2,M3]

Topologies:
  S0  single_agent
  M1  planner -> executor_agent
  M2  coordinator -> {worker per domain} -> integrator -> executor
  M3  coordinator -> {worker per domain} -> integrator -> verifier -> executor

Safety contract (validated): EXACTLY ONE write-capable node per system, and it is
the sink that emits the scored ``prediction``. Workers are read-only. This encodes
"only one executor may invoke state-changing tools".

This module is the declarative spec + validator (imported and validated at load).
Execution lives in execution.py; the worker stage fans out to one worker per
relevant domain AT RUNTIME (WorkBench tasks touch 1–5 domains), so the fan-out
width is dynamic rather than baked into the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

WRITE, READ, NONE = "write", "read", "none"
REACT, LLM, DET = "react", "llm", "det"


@dataclass(frozen=True)
class Stage:
    id: str
    kind: str           # react | llm | det
    role: str
    emits: str          # prediction | plan | subgoals | evidence | proposed_actions | verifier_decision
    inputs: tuple[str, ...] = ()
    tool_access: str = NONE   # write | read | none
    fanout: str = ""          # "" or "domains" (worker stage expands per relevant domain)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    stages: tuple[Stage, ...]
    sink: str
    comm_edges: tuple[tuple[str, str], ...] = ()  # message edges subject to evidence transforms
    note: str = field(default="", compare=False)


class SpecError(ValueError):
    """A malformed MAS system spec (fail loud at import)."""


_VALID_EMITS = {"prediction", "plan", "subgoals", "evidence", "proposed_actions", "verifier_decision"}
_KIND_ACCESS = {  # which tool access each node kind may declare
    REACT: {WRITE, READ},
    LLM: {NONE},
    DET: {WRITE, NONE},
}


def validate_spec(spec: SystemSpec) -> None:
    if not spec.stages:
        raise SpecError(f"{spec.name}: no stages")
    ids: list[str] = []
    seen: set[str] = set()
    for st in spec.stages:
        if st.id in seen:
            raise SpecError(f"{spec.name}: duplicate stage id {st.id!r}")
        seen.add(st.id)
        ids.append(st.id)
        if st.emits not in _VALID_EMITS:
            raise SpecError(f"{spec.name}.{st.id}: bad emits {st.emits!r}")
        if st.kind not in _KIND_ACCESS or st.tool_access not in _KIND_ACCESS[st.kind]:
            raise SpecError(f"{spec.name}.{st.id}: kind {st.kind!r} cannot have access {st.tool_access!r}")
        for src in st.inputs:
            if src not in seen:  # inputs must reference EARLIER stages (topo by construction)
                raise SpecError(f"{spec.name}.{st.id}: input {src!r} not a prior stage")
        if st.fanout and st.fanout != "domains":
            raise SpecError(f"{spec.name}.{st.id}: bad fanout {st.fanout!r}")
        if st.fanout and st.role != "worker":
            raise SpecError(f"{spec.name}.{st.id}: only the worker stage may fan out")
    # sink present and emits the scored prediction
    if spec.sink not in seen:
        raise SpecError(f"{spec.name}: sink {spec.sink!r} not a stage")
    sink = next(s for s in spec.stages if s.id == spec.sink)
    if sink.emits != "prediction":
        raise SpecError(f"{spec.name}: sink must emit 'prediction'")
    # EXACTLY ONE write-capable node, and it is the sink (only-one-executor contract)
    writers = [s for s in spec.stages if s.tool_access == WRITE]
    if len(writers) != 1:
        raise SpecError(f"{spec.name}: exactly one write-capable node required, got {[w.id for w in writers]}")
    if writers[0].id != spec.sink:
        raise SpecError(f"{spec.name}: the write-capable node must be the sink")
    # comm edges must reference real stages
    for a, b in spec.comm_edges:
        if a not in seen or b not in seen:
            raise SpecError(f"{spec.name}: comm edge {(a, b)} references unknown stage")


# --------------------------------------------------------------------------- specs
S0 = SystemSpec(
    name="S0",
    stages=(Stage("single_agent", REACT, "single_agent", "prediction", (), WRITE),),
    sink="single_agent",
    note="Official-style single ReAct agent over all tools (the control).",
)

M1 = SystemSpec(
    name="M1",
    stages=(
        Stage("planner", LLM, "planner", "plan", (), NONE),
        Stage("executor_agent", REACT, "executor_agent", "prediction", ("planner",), WRITE),
    ),
    sink="executor_agent",
    comm_edges=(("planner", "executor_agent"),),
    note="Plan then act.",
)

M2 = SystemSpec(
    name="M2",
    stages=(
        Stage("coordinator", LLM, "coordinator", "subgoals", (), NONE),
        Stage("worker", REACT, "worker", "evidence", ("coordinator",), READ, fanout="domains"),
        Stage("integrator", LLM, "integrator", "proposed_actions", ("worker",), NONE),
        Stage("executor", DET, "executor", "prediction", ("integrator",), WRITE),
    ),
    sink="executor",
    comm_edges=(("worker", "integrator"),),
    note="Communicating DAG: coordinator -> domain workers -> integrator -> executor.",
)

M3 = SystemSpec(
    name="M3",
    stages=(
        Stage("coordinator", LLM, "coordinator", "subgoals", (), NONE),
        Stage("worker", REACT, "worker", "evidence", ("coordinator",), READ, fanout="domains"),
        Stage("integrator", LLM, "integrator", "proposed_actions", ("worker",), NONE),
        Stage("verifier", LLM, "verifier", "verifier_decision", ("integrator", "worker"), NONE),
        Stage("executor", DET, "executor", "prediction", ("verifier",), WRITE),
    ),
    sink="executor",
    comm_edges=(("worker", "integrator"), ("worker", "verifier")),
    note="Verified DAG: adds a safety verifier before the executor.",
)

SYSTEMS: dict[str, SystemSpec] = {"S0": S0, "M1": M1, "M2": M2, "M3": M3}

for _spec in SYSTEMS.values():
    validate_spec(_spec)  # fail loud at import if any topology is ill-formed
