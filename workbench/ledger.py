"""Token, tool-call, and wall-time accounting for WorkBench execution.

Every MAS run threads one ``Ledger``. It records, per role and in aggregate:
  * LLM calls and prompt/completion tokens,
  * read-tool and write-tool call counts,
  * wall time per phase.

Callers supply token counts from model responses. Token and tool-call counts are
additive; wall time is measured per phase.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class RoleTokens:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Ledger:
    llm: dict[str, RoleTokens] = field(default_factory=lambda: defaultdict(RoleTokens))
    read_tool_calls: int = 0
    write_tool_calls: int = 0
    wall: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # ---- LLM accounting -------------------------------------------------------
    def add_llm(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        rt = self.llm[role]
        rt.calls += 1
        rt.prompt_tokens += int(prompt_tokens)
        rt.completion_tokens += int(completion_tokens)

    # ---- tool accounting ------------------------------------------------------
    def add_tool_call(self, *, write: bool) -> None:
        if write:
            self.write_tool_calls += 1
        else:
            self.read_tool_calls += 1

    # ---- wall time ------------------------------------------------------------
    @contextmanager
    def timed(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall[phase] += time.perf_counter() - t0

    # ---- aggregates -----------------------------------------------------------
    @property
    def total_llm_calls(self) -> int:
        return sum(rt.calls for rt in self.llm.values())

    @property
    def total_prompt_tokens(self) -> int:
        return sum(rt.prompt_tokens for rt in self.llm.values())

    @property
    def total_completion_tokens(self) -> int:
        return sum(rt.completion_tokens for rt in self.llm.values())

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_tool_calls(self) -> int:
        return self.read_tool_calls + self.write_tool_calls

    @property
    def total_wall(self) -> float:
        return sum(self.wall.values())

    def to_dict(self) -> dict:
        return {
            "llm_by_role": {r: asdict(rt) | {"total_tokens": rt.total_tokens} for r, rt in self.llm.items()},
            "total_llm_calls": self.total_llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "read_tool_calls": self.read_tool_calls,
            "write_tool_calls": self.write_tool_calls,
            "total_tool_calls": self.total_tool_calls,
            "wall_by_phase": dict(self.wall),
            "total_wall_s": self.total_wall,
        }

    def merge(self, other: "Ledger") -> None:
        """Fold another ledger into this one (for aggregation across tasks)."""
        for role, rt in other.llm.items():
            self.llm[role].calls += rt.calls
            self.llm[role].prompt_tokens += rt.prompt_tokens
            self.llm[role].completion_tokens += rt.completion_tokens
        self.read_tool_calls += other.read_tool_calls
        self.write_tool_calls += other.write_tool_calls
        for phase, w in other.wall.items():
            self.wall[phase] += w
