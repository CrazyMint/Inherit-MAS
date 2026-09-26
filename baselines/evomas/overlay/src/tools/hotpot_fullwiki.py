"""Native smolagents retrieval tool for the pinned HotpotQA FullWiki task."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


from baselines.evomas import hotpot as native_baseline


def _manifest_path() -> Path:
    raw = os.environ.get("EVOMAS_HOTPOT_MANIFEST", "").strip()
    if not raw:
        raise RuntimeError("EVOMAS_HOTPOT_MANIFEST is required for hotpot_fullwiki")
    return Path(raw).resolve()


@lru_cache(maxsize=8)
def _examples(path: str):
    return native_baseline.runtime_examples(path)


def make_hotpot_search(task: str, *, searcher=None):
    """Create one task-scoped tool and retrieval budget for a native agent."""
    from smolagents import tool

    task_id = native_baseline.task_id_from_prompt(task)
    examples = _examples(str(_manifest_path()))
    if task_id not in examples:
        raise ValueError(f"task {task_id!r} is absent from the frozen Hotpot manifest")
    cache_dir = Path(os.environ.get(
        "EVOMAS_HOTPOT_CACHE",
        str(Path.cwd() / "cache" / "hotpot_fullwiki"),
    ))
    call_budget = 4
    session = native_baseline.new_retrieval_session(
        examples[task_id], cache_dir=cache_dir, searcher=searcher, call_budget=call_budget,
    )

    @tool
    def hotpot_search(query: str) -> str:
        """Search the pinned FullWiki BM25 index for evidence.

        Args:
            query: A focused entity, relation, or bridge-fact search query.

        Returns:
            JSON results with exact article titles and sentence ids for citation.
        """
        return json.dumps(session.search(query), ensure_ascii=True, sort_keys=True)

    return hotpot_search


def get_hotpot_tools(task: str) -> list[Any]:
    return [make_hotpot_search(task)]
