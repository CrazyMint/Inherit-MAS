"""One FullWiki ReAct agent; GPT retains the evaluated diagnostic judge."""
from __future__ import annotations

from hotpot_fullwiki.executor import execute_single


def run(example, *, models, retriever, backbone="gpt-4o-mini"):
    strategy = "bridge_first" if backbone == "gpt-4o-mini" else "independent"
    result = execute_single(example, models=models, retriever=retriever, strategy=strategy)
    record = {"task_id": example.id, "prediction": result.prediction,
              "execution": result.trace, "snapshots": result.snapshots,
              "usage": dict(result.snapshots["single"]["usage"]),
              "task_failure": "", "strategy": strategy, "node_output_reuse": False}
    if backbone == "gpt-4o-mini":
        from .judge import judge
        judgment, attempts = judge(models, example.question, None, result.prediction, result.trace)
        record.update(judgment=judgment, judge_attempts=attempts)
        for attempt in attempts:
            for usage in attempt["usage"]:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "estimated_usd", "wall_s"):
                    record["usage"][key] = record["usage"].get(key, 0) + usage.get(key, 0)
                record["usage"]["calls"] += 1
                record["usage"].setdefault("records", []).append(usage)
    return record
