"""HotpotQA execution and official scoring commands."""
from __future__ import annotations

import json
from pathlib import Path

from hotpot_fullwiki.common import atomic_json
from hotpot_fullwiki.manifests import load


def run(manifest_path: Path, out: Path, *, usd_cap: float, workers: int,
        backbone: str = "gpt-4o-mini") -> dict:
    from inherit_mas.release_config import ensure_hf_home
    ensure_hf_home()
    from hotpot_fullwiki.loader import load_examples, open_snapshot
    from hotpot_fullwiki.retrieval import BM25Retriever
    from hotpot_fullwiki.run_select_edit import run as execute

    manifest = load(manifest_path)
    snapshot = open_snapshot()
    if manifest["dataset_revision"] != snapshot.digest:
        raise RuntimeError("manifest/dataset mismatch")
    example = load_examples(manifest["rows"][:1], snapshot)[0].runtime()
    # Check retrieval before making paid model calls.
    BM25Retriever(cache_dir=out / "retrieval_cache").preflight(example.question, example)
    kwargs = {"usd_cap": usd_cap, "workers": workers}
    if backbone != "gpt-4o-mini":
        kwargs["backbone"] = backbone
    return execute(manifest_path, out, **kwargs)


def score(manifest_path: Path, out: Path) -> dict:
    manifest = load(manifest_path)
    config = json.loads((out / "RUN_CONFIG.json").read_text())
    seal = json.loads((out / "SEALED.json").read_text())
    status = json.loads((out / "RUN_STATUS.json").read_text())
    if (status["status"] != "sealed"
            or seal.get("config_digest") != config.get("config_digest")
            or config.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("expected_units") != len(manifest["rows"])):
        raise RuntimeError("only a complete, matching run can be scored")

    from inherit_mas.release_config import ensure_hf_home
    ensure_hf_home()
    from hotpot_fullwiki.loader import load_examples
    from hotpot_fullwiki.analyze import score_prediction
    from hotpot_fullwiki.select_edit import CONDITION

    examples = {e.id: e for e in load_examples(manifest["rows"])}
    rows = []
    for item in manifest["rows"]:
        unit = json.loads((out / "units" / CONDITION / f"{item['id']}.json").read_text())
        if (unit.get("config_digest") != config["config_digest"]
                or unit.get("task_id") != item["id"]
                or unit.get("status") not in {"complete", "task_failure"}):
            raise RuntimeError(f"invalid task record: {item['id']}")
        record = unit.get("record") or {}
        selected = next((c for c in record.get("candidates", [])
                         if c.get("candidate_index") == record.get("selected_candidate")
                         and c.get("status") == "complete"), None)
        prediction = selected.get("prediction") if selected else None
        if unit["status"] == "task_failure":
            prediction = None
        rows.append({"task_id": item["id"], **score_prediction(examples[item["id"]], prediction)})
    names = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")
    report = {"benchmark": "hotpotqa", "n": len(rows),
              "metrics": {name: sum(row[name] for row in rows) / len(rows) for name in names},
              "per_task": rows}
    ledger = out / "api_calls.jsonl"
    calls = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    report["usage"] = {"calls": len(calls),
                       "tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
                       "estimated_usd": sum(float(row.get("estimated_usd", 0)) for row in calls)}
    atomic_json(out / "SCORE.json", report)
    return report

