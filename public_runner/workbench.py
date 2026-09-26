"""Run Inherit-MAS on public WorkBench tasks and score sealed runs offline."""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, digest


ROOT = Path(__file__).resolve().parents[1]
CONDITION = "inherit_mas_select_edit_v1"


def _workbench_path() -> None:
    path = str(ROOT / "workbench")
    if path not in sys.path:
        sys.path.insert(0, path)


def _runtime():
    _workbench_path()
    from task_evolution.controller import PublicTask
    from task_evolution.select_edit import run_condition
    from task_evolution.models import APIModels
    return PublicTask, run_condition, APIModels


def _manifest(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("manifest_digest") != digest(
            {key: row for key, row in value.items() if key != "manifest_digest"}):
        raise ValueError("invalid manifest digest")
    tasks = value.get("tasks")
    n = value.get("n")
    if type(n) is not int or n <= 0 or not isinstance(tasks, list) or len(tasks) != n:
        raise ValueError("manifest n must match a nonempty task list")
    ids = []
    for item in tasks:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"] or not isinstance(item.get("task"), str)
                or not item["task"]):
            raise ValueError("manifest tasks require nonempty id and task strings")
        ids.append(item["id"])
    if len(set(ids)) != n:
        raise ValueError("duplicate manifest task id")
    return value


def _validate_limits(usd_cap: float, workers: int) -> None:
    if (isinstance(usd_cap, bool) or not isinstance(usd_cap, (int, float))
            or not math.isfinite(usd_cap) or usd_cap <= 0):
        raise ValueError("usd_cap must be positive and finite")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")


def _config(manifest: dict, usd_cap: float, workers: int, backbone: str = "gpt-4o-mini",
            model_identity: dict | None = None) -> dict:
    value = {"schema": "workbench_inherit_mas_config/1",
             "manifest_digest": manifest["manifest_digest"],
             "condition": CONDITION, "worker_model": backbone,
             "meta_judge_model": "gpt-5.4-mini", "max_candidates": 5,
             "controller": "inherit_mas.select_edit.run_select_edit",
             "parent": "latest-complete-candidate-v1",
             "judge": "graph-aware-node-attribution-v1",
             "select": "meta-model-discard-set-v1",
             "edit": "one-validated-edit-with-node-insertion-v1",
             "usd_cap": float(usd_cap), "workers": workers}
    if model_identity is not None:
        value["model_settings"] = model_identity
    value["config_digest"] = digest(value)
    return value


def _usage(record: dict) -> dict:
    worker_live = worker_reused = meta_tokens = 0
    usd = 0.0
    for candidate in record.get("candidates", []):
        execution = candidate.get("execution", {})
        worker_live += int(execution.get("live_tokens", 0))
        worker_reused += int(execution.get("reused_tokens", 0))
        for node in execution.get("nodes", {}).values():
            usd += float(node.get("usage", {}).get("estimated_usd", 0.0))
        for key in ("proposal_attempts", "judge_attempts"):
            for attempt in candidate.get(key, []):
                for row in attempt.get("usage", []):
                    meta_tokens += int(row.get("total_tokens", 0))
                    usd += float(row.get("estimated_usd", 0.0))
    return {"worker_live_tokens": worker_live, "worker_reused_tokens": worker_reused,
            "meta_judge_tokens": meta_tokens, "estimated_usd": usd}


def _unit_path(out: Path, task_id: str) -> Path:
    return out / "units" / f"{digest(task_id)}.json"


def _load_units(out: Path, manifest: dict, config: dict) -> dict[str, dict]:
    expected = {item["id"] for item in manifest["tasks"]}
    units = {}
    for path in sorted((out / "units").glob("*.json")):
        row = json.loads(path.read_text())
        task_id = row.get("task_id") if isinstance(row, dict) else None
        if not isinstance(task_id, str) or task_id not in expected or task_id in units:
            raise RuntimeError("unexpected or duplicate unit task id")
        if (path != _unit_path(out, task_id)
                or row.get("config_digest") != config["config_digest"]
                or row.get("manifest_digest") != manifest["manifest_digest"]):
            raise RuntimeError("unit/config/manifest identity mismatch")
        status = row.get("status")
        record = row.get("record")
        if status == "complete":
            if (row.get("infra_failure") or row.get("task_failure")
                    or not isinstance(record, dict) or record.get("task_id") != task_id
                    or "selected_candidate" not in record
                    or not isinstance(record.get("candidates"), list)):
                raise RuntimeError("invalid complete unit")
            candidates = record["candidates"]
            indices = [item.get("candidate_index") for item in candidates
                       if isinstance(item, dict)]
            if (len(indices) != len(candidates) or any(type(i) is not int for i in indices)
                    or len(set(indices)) != len(indices)
                    or (record.get("selected_candidate") is not None
                        and record["selected_candidate"] not in indices)):
                raise RuntimeError("invalid candidate selection")
            for candidate in candidates:
                if candidate.get("status") not in ("complete", "invalid_proposal"):
                    raise RuntimeError("invalid candidate status")
                if candidate.get("status") == "complete" and (
                        not isinstance(candidate.get("prediction"), list)
                        or any(not isinstance(action, str) for action in candidate["prediction"])):
                    raise RuntimeError("invalid candidate prediction")
            selected = record["selected_candidate"]
            completed = [item["candidate_index"] for item in candidates if item["status"] == "complete"]
            if (completed and (type(selected) is not int or selected not in completed)) or (
                    not completed and selected is not None):
                raise RuntimeError("invalid candidate selection")
        elif status in ("task_failure", "infra_failure"):
            other = "infra_failure" if status == "task_failure" else "task_failure"
            if not row.get(status) or row.get(other) or record is not None:
                raise RuntimeError("invalid failed unit")
        else:
            raise RuntimeError("invalid unit status")
        history = row.get("attempt_history", [])
        if not isinstance(history, list):
            raise RuntimeError("invalid attempt history")
        for attempt in [*history, row]:
            if not isinstance(attempt, dict):
                raise RuntimeError("invalid attempt history")
            if attempt is not row and (
                    attempt.get("status") != "infra_failure"
                    or not attempt.get("infra_failure") or attempt.get("task_failure")
                    or attempt.get("record") is not None or "attempt_history" in attempt
                    or attempt.get("task_id") != task_id
                    or attempt.get("config_digest") != config["config_digest"]
                    or attempt.get("manifest_digest") != manifest["manifest_digest"]):
                raise RuntimeError("invalid attempt history")
            cost = attempt.get("cost")
            if not isinstance(cost, dict) or "estimated_usd" not in cost:
                raise RuntimeError("missing unit cost")
            for key in ("estimated_usd", "worker_live_tokens", "worker_reused_tokens", "meta_judge_tokens"):
                number = cost.get(key, 0)
                if (isinstance(number, bool) or not isinstance(number, (int, float))
                        or not math.isfinite(number) or number < 0):
                    raise RuntimeError("invalid unit cost")
        units[task_id] = row
    return units


def _totals(units) -> dict:
    rows = list(units)
    attempts = [attempt for row in rows for attempt in [*row.get("attempt_history", []), row]]
    return {**{key: sum(attempt["cost"].get(key, 0) for attempt in attempts) for key in (
        "estimated_usd", "worker_live_tokens", "worker_reused_tokens", "meta_judge_tokens")},
        "unknown_cost_units": sum(any(attempt["cost"].get("unknown", False)
            for attempt in [*row.get("attempt_history", []), row]) for row in rows)}


def run(manifest_path: Path, out: Path, *, usd_cap: float, workers: int,
        backbone: str = "gpt-4o-mini") -> dict:
    _validate_limits(usd_cap, workers)
    manifest = _manifest(manifest_path)
    out = Path(out)
    if backbone not in {"gpt-4o-mini", "qwen3-32b"}:
        raise ValueError("unsupported backbone")
    from public_runner.models import model_settings
    model_identity = model_settings(backbone)
    config = _config(manifest, usd_cap, workers, backbone, model_identity)
    config_path = out / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("run directory has a different configuration")
    if not config_path.exists() and (out / "units").exists() and any((out / "units").iterdir()):
        raise RuntimeError("existing units have no run configuration")
    existing = _load_units(out, manifest, config)
    atomic_json(config_path, config)
    pending = [item for item in manifest["tasks"] if item["id"] not in existing
               or existing[item["id"]]["status"] == "infra_failure"]
    started = time.time()
    records = dict(existing)
    cap_stop = bool(pending and _totals(records.values())["estimated_usd"] >= usd_cap)
    if pending and not cap_stop:
        PublicTask, run_condition, APIModels = _runtime()
        if backbone == "gpt-4o-mini":
            models = APIModels()
        else:
            from hotpot_fullwiki.models import CallLedger
            from public_runner.models import make_models
            # WorkBench's cap is checked between bounded task waves.
            models = make_models(CallLedger(out / "api_calls.jsonl", float("inf")), backbone)

        def execute(item: dict) -> dict:
            unit_started = time.time()
            try:
                record = run_condition(
                    PublicTask(item["id"], item["task"]), models=models,
                    run_dir=out / "task_records" / CONDITION / digest(item["id"]),
                )
                value = {"status": "complete", "task_id": item["id"], "record": record,
                         "task_failure": "", "infra_failure": "", "cost": _usage(record)}
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                task_failure = message if any(marker in message.lower() for marker in (
                    "content_filter", "content filter", "context_length")) else ""
                value = {"status": "task_failure" if task_failure else "infra_failure",
                         "task_id": item["id"], "record": None,
                         "task_failure": task_failure,
                         "infra_failure": "" if task_failure else message,
                         "cost": {"estimated_usd": 0.0, "unknown": True}}
            value.update({"config_digest": config["config_digest"],
                          "manifest_digest": manifest["manifest_digest"],
                          "wall_s": time.time() - unit_started})
            previous = existing.get(item["id"])
            if previous is not None:
                # Commit prior failures together with progress so retries retain their costs.
                value["attempt_history"] = [*previous.get("attempt_history", []),
                    {key: entry for key, entry in previous.items() if key != "attempt_history"}]
            atomic_json(_unit_path(out, item["id"]), value)
            return value

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Check the soft cost cap between bounded waves of at most workers tasks.
            while pending:
                if _totals(records.values())["estimated_usd"] >= usd_cap:
                    cap_stop = True
                    break
                batch, pending = pending[:workers], pending[workers:]
                futures = [pool.submit(execute, item) for item in batch]
                for future in as_completed(futures):
                    value = future.result()
                    records[value["task_id"]] = value
                atomic_json(out / "LIVE_STATUS.json", {
                    "status": "running", "expected": manifest["n"],
                    "recorded": len(records), "spent_usd": _totals(records.values())["estimated_usd"],
                    "elapsed_s": time.time() - started,
                })
    records = list(_load_units(out, manifest, config).values())
    infra = [row["task_id"] for row in records if row["status"] == "infra_failure"]
    complete = len(records) == manifest["n"] and not infra and not cap_stop
    report = {"status": "sealed" if complete else ("cap_stop" if cap_stop else "incomplete"),
              "expected": manifest["n"], "recorded": len(records), "infra_failures": infra,
              "spent_usd": _totals(records)["estimated_usd"], "usage": _totals(records),
              "elapsed_s": time.time() - started, "config_digest": config["config_digest"]}
    atomic_json(out / "RUN_STATUS.json", report)
    if complete:
        atomic_json(out / "SEALED.json", {"manifest_digest": manifest["manifest_digest"],
                    "config_digest": config["config_digest"], "expected": manifest["n"],
                    "sealed_at": time.time()})
    else:
        (out / "SEALED.json").unlink(missing_ok=True)
    return report


def score(manifest_path: Path, out: Path) -> dict:
    manifest = _manifest(manifest_path)
    out = Path(out)
    config = json.loads((out / "RUN_CONFIG.json").read_text())
    seal = json.loads((out / "SEALED.json").read_text())
    _validate_limits(config.get("usd_cap"), config.get("workers"))
    if (config != _config(manifest, config["usd_cap"], config["workers"],
                          config.get("worker_model", "gpt-4o-mini"), config.get("model_settings"))
            or seal.get("config_digest") != config["config_digest"]
            or seal.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("expected") != manifest["n"]):
        raise RuntimeError("seal/config/manifest mismatch")
    units = _load_units(out, manifest, config)
    if len(units) != manifest["n"] or any(row["status"] == "infra_failure" for row in units.values()):
        raise RuntimeError("sealed run has missing or infrastructure-failure units")
    # Scorer-only gold is loaded after all run identity and completeness checks.
    _workbench_path()
    import loader
    import wb_env as W
    public_tasks, gold = loader.load_tasks()
    public = {task.id: task for task in public_tasks}
    if "dataset_pin" in manifest and manifest["dataset_pin"] != loader.dataset_pin():
        raise RuntimeError("manifest dataset pin differs from the official dataset")
    per_task = []
    for item in manifest["tasks"]:
        task_id = item["id"]
        if task_id not in gold or task_id not in public or public[task_id].task != item["task"]:
            raise RuntimeError("manifest task differs from the official dataset")
        unit = units[task_id]
        record = unit.get("record") or {}
        index = record.get("selected_candidate")
        candidate = next((row for row in record.get("candidates", [])
                          if row["candidate_index"] == index), None)
        prediction = []
        error = unit.get("task_failure") or "missing_selected"
        if candidate is not None and candidate.get("status") == "complete":
            prediction, error = candidate["prediction"], ""
        per_task.append({"task_id": task_id, "selected_candidate": index,
                         **W.score_prediction(prediction, gold[task_id], error)})
    correct = sum(row["correct"] for row in per_task)
    result = {"manifest_digest": manifest["manifest_digest"],
              "config_digest": config["config_digest"], "n": manifest["n"],
              "correct": correct, "completion": correct / manifest["n"],
              "side_effects": sum(row["unwanted_side_effect"] for row in per_task),
              "usage": _totals(units.values()), "per_task": per_task}
    atomic_json(out / "SCORE.json", result)
    return result
