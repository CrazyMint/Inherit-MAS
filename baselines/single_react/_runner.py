"""Shared immutable run ledger for the independent ReAct and EvoAgent tasks."""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from hotpot_fullwiki import config as hotpot_config
from hotpot_fullwiki.common import atomic_json, digest, file_sha256
from public_runner import common

ROOT = Path(__file__).resolve().parents[2]
BACKBONES = ("gpt-4o-mini", "qwen3-32b")


def _manifest(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or ("tasks" in raw) == ("rows" in raw):
        raise ValueError("manifest must contain exactly one of tasks or rows")
    benchmark = "workbench" if "tasks" in raw else "hotpotqa"
    rows = raw["tasks" if benchmark == "workbench" else "rows"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid manifest rows")
    if type(raw.get("n")) is not int:
        raise ValueError("manifest n must be an integer")
    value = common.read_manifest(Path(path), benchmark)
    if benchmark == "workbench" and any(
            not isinstance(row.get("task"), str) or not row["task"] for row in rows):
        raise ValueError("manifest requires nonempty public task text")
    return benchmark, value, rows


def _need_meta(method, benchmark, backbone):
    return method == "single-react" and benchmark == "hotpotqa" and backbone == "gpt-4o-mini"


def _model_kwargs(method, benchmark, backbone):
    kwargs = {"need_meta": _need_meta(method, benchmark, backbone)}
    if (method, benchmark, backbone) == ("single-react", "workbench", "gpt-4o-mini"):
        kwargs["request_timeout_s"] = 300
    return kwargs


def _config(method, benchmark, manifest, usd_cap, workers, backbone, *, model_identity=None):
    if (isinstance(usd_cap, bool) or not isinstance(usd_cap, (int, float))
            or not math.isfinite(usd_cap) or usd_cap <= 0):
        raise ValueError("usd_cap must be positive and finite")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if backbone not in BACKBONES:
        raise ValueError("unsupported backbone")
    if model_identity is None:
        from public_runner.models import model_settings
        model_identity = model_settings(backbone, **_model_kwargs(method, benchmark, backbone))
    if not isinstance(model_identity, dict) or not model_identity:
        raise ValueError("missing model identity")
    settings = {"schema": "public_baseline_config/1", "method": method,
        "benchmark": benchmark, "manifest_digest": manifest["manifest_digest"],
        "worker_model": backbone, "temperature": 0,
        "enable_thinking": False if backbone == "qwen3-32b" else None,
        "judge_model": "gpt-5.4-mini" if _need_meta(method, benchmark, backbone) else None,
        "node_output_reuse": False, "usd_cap": float(usd_cap), "workers": workers,
        "model_settings": model_identity}
    if benchmark == "workbench":
        settings.update(max_iterations=20, max_tokens=8192, task_ceiling_s=1200,
                        tools="all_official", act_without_confirmation=method == "evoagent")
        if method == "single-react":
            settings.update(transient_request_attempts=3 if backbone == "gpt-4o-mini" else 1,
                            request_timeout_s=300 if backbone == "gpt-4o-mini" else 240)
    else:
        settings.update(max_tool_turns=hotpot_config.MAX_TOOL_TURNS,
            max_tokens=hotpot_config.WORKER_MAX_TOKENS,
            retrieval_calls_per_agent=hotpot_config.RETRIEVAL_CALLS_PER_CANDIDATE,
            retrieval_top_k=hotpot_config.RETRIEVAL_TOP_K,
            index=hotpot_config.BEIR_INDEX, index_md5=hotpot_config.BEIR_INDEX_MD5,
            bm25_k1=hotpot_config.BM25_K1, bm25_b=hotpot_config.BM25_B)
    if method == "evoagent":
        from baselines.evoagent.core import UPSTREAM_COMMIT, NLP_ITERATIONS, MAX_ROLE_PROPOSALS
        settings.update(upstream_commit=UPSTREAM_COMMIT, population_per_iteration=1,
            iterations=1 if benchmark == "workbench" else NLP_ITERATIONS,
            quality_selection=benchmark == "hotpotqa", max_role_proposals=MAX_ROLE_PROPOSALS,
            role_max_tokens=1600, role_model=backbone, integration_model=backbone,
            quality_model=backbone if benchmark == "hotpotqa" else None)
    elif benchmark == "hotpotqa":
        settings.update(strategy="bridge_first" if backbone == "gpt-4o-mini" else "independent",
                        judge_max_tokens=900 if backbone == "gpt-4o-mini" else None,
                        judge_max_attempts=2 if backbone == "gpt-4o-mini" else 0)
    settings["config_digest"] = digest(settings)
    return settings


def _workbench_path():
    path = str(ROOT / "workbench")
    if path not in sys.path:
        sys.path.insert(0, path)


def _prepare(method, benchmark, manifest, rows, out, backbone, ledger):
    if benchmark == "workbench":
        _workbench_path()
        import loader
        if "dataset_pin" in manifest and manifest["dataset_pin"] != loader.dataset_pin():
            raise RuntimeError("manifest dataset pin differs from official dataset")
        runtime = {row["id"]: SimpleNamespace(id=row["id"], task=row["task"],
            source=row.get("source", ""), base_template=row.get("base_template", "")) for row in rows}
        if method == "evoagent":
            from baselines.evoagent.workbench_adapter import run as execute
        else:
            from .workbench_adapter import run as execute
        retriever = None
    else:
        from inherit_mas.release_config import ensure_hf_home
        from hotpot_fullwiki.loader import load_examples, open_snapshot
        from hotpot_fullwiki.retrieval import BM25Retriever
        ensure_hf_home()
        snapshot = open_snapshot()
        if manifest.get("dataset_revision") != snapshot.digest:
            raise RuntimeError("manifest/dataset mismatch")
        runtime = {example.id: example.runtime() for example in load_examples(rows, snapshot)}
        retriever = BM25Retriever(cache_dir=out / "retrieval_cache")
        first = runtime[rows[0]["id"]]
        retriever.preflight(first.question, first)
        if method == "evoagent":
            from baselines.evoagent.hotpot_adapter import run as execute
        else:
            from .hotpot_adapter import run as execute
    from public_runner.models import make_models
    models = make_models(ledger, backbone, **_model_kwargs(method, benchmark, backbone))

    def run_item(row):
        kwargs = {"models": models}
        if method == "single-react":
            kwargs["backbone"] = backbone
        if benchmark == "hotpotqa":
            kwargs["retriever"] = retriever
        record = execute(runtime[row["id"]], **kwargs)
        record.update(schema=f"public_{method.replace('-', '_')}_{benchmark}/1",
                      method=method, system=f"{method}:{backbone}", backbone_model=backbone,
                      separate_meta_model=_need_meta(method, benchmark, backbone))
        return record
    return run_item


def _prediction(record, benchmark):
    prediction = record.get("prediction")
    if "candidates" in record:
        candidates = record["candidates"]
        if not isinstance(candidates, list):
            raise RuntimeError("invalid candidate list")
        indices = [row.get("candidate_index") for row in candidates if isinstance(row, dict)]
        if len(indices) != len(candidates) or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
            raise RuntimeError("invalid candidate indices")
        selected = [row for row in candidates if row["candidate_index"] == record.get("selected_candidate")]
        if len(selected) != 1 or selected[0].get("status") != "complete":
            raise RuntimeError("invalid candidate selection")
        prediction = selected[0].get("prediction")
    if benchmark == "workbench":
        if not isinstance(prediction, list) or any(not isinstance(action, str) for action in prediction):
            raise RuntimeError("invalid WorkBench prediction")
    elif not isinstance(prediction, dict):
        raise RuntimeError("invalid HotpotQA prediction")
    return prediction


def _unit_path(out, task_id):
    return out / "units" / f"{digest(task_id)}.json"


def _units(out, rows, config):
    ids = {row["id"] for row in rows}
    result = {}
    for path in sorted((out / "units").glob("*.json")):
        unit = json.loads(path.read_text())
        task_id = unit.get("task_id")
        if (task_id not in ids or task_id in result or path != _unit_path(out, task_id)
                or unit.get("config_digest") != config["config_digest"]
                or unit.get("manifest_digest") != config["manifest_digest"]):
            raise RuntimeError("unit/config/manifest identity mismatch")
        status = unit.get("status")
        if status not in {"complete", "task_failure", "infra_failure", "cap_stop"}:
            raise RuntimeError("invalid unit status")
        if status == "complete" and (unit.get("task_failure") or unit.get("infra_failure")):
            raise RuntimeError("invalid complete unit")
        if status == "task_failure" and (not unit.get("task_failure") or unit.get("infra_failure")):
            raise RuntimeError("invalid task failure")
        if status == "infra_failure" and not unit.get("infra_failure"):
            raise RuntimeError("invalid infrastructure failure")
        record = unit.get("record")
        if record is not None:
            if not isinstance(record, dict) or record.get("task_id") != task_id:
                raise RuntimeError("invalid record task identity")
            _prediction(record, config["benchmark"])
        elif status == "complete":
            raise RuntimeError("missing complete record")
        usage = unit.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("missing usage")
        for key in ("estimated_usd", "total_tokens", "calls"):
            value = usage.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("invalid usage")
        result[task_id] = unit
    return result


def _usage(out, units):
    path = out / "api_calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for row in calls:
        for key in ("estimated_usd", "total_tokens"):
            value = row.get(key, 0)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise RuntimeError("invalid API call ledger usage")
    rows = calls if calls else [unit["usage"] for unit in units.values()]
    return {"estimated_usd": sum(float(row.get("estimated_usd", 0)) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
            "calls": len(calls) if calls else sum(int(row.get("calls", 0)) for row in rows),
            "unknown_cost_units": sum(bool(row["usage"].get("unknown")) for row in units.values()),
            "unknown_usage_calls": sum(bool(row.get("unknown_usage")) for row in calls)}


def _seal(out, manifest, config, rows):
    return {"manifest_digest": manifest["manifest_digest"], "config_digest": config["config_digest"],
        "expected": len(rows), "unit_sha256": {
            _unit_path(out, row["id"]).name: file_sha256(_unit_path(out, row["id"])) for row in rows},
        "ledger_sha256": file_sha256(out / "api_calls.jsonl") if (out / "api_calls.jsonl").exists() else None}


def run(method, manifest_path, out, *, usd_cap, workers, backbone):
    benchmark, manifest, rows = _manifest(manifest_path)
    config = _config(method, benchmark, manifest, usd_cap, workers, backbone)
    out = Path(out)
    common.bind_run(out, benchmark, manifest, method=method, backbone=backbone)
    config_path = out / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("run directory has a different configuration")
    units = _units(out, rows, config)
    if not config_path.exists() and units:
        raise RuntimeError("existing units have no run configuration")
    if (out / "SEALED.json").exists():
        if json.loads((out / "SEALED.json").read_text()) != _seal(out, manifest, config, rows):
            raise RuntimeError("sealed unit/ledger identity mismatch")
    atomic_json(config_path, config)
    pending = [row for row in rows if row["id"] not in units]
    started = time.time()
    cap_stop = any(unit["status"] == "cap_stop" for unit in units.values())
    cap_stop |= bool(pending and _usage(out, units)["estimated_usd"] >= usd_cap)
    if pending and not cap_stop:
        from hotpot_fullwiki.models import BudgetExhausted, CallLedger
        ledger = CallLedger(out / "api_calls.jsonl", cap_usd=usd_cap)
        execute = _prepare(method, benchmark, manifest, rows, out, backbone, ledger)

        def execute_unit(row):
            record = None
            failure = infra = ""
            status = "complete"
            try:
                record = execute(row)
                _prediction(record, benchmark)
                failure = str(record.get("task_failure", ""))
                infra = str(record.get("infra_failure", ""))
                status = "infra_failure" if infra else ("task_failure" if failure else "complete")
            except BudgetExhausted as exc:
                status, failure = "cap_stop", str(exc)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                is_task = isinstance(exc, TimeoutError) or any(
                    marker in message.lower() for marker in ("content_filter", "content filter",
                    "context_length", "maximum context length", "context window", "apitimeouterror",
                    "request_timeout", "invalid_model_tool_call", "structuredcallerror"))
                status = "task_failure" if is_task else "infra_failure"
                failure, infra = (message, "") if is_task else ("", message)
                record = None
            unit = {"task_id": row["id"], "status": status, "record": record,
                "task_failure": failure, "infra_failure": infra,
                "usage": record.get("usage", {}) if record is not None else {"unknown": status != "cap_stop"},
                "config_digest": config["config_digest"], "manifest_digest": manifest["manifest_digest"]}
            atomic_json(_unit_path(out, row["id"]), unit)
            return unit

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending and not cap_stop:
                if _usage(out, units)["estimated_usd"] >= usd_cap:
                    cap_stop = True
                    break
                batch, pending = pending[:workers], pending[workers:]
                for unit in pool.map(execute_unit, batch):
                    units[unit["task_id"]] = unit
                    cap_stop |= unit["status"] == "cap_stop"
    units = _units(out, rows, config)
    infra = [task_id for task_id, unit in units.items() if unit["status"] == "infra_failure"]
    complete = len(units) == len(rows) and not infra and not cap_stop
    usage = _usage(out, units)
    report = {"status": "sealed" if complete else ("cap_stop" if cap_stop else "incomplete"),
        "method": method, "benchmark": benchmark, "expected": len(rows), "recorded": len(units),
        "infra_failures": infra, "usage": usage, "spent_usd": usage["estimated_usd"],
        "config_digest": config["config_digest"], "elapsed_s": time.time() - started}
    atomic_json(out / "RUN_STATUS.json", report)
    if complete:
        atomic_json(out / "SEALED.json", _seal(out, manifest, config, rows))
    else:
        (out / "SEALED.json").unlink(missing_ok=True)
    return report


def score(method, manifest_path, out):
    benchmark, manifest, rows = _manifest(manifest_path)
    out = Path(out)
    config = json.loads((out / "RUN_CONFIG.json").read_text())
    common.verify_binding(out, benchmark, method=method, backbone=config.get("worker_model"))
    if not isinstance(config.get("model_settings"), dict) or not config["model_settings"]:
        raise RuntimeError("configuration missing model identity")
    if config != _config(method, benchmark, manifest, config.get("usd_cap"),
                         config.get("workers"), config.get("worker_model"),
                         model_identity=config["model_settings"]):
        raise RuntimeError("configuration mismatch")
    units = _units(out, rows, config)
    if (len(units) != len(rows) or any(unit["status"] not in {"complete", "task_failure"} for unit in units.values())
            or json.loads((out / "RUN_STATUS.json").read_text()).get("status") != "sealed"
            or json.loads((out / "SEALED.json").read_text()) != _seal(out, manifest, config, rows)):
        raise RuntimeError("only a complete matching sealed run can be scored")
    per_task = []
    if benchmark == "workbench":
        _workbench_path()
        import loader
        import wb_env as W
        tasks, gold = loader.load_tasks()
        public = {task.id: task for task in tasks}
        if "dataset_pin" in manifest and manifest["dataset_pin"] != loader.dataset_pin():
            raise RuntimeError("manifest dataset pin differs from official dataset")
        for row in rows:
            task_id = row["id"]
            if task_id not in public or task_id not in gold or public[task_id].task != row["task"]:
                raise RuntimeError("manifest task differs from official dataset")
            unit = units[task_id]
            prediction = _prediction(unit["record"], benchmark) if unit["record"] else []
            per_task.append({"task_id": task_id, **W.score_prediction(prediction, gold[task_id], unit["task_failure"])})
        correct = sum(row["correct"] for row in per_task)
        metrics = {"correct": correct, "completion": correct / len(rows),
                   "side_effects": sum(row["unwanted_side_effect"] for row in per_task)}
    else:
        from inherit_mas.release_config import ensure_hf_home
        from hotpot_fullwiki.loader import load_examples, open_snapshot
        from hotpot_fullwiki.analyze import score_prediction
        ensure_hf_home()
        snapshot = open_snapshot()
        if manifest.get("dataset_revision") != snapshot.digest:
            raise RuntimeError("manifest/dataset mismatch")
        examples = {example.id: example for example in load_examples(rows, snapshot)}
        for row in rows:
            unit = units[row["id"]]
            prediction = _prediction(unit["record"], benchmark) if unit["status"] == "complete" else None
            per_task.append({"task_id": row["id"], **score_prediction(examples[row["id"]], prediction)})
        metrics = {"metrics": {name: sum(row[name] for row in per_task) / len(rows)
                    for name in ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")}}
    report = {"method": method, "benchmark": benchmark, "n": len(rows), **metrics,
        "per_task": per_task, "usage": _usage(out, units), "config_digest": config["config_digest"],
        "manifest_digest": manifest["manifest_digest"]}
    atomic_json(out / "SCORE.json", report)
    return report
