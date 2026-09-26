"""Public, resumable execution and offline scoring for graph baselines."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hotpot_fullwiki import config
from hotpot_fullwiki.common import atomic_json, digest, file_sha256
from public_runner.common import read_manifest
from public_runner.workbench import _validate_limits


def run(manifest_path, out, *, usd_cap, workers, backbone, method, controller):
    _validate_limits(usd_cap, workers)
    manifest = read_manifest(Path(manifest_path), "hotpotqa")
    out = Path(out)
    from public_runner.models import make_models, model_settings
    settings = model_settings(backbone)
    sources = [Path(controller.__file__), *Path(__file__).parent.glob("*.py")]
    shared = Path(__file__).parents[1]
    sources += [shared / name for name in ("controller.py", "executor.py", "config.py",
                                           "schemas.py", "retrieval.py", "models.py")]
    value = {"schema": "hotpot_graph_baseline_config/1", "method": method,
             "manifest_digest": manifest["manifest_digest"], "backbone": backbone,
             "models": settings, "usd_cap": float(usd_cap), "workers": workers,
             "upstream_commit": controller.UPSTREAM_COMMIT,
             "controller_settings": {name: getattr(controller, name) for name in dir(controller)
                                     if name.isupper() and isinstance(getattr(controller, name),
                                                                     (str, int, float, bool))},
             "retrieval_calls_per_candidate": config.RETRIEVAL_CALLS_PER_CANDIDATE,
             "execution_reuse": False,
             "source_sha256": {str(p.relative_to(shared.parents[0])): file_sha256(p)
                               for p in sorted(set(sources))}}
    value["config_digest"] = digest(value)
    target = out / "RUN_CONFIG.json"
    if target.exists() and json.loads(target.read_text()) != value:
        raise RuntimeError("run directory has a different configuration")
    if not target.exists() and any((out / "units").rglob("*.json")):
        raise RuntimeError("existing units have no run configuration")
    atomic_json(target, value)
    existing = _units(out, manifest, value)
    pending = [row for row in manifest["rows"] if row["id"] not in existing]
    from hotpot_fullwiki.models import BudgetExhausted, CallLedger
    ledger = CallLedger(out / "api_calls.jsonl", usd_cap)
    cap_hit = ""
    started = time.time()
    if pending:
        from inherit_mas.release_config import ensure_hf_home
        from hotpot_fullwiki.loader import load_examples, open_snapshot
        from hotpot_fullwiki.retrieval import BM25Retriever
        ensure_hf_home()
        snapshot = open_snapshot()
        if manifest["dataset_revision"] != snapshot.digest:
            raise RuntimeError("manifest/dataset mismatch")
        examples = {e.id: e.runtime() for e in load_examples(manifest["rows"], snapshot)}
        retriever = BM25Retriever(cache_dir=out / "retrieval_cache")
        first = examples[pending[0]["id"]]
        retriever.preflight(first.question, first)
        models = make_models(ledger, backbone)

        def execute(row):
            task_id = row["id"]
            try:
                record = controller.run(examples[task_id], models=models, retriever=retriever,
                                        run_dir=out / "task_records" / method / task_id)
                unit = {"status": "complete", "record": record,
                        "task_failure": "", "infra_failure": ""}
            except BudgetExhausted:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                task_failure = any(x in error.lower() for x in
                                   ("content_filter", "content filter", "context_length"))
                unit = {"status": "task_failure" if task_failure else "infra_failure",
                        "record": None, "task_failure": error if task_failure else "",
                        "infra_failure": "" if task_failure else error}
            unit.update(task_id=task_id, condition=method,
                        config_digest=value["config_digest"],
                        manifest_digest=manifest["manifest_digest"])
            atomic_json(out / "units" / method / f"{task_id}.json", unit)
            return unit

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending and not cap_hit:
                batch, pending = pending[:workers], pending[workers:]
                futures = [pool.submit(execute, row) for row in batch]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BudgetExhausted as exc:
                        cap_hit = str(exc)
    units = _units(out, manifest, value)
    infra = [row["task_id"] for row in units.values() if row["status"] == "infra_failure"]
    sealed = len(units) == manifest["n"] and not infra
    report = {"status": "sealed" if sealed else ("cap_stop" if cap_hit else "incomplete"),
              "expected_units": manifest["n"], "recorded_units": len(units),
              "infra_failures": infra, "cap_hit": cap_hit, "spent_usd": ledger.spent_usd,
              "config_digest": value["config_digest"], "elapsed_s": time.time() - started}
    atomic_json(out / "RUN_STATUS.json", report)
    if sealed:
        atomic_json(out / "SEALED.json", {"manifest_digest": manifest["manifest_digest"],
                    "config_digest": value["config_digest"], "expected_units": manifest["n"],
                    "sealed_at": time.time()})
    else:
        (out / "SEALED.json").unlink(missing_ok=True)
    return report


def _units(out, manifest, config_value):
    expected = {row["id"] for row in manifest["rows"]}
    units = {}
    method = config_value["method"]
    for path in (out / "units" / method).glob("*.json"):
        row = json.loads(path.read_text())
        task_id = row.get("task_id")
        if (task_id not in expected or task_id in units
                or path.name != f"{task_id}.json" or row.get("condition") != method
                or row.get("config_digest") != config_value["config_digest"]
                or row.get("manifest_digest") != manifest["manifest_digest"]
                or row.get("status") not in {"complete", "task_failure", "infra_failure"}):
            raise RuntimeError("invalid unit identity/status")
        record = row.get("record")
        if row["status"] == "complete":
            if not isinstance(record, dict) or record.get("task_id") != task_id:
                raise RuntimeError("invalid complete record")
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise RuntimeError("missing candidate list")
            indices = [c.get("candidate_index") for c in candidates]
            if any(type(i) is not int for i in indices) or len(indices) != len(set(indices)):
                raise RuntimeError("invalid candidate indices")
            complete = [c["candidate_index"] for c in candidates if c.get("status") == "complete"]
            selected = record.get("selected_candidate")
            if (complete and (type(selected) is not int or selected not in complete)) or (not complete and selected is not None):
                raise RuntimeError("invalid selected candidate")
        elif record is not None or not row.get(row["status"]):
            raise RuntimeError("invalid failure record")
        units[task_id] = row
    return units


def score(manifest_path, out, *, method):
    manifest = read_manifest(Path(manifest_path), "hotpotqa")
    out = Path(out)
    cfg = json.loads((out / "RUN_CONFIG.json").read_text())
    seal = json.loads((out / "SEALED.json").read_text())
    status = json.loads((out / "RUN_STATUS.json").read_text())
    if (cfg.get("method") != method or cfg.get("manifest_digest") != manifest["manifest_digest"]
            or cfg.get("config_digest") != digest({k: v for k, v in cfg.items() if k != "config_digest"})
            or seal.get("config_digest") != cfg["config_digest"]
            or seal.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("expected_units") != manifest["n"] or status.get("status") != "sealed"):
        raise RuntimeError("seal/config/manifest mismatch")
    units = _units(out, manifest, cfg)
    if len(units) != manifest["n"] or any(u["status"] == "infra_failure" for u in units.values()):
        raise RuntimeError("sealed run has missing or failed infrastructure units")
    source_root = Path(__file__).resolve().parents[2]
    sources = cfg.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("missing source identity")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise RuntimeError("invalid source identity path")
        if file_sha256(source_root / path) != expected:
            raise RuntimeError("source code changed since execution")
    from inherit_mas.release_config import ensure_hf_home
    from hotpot_fullwiki.loader import load_examples
    from hotpot_fullwiki.analyze import score_prediction
    ensure_hf_home()
    examples = {e.id: e for e in load_examples(manifest["rows"])}
    rows = []
    for task_id, unit in units.items():
        record = unit.get("record") or {}
        selected = next((c for c in record.get("candidates", [])
                         if c.get("candidate_index") == record.get("selected_candidate")), {})
        prediction = selected.get("prediction") if unit["status"] == "complete" else None
        rows.append({"task_id": task_id, **score_prediction(examples[task_id], prediction)})
    names = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")
    report = {"benchmark": "hotpotqa", "method": method, "backbone": cfg["backbone"],
              "n": len(rows), "metrics": {n: sum(row[n] for row in rows) / len(rows) for n in names},
              "per_task": rows, "usage": ledger_totals(out)}
    atomic_json(out / "SCORE.json", report)
    return report


def ledger_totals(out):
    path = Path(out) / "api_calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for row in calls:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            raise RuntimeError("malformed provider ledger")
        tokens, cost = row.get("total_tokens"), row.get("estimated_usd")
        if (type(tokens) is not int or tokens < 0 or isinstance(cost, bool)
                or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0):
            raise RuntimeError("malformed provider usage")
    totals = {"calls": len(calls), "total_tokens": sum(c["total_tokens"] for c in calls),
              "estimated_usd": sum(c["estimated_usd"] for c in calls),
              "unknown_usage_calls": sum(bool(c.get("unknown_usage")) for c in calls), "by_model": {}}
    for model in sorted({c["model"] for c in calls}):
        subset = [c for c in calls if c["model"] == model]
        totals["by_model"][model] = {"calls": len(subset),
            "total_tokens": sum(c["total_tokens"] for c in subset),
            "estimated_usd": sum(c["estimated_usd"] for c in subset)}
    return totals
