#!/usr/bin/env python3
"""Resumable runner for the benchmark-neutral-controller HotpotQA transfer."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from inherit_mas.adapters import adapter_contract_digest_payload

from . import config, manifests
from .common import atomic_json, digest
from .loader import load_examples, open_snapshot
from .models import BudgetExhausted, CallLedger, Models
from .retrieval import BM25Retriever
from .revision_v3 import (ADAPTER, CONDITION, JUDGE_MAX_TOKENS,
                          MAX_CANDIDATES, REFINER_MAX_TOKENS, run_condition)


def _is_task_failure(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("content_filter", "content filter", "context_length"))


def run(manifest_path: Path, out: Path, *, usd_cap: float, workers: int = 4,
        backbone: str = "gpt-4o-mini") -> dict:
    manifest = manifests.load(manifest_path)
    snapshot = open_snapshot()
    if manifest["dataset_revision"] != snapshot.digest:
        raise RuntimeError("manifest/snapshot mismatch")
    examples = load_examples(manifest["rows"], snapshot)
    runtime = {example.id: example.runtime() for example in examples}
    ledger = CallLedger(out / "api_calls.jsonl", usd_cap)
    if backbone == "gpt-4o-mini":
        models = Models.api(ledger=ledger)
    elif backbone == "qwen3-32b":
        from public_runner.models import make_models
        models = make_models(ledger, backbone)
    else:
        raise ValueError("unsupported backbone")
    retriever = BM25Retriever(cache_dir=out / "retrieval_cache")
    config_payload = {
        "schema": "inherit_masization_run_config/1",
        "revision": "general-controller-v3", "manifest_digest": manifest["manifest_digest"],
        "condition": CONDITION, "adapter": adapter_contract_digest_payload(ADAPTER),
        "worker_model": backbone, "meta_judge_model": config.META_MODEL,
        "retriever": retriever.fingerprint,
        "retrieval_calls_per_candidate": config.RETRIEVAL_CALLS_PER_CANDIDATE,
        "max_candidates": MAX_CANDIDATES, "judge_max_tokens": JUDGE_MAX_TOKENS,
        "refiner_max_tokens": REFINER_MAX_TOKENS,
        "selector": "generic-validity-then-gold-free-quality-v1",
        "escape": "generic-invalid-output-or-stagnation-v1", "usd_cap": float(usd_cap),
    }
    if backbone == "qwen3-32b":
        config_payload["model_settings"] = models.settings
    config_payload["config_digest"] = digest(config_payload)
    config_path = out / "RUN_CONFIG.json"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("config_digest") != config_payload["config_digest"]:
            raise RuntimeError("run directory has a different configuration")
    else:
        atomic_json(config_path, config_payload)

    unit_dir = out / "units" / CONDITION
    pending, records = [], []
    for row in manifest["rows"]:
        path = unit_dir / f"{row['id']}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("config_digest") != config_payload["config_digest"]:
                raise RuntimeError(f"stale unit {path}")
            records.append(value)
        else:
            pending.append(row["id"])

    started = time.time()

    def execute(task_id: str) -> dict:
        path = unit_dir / f"{task_id}.json"
        unit_started = time.time()
        try:
            record = run_condition(runtime[task_id], models=models, retriever=retriever,
                                   run_dir=out / "task_records" / CONDITION / task_id)
            value = {"status": "complete", "condition": CONDITION, "task_id": task_id,
                     "record": record, "task_failure": "", "infra_failure": ""}
        except BudgetExhausted:
            raise
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            task_failure = failure if _is_task_failure(exc) else ""
            value = {"status": "task_failure" if task_failure else "infra_failure",
                     "condition": CONDITION, "task_id": task_id, "record": None,
                     "task_failure": task_failure,
                     "infra_failure": "" if task_failure else failure}
        value.update({"config_digest": config_payload["config_digest"],
                      "wall_s": time.time() - unit_started})
        atomic_json(path, value)
        return value

    cap_hit = ""
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(execute, task_id): task_id for task_id in pending}
        try:
            for future in as_completed(futures):
                records.append(future.result())
        except BudgetExhausted as exc:
            cap_hit = str(exc)
            for future in futures:
                future.cancel()

    expected = len(manifest["rows"])
    scorable = sum(row["status"] in {"complete", "task_failure"} for row in records)
    infra = sum(row["status"] == "infra_failure" for row in records)
    status = "sealed" if scorable == expected and infra == 0 else ("cap_stop" if cap_hit else "incomplete")
    report = {"schema": "inherit_masization_run_status/1", "status": status,
              "expected_units": expected, "recorded_units": len(records),
              "scorable_units": scorable, "infra_failures": infra, "cap_hit": cap_hit,
              "spent_usd": ledger.spent_usd, "elapsed_wall_s": time.time() - started,
              "config_digest": config_payload["config_digest"]}
    atomic_json(out / "RUN_STATUS.json", report)
    if status == "sealed":
        atomic_json(out / "SEALED.json", {"config_digest": config_payload["config_digest"],
                    "manifest_digest": manifest["manifest_digest"], "sealed_at": time.time(),
                    "expected_units": expected})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usd-cap", type=float, required=True)
    concurrency = parser.add_mutually_exclusive_group()
    concurrency.add_argument("--task-concurrency", type=int, default=4,
                             help="Maximum benchmark tasks run concurrently, not LLM agents per task (default: 4)")
    concurrency.add_argument("--workers", dest="task_concurrency", type=int,
                             default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.task_concurrency < 1:
        parser.error("--task-concurrency must be positive")
    print(json.dumps(run(args.manifest, args.out, usd_cap=args.usd_cap,
                         workers=args.task_concurrency), indent=2))


if __name__ == "__main__":
    main()
