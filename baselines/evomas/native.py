"""Serial run orchestration around the unmodified native evolutionary controller."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, digest, file_sha256
from public_runner.common import read_manifest
from .runtime_lock import COMMIT, verify

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
DOMAINS = ("analytics", "calendar", "customer_relationship_manager", "email",
           "multi_domain", "project_management")
SEEDS = ("single_codeagent", "debate", "majority_vote", "peer_review", "smoa")


def settings(benchmark: str, backbone: str) -> dict:
    if benchmark not in {"workbench", "hotpotqa"} or backbone not in {"gpt-4o-mini", "qwen3-32b"}:
        raise ValueError("invalid benchmark or backbone")
    if benchmark == "hotpotqa" and backbone != "gpt-4o-mini":
        raise ValueError("Qwen HotpotQA uses the evaluated graph adapter, not native EvoMAS")
    return {"num_parents": 2, "evolution_steps": 3 if benchmark == "workbench" else 2,
            "agent_max_steps": 8 if benchmark == "workbench" else 20,
            "agent_max_tokens": 1024 if benchmark == "workbench" else None,
            "worker_model": "azure:gpt-4o-mini" if backbone == "gpt-4o-mini" else "openai:qwen3-32b",
            "meta_model": "azure:gpt-5.4-mini", "judge_model": "azure:gpt-5.4-mini",
            "meta_temperature": 0.7, "judge_temperature": 0,
            "worker_temperature": "native_config" if backbone == "gpt-4o-mini" else 0,
            "meta_max_tokens": 8192, "mutation_probability": 0.8, "seed": 42,
            "memory_evolution": True, "workers": 1, "batch_size": 1,
            "retrieval_calls_per_agent": 4 if benchmark == "hotpotqa" else None}


def unit_path(out: Path, task_id: str) -> Path:
    return out / "units" / task_id.replace("#", "__")


def state_digest(out: Path) -> str:
    root = out / "state"
    value = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        value.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return value.hexdigest()


def verify_resume(out: Path, rows: list[dict]) -> int:
    last = json.loads((out / "STATE_INITIALIZED.json").read_text())["state_digest"]
    first = len(rows)
    for ordinal, row in enumerate(rows):
        path = unit_path(out, row["id"]) / "DONE.json"
        if not path.exists():
            first = min(first, ordinal)
            continue
        done = json.loads(path.read_text())
        if ordinal > first or done.get("ordinal") != ordinal or done.get("state_before") != last:
            raise RuntimeError("invalid dependent EvoMAS task/state chain")
        last = done["state_after"]
    if state_digest(out) != last:
        raise RuntimeError("EvoMAS state does not match the completed serial chain")
    return first


def normalize_pool(source: Path, target: Path, worker: str, *, hotpot=False) -> None:
    import yaml
    value = yaml.safe_load(source.read_text())
    if hotpot:
        value["name"] = f"{value.get('name', source.stem)}_hotpot_fullwiki"
        value["description"] = f"HotpotQA FullWiki seed adapted from upstream {source.name}"
    for agent in (value.get("agents") or {}).values():
        agent["model_id"] = worker
        if hotpot:
            agent["tools"] = ["hotpot_fullwiki"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(value, sort_keys=False))


def prepare(out: Path, manifest_path: Path, manifest: dict, benchmark: str, config: dict) -> None:
    if not (out / "STATE_INITIALIZED.json").exists():
        if (out / "state").exists() and any((out / "state").iterdir()):
            raise RuntimeError("uncommitted EvoMAS state exists")
        if benchmark == "workbench":
            for domain in DOMAINS:
                files = sorted((UPSTREAM / "mas_pools/workbench" / domain).glob("*.yaml"))
                if not files:
                    raise RuntimeError(f"missing native seed pool for {domain}")
                for source in files:
                    normalize_pool(source, out / "state/pools" / domain / source.name, config["worker_model"])
        else:
            for name in SEEDS:
                normalize_pool(UPSTREAM / "mas_pools/bbeh" / f"{name}.yaml",
                               out / "state/pools/hotpot_fullwiki" / f"{name}.yaml",
                               config["worker_model"], hotpot=True)
        atomic_json(out / "STATE_INITIALIZED.json", {"state_digest": state_digest(out)})
    dataset = out / "dataset"
    if benchmark == "hotpotqa":
        from .hotpot import stage_evomas_dataset
        stage_evomas_dataset(manifest_path, dataset / "hotpot_fullwiki/test.json")
        os.environ["EVOMAS_HOTPOT_MANIFEST"] = str(manifest_path)
        os.environ["EVOMAS_HOTPOT_CACHE"] = str(out / "retrieval_cache")
    else:
        for domain in DOMAINS:
            selected = [row for row in manifest["tasks"] if row["id"].split("#")[0] == domain]
            if not selected:
                continue
            upstream = json.loads((UPSTREAM / "dataset/workbench" / domain / "test.json").read_text())
            for row in selected:
                _, index = row["id"].split("#")
                source = upstream[int(index)]
                if (source.get("query") or source.get("q") or "").strip() != row["task"].strip():
                    raise ValueError(f"task text differs from pinned EvoMAS dataset: {row['id']}")
            # Native task sampling indexes integer WorkBench IDs positionally.
            # Retain all public rows, but never stage reference actions.
            staged = [{"id": index, "query": source.get("query") or source.get("q") or "",
                       "gt": "__SEALED_OFFLINE__", "tag": source.get("tag", [domain]),
                       "source": source.get("source", "WORKBENCH")}
                      for index, source in enumerate(upstream)]
            atomic_json(dataset / "workbench" / domain / "test.json", staged)


def usage(out: Path) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0}
    for path in (out / "units").glob("*/live_calls.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                totals["calls"] += int(row.get("api_calls", 1))
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    totals[key] += int(row.get(key, 0))
                totals["cost_usd"] += float(row.get("cost_usd") or 0)
    return totals


def audit(out: Path, worker: str) -> dict:
    import yaml
    seen = set()
    count = 0
    for unit in (out / "units").iterdir():
        for path in (unit / "candidates").glob("*.yaml"):
            value = yaml.safe_load(path.read_text()) or {}
            for agent in (value.get("agents") or {}).values():
                if agent.get("model_id") != worker:
                    raise RuntimeError("candidate escaped the worker palette")
        for path in (unit / "trajectories").glob("*.json"):
            for row in json.loads(path.read_text()).get("trace", []):
                model = (row.get("metadata") or {}).get("model_id")
                if model:
                    seen.add(model)
                    count += 1
    if seen - {worker}:
        raise RuntimeError(f"worker trace escaped palette: {sorted(seen)}")
    return {"expected_worker": worker, "observed_workers": sorted(seen), "worker_traces": count,
            "pass": not (seen - {worker})}


def run(manifest_path: Path, out: Path, *, benchmark: str, backbone: str, usd_cap: float) -> dict:
    manifest = read_manifest(manifest_path, benchmark)
    cfg = settings(benchmark, backbone)
    lock = verify(UPSTREAM)
    from public_runner.models import model_settings
    config = {"schema": "public_native_evomas/1", "method": "evomas", "benchmark": benchmark,
              "backbone": backbone, "manifest_digest": manifest["manifest_digest"],
              "settings": cfg, "model_routing": model_settings(backbone),
              "upstream_commit": COMMIT, "runtime_digest": digest(lock),
              "usd_cap": usd_cap, "cap_boundary": "between_serial_tasks"}
    if benchmark == "hotpotqa":
        from .hotpot import RemoteBM25Retriever
        url = os.environ.get("HOTPOT_BM25_URL", "").strip()
        if not url:
            raise RuntimeError("native Hotpot EvoMAS requires HOTPOT_BM25_URL; start scripts/serve_bm25.py")
        retriever = RemoteBM25Retriever(url)
        os.environ["EVOMAS_BM25_DIGEST"] = retriever.fingerprint_digest
        config["retrieval"] = {"service_digest": digest(url), "retriever_digest": retriever.fingerprint_digest,
                               "calls_per_agent": 4}
    config["config_digest"] = digest(config)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "RUN_CONFIG.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("EvoMAS configuration changed; use a fresh output directory")
    atomic_json(path, config)
    prepare(out, manifest_path, manifest, benchmark, cfg)
    rows = manifest["tasks" if benchmark == "workbench" else "rows"]
    first = verify_resume(out, rows)
    os.environ["EVOMAS_AGENT_MAX_STEPS"] = str(cfg["agent_max_steps"])
    os.environ["META_MODEL_MAX_TOKENS"] = str(cfg["meta_max_tokens"])
    if cfg["agent_max_tokens"]:
        os.environ["EVOMAS_AGENT_MAX_TOKENS"] = str(cfg["agent_max_tokens"])
    else:
        os.environ.pop("EVOMAS_AGENT_MAX_TOKENS", None)
    sys.path.insert(0, str(UPSTREAM))
    from .model_bridge import install
    install(backbone)
    loader = importlib.import_module("src.dataset.load_dataset")
    def dataset_path(self):
        name = self.dataset_name
        relative = Path("workbench") / name.removeprefix("workbench_") if name.startswith("workbench_") else Path(name)
        return str(out / "dataset" / relative / "test.json")
    loader.Dataset._find_dataset_path = dataset_path
    from main import run_evolution_pipeline
    os.chdir(UPSTREAM)
    failure = None
    cap_stop = False
    for ordinal in range(first, len(rows)):
        if usage(out)["cost_usd"] >= usd_cap:
            cap_stop = True
            break
        item = rows[ordinal]
        task_id = item["id"]
        unit = unit_path(out, task_id)
        unit.mkdir(parents=True, exist_ok=True)
        checkpoint = unit / "STATE_BEFORE"
        shutil.rmtree(checkpoint, ignore_errors=True)
        shutil.copytree(out / "state", checkpoint)
        before = state_digest(out)
        os.environ["EVOMAS_LIVE_LEDGER"] = str(unit / "live_calls.jsonl")
        try:
            domain, native_id = task_id.split("#") if benchmark == "workbench" else ("hotpot_fullwiki", task_id)
            result = run_evolution_pipeline(
                dataset_name=f"workbench_{domain}" if benchmark == "workbench" else domain,
                pool_dir=str(out / "state/pools" / domain), num_eval_tasks=1,
                max_steps=cfg["evolution_steps"], num_parents=2, seed=42, output_dir=str(unit),
                meta_model_id=cfg["meta_model"], model_list=[cfg["worker_model"]],
                llm_as_judge=cfg["judge_model"],
                task_ids=[int(native_id) if benchmark == "workbench" else native_id],
                memory_path=str(out / "state/memory.json"), memory_evolution=True, batch_size=1, workers=1)
            from .model_bridge import provider_failures
            if provider_failures:
                raise RuntimeError("provider failure inside native task; refusing to seal unknown usage")
            trace_audit = audit(out, cfg["worker_model"])
            if not trace_audit["worker_traces"]:
                raise RuntimeError("native task produced no auditable worker execution")
            if len(list((unit / "trajectories").glob("*__final_selected.json"))) != 1:
                raise RuntimeError("native task lacks its unique fresh final execution")
            atomic_json(unit / "DONE.json", {"task_id": task_id, "ordinal": ordinal,
                        "state_before": before, "state_after": state_digest(out), "result": result})
            (unit / "FAILED.json").unlink(missing_ok=True)
            shutil.rmtree(checkpoint)
            atomic_json(out / "WORKER_AUDIT.json", trace_audit)
        except Exception as exc:
            shutil.rmtree(out / "state")
            shutil.copytree(checkpoint, out / "state")
            failure = f"{type(exc).__name__}: {exc}"
            atomic_json(unit / "FAILED.json", {"task_id": task_id, "ordinal": ordinal, "error": failure})
            break
    completed = verify_resume(out, rows)
    status = {"status": "sealed" if completed == len(rows) else "cap_stop" if cap_stop else "incomplete",
              "expected_units": len(rows), "completed_units": completed, "failure": failure,
              "config_digest": config["config_digest"], "usage": usage(out),
              "state_digest": state_digest(out)}
    atomic_json(out / "RUN_STATUS.json", status)
    if status["status"] == "sealed":
        artifacts = {str(p.relative_to(out)): file_sha256(p) for p in (out / "units").rglob("*") if p.is_file()}
        atomic_json(out / "SEALED.json", {"config_digest": config["config_digest"], "artifacts": artifacts,
                                         "state_digest": state_digest(out)})
    else:
        (out / "SEALED.json").unlink(missing_ok=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--usd-cap", type=float, required=True)
    args = parser.parse_args()
    status = run(args.manifest.resolve(), args.out.resolve(), benchmark=args.benchmark,
                 backbone=args.backbone, usd_cap=args.usd_cap)
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if status["status"] == "sealed" else 2)
