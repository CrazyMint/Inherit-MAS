"""Pinned native TacoMAS WorkBench runner and offline official scorer."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, digest, file_sha256
from public_runner.common import read_manifest
from public_runner.workbench import _validate_limits, _workbench_path
from .runtime_lock import COMMIT, verify_lock
from .native_transport import read_calls, usage_summary

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).parent / "upstream"
NATIVE_SETTINGS = {"controller": "native AgentState/GraphManager/EvolutionController",
    "rounds": 10, "slow_interval": 2, "initial_population": 5, "population_min": 2,
    "population_max": 20, "birth_death_pairs": 2, "edge_edits": 8, "worker_iterations": 20,
    "skip_meta_init": True, "instance_attempts": 1, "worker_temperature": 0,
    "meta_temperature": 0.3, "judge_temperature": 0, "compiler_tokens": 1200,
    "worker_tool_calls_per_round": 6, "execution_reuse": False, "typed_graph_projection": False}


def stage_public_dataset(rows, output):
    if any(not isinstance(row.get("task"), str) or not row["task"].strip() for row in rows):
        raise ValueError("WorkBench tasks must have nonempty task text")
    payload = {"dataset_id": "workbench-official", "instances": [
        {"id": row["id"], "question": row["task"], "source": str(row.get("source") or row["id"].split("#", 1)[0]),
         "base_template": str(row.get("base_template", "")), "expected_output": None}
        for row in rows]}
    atomic_json(Path(output), payload)
    return payload


def _unit_path(out, task_id):
    return out / "units" / f"{digest(task_id)}.json"


def _load_units(out, manifest, config):
    expected = {row["id"] for row in manifest["tasks"]}
    units = {}
    for path in (out / "units").glob("*.json"):
        unit = json.loads(path.read_text())
        task_id = unit.get("task_id")
        if (task_id not in expected or task_id in units or path != _unit_path(out, task_id)
                or unit.get("config_digest") != config["config_digest"]
                or unit.get("manifest_digest") != manifest["manifest_digest"]
                or unit.get("status") not in {"complete", "task_failure", "infra_failure"}):
            raise RuntimeError("invalid native unit identity/status")
        record = unit.get("record")
        if not isinstance(record, dict) or record.get("task_id") != task_id:
            raise RuntimeError("invalid native task record")
        if (not isinstance(record.get("prediction"), list)
                or any(not isinstance(action, str) for action in record["prediction"])):
            raise RuntimeError("invalid action prediction")
        if unit["status"] == "complete" and (unit.get("failure")
                or type(record.get("fast_rounds")) is not int or record["fast_rounds"] <= 0
                or type(unit.get("successful_calls")) is not int or unit["successful_calls"] <= 0):
            raise RuntimeError("invalid completed native evolution")
        if unit["status"] != "complete" and not unit.get("failure"):
            raise RuntimeError("failure record lacks its error")
        units[task_id] = unit
    return units


def _calls(out):
    return read_calls(out)


def _sources():
    files = [p for p in Path(__file__).parent.rglob("*.py")
             if "upstream" not in p.parts and "__pycache__" not in p.parts]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(files)}


def native_preflight(upstream, python):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(upstream))),
               PYTHON_DOTENV_DISABLED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    result = subprocess.run([python, "-m", "baselines.tacomas.native_child", "--preflight", str(upstream)],
                            cwd=upstream, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("native TacoMAS preflight failed; install baselines/tacomas/requirements.txt "
                           "in the environment selected by INHERIT_TACOMAS_PYTHON")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run(manifest_path, out, *, usd_cap, workers=4, backbone="gpt-4o-mini"):
    _validate_limits(usd_cap, workers)
    manifest = read_manifest(Path(manifest_path), "workbench")
    if any(not isinstance(row.get("task"), str) or not row["task"].strip() for row in manifest["tasks"]):
        raise ValueError("WorkBench tasks must have nonempty task text")
    out = Path(out).resolve()
    from public_runner.models import model_settings
    runtime = verify_lock(UPSTREAM)
    python = os.getenv("INHERIT_TACOMAS_PYTHON", sys.executable)
    python = shutil.which(python)
    if python is None:
        raise RuntimeError("INHERIT_TACOMAS_PYTHON does not resolve to an executable")
    preflight = native_preflight(UPSTREAM, python)
    models = model_settings(backbone)
    config = {"schema": "native_tacomas_workbench_config/1", "backbone": backbone,
        "manifest_digest": manifest["manifest_digest"], "upstream_commit": COMMIT,
        "runtime_digest": digest(runtime), "models": models, "settings": NATIVE_SETTINGS,
        "memory_top_k_synthesis_model": "gpt-5.4-mini" if backbone == "gpt-4o-mini" else backbone,
        "worker_model": backbone, "compiler_model": backbone, "meta_judge_model": "gpt-5.4-mini",
        "usd_cap": float(usd_cap), "workers": workers, "native_python": python,
        "source_sha256": _sources(), "native_preflight": preflight}
    child_config = {"out": str(out), "upstream": str(UPSTREAM), "backbone": backbone, "usd_cap": float(usd_cap)}
    if backbone == "qwen3-32b":
        raw = os.getenv("INHERIT_QWEN_TOKENIZER_PATH")
        if not raw:
            raise RuntimeError("native TacoMAS needs INHERIT_QWEN_TOKENIZER_PATH with local tokenizer files")
        tokenizer = Path(raw).expanduser().resolve()
        context = int(os.getenv("INHERIT_QWEN_CONTEXT_TOKENS", "32768"))
        if context < 256:
            raise ValueError("Qwen context limit must be at least 256")
        config["qwen_tokenizer"] = {name: file_sha256(tokenizer / name)
                                    for name in ("tokenizer.json", "tokenizer_config.json")}
        config["qwen_context_tokens"] = context
        child_config.update(tokenizer_path=str(tokenizer), context_tokens=context,
                            qwen_url=models["qwen"]["urls"][0])
    config["config_digest"] = digest(config)
    target = out / "RUN_CONFIG.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise RuntimeError("run directory has a different configuration")
    if not target.exists() and any((out / "units").glob("*.json")):
        raise RuntimeError("existing units have no configuration")
    atomic_json(target, config)
    units = _load_units(out, manifest, config)
    stage_public_dataset(manifest["tasks"], out / "workbench_public.json")
    child_path = out / "CHILD_CONFIG.json"
    atomic_json(child_path, child_config)
    started = time.time()
    pending = [i for i, row in enumerate(manifest["tasks"]) if row["id"] not in units]

    def execute(index):
        log = out / "logs" / f"{index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(UPSTREAM), env.get("PYTHONPATH", "")))
        with log.open("w") as stream:
            result = subprocess.run([python, "-m", "baselines.tacomas.native_child", str(child_path), str(index)],
                                    cwd=UPSTREAM, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record_path = out / "child_records" / f"{index}.json"
        if result.returncode or not record_path.exists():
            task_id = manifest["tasks"][index]["id"]
            unit = {"task_id": task_id, "status": "infra_failure",
                    "record": {"task_id": task_id, "prediction": []},
                    "failure": f"native subprocess failed; inspect {log}"}
        else:
            unit = json.loads(record_path.read_text())
        if unit["status"] == "cap_stop":
            return
        unit.update(config_digest=config["config_digest"], manifest_digest=manifest["manifest_digest"])
        atomic_json(_unit_path(out, unit["task_id"]), unit)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending and not (out / "CAP_STOP.json").exists() and not (out / "INFRA_STOP.json").exists():
            batch, pending = pending[:workers], pending[workers:]
            for future in [pool.submit(execute, i) for i in batch]:
                future.result()
    units = _load_units(out, manifest, config)
    infra = [u["task_id"] for u in units.values() if u["status"] == "infra_failure"]
    sealed = len(units) == manifest["n"] and not infra
    calls = _calls(out)
    report = {"status": "sealed" if sealed else ("cap_stop" if (out / "CAP_STOP.json").exists() else "incomplete"),
              "expected": manifest["n"], "recorded": len(units), "infra_failures": infra,
              "spent_usd": sum(c.get("estimated_cost_usd", 0) for c in calls),
              "tokens": sum(c.get("total_tokens", 0) for c in calls), "calls": len(calls),
              "config_digest": config["config_digest"], "elapsed_s": time.time() - started}
    report["usage"] = usage_summary(calls)
    atomic_json(out / "RUN_STATUS.json", report)
    if sealed:
        atomic_json(out / "SEALED.json", {"manifest_digest": manifest["manifest_digest"],
                    "config_digest": config["config_digest"], "expected": manifest["n"], "sealed_at": time.time()})
    else:
        (out / "SEALED.json").unlink(missing_ok=True)
    return report


def score(manifest_path, out):
    manifest = read_manifest(Path(manifest_path), "workbench")
    out = Path(out)
    config = json.loads((out / "RUN_CONFIG.json").read_text())
    seal = json.loads((out / "SEALED.json").read_text())
    status = json.loads((out / "RUN_STATUS.json").read_text())
    if (config.get("settings") != NATIVE_SETTINGS or config.get("upstream_commit") != COMMIT
            or config.get("source_sha256") != _sources()
            or config.get("config_digest") != digest({k: v for k, v in config.items() if k != "config_digest"})
            or config.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("config_digest") != config["config_digest"]
            or seal.get("manifest_digest") != manifest["manifest_digest"]
            or seal.get("expected") != manifest["n"] or status.get("status") != "sealed"):
        raise RuntimeError("seal/config/manifest mismatch")
    units = _load_units(out, manifest, config)
    if len(units) != manifest["n"] or any(u["status"] == "infra_failure" for u in units.values()):
        raise RuntimeError("sealed run has missing or infrastructure-failure units")
    _workbench_path()
    import loader
    import wb_env as W
    tasks, gold = loader.load_tasks()
    public = {task.id: task.task for task in tasks}
    if manifest.get("dataset_pin") != loader.dataset_pin():
        raise RuntimeError("manifest dataset pin mismatch")
    per_task = []
    for task in manifest["tasks"]:
        task_id, unit = task["id"], units[task["id"]]
        if public.get(task_id) != task["task"]:
            raise RuntimeError("manifest task differs from official dataset")
        prediction = unit["record"]["prediction"]
        per_task.append({"task_id": task_id, **W.score_prediction(prediction, gold[task_id], unit["failure"])})
    correct = sum(row["correct"] for row in per_task)
    report = {"benchmark": "workbench", "method": "tacomas", "backbone": config["backbone"],
              "n": manifest["n"], "correct": correct, "completion": correct / manifest["n"],
              "per_task": per_task, "usage": usage_summary(_calls(out))}
    atomic_json(out / "SCORE.json", report)
    return report
