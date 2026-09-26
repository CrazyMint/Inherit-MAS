"""Public EvoMAS dispatch and sealed offline scoring."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, digest, file_sha256
from public_runner.common import read_manifest
from public_runner.workbench import _validate_limits, _workbench_path
from .native import ROOT, settings, unit_path, usage, verify_resume


def benchmark_for(path: Path) -> str:
    return "workbench" if "tasks" in json.loads(path.read_text()) else "hotpotqa"


def build_run_command(manifest_path, out, *, usd_cap, backbone="gpt-4o-mini") -> list[str]:
    return [os.environ.get("INHERIT_EVOMAS_PYTHON", sys.executable), "-m", "baselines.evomas.native", "--manifest", str(Path(manifest_path).resolve()),
            "--out", str(Path(out).resolve()), "--benchmark", benchmark_for(Path(manifest_path)),
            "--backbone", backbone, "--usd-cap", str(usd_cap)]


def run(manifest_path, out, *, usd_cap, workers=1, backbone="gpt-4o-mini") -> dict:
    _validate_limits(usd_cap, workers)
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    benchmark = benchmark_for(manifest_path)
    if benchmark == "hotpotqa" and backbone == "qwen3-32b":
        from hotpot_fullwiki.baselines import evomas, runner
        return runner.run(manifest_path, out, usd_cap=usd_cap, workers=workers,
                          backbone=backbone, method="evomas_adapted", controller=evomas)
    settings(benchmark, backbone)
    if workers != 1:
        raise ValueError("native EvoMAS is one serial cross-task trajectory; --task-concurrency must be 1")
    read_manifest(manifest_path, benchmark)
    from .runtime_lock import verify
    verify(Path(__file__).parent / "upstream")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with (out / "native.log").open("a") as log:
        result = subprocess.run(build_run_command(manifest_path, out, usd_cap=usd_cap, backbone=backbone),
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    status_path = out / "RUN_STATUS.json"
    if result.returncode not in {0, 2} or not status_path.exists():
        raise RuntimeError(f"native EvoMAS exited with code {result.returncode}; inspect {out / 'native.log'}")
    return json.loads(status_path.read_text())


def _verify_seal(manifest_path: Path, out: Path, benchmark: str) -> tuple[dict, dict]:
    manifest = read_manifest(manifest_path, benchmark)
    cfg = json.loads((out / "RUN_CONFIG.json").read_text())
    seal = json.loads((out / "SEALED.json").read_text())
    status = json.loads((out / "RUN_STATUS.json").read_text())
    if (cfg.get("method") != "evomas" or cfg.get("benchmark") != benchmark
            or cfg.get("manifest_digest") != manifest["manifest_digest"]
            or cfg.get("config_digest") != digest({k: v for k, v in cfg.items() if k != "config_digest"})
            or seal.get("config_digest") != cfg["config_digest"]
            or status.get("config_digest") != cfg["config_digest"] or status.get("status") != "sealed"
            or cfg.get("settings") != settings(benchmark, cfg.get("backbone"))):
        raise RuntimeError("native EvoMAS seal/config/manifest mismatch")
    paths = {str(p.relative_to(out)): file_sha256(p) for p in (out / "units").rglob("*") if p.is_file()}
    if paths != seal.get("artifacts"):
        raise RuntimeError("sealed native EvoMAS artifacts changed")
    rows = manifest["tasks" if benchmark == "workbench" else "rows"]
    if verify_resume(out, rows) != len(rows):
        raise RuntimeError("sealed native EvoMAS run has incomplete serial state")
    return manifest, cfg


def score(manifest_path, out) -> dict:
    manifest_path, out = Path(manifest_path), Path(out)
    benchmark = benchmark_for(manifest_path)
    config = json.loads((out / "RUN_CONFIG.json").read_text())
    if benchmark == "hotpotqa" and config.get("backbone") == "qwen3-32b":
        from hotpot_fullwiki.baselines.runner import score as score_graph
        return score_graph(manifest_path, out, method="evomas_adapted")
    manifest, config = _verify_seal(manifest_path, out, benchmark)
    predictions = {}
    for row in manifest["tasks" if benchmark == "workbench" else "rows"]:
        paths = sorted((unit_path(out, row["id"]) / "trajectories").glob("*__final_selected.json"))
        if len(paths) != 1:
            raise RuntimeError(f"missing or ambiguous fresh final trajectory for {row['id']}")
        predictions[row["id"]] = json.loads(paths[0].read_text()).get("final_result", "")
    if benchmark == "hotpotqa":
        from .hotpot import scorer_examples, score_records
        report = score_records(scorer_examples(manifest_path), predictions)
    else:
        _workbench_path()
        from loader import load_tasks
        from wb_env import score_prediction
        from .actions import extract_calls
        _, gold = load_tasks()
        rows = [{"task_id": row["id"], **score_prediction(extract_calls(predictions[row["id"]]), gold[row["id"]], "")}
                for row in manifest["tasks"]]
        report = {"n": len(rows), "completion": sum(int(row["correct"]) for row in rows) / len(rows),
                  "per_task": rows}
    report.update(benchmark=benchmark, method="evomas", backbone=config["backbone"], usage=usage(out))
    atomic_json(out / "SCORE.json", report)
    return report
