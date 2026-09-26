"""One isolated native WorkBench evolution followed by one action compiler."""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

from hotpot_fullwiki.common import atomic_json
from inherit_mas import release_config
from .native_transport import (BudgetStop, install_role_overlays, install_transport,
                               read_calls, role_scope)


def native_arguments(index, backbone, worker_endpoint, meta_endpoint):
    return ["--start", str(index), "--end", str(index + 1),
            "--agent_name", "tacomas", "--n_base_agents", "1",
            "--min_iterations_per_agent", "1", "--max_iterations_per_agent", "20",
            "--agent_model", f"openai/{backbone}", "--agent_api_base", worker_endpoint,
            "--agent_temperature", "0", "--meta_model", "openai/gpt-5.4-mini",
            "--meta_api_base", meta_endpoint, "--meta_temperature", "0.3",
            "--max_fast_rounds", "10", "--fast_steps_per_window", "1",
            "--bd_check_interval", "2", "--graph_rewire_interval", "2",
            "--init_n_min", "5", "--init_n_max", "5", "--pop_n_min", "2", "--pop_n_max", "20",
            "--max_birth_death_pairs", "2", "--max_edge_edits", "8",
            "--instance_retries", "1", "--per_instance_sleep", "0", "--skip_meta_init"]


def preflight(upstream):
    sys.path.insert(0, str(Path(upstream).resolve()))
    from tacomas.meta_evolution.schemas import AgentState
    from tacomas.meta_evolution.graph_manager import GraphManager
    from tacomas.meta_evolution.evolution_controller import EvolutionController
    from tacomas.env.workbench import READ_TOOL_ALIASES, PUBLIC_TOOL_NAMES
    install_role_overlays()
    if len(READ_TOOL_ALIASES) != 13 or len(PUBLIC_TOOL_NAMES) != 14:
        raise RuntimeError("native WorkBench read-only tool contract changed")
    return {"controller": [cls.__name__ for cls in (AgentState, GraphManager, EvolutionController)],
            "read_only_tools": len(READ_TOOL_ALIASES), "write_tools": 0,
            "python_version": list(sys.version_info[:3])}


def run(config_path, index):
    config = json.loads(Path(config_path).read_text())
    root, upstream = Path(config["out"]), Path(config["upstream"])
    output = root / "native" / str(index)
    dataset = json.loads((root / "workbench_public.json").read_text())
    item = dataset["instances"][index]
    sys.path.insert(0, str(upstream))
    document = release_config.credentials()
    meta = release_config.deployment(document, "5.4-mini")
    meta_endpoint = release_config.model_endpoint(document, meta)
    meta_key = release_config.api_key(meta["api_key_env"])
    if config["backbone"] == "gpt-4o-mini":
        worker = release_config.deployment(document, "gpt-4o-mini")
        worker_endpoint = release_config.model_endpoint(document, worker)
        worker_key = release_config.api_key(worker["api_key_env"])
    else:
        worker_endpoint, worker_key = config["qwen_url"], "EMPTY"
    os.environ.update({"ALLOW_DIRECT_RUN": "1", "DATASET_ID": "workbench-official",
        "DATASET_LOCAL_PATH": str(root / "workbench_public.json"),
        "DATASET_TEMPLATE_PATH": str(upstream / "prompts/dataset-shared/workbench_role_aware.yaml"),
        "DATASET_ENV_NAME": "workbench-official", "DATASET_TOOLS": "",
        "JUDGE_MODEL": "openai/gpt-5.4-mini", "JUDGE_API_BASE": meta_endpoint, "JUDGE_API_KEY": meta_key,
        "TACOMAS_AGENT_API_KEY": worker_key, "TACOMAS_META_API_KEY": meta_key,
        "TACOMAS_AGENT_API_BASE": worker_endpoint, "MAX_TOOL_CALLS_PER_SUBAGENT_ROUND": "6",
        "DISABLE_LLM_DETAIL_LOG": "1", "TACOMAS_USAGE_LEDGER": "1",
        "PYTHON_DOTENV_DISABLED": "1", "USE_DMX_ALL_NONSTREAM": "0", "USE_DMX_META_NONSTREAM": "0",
        "TACOMAS_OUTPUT_ROOT": str(output), "TACOMAS_INSTANCE_IDX": str(index), "RUN_TAG": "public"})
    completion = install_transport(config)
    install_role_overlays()
    sys.argv = [str(upstream / "scripts/run_evolution.py"),
                *native_arguments(index, config["backbone"], worker_endpoint, meta_endpoint)]
    record = {"task_id": item["id"], "prediction": [], "compiler_error": "", "execution_error": ""}
    failure = ""
    try:
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError("native runner failed") from exc
        traces = sorted(output.glob("evolution_trace_*"))
        if len(traces) != 1:
            raise RuntimeError("expected one native task trace")
        trace = traces[0]
        health = json.loads((trace / "summary.json").read_text())
        record.update(trace_dir=str(trace), fast_rounds=int(health.get("evolution_steps", 0)),
                      slow_updates=int(health.get("slow_updates", 0)))
        if health.get("instances_failed") or record["fast_rounds"] <= 0:
            raise RuntimeError("native task did not finish")
        instance = json.loads((trace / f"instance_{index}.json").read_text())
        plan = str((instance.get("runtime_result") or {}).get("final_answer", ""))
        record["evolved_plan"] = plan
        from . import workbench_executor
        workbench_executor.MODEL = f"openai/{config['backbone']}"
        compile_fn = role_scope(lambda: workbench_executor.compile_actions(
            item["question"], plan, completion_fn=completion), "final_action_compiler")
        try:
            compiled = compile_fn()
            executed = workbench_executor.execute_actions(compiled["actions"])
            record.update(compiled_actions=compiled["actions"], compiler_reason=compiled["reason"],
                          compiler_usage=compiled["usage"], prediction=executed["prediction"],
                          executor_trace=executed["trace"], execution_error=executed["execution_error"])
        except (ValueError, TypeError, KeyError) as exc:
            record["compiler_error"] = type(exc).__name__
    except BaseException as exc:
        failure = type(exc).__name__
    calls = read_calls(root)
    calls = [call for call in calls if call.get("instance_idx") == index]
    successful_calls = sum(call["status"] == "success" for call in calls)
    if not successful_calls and not failure:
        failure = "native task has no successful provider calls"
    if (root / "INFRA_STOP.json").exists():
        status, failure = "infra_failure", failure or "provider infrastructure failure"
    elif (root / "CAP_STOP.json").exists() and failure:
        status = "cap_stop"
    elif failure:
        status = "task_failure" if any(call.get("task_failure") for call in calls) else "infra_failure"
    else:
        status = "task_failure" if record["compiler_error"] or record["execution_error"] else "complete"
        failure = record["compiler_error"] or record["execution_error"]
    atomic_json(root / "child_records" / f"{index}.json", {
        "task_id": item["id"], "status": status, "record": record,
        "failure": failure, "calls": len(calls), "successful_calls": successful_calls})


if __name__ == "__main__":
    if sys.argv[1] == "--preflight":
        print(json.dumps(preflight(sys.argv[2]), sort_keys=True))
    else:
        run(sys.argv[1], int(sys.argv[2]))
