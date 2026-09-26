"""Run Inherit-MAS or a baseline, then score completed local runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from public_runner.common import ROOT, bind_run, read_manifest, verify_binding
from public_runner.baselines import BACKBONES, METHODS, runtime as baseline_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    online = commands.add_parser("run", help="Execute benchmark tasks using configured API or local models")
    online.add_argument("benchmark", choices=("workbench", "hotpotqa"))
    online.add_argument("--method", choices=METHODS, default="inherit-mas")
    online.add_argument("--backbone", choices=BACKBONES, default="gpt-4o-mini")
    online.add_argument("--out", type=Path, required=True)
    online.add_argument("--manifest", type=Path)
    online.add_argument("--limit", type=int, help="Run only the first N manifest tasks")
    concurrency = online.add_mutually_exclusive_group()
    concurrency.add_argument("--task-concurrency", type=int, default=1,
                             help="Maximum benchmark tasks run concurrently, not LLM agents per task (default: 1)")
    concurrency.add_argument("--workers", dest="task_concurrency", type=int,
                             default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    online.add_argument("--usd-cap", type=float, required=True, help="Estimated API budget; in-flight requests may exceed it")
    online.add_argument("--dry-run", action="store_true", help="Validate task selection without model calls or writes")
    offline = commands.add_parser("score", help="Write official scores for a completed run; no model calls")
    offline.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if args.command == "run":
        if args.task_concurrency < 1 or args.usd_cap <= 0 or not math.isfinite(args.usd_cap):
            parser.error("--task-concurrency and --usd-cap must be positive; --usd-cap must be finite")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        path = args.manifest or ROOT / "data" / f"{args.benchmark}.json"
        manifest = read_manifest(path, args.benchmark, args.limit)
        if args.dry_run:
            print(json.dumps({"benchmark": args.benchmark, "tasks": manifest["n"],
                              "method": args.method, "backbone": args.backbone,
                              "task_concurrency": args.task_concurrency, "usd_cap": args.usd_cap,
                              "output": str(out), "model_calls": False}, indent=2))
            return 0
        method, backbone = args.method, args.backbone
        path = bind_run(out, args.benchmark, manifest, method, backbone)
        benchmark = args.benchmark
    else:
        binding = out / "RUN_BINDING.json"
        if not binding.is_file():
            parser.error(f"not a run directory: {out} (missing RUN_BINDING.json)")
        try:
            value = json.loads(binding.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read run metadata: {exc}")
        benchmark = value.get("benchmark") if isinstance(value, dict) else None
        if not isinstance(benchmark, str) or benchmark not in {"workbench", "hotpotqa"}:
            parser.error("run metadata has an unsupported or missing benchmark")
        method = value.get("method", "inherit-mas")
        backbone = value.get("backbone", "gpt-4o-mini")
        if method not in METHODS or backbone not in BACKBONES:
            parser.error("run metadata has an unsupported method or backbone")
        required = ("INPUT_MANIFEST.json", "RUN_CONFIG.json", "RUN_STATUS.json", "SEALED.json")
        missing = [name for name in required if not (out / name).is_file()]
        if missing:
            parser.error(f"run is not ready for scoring: missing {', '.join(missing)}; complete the run first")
        path = verify_binding(out, benchmark, method, backbone)
    if method != "inherit-mas":
        runtime = baseline_runtime(method)
    elif benchmark == "workbench":
        from public_runner import workbench as runtime
    else:
        from public_runner import hotpot as runtime
    if args.command == "run":
        kwargs = {"usd_cap": args.usd_cap, "workers": args.task_concurrency}
        if method != "inherit-mas" or backbone != "gpt-4o-mini":
            kwargs["backbone"] = backbone
        value = runtime.run(path, out, **kwargs)
        code = 0 if value.get("status") == "sealed" else 2
    else:
        value = runtime.score(path, out)
        code = 0
    print(json.dumps({k: v for k, v in value.items() if k != "per_task"}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
