"""Launch one local Qwen3-32B replica and record its serving configuration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3-32B model directory")
    parser.add_argument("--devices", required=True, help="Two GPU IDs, for example 0,1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("runs/qwen/serve_provenance.json"))
    args = parser.parse_args()
    if (len(args.devices.split(",")) != 2 or len(set(args.devices.split(","))) != 2
            or any(not x.isdigit() for x in args.devices.split(","))):
        parser.error("--devices must name two GPU IDs")
    model = args.model.expanduser().resolve()
    names = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "model.safetensors.index.json")
    if any(not (model / name).is_file() for name in names):
        parser.error("model directory lacks required weights index or tokenizer/configuration files")
    version = importlib.metadata.version("vllm")
    if version != "0.21.0":
        parser.error("this serving configuration requires vLLM 0.21.0 in a separate environment")
    flags = ["--model", str(model), "--served-model-name", "qwen3-32b", "--host", "127.0.0.1",
             "--port", str(args.port), "--tensor-parallel-size", "2", "--dtype", "bfloat16",
             "--max-model-len", "32768", "--max-num-seqs", "2", "--gpu-memory-utilization", "0.85",
             "--generation-config", "vllm", "--seed", "0", "--no-enable-prefix-caching",
             "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    payload = {"model": "qwen3-32b", "model_path": str(model), "vllm_version": version,
               "python": platform.python_version(), "cuda_visible_devices": args.devices,
               "batch_invariant": True, "enable_thinking": False, "flags": flags,
               "metadata_sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                                   for name in names},
               "weight_files": [{"name": path.name, "bytes": path.stat().st_size}
                                for path in sorted(model.glob("model-*-of-*.safetensors"))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.devices, "VLLM_BATCH_INVARIANT": "1"}
    os.execvpe(sys.executable, [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *flags], env)


if __name__ == "__main__":
    main()
