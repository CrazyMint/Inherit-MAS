"""Lazy dispatch to benchmark baseline runners."""
from importlib import import_module

METHODS = ("inherit-mas", "single-react", "evoagent", "evomas", "tacomas")
BACKBONES = ("gpt-4o-mini", "qwen3-32b")


def runtime(method: str):
    if method not in METHODS or method == "inherit-mas":
        raise ValueError(f"unsupported baseline: {method}")
    return import_module(f"baselines.{method.replace('-', '_')}.runner")
