"""Manifest selection and resume identity for public commands."""
from __future__ import annotations

import json
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, digest, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path, benchmark: str, limit: int | None = None) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop("manifest_digest", None)
    if not expected or digest(value) != expected:
        raise ValueError("manifest checksum mismatch")
    key = "tasks" if benchmark == "workbench" else "rows"
    rows = value.get(key)
    if not isinstance(rows, list) or not rows or value.get("n") != len(rows):
        raise ValueError("invalid manifest task count")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(x, str) or not x or "/" in x or "\\" in x or x in {".", ".."} for x in ids):
        raise ValueError("invalid task identifier")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task identifiers")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        value[key] = rows[:limit]
        value["n"] = len(value[key])
    value["manifest_digest"] = digest(value)
    return value


def source_identity() -> dict:
    paths = [ROOT / "run.py"]
    for name in ("public_runner", "inherit_mas", "workbench", "hotpot", "hotpot_fullwiki", "baselines"):
        paths.extend(p for p in (ROOT / name).rglob("*.py")
                     if "tests" not in p.parts and not p.name.startswith("test_"))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


def binding_for(benchmark: str, manifest: dict, method: str = "inherit-mas",
                backbone: str = "gpt-4o-mini") -> dict:
    return {"benchmark": benchmark, "manifest_digest": manifest["manifest_digest"],
            "method": method, "backbone": backbone,
            "source_files": source_identity()}


def bind_run(out: Path, benchmark: str, manifest: dict, method: str = "inherit-mas",
             backbone: str = "gpt-4o-mini") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "INPUT_MANIFEST.json"
    binding = out / "RUN_BINDING.json"
    expected = binding_for(benchmark, manifest, method, backbone)
    if binding.exists() and json.loads(binding.read_text()) != expected:
        raise RuntimeError("run directory belongs to different code, benchmark or task selection")
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise RuntimeError("run directory has a different input manifest")
    if not binding.exists() and any(p.name not in {"INPUT_MANIFEST.json"} for p in out.iterdir()):
        raise RuntimeError("refusing to adopt an existing unbound run directory")
    atomic_json(binding, expected)
    atomic_json(target, manifest)
    return target


def verify_binding(out: Path, benchmark: str, method: str = "inherit-mas",
                   backbone: str = "gpt-4o-mini") -> Path:
    path = out / "INPUT_MANIFEST.json"
    manifest = read_manifest(path, benchmark)
    if json.loads((out / "RUN_BINDING.json").read_text()) != binding_for(benchmark, manifest, method, backbone):
        raise RuntimeError("run/code/manifest identity mismatch")
    return path
