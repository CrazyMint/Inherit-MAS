"""Verify the fetched native runtime, excluding generated run artifacts."""
import json
import subprocess
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, file_sha256

COMMIT = "6f0d545f2493cf95d2eb6a325d1a6686acf658eb"


def identity(upstream):
    upstream = Path(upstream)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("TacoMAS upstream commit mismatch")
    paths = [upstream / "scripts/run_evolution.py", upstream / "paired_accounting.py"]
    paths += list((upstream / "tacomas").rglob("*.py"))
    paths += list((upstream / "prompts").rglob("*.yaml"))
    paths += list((upstream / "tacomas/skill/workbench_playbook").rglob("*.md"))
    return {"commit": commit,
            "files": {str(p.relative_to(upstream)): file_sha256(p) for p in sorted(paths)}}


def write_lock(upstream):
    value = identity(upstream)
    atomic_json(Path(upstream) / ".runtime-lock.json", value)
    return value


def verify_lock(upstream):
    path = Path(upstream) / ".runtime-lock.json"
    if not path.is_file() or json.loads(path.read_text()) != identity(upstream):
        raise RuntimeError("TacoMAS runtime differs from setup; rerun scripts/setup_baselines.py")
    return json.loads(path.read_text())
