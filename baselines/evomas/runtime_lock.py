"""Portable integrity lock for the locally assembled, non-redistributed runtime."""
from __future__ import annotations

import json
from pathlib import Path

from hotpot_fullwiki.common import atomic_json, file_sha256

COMMIT = "93fd9d6766b093f6bbdeeffc35896910cbece6e2"


def sources(root: Path) -> dict:
    paths = [root / "main.py"]
    for directory, extensions in (("src", {".py", ".md"}),
                                  ("mas_pools/workbench", {".yaml", ".json"}),
                                  ("mas_pools/bbeh", {".yaml", ".json"}),
                                  ("dataset/workbench", {".json", ".csv"})):
        paths.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in extensions)
    if len(paths) < 10 or not paths[0].is_file():
        raise RuntimeError("EvoMAS runtime is not assembled; run scripts/setup_baselines.py")
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def write_lock(root: Path) -> None:
    atomic_json(root / "RUNTIME_LOCK.json", {"commit": COMMIT, "files": sources(root)})


def verify(root: Path) -> dict:
    path = root / "RUNTIME_LOCK.json"
    if not path.exists():
        raise RuntimeError("EvoMAS runtime is not assembled; run scripts/setup_baselines.py")
    lock = json.loads(path.read_text())
    if lock != {"commit": COMMIT, "files": sources(root)}:
        raise RuntimeError("EvoMAS runtime source lock mismatch; repeat setup on a clean tree")
    return lock


verify_lock = verify
