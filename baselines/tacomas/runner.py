"""Public CLI dispatch for the two evaluated TacoMAS integrations."""
import json
from pathlib import Path

from . import hotpot, workbench


def _adapter(manifest_path):
    value = json.loads(Path(manifest_path).read_text())
    if "tasks" in value and "rows" not in value:
        return workbench
    if "rows" in value and "tasks" not in value:
        return hotpot
    raise ValueError("manifest must identify exactly one benchmark")


def run(manifest_path, out, *, usd_cap, workers=4, backbone="gpt-4o-mini"):
    return _adapter(manifest_path).run(manifest_path, out, usd_cap=usd_cap,
                                       workers=workers, backbone=backbone)


def score(manifest_path, out):
    return _adapter(manifest_path).score(manifest_path, out)
