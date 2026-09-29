"""Additive candidate-config dump for dirty-subgraph (search-efficiency) analysis.

Persists every candidate MAS config evaluated during evolution — adapted
parents (GENERATE) and mutation/crossover offspring — plus a lineage manifest,
under <output_dir>/candidates/ for offline comparison with per-node token
counts from <output_dir>/trajectories/.

Purely observational, same contract as cost_ledger: it does NOT feed the
reward, selection, or any experimental decision, and every entry point
swallows its own errors so instrumentation can never break a run.

Labels match the cost-ledger / trajectory naming (parent{i}, step{N}_mutation,
step{N}_crossover) so <task>__<label>.json trajectory files join directly.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()


def dump_candidate(output_dir, label: str, config_yaml: Optional[str] = None,
                   meta: Optional[Dict[str, Any]] = None) -> None:
    """Write <output_dir>/candidates/<label>.yaml and merge meta into manifest.json.

    Call with config_yaml=None to only merge metadata into an existing entry
    (e.g. mark is_final_best after STEP 4). Labels starting with "_" are
    manifest-only entries (run-level metadata), no YAML written.
    """
    try:
        cand_dir = Path(output_dir) / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            if config_yaml is not None and not label.startswith("_"):
                (cand_dir / f"{label}.yaml").write_text(config_yaml)
            manifest_path = cand_dir / "manifest.json"
            manifest: Dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except Exception:
                    manifest = {}
            entry = manifest.get(label, {"label": label})
            if meta:
                entry.update({k: v for k, v in meta.items() if v is not None})
            manifest[label] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    except Exception:
        pass  # instrumentation must never break the run
