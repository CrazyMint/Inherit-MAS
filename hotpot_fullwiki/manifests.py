"""Deterministic FullWiki manifests selected from public metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config
from .common import atomic_json, digest
from .loader import open_snapshot, public_manifest_rows

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manifests"
VERSION = "hotpot-fullwiki-manifest-v1"
SIZES = {"engineering_smoke": config.SMOKE_N,
         "calibration": config.CALIBRATION_N,
         "powered_pool": config.POWERED_POOL_N}


def _rank(stage: str, row: dict) -> str:
    return hashlib.blake2b(f"{VERSION}\0{stage}\0{row['id']}".encode(), digest_size=16).hexdigest()


def build() -> dict[str, dict]:
    snapshot = open_snapshot()
    rows = [row for row in public_manifest_rows(snapshot)
            if row["source"] == "train" and row["split"] == "adaptive_search"]
    used: set[str] = set()
    payloads = {}
    for stage, size in SIZES.items():
        eligible = [row for row in rows if row["id"] not in used]
        selected = sorted(eligible, key=lambda row: (_rank(stage, row), row["id"]))[:size]
        if len(selected) != size:
            raise RuntimeError(f"not enough rows for {stage}")
        used.update(row["id"] for row in selected)
        payload = {"version": VERSION, "stage": stage, "n": size,
                   "dataset_revision": snapshot.digest,
                   "selection": "public-metadata-only deterministic hash; pairwise disjoint",
                   "rows": selected}
        payload["manifest_digest"] = digest(payload)
        payloads[stage] = payload
    return payloads


def write_all(out: Path = OUT) -> dict[str, dict]:
    payloads = build()
    for name, payload in payloads.items():
        atomic_json(out / f"{name}.json", payload)
    return payloads


def load(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    expected = value.pop("manifest_digest", None)
    if expected != digest(value):
        raise ValueError("manifest digest mismatch")
    value["manifest_digest"] = expected
    return value


if __name__ == "__main__":
    print(json.dumps({key: {"n": value["n"], "digest": value["manifest_digest"]}
                      for key, value in write_all().items()}, indent=2))
