#!/usr/bin/env python
"""Materialize the pinned HotpotQA snapshot the runners expect.

The manifest builder and loader read Arrow files from
  $HF_HOME/datasets/hotpot_qa/distractor/0.0.0/<DATASET_REVISION>/
and verify them by SHA-256 (hotpot/loader.py). This script downloads exactly
that revision with `datasets`, then opens it through the project's own loader
so a mismatch fails loudly here rather than at launch time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inherit_mas.release_config import ensure_hf_home  # noqa: E402

ensure_hf_home()
from hotpot import loader  # noqa: E402


def main() -> int:
    target = loader.default_snapshot_root()
    try:
        snap = loader.open_pinned_snapshot(target)
        print(f"already present and verified: {target} (digest {snap.digest[:12]}...)")
        return 0
    except loader.LoaderError:
        pass
    from datasets import load_dataset  # noqa: PLC0415
    print(f"downloading hotpotqa/hotpot_qa[{loader.DATASET_CONFIG}] @ {loader.DATASET_REVISION} "
          f"into HF_HOME={os.environ['HF_HOME']} (~570 MB)")
    load_dataset("hotpotqa/hotpot_qa", loader.DATASET_CONFIG,
                 revision=loader.DATASET_REVISION)
    # Recent `datasets` versions cache hub repos as <owner>___<name>/...; the
    # loader expects the older hotpot_qa/... layout. Bridge with a symlink.
    produced = (Path(os.environ["HF_HOME"]) / "datasets" / "hotpotqa___hotpot_qa" /
                loader.DATASET_CONFIG / loader.DATASET_BUILDER_VERSION / loader.DATASET_REVISION)
    if produced.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(produced, target_is_directory=True)
        print(f"linked {target} -> {produced}")
    snap = loader.open_pinned_snapshot(target)
    print(f"verified pinned snapshot at {target} (digest {snap.digest[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
