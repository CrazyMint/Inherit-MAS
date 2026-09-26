"""Pinned HotpotQA distractor loader with separate public and scoring views.

``PublicExample`` exposes questions and context without answers or supporting
facts. ``Example`` adds those fields for scoring. Projection helpers validate raw
records, snapshot hashes identify the data, and ID hashes determine partitions.
This module reads local snapshots and does not download datasets.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

Paragraph = Tuple[str, List[str]]

DATASET_REPO = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATASET_BUILDER_VERSION = "0.0.0"
SOURCE_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class PublicExample:
    """AGENT-visible: NO gold answer / supporting_facts exist on this object."""
    id: str
    question: str
    context: List[Paragraph]          # ordered [(title, [sentences]), ...]
    level: str = ""
    qtype: str = ""

    def paragraphs_by_title(self) -> Dict[str, List[str]]:
        return {t: s for t, s in self.context}


@dataclass(frozen=True)
class Example(PublicExample):
    """Scorer-only view (gated): gold answer + supporting facts. NEVER shown to an
    agent/proposer/router/acceptance prompt."""
    answer: str = ""
    supporting_facts: List[Tuple[str, int]] = field(default_factory=list)


class LoaderError(ValueError):
    """A malformed raw HotpotQA record."""


_REQUIRED_PUBLIC_COLS = ("question", "context")


def _norm_context(raw_context: Any) -> List[Paragraph]:
    """Normalize the HF distractor context to [(title, [sentences]), ...] with
    STRICT validation: equal parallel-list lengths, unique titles, string sentences."""
    if isinstance(raw_context, dict):
        if "title" not in raw_context or "sentences" not in raw_context:
            raise LoaderError("context dict needs 'title' and 'sentences'")
        titles, sents = raw_context["title"], raw_context["sentences"]
        if len(titles) != len(sents):
            raise LoaderError(f"context parallel lists unequal ({len(titles)} vs {len(sents)})")
        pairs = list(zip(titles, sents))
    else:
        pairs = list(raw_context)
    out = []
    for pr in pairs:
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            raise LoaderError(f"bad paragraph {pr!r}")
        t, s = pr
        if not isinstance(s, (list, tuple)):
            raise LoaderError(f"paragraph sentences must be a list: {t!r}")
        out.append((str(t), [str(x) for x in s]))
    seen = [t for t, _ in out]
    if len(seen) != len(set(seen)):
        raise LoaderError(f"duplicate context titles: {seen}")
    if not out:
        raise LoaderError("context is empty")
    return out


def _norm_supporting(raw_sf: Any, context: List[Paragraph]) -> List[Tuple[str, int]]:
    """Normalize supporting_facts to [(title, sent_id), ...] and VALIDATE each
    (title, sent_id) resolves to a real sentence in the context."""
    if isinstance(raw_sf, dict):
        if "title" not in raw_sf or "sent_id" not in raw_sf:
            raise LoaderError("supporting_facts dict needs 'title' and 'sent_id'")
        if len(raw_sf["title"]) != len(raw_sf["sent_id"]):
            raise LoaderError("supporting_facts parallel lists unequal")
        pairs = list(zip(raw_sf["title"], raw_sf["sent_id"]))
    else:
        pairs = list(raw_sf)
    by_title = {t: len(s) for t, s in context}
    out = []
    for t, i in pairs:
        if isinstance(i, bool) or not isinstance(i, int):   # sent_id must be a non-bool int
            raise LoaderError(f"supporting-fact sent_id must be a non-bool int: ({t!r}, {i!r})")
        t = str(t)
        if t not in by_title or not (0 <= i < by_title[t]):
            raise LoaderError(f"invalid supporting fact ({t!r}, {i}) — no such sentence")
        out.append((t, i))
    return out


def _require_id(raw: Dict[str, Any]) -> str:
    rid = raw.get("id") or raw.get("_id") or raw.get("question_id")
    if not rid or not str(rid).strip():
        raise LoaderError("record has an empty/missing id")
    return str(rid)


def project_public(raw: Dict[str, Any]) -> PublicExample:
    """Raw HF record -> PublicExample with STRICT validation (required public
    columns, id, context). Reads ONLY public fields; gold is never touched."""
    for col in _REQUIRED_PUBLIC_COLS:
        if col not in raw:
            raise LoaderError(f"missing required public column {col!r}")
    if not str(raw["question"]).strip():
        raise LoaderError("empty question")
    return PublicExample(id=_require_id(raw), question=str(raw["question"]),
                         context=_norm_context(raw["context"]),
                         level=str(raw.get("level", "")), qtype=str(raw.get("type", "")))


def project_full(raw: Dict[str, Any]) -> Example:
    """Raw HF record -> Example WITH gold (scorer-only; gated). Validates gold
    supporting facts against the context."""
    p = project_public(raw)
    if "answer" not in raw:
        raise LoaderError("full record missing gold 'answer'")
    if "supporting_facts" not in raw:                   # required on full records
        raise LoaderError("full record missing gold 'supporting_facts'")
    return Example(id=p.id, question=p.question, context=p.context, level=p.level,
                   qtype=p.qtype, answer=str(raw["answer"]),
                   supporting_facts=_norm_supporting(raw["supporting_facts"], p.context))


def load_public_examples(records: List[Dict[str, Any]]) -> List[PublicExample]:
    """Project raw records to public examples and reject duplicate IDs."""
    out = [project_public(r) for r in records]
    ids = [e.id for e in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example ids in HotpotQA records")
    return out


# ------------------------------------------------------- pinning + deterministic splits
def combined_sha256(records: List[Dict[str, Any]]) -> str:
    """Pin a snapshot: sha256 over the canonical JSON of the (id-sorted) raw
    records — so a changed dataset revision / preprocessing fails the pin."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: str(x.get("id") or x.get("_id"))):
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


SPLITS = ("development", "adaptive_search", "confirmation", "final_report")


def split_of(example_id: str) -> str:
    """Deterministic disjoint partition by id hash (fixed before any search)."""
    b = int(hashlib.blake2b(str(example_id).encode(), digest_size=8).hexdigest(), 16) % 10
    if b <= 3:
        return "development"          # 0-3
    if b <= 6:
        return "adaptive_search"      # 4-6
    if b <= 8:
        return "confirmation"         # 7-8
    return "final_report"             # 9 (untouched)


def source_split_of(source: str, example_id: str) -> str:
    """Deterministic partition within each official source split.

    Official training examples are the only development/search pool. Official
    validation examples are reserved for confirmation/final reporting, so model
    selection cannot gradually consume the held-out source split.
    """
    if source not in SOURCE_SPLITS:
        raise ValueError(f"unknown official source split {source!r}")
    bucket = int(hashlib.blake2b(
        f"hotpot-source-v1\0{source}\0{example_id}".encode(), digest_size=8
    ).hexdigest(), 16) % 10
    if source == "train":
        return "development" if bucket < 5 else "adaptive_search"
    return "confirmation" if bucket < 5 else "final_report"


@dataclass(frozen=True)
class PinnedSnapshot:
    root: Path
    train_files: Tuple[Path, ...]
    validation_files: Tuple[Path, ...]
    file_sha256: Dict[str, str]

    @property
    def digest(self) -> str:
        payload = json.dumps({
            "repo": DATASET_REPO,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "files": self.file_sha256,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_snapshot_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return (hf_home / "datasets" / "hotpot_qa" / DATASET_CONFIG /
            DATASET_BUILDER_VERSION / DATASET_REVISION)


def open_pinned_snapshot(root: str | Path | None = None) -> PinnedSnapshot:
    """Open the already-materialized pinned Arrow snapshot and verify its shape.

    This never falls back to a network download or to a moving ``main`` ref.
    """
    base = Path(root) if root is not None else default_snapshot_root()
    train = tuple(sorted(base.glob("hotpot_qa-train-*.arrow")))
    validation = tuple(sorted(base.glob("hotpot_qa-validation*.arrow")))
    if len(train) != 2 or len(validation) != 1:
        raise LoaderError(
            f"pinned snapshot incomplete at {base}: want 2 train + 1 validation "
            "Arrow files. run: python scripts/download_hotpotqa.py (about 570 MB), "
            "or point HF_HOME at an existing copy."
        )
    files = train + validation
    hashes = {p.name: _sha256_file(p) for p in files}
    return PinnedSnapshot(base, train, validation, hashes)


def _datasets_for(snapshot: PinnedSnapshot, source: str):
    if source not in SOURCE_SPLITS:
        raise ValueError(f"unknown source {source!r}")
    from datasets import Dataset, concatenate_datasets

    files = snapshot.train_files if source == "train" else snapshot.validation_files
    parts = [Dataset.from_file(str(path)) for path in files]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def public_manifest_rows(snapshot: PinnedSnapshot) -> List[Dict[str, str]]:
    """Build split membership from public metadata only; never read gold columns."""
    rows: List[Dict[str, str]] = []
    seen = set()
    for source in SOURCE_SPLITS:
        ds = _datasets_for(snapshot, source).select_columns(["id", "type", "level"])
        for row in ds:
            rid = _require_id(row)
            if rid in seen:
                raise LoaderError(f"duplicate id across snapshot: {rid}")
            seen.add(rid)
            rows.append({
                "id": rid,
                "source": source,
                "split": source_split_of(source, rid),
                "type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
            })
    return rows


def load_examples(snapshot: PinnedSnapshot, rows: Iterable[Dict[str, str]]) -> List[Example]:
    """Load scorer-side examples named by a frozen manifest, preserving its order."""
    wanted = list(rows)
    by_source: Dict[str, set[str]] = {s: set() for s in SOURCE_SPLITS}
    for row in wanted:
        source, rid = row["source"], row["id"]
        if source not in by_source:
            raise LoaderError(f"manifest has unknown source {source!r}")
        by_source[source].add(rid)
    found: Dict[str, Example] = {}
    for source, ids in by_source.items():
        if not ids:
            continue
        ds = _datasets_for(snapshot, source)
        id_col = ds["id"]
        index = {rid: i for i, rid in enumerate(id_col) if rid in ids}
        missing = ids - set(index)
        if missing:
            raise LoaderError(f"manifest ids absent from {source}: {sorted(missing)[:5]}")
        for rid, i in index.items():
            found[rid] = project_full(ds[int(i)])
    return [found[row["id"]] for row in wanted]


def split(examples, which: str):
    if which not in SPLITS:
        raise ValueError(f"unknown split {which!r}; want {SPLITS}")
    return [e for e in examples if split_of(e.id) == which]
