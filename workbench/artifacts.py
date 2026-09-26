"""Immutable per-task artifact store and reference resolver.

Bottleneck this addresses: workers retrieve a correct-but-large read-tool result and
then TRUNCATE while copying it into evidence JSON (budget_exhausted / schema_invalid),
and the bulky worker->integrator text channel bloats the integrator input. Instead:

  * Every successful read-tool result is stored ONCE as an immutable Artifact
    (artifact_id, tool, canonical args, result bytes, environment digest, content
    digest, task id).
  * Workers return compact EvidenceRefs (artifact_id + optional JSON path + concise
    fact + provenance + confidence) — never the bulk value.
  * A deterministic resolver dereferences refs immediately before the sole executor
    calls the write tool, REJECTING unknown / cross-task / stale / type-incompatible /
    out-of-scope references.

This channel also aligns with the future state-aware cache: a cached tool transition
can return an artifact reference instead of re-serializing large environment state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    tool: str
    args: dict
    result: str          # the read-tool result bytes (JSON/text the tool returned)
    env_digest: str      # environment snapshot digest at creation
    content_digest: str  # digest of `result`
    task_id: str


class ArtifactStore:
    """Per-task, append-only, immutable. artifact_ids are deterministic (art001, ...)."""

    def __init__(self, task_id: str, env_digest: str):
        self.task_id = task_id
        self.env_digest = env_digest
        self._by_id: dict[str, Artifact] = {}
        self._by_key: dict[str, str] = {}   # (tool, canon args) -> id (dedupe identical reads)
        self._n = 0

    def put(self, tool: str, args: dict, result: str) -> str:
        key = tool + "|" + _canon(args)
        if key in self._by_key:                 # identical read -> same immutable artifact
            return self._by_key[key]
        self._n += 1
        aid = f"art{self._n:03d}"
        self._by_id[aid] = Artifact(aid, tool, dict(args), result, self.env_digest,
                                    _digest(result), self.task_id)
        self._by_key[key] = aid
        return aid

    def get(self, aid: str) -> Artifact | None:
        return self._by_id.get(aid)

    def __len__(self):
        return len(self._by_id)


# ------------------------------------------------------------------ JSON path select
def select_path(result: str, path: str | None):
    """Deterministically select a value from a read-tool result via a small path
    grammar: dotted keys + [i] indices, optional leading '$' or '$.'. e.g.
    "events[0].event_id", "$.total", "0.customer_email". Returns (value, ok)."""
    try:
        obj = json.loads(result)
    except Exception:  # noqa: BLE001 — a plain-string result: only the empty path is valid
        return (result, True) if not path else (None, False)
    if not path:
        return obj, True
    p = path.strip()
    if p.startswith("$"):
        p = p[1:]
    p = p.lstrip(".")
    cur = obj
    for raw in _tokens(p):
        if isinstance(raw, int):
            if not isinstance(cur, list) or not (-len(cur) <= raw < len(cur)):
                return None, False
            cur = cur[raw]
        else:
            if not isinstance(cur, dict) or raw not in cur:
                return None, False
            cur = cur[raw]
    return cur, True


def _tokens(path: str):
    out: list = []
    for part in path.split("."):
        if not part:
            continue
        while "[" in part:                      # split "events[0]" -> "events", 0
            head, _, rest = part.partition("[")
            idx, _, part = rest.partition("]")
            if head:
                out.append(head)
            try:
                out.append(int(idx))
            except ValueError:
                out.append(idx)
        if part:
            out.append(part)
    return out


def _is_scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


# ------------------------------------------------------------------ resolve a reference
def resolve_ref(store: ArtifactStore, ref: dict, *, task_id: str, env_digest: str, scope: set | None = None):
    """Dereference a validated artifact reference to a concrete SCALAR write-arg value.
    Returns (value, None) on success or (None, reason) on rejection. Reasons:
    unknown_reference | cross_task | stale | out_of_scope | type_incompatible."""
    aid = ref.get("artifact_id") or ref.get("$artifact")
    path = ref.get("path") or ref.get("$path")
    art = store.get(aid) if aid else None
    if art is None:
        return None, "unknown_reference"
    if art.task_id != task_id:
        return None, "cross_task"
    if art.env_digest != env_digest:
        return None, "stale"
    if scope is not None and aid not in scope:      # only refs surfaced by this run's workers
        return None, "out_of_scope"
    val, ok = select_path(art.result, path)
    if not ok:
        return None, "type_incompatible"
    if not _is_scalar(val):                          # a write-arg must resolve to a scalar
        return None, "type_incompatible"
    return ("" if val is None else str(val)), None


def is_ref(v) -> bool:
    return isinstance(v, dict) and ("artifact_id" in v or "$artifact" in v)


# ------------------------------------------------------------------ typed manifest
@dataclass(frozen=True)
class ManifestEntry:
    """A TYPED catalog entry for one produced artifact — the deterministic, bulk-free
    handle the aggregation planner/reducer use. Carries provenance + shape + digests,
    NEVER the table rows themselves."""
    artifact_id: str
    tool: str
    args: dict
    task_id: str
    env_digest: str          # snapshot digest
    content_digest: str
    kind: str                # "table" | "scalar" | "opaque"
    columns: tuple | None    # sorted column names for a table, else None
    n_rows: int | None


def _shape(result: str):
    """Deterministic (kind, columns, n_rows) for an artifact result — no bulk data."""
    try:
        obj = json.loads(result)
    except Exception:  # noqa: BLE001
        return "opaque", None, None
    if isinstance(obj, dict):
        obj = [obj]
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        cols = tuple(sorted({k for r in obj for k in r}))
        return "table", cols, len(obj)
    if isinstance(obj, list) and not obj:
        return "table", (), 0
    return "scalar", None, None


def build_manifest(store: ArtifactStore, artifact_ids=None) -> tuple:
    """Deterministically expose ALL produced artifacts (or a given subset) as a typed
    manifest, INDEPENDENT of whichever EvidenceRefs a worker chose to surface. Sorted by
    artifact_id for a stable identity."""
    ids = sorted(store._by_id if artifact_ids is None else artifact_ids)
    out = []
    for aid in ids:
        a = store.get(aid)
        if a is None:
            continue
        kind, cols, n = _shape(a.result)
        out.append(ManifestEntry(a.artifact_id, a.tool, dict(a.args), a.task_id, a.env_digest,
                                 a.content_digest, kind, cols, n))
    return tuple(out)


def manifest_index(manifest) -> dict:
    return {e.artifact_id: e for e in manifest}


def manifest_digest(manifest) -> str:
    """A stable digest binding the manifest's COMPLETE canonical identity — every
    ManifestEntry field (artifact_id, tool, canonical args, task_id, env_digest,
    content_digest, kind, columns, n_rows) — into planner/reducer request + cache
    identities. A change to any of these (e.g. a different tool or different args that
    produced the same bytes) yields a different digest."""
    payload = [[e.artifact_id, e.tool, _canon(e.args), e.task_id, e.env_digest,
                e.content_digest, e.kind, list(e.columns) if e.columns is not None else None, e.n_rows]
               for e in sorted(manifest, key=lambda e: e.artifact_id)]
    return _digest(_canon(payload))
