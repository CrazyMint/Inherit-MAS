"""Shared exact-snapshot cache for benchmark-neutral Inherit-MAS V2.

The cache is deliberately semantic-free: callers provide the actual ordered
messages and every execution determinant. A hit requires an identical canonical
request digest. Stored snapshots are immutable and protected by a second digest;
corruption or conflicting values fail loudly instead of degrading to a miss.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "inherit_mas_resolved_request/2"
ENTRY_SCHEMA = "inherit_mas_exact_snapshot/2"
_KEY = re.compile(r"^[0-9a-f]{64}$")


class CacheError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    """Canonical JSON used by both request and value identities."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CacheError(f"cache payload is not canonical JSON: {exc}") from exc


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def resolved_request(
    *,
    benchmark: str,
    task_id: str,
    messages: list[dict],
    model: dict,
    decode: dict,
    tools: list | dict,
    environment: str | dict,
    node_function: str | dict,
) -> dict:
    """Build the cache identity from the request that is actually executed.

    Ordered message and tool lists remain ordered. Dictionary keys are sorted
    only during canonical serialization. ``node_function`` binds parsing and
    deterministic post-processing that turn the raw model response into the
    stored node snapshot.
    """
    if not isinstance(benchmark, str) or not benchmark:
        raise CacheError("benchmark must be a non-empty string")
    if not isinstance(task_id, str) or not task_id:
        raise CacheError("task_id must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise CacheError("messages must be a non-empty ordered list")
    normalized_messages = []
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or not isinstance(message["role"], str)
                or not isinstance(message["content"], str)):
            raise CacheError("each message must contain string role/content fields")
        normalized_messages.append(dict(message))
    if not isinstance(model, dict) or not model:
        raise CacheError("model fingerprint must be a non-empty mapping")
    if not isinstance(decode, dict):
        raise CacheError("decode must be a mapping")
    body = {
        "schema": REQUEST_SCHEMA,
        "benchmark": benchmark,
        "task_id": task_id,
        "messages": normalized_messages,
        "model": copy.deepcopy(model),
        "decode": copy.deepcopy(decode),
        "tools": copy.deepcopy(tools),
        "environment": copy.deepcopy(environment),
        "node_function": copy.deepcopy(node_function),
    }
    canonical_json(body)
    return body


class ExactSnapshotStore:
    """Persistent, content-addressed, fail-loud snapshot store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(request: dict) -> str:
        if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
            raise CacheError(f"request must use schema {REQUEST_SCHEMA}")
        return content_digest(request)

    def path(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise CacheError("cache key must be a 64-character lowercase hex digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            raise CacheError(f"cannot read cache entry {path}: {exc}") from exc
        required = {"schema", "key", "snapshot_digest", "snapshot"}
        if not isinstance(record, dict) or set(record) != required:
            raise CacheError(f"invalid cache record shape: {path}")
        if record["schema"] != ENTRY_SCHEMA or record["key"] != key:
            raise CacheError(f"cache key/schema mismatch: {path}")
        snapshot = record["snapshot"]
        if not isinstance(snapshot, dict) or snapshot.get("request_digest") != key:
            raise CacheError(f"snapshot request digest mismatch: {path}")
        if record["snapshot_digest"] != content_digest(snapshot):
            raise CacheError(f"snapshot content digest mismatch: {path}")
        return copy.deepcopy(snapshot)

    def put(self, key: str, snapshot: dict) -> None:
        self.path(key)  # Validate before inspecting the snapshot.
        if not isinstance(snapshot, dict) or snapshot.get("request_digest") != key:
            raise CacheError("snapshot request digest does not match its cache key")
        canonical_json(snapshot)
        previous = self.get(key)
        if previous is not None:
            if previous != snapshot:
                raise CacheError("identical request produced conflicting snapshots")
            return
        record = {
            "schema": ENTRY_SCHEMA,
            "key": key,
            "snapshot_digest": content_digest(snapshot),
            "snapshot": copy.deepcopy(snapshot),
        }
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(canonical_json(record))
            os.replace(temporary, path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise CacheError(f"cannot persist cache entry {path}: {exc}") from exc


__all__ = ["CacheError", "ENTRY_SCHEMA", "ExactSnapshotStore",
           "REQUEST_SCHEMA", "canonical_json", "content_digest",
           "resolved_request"]
