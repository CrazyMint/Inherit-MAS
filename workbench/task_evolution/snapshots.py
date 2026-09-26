"""Content-addressed node snapshots with full read-artifact rehydration.

V2 uses the benchmark-neutral exact store. The local aliases preserve the
controller API and record-digest helpers without making V1 cache files part of
the V2 namespace.
"""
from __future__ import annotations

from inherit_mas.cache import (CacheError as SnapshotError,
                                    ExactSnapshotStore as SnapshotStore,
                                    canonical_json as canonical,
                                    content_digest as digest)


class RoundArtifacts:
    """Rehydrates full immutable tool observations from live or cached workers."""

    def __init__(self):
        self.by_id: dict[str, dict] = {}

    def ingest(self, node_id: str, tool_events: list[dict]) -> list[str]:
        ids = []
        for index, event in enumerate(tool_events):
            artifact_id = f"{node_id}:{index}:{digest(event)[:12]}"
            existing = self.by_id.get(artifact_id)
            if existing is not None and existing != event:
                raise SnapshotError("artifact id collision")
            self.by_id[artifact_id] = event
            ids.append(artifact_id)
        return ids

    def manifest(self) -> list[dict]:
        return [{"artifact_id": aid, "content_digest": digest(event)} for aid, event in sorted(self.by_id.items())]
