from __future__ import annotations

import json

import pytest

from .cache import (CacheError, ExactSnapshotStore, content_digest,
                    resolved_request)


def request(**overrides):
    values = {
        "benchmark": "bench",
        "task_id": "task-1",
        "messages": [{"role": "system", "content": "be exact"},
                     {"role": "user", "content": "solve"}],
        "model": {"id": "worker", "revision": "r1"},
        "decode": {"temperature": 0, "max_tokens": 100},
        "tools": [{"name": "search", "schema": {"q": "string"}}],
        "environment": {"state": "pristine", "revision": "e1"},
        "node_function": {"runtime": "v2", "parser": "evidence/1"},
    }
    values.update(overrides)
    return resolved_request(**values)


def snapshot(key, text="answer"):
    return {"request_digest": key, "output": text, "tool_events": [],
            "usage": {"total_tokens": 12}, "error": ""}


def test_request_key_covers_every_execution_determinant(tmp_path):
    store = ExactSnapshotStore(tmp_path)
    base = store.key(request())
    changes = [
        request(task_id="task-2"),
        request(messages=[{"role": "system", "content": "be exact"},
                          {"role": "user", "content": "different"}]),
        request(model={"id": "worker", "revision": "r2"}),
        request(decode={"temperature": 0, "max_tokens": 101}),
        request(tools=[{"name": "lookup", "schema": {"q": "string"}}]),
        request(environment={"state": "changed", "revision": "e1"}),
        request(node_function={"runtime": "v2", "parser": "evidence/2"}),
    ]
    assert all(store.key(value) != base for value in changes)


def test_message_order_is_significant_but_mapping_order_is_not(tmp_path):
    store = ExactSnapshotStore(tmp_path)
    first = request(model={"id": "worker", "revision": "r1"})
    reordered_mapping = request(model={"revision": "r1", "id": "worker"})
    reversed_messages = request(messages=list(reversed(first["messages"])))
    assert store.key(first) == store.key(reordered_mapping)
    assert store.key(first) != store.key(reversed_messages)


def test_hit_returns_exact_snapshot_and_conflicting_put_fails(tmp_path):
    store = ExactSnapshotStore(tmp_path)
    key = store.key(request())
    value = snapshot(key)
    assert store.get(key) is None
    store.put(key, value)
    assert store.get(key) == value
    store.put(key, value)
    with pytest.raises(CacheError, match="conflicting"):
        store.put(key, snapshot(key, "different"))


def test_corruption_fails_loud_instead_of_becoming_a_miss(tmp_path):
    store = ExactSnapshotStore(tmp_path)
    key = store.key(request())
    store.put(key, snapshot(key))
    path = store.path(key)
    record = json.loads(path.read_text())
    record["snapshot"]["output"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(CacheError, match="content digest"):
        store.get(key)


def test_non_json_request_is_rejected():
    with pytest.raises(CacheError, match="canonical JSON"):
        request(environment={"bad": object()})
    with pytest.raises(CacheError, match="canonical JSON"):
        request(environment={"bad": float("nan")})
    assert len(content_digest({"stable": True})) == 64
