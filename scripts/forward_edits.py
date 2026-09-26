"""Apply pinned, hash-checked line edits without storing the original source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def apply_edits(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "forward-line-edits/1" or not isinstance(manifest.get("files"), list):
        raise RuntimeError("unsupported forward-edit manifest")
    root = root.resolve()
    pending = {}
    for spec in manifest["files"]:
        relative = Path(spec["path"])
        target = root / relative
        if (relative.is_absolute() or ".." in relative.parts or target in pending
                or target.resolve() != target or not target.is_file()):
            raise RuntimeError("invalid forward-edit target")
        original = target.read_bytes()
        if hashlib.sha256(original).hexdigest() != spec["before_sha256"]:
            raise RuntimeError(f"pinned source mismatch: {relative}")
        lines = original.decode("utf-8").splitlines(keepends=True)
        cursor, chunks = 0, []
        for edit in spec["edits"]:
            start, end, replacement = edit["start"], edit["end"], edit["replacement"]
            if (type(start) is not int or type(end) is not int or not isinstance(replacement, str)
                    or not cursor <= start <= end <= len(lines)):
                raise RuntimeError(f"invalid forward-edit range: {relative}")
            chunks.extend(("".join(lines[cursor:start]), replacement))
            cursor = end
        chunks.append("".join(lines[cursor:]))
        updated = "".join(chunks).encode("utf-8")
        if hashlib.sha256(updated).hexdigest() != spec["after_sha256"]:
            raise RuntimeError(f"forward-edit result mismatch: {relative}")
        pending[target] = updated
    # Validate the whole set before changing any file in the temporary checkout.
    for target, updated in pending.items():
        target.write_bytes(updated)
