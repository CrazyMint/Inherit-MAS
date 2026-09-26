"""Fetch pinned external code and apply the bundled benchmark adapters."""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "evomas": ("https://github.com/amazon-science/EvoMAS.git", "93fd9d6766b093f6bbdeeffc35896910cbece6e2"),
    "tacomas": ("https://github.com/chenxu2-gif/TacoMAS-MultiAgent.git", "6f0d545f2493cf95d2eb6a325d1a6686acf658eb"),
}


def git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed; check the repository, pinned commit, and adapter patch")
    return result.stdout.strip()


def scrub_credentials(root: Path) -> None:
    path = root / "scripts" / "run_evolution.py"
    text = path.read_text()
    text = re.sub(r'(?m)^(\s*)os\.environ\.setdefault\(\s*"([A-Z0-9_]*_KEY)"\s*,\s*"[^"\n]{8,}"\s*\)\s*$',
                  r'\1# \2 must be supplied through the environment.', text)
    text = re.sub(r'(?m)^(\s*#?\s*)([A-Z0-9_]*_KEY)\s*=\s*os\.environ\.get\(\s*"([A-Z0-9_]*_KEY)"\s*,\s*"[^"\n]{8,}"\s*\)\s*$',
                  r'\1\2 = os.environ.get("\3")', text)
    for match in re.finditer(r'"(?:tvly|sk-|pk-|AIza)[A-Za-z0-9._-]{12,}"', text):
        if not re.search(r"YOUR|HERE|EXAMPLE|XXXX|\.\.\.", match.group(), re.I):
            raise RuntimeError("upstream credential cleanup failed; refusing to assemble baseline")
    path.write_text(text)


def setup(name: str, source: str | None = None) -> Path:
    url, commit = PINS[name]
    parent = ROOT / "baselines" / name
    target = parent / "upstream"
    sys.path.insert(0, str(ROOT))
    locks = importlib.import_module(f"baselines.{name}.runtime_lock")
    overlay = parent / "overlay"
    suffix = ".edits.json" if name == "tacomas" else ".patch"
    patch = ROOT / "baselines" / "patches" / f"{name}{suffix}"
    if not overlay.is_dir() or not patch.is_file():
        raise RuntimeError(f"missing bundled {name} adapter files")
    fingerprint = {"commit": commit, "patch": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "overlay": {str(path.relative_to(overlay)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(overlay.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}}
    if name == "evomas":
        from baselines.evomas.prepare import source_identity
        fingerprint["workbench_data"] = source_identity()
    else:
        from scripts import forward_edits
        fingerprint["edit_applier"] = hashlib.sha256(Path(forward_edits.__file__).read_bytes()).hexdigest()
        if json.loads(patch.read_text()).get("upstream_commit") != commit:
            raise RuntimeError("forward-edit manifest commit mismatch")
    if target.exists():
        locks.verify_lock(target)
        stamp = target / "PUBLIC_SETUP.json"
        if not stamp.is_file() or json.loads(stamp.read_text()) != fingerprint:
            raise RuntimeError(f"{name} adapter setup changed; move the existing upstream directory aside and rerun setup")
        return target
    with tempfile.TemporaryDirectory(prefix=".baseline-stage-", dir=parent) as temp:
        stage = Path(temp) / "upstream"
        stage.mkdir()
        git(stage, "init", "-q")
        git(stage, "fetch", "-q", "--depth", "1", source or url, commit)
        git(stage, "checkout", "-q", "--detach", "FETCH_HEAD")
        if git(stage, "rev-parse", "HEAD") != commit:
            raise RuntimeError("upstream commit mismatch")
        if name == "tacomas":
            forward_edits.apply_edits(stage, patch)
        else:
            git(stage, "apply", "--whitespace=nowarn", str(patch))
        for path in sorted(overlay.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("adapter overlays cannot contain symlinks")
            if path.is_file() and "__pycache__" not in path.parts:
                destination = stage / path.relative_to(overlay)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        if name == "tacomas":
            scrub_credentials(stage)
        if name == "evomas":
            from baselines.evomas.prepare import prepare_workbench
            prepare_workbench(stage)
        locks.write_lock(stage)
        locks.verify_lock(stage)
        (stage / "PUBLIC_SETUP.json").write_text(json.dumps(fingerprint, sort_keys=True, indent=2) + "\n")
        stage.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", metavar="{evomas,tacomas}")
    parser.add_argument("--source", action="append", default=[], metavar="METHOD=LOCAL_GIT_REPO",
                        help="Optional local upstream checkout, instead of downloading")
    args = parser.parse_args()
    if any(name not in PINS for name in args.methods):
        parser.error("methods must be evomas or tacomas")
    sources = {}
    for entry in args.source:
        name, separator, path = entry.partition("=")
        if name not in PINS or not separator or not Path(path).is_dir():
            parser.error("--source must be evomas=PATH or tacomas=PATH to a local Git repository")
        sources[name] = str(Path(path).resolve())
    try:
        for name in (args.methods or PINS):
            target = setup(name, sources.get(name))
            print(f"Ready: {name} at {target}")
    except (RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
