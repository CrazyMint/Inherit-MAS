"""No-download setup checks."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("public_setup", Path(__file__).parents[1] / "scripts/setup_baselines.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


@pytest.mark.parametrize("args,expected", [([], ["evomas", "tacomas"]), (["evomas"], ["evomas"]),
                                           (["tacomas", "evomas"], ["tacomas", "evomas"])])
def test_setup_selects_methods_without_network(monkeypatch, args, expected):
    calls = []
    monkeypatch.setattr(setup, "setup", lambda name, source: calls.append(name) or Path(name))
    monkeypatch.setattr(sys, "argv", ["setup_baselines.py", *args])
    assert setup.main() == 0
    assert calls == expected


def test_unknown_setup_method_fails_before_fetch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["setup_baselines.py", "unknown"])
    with pytest.raises(SystemExit) as exc:
        setup.main()
    assert exc.value.code == 2


def test_credential_scrub_uses_assignment_not_secret_value(tmp_path):
    path = tmp_path / "scripts/run_evolution.py"
    path.parent.mkdir()
    fake = "tvly-" + "a" * 32
    path.write_text('import os\nos.environ.setdefault("TAVILY_API_KEY", "' + fake + '")\n')
    setup.scrub_credentials(tmp_path)
    assert fake not in path.read_text()
    assert "TAVILY_API_KEY must be supplied" in path.read_text()


def _forward_manifest(tmp_path):
    old, new = b"first\nsecond\n", b"first\nreplacement\n"
    (tmp_path / "a.py").write_bytes(old)
    manifest = {"schema": "forward-line-edits/1", "files": [{
        "path": "a.py", "before_sha256": hashlib.sha256(old).hexdigest(),
        "after_sha256": hashlib.sha256(new).hexdigest(),
        "edits": [{"start": 1, "end": 2, "replacement": "replacement\n"}]}]}
    return old, new, manifest


def test_forward_edits_contain_no_original_source(tmp_path):
    from scripts.forward_edits import apply_edits
    _, new, manifest = _forward_manifest(tmp_path)
    path = tmp_path / "edits.json"
    path.write_text(json.dumps(manifest))
    assert "second" not in path.read_text()
    apply_edits(tmp_path, path)
    assert (tmp_path / "a.py").read_bytes() == new


@pytest.mark.parametrize("fault", ["source_hash", "result_hash", "range", "overlap", "path", "duplicate"])
def test_forward_edits_fail_closed(tmp_path, fault):
    from scripts.forward_edits import apply_edits
    old, _, manifest = _forward_manifest(tmp_path)
    spec = manifest["files"][0]
    if fault == "source_hash":
        spec["before_sha256"] = "0" * 64
    elif fault == "result_hash":
        spec["after_sha256"] = "0" * 64
    elif fault == "range":
        spec["edits"][0]["start"] = -1
    elif fault == "overlap":
        spec["edits"].append(spec["edits"][0])
    elif fault == "path":
        spec["path"] = "../outside.py"
    else:
        manifest["files"].append(spec)
    path = tmp_path / "edits.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        apply_edits(tmp_path, path)
    assert (tmp_path / "a.py").read_bytes() == old


def test_taco_release_omits_upstream_prompt_and_context_patch():
    root = Path(__file__).parents[1]
    assert not (root / "baselines/patches/tacomas.patch").exists()
    assert not (root / "baselines/tacomas/overlay/prompts/dataset-shared/workbench_role_aware.yaml").exists()
    manifest = json.loads((root / "baselines/patches/tacomas.edits.json").read_text())
    assert manifest["upstream_commit"] == setup.PINS["tacomas"][1]
    for spec in manifest["files"]:
        assert set(spec) == {"path", "before_sha256", "after_sha256", "edits"}
        assert all(set(edit) == {"start", "end", "replacement"} for edit in spec["edits"])


def test_evomas_excerpts_have_separate_license_notice():
    root = Path(__file__).parents[1]
    license_text = (root / "baselines/patches/EVOMAS_LICENSE.txt").read_text()
    notice = (root / "baselines/patches/EVOMAS_NOTICE.md").read_text()
    assert "Attribution-NonCommercial 4.0 International" in license_text
    assert "evomas.patch" in notice and "not the MIT license" in notice
