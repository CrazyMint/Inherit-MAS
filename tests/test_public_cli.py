"""Offline regression tests for the public run and score commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run as cli
from hotpot_fullwiki.common import digest
from public_runner import common, hotpot


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def manifest_file(tmp_path: Path, benchmark: str = "hotpotqa", *, ids=None,
                  count: int | None = None) -> Path:
    ids = ["task-a", "task-b"] if ids is None else ids
    key = "tasks" if benchmark == "workbench" else "rows"
    rows = [{"id": task_id, "task": "A public task"} for task_id in ids]
    value = {"dataset_revision": "snapshot-digest", "n": len(rows) if count is None else count,
             key: rows}
    value["manifest_digest"] = digest(value)
    return write_json(tmp_path / "manifest.json", value)


@pytest.mark.parametrize("benchmark", ["workbench", "hotpotqa"])
def test_manifest_limit_updates_count_and_digest(tmp_path, benchmark):
    path = manifest_file(tmp_path, benchmark)
    full = common.read_manifest(path, benchmark)
    limited = common.read_manifest(path, benchmark, limit=1)
    key = "tasks" if benchmark == "workbench" else "rows"
    assert limited["n"] == 1
    assert limited[key] == full[key][:1]
    assert limited["manifest_digest"] != full["manifest_digest"]
    assert limited["manifest_digest"] == digest(
        {key: value for key, value in limited.items() if key != "manifest_digest"})
    assert json.loads(path.read_text())["n"] == 2


@pytest.mark.parametrize("benchmark", ["workbench", "hotpotqa"])
def test_manifest_rejects_changed_content_without_new_digest(tmp_path, benchmark):
    path = manifest_file(tmp_path, benchmark)
    value = json.loads(path.read_text())
    value["n"] = 100
    write_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        common.read_manifest(path, benchmark)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_manifest_rejects_incorrect_count(tmp_path, count):
    with pytest.raises(ValueError, match="task count"):
        common.read_manifest(manifest_file(tmp_path, count=count), "hotpotqa")


def test_manifest_rejects_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="task count"):
        common.read_manifest(manifest_file(tmp_path, ids=[]), "hotpotqa")


def test_manifest_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        common.read_manifest(manifest_file(tmp_path, ids=["same", "same"]), "hotpotqa")


@pytest.mark.parametrize("task_id", ["", ".", "..", "../escape", "a/b", "a\\b", 7, None])
def test_manifest_rejects_unsafe_identifiers(tmp_path, task_id):
    with pytest.raises(ValueError, match="identifier"):
        common.read_manifest(manifest_file(tmp_path, ids=[task_id]), "hotpotqa")


@pytest.mark.parametrize("limit", [0, -1])
def test_manifest_rejects_nonpositive_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="positive"):
        common.read_manifest(manifest_file(tmp_path), "hotpotqa", limit=limit)


def test_binding_allows_same_code_resume_and_rejects_changed_code(tmp_path, monkeypatch):
    manifest = common.read_manifest(manifest_file(tmp_path), "hotpotqa")
    out = tmp_path / "run"
    monkeypatch.setattr(common, "source_identity", lambda: {"core.py": "original"})
    path = common.bind_run(out, "hotpotqa", manifest)
    assert common.bind_run(out, "hotpotqa", manifest) == path
    assert common.verify_binding(out, "hotpotqa") == path
    original_binding = (out / "RUN_BINDING.json").read_text()
    monkeypatch.setattr(common, "source_identity", lambda: {"core.py": "changed"})
    with pytest.raises(RuntimeError, match="different code"):
        common.bind_run(out, "hotpotqa", manifest)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        common.verify_binding(out, "hotpotqa")
    assert (out / "RUN_BINDING.json").read_text() == original_binding


def test_binding_rejects_changed_selection_and_unbound_output(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    full = common.read_manifest(path, "hotpotqa")
    subset = common.read_manifest(path, "hotpotqa", limit=1)
    monkeypatch.setattr(common, "source_identity", lambda: {"core.py": "original"})
    out = tmp_path / "run"
    common.bind_run(out, "hotpotqa", full)
    with pytest.raises(RuntimeError, match="task selection"):
        common.bind_run(out, "hotpotqa", subset)
    unbound = tmp_path / "unbound"
    write_json(unbound / "RUN_CONFIG.json", {"old": True})
    with pytest.raises(RuntimeError, match="unbound"):
        common.bind_run(unbound, "hotpotqa", full)


@pytest.mark.parametrize("change", [{"method": "evoagent"}, {"backbone": "qwen3-32b"}])
def test_binding_rejects_different_method_or_backbone(tmp_path, monkeypatch, change):
    manifest = common.read_manifest(manifest_file(tmp_path), "hotpotqa")
    monkeypatch.setattr(common, "source_identity", lambda: {"core.py": "original"})
    out = tmp_path / "run"
    common.bind_run(out, "hotpotqa", manifest)
    with pytest.raises(RuntimeError, match="different code"):
        common.bind_run(out, "hotpotqa", manifest, **change)


def forbidden(*args, **kwargs):
    raise AssertionError("dry commands must not execute or create a run")


@pytest.mark.parametrize("state", ["absent", "empty", "bad_json", "bad_benchmark", "bad_type", "incomplete"])
def test_score_directory_errors_are_clear(tmp_path, monkeypatch, capsys, state):
    out = tmp_path / "run"
    if state != "absent":
        out.mkdir()
    if state == "bad_json":
        (out / "RUN_BINDING.json").write_text("{")
    elif state == "bad_benchmark":
        write_json(out / "RUN_BINDING.json", {"benchmark": "unknown"})
    elif state == "bad_type":
        write_json(out / "RUN_BINDING.json", {"benchmark": []})
    elif state == "incomplete":
        write_json(out / "RUN_BINDING.json", {"benchmark": "workbench"})
    monkeypatch.setattr(cli, "verify_binding", forbidden)
    monkeypatch.setattr(sys, "argv", ["run.py", "score", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    expected = {"absent": "not a run directory", "empty": "not a run directory",
                "bad_json": "cannot read run metadata", "bad_benchmark": "unsupported", "bad_type": "unsupported",
                "incomplete": "complete the run first"}
    assert expected[state] in err
    assert "Traceback" not in err
    assert not (out / "SCORE.json").exists()


@pytest.mark.parametrize("args", [["--help"], ["run", "--help"], ["score", "--help"]])
def test_help_does_not_execute_or_write(tmp_path, monkeypatch, capsys, args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "bind_run", forbidden)
    monkeypatch.setattr(cli, "verify_binding", forbidden)
    monkeypatch.setattr(sys, "argv", ["run.py", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text
    if args == ["run", "--help"]:
        assert "--task-concurrency" in help_text
        assert "--workers" not in help_text
        assert "not LLM agents per task" in " ".join(help_text.split())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("benchmark", ["workbench", "hotpotqa"])
@pytest.mark.parametrize("flags,expected_concurrency", [([], 1),
    (["--task-concurrency", "3"], 3), (["--workers", "2"], 2)])
def test_dry_run_validates_selection_without_runtime_or_writes(tmp_path, monkeypatch, capsys,
                                                            benchmark, flags, expected_concurrency):
    path = manifest_file(tmp_path, benchmark)
    out = tmp_path / "not-created"
    monkeypatch.setattr(cli, "bind_run", forbidden)
    monkeypatch.setattr(hotpot, "run", forbidden)
    fake_workbench = ModuleType("public_runner.workbench")
    fake_workbench.run = forbidden
    monkeypatch.setitem(sys.modules, "public_runner.workbench", fake_workbench)
    monkeypatch.setattr(sys, "argv", ["run.py", "run", benchmark, "--manifest", str(path),
                                    "--out", str(out), "--usd-cap", "1", "--limit", "1",
                                    "--dry-run", *flags])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == 1
    assert result["model_calls"] is False
    assert result["task_concurrency"] == expected_concurrency
    assert "workers" not in result
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("flags", [["--task-concurrency", "0"],
    ["--task-concurrency", "-1"], ["--workers", "0"],
    ["--task-concurrency", "2", "--workers", "3"]])
def test_invalid_task_concurrency_rejected_before_loading(tmp_path, monkeypatch, capsys, flags):
    monkeypatch.setattr(cli, "read_manifest", forbidden)
    monkeypatch.setattr(sys, "argv", ["run.py", "run", "workbench", "--out", str(tmp_path),
                                    "--usd-cap", "1", *flags])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--task-concurrency" in capsys.readouterr().err


@pytest.mark.parametrize("method", cli.METHODS)
@pytest.mark.parametrize("benchmark", ["workbench", "hotpotqa"])
def test_task_concurrency_reaches_runner(tmp_path, monkeypatch, capsys, method, benchmark):
    from public_runner import workbench

    path = manifest_file(tmp_path, benchmark)
    out = tmp_path / "run"
    monkeypatch.setattr(cli, "bind_run", lambda *args: path)

    def execute(manifest_path, output, *, workers, usd_cap, **kwargs):
        assert (manifest_path, output, workers, usd_cap) == (path, out, 3, 1.0)
        return {"status": "sealed"}

    monkeypatch.setattr(cli, "baseline_runtime", lambda name: SimpleNamespace(run=execute))
    monkeypatch.setattr(hotpot, "run", execute)
    monkeypatch.setattr(workbench, "run", execute)
    monkeypatch.setattr(sys, "argv", ["run.py", "run", benchmark, "--method", method,
        "--manifest", str(path), "--out", str(out), "--usd-cap", "1", "--task-concurrency", "3"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "sealed"


@pytest.mark.parametrize("module_name", ["run_select_edit", "run_revision_v3"])
@pytest.mark.parametrize("flags,expected_concurrency", [([], 4),
    (["--task-concurrency", "3"], 3), (["--workers", "2"], 2)])
def test_standalone_hotpot_task_concurrency(tmp_path, monkeypatch, capsys, module_name,
                                          flags, expected_concurrency):
    from importlib import import_module

    module = import_module(f"hotpot_fullwiki.{module_name}")

    def execute(manifest_path, output, *, workers, usd_cap):
        assert workers == expected_concurrency
        return {"status": "sealed"}

    monkeypatch.setattr(module, "run", execute)
    monkeypatch.setattr(sys, "argv", [module_name, "--manifest", str(tmp_path / "unused.json"),
        "--out", str(tmp_path), "--usd-cap", "1", *flags])
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "sealed"


@pytest.mark.parametrize("method", cli.METHODS)
@pytest.mark.parametrize("backbone", cli.BACKBONES)
def test_baseline_dry_run_needs_no_models_or_upstream(tmp_path, monkeypatch, capsys, method, backbone):
    path = manifest_file(tmp_path)
    out = tmp_path / "not-created"
    monkeypatch.setattr(cli, "baseline_runtime", forbidden)
    monkeypatch.setattr(cli, "bind_run", forbidden)
    monkeypatch.setattr(sys, "argv", ["run.py", "run", "hotpotqa", "--method", method,
        "--backbone", backbone, "--manifest", str(path), "--out", str(out), "--usd-cap", "1", "--dry-run"])
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["method"], report["backbone"]) == (method, backbone)
    assert not out.exists()


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_hotpot_retrieval_preflight_precedes_execution(tmp_path, monkeypatch, fail_preflight):
    from inherit_mas import release_config
    from hotpot_fullwiki import loader, retrieval

    path = manifest_file(tmp_path)
    out = tmp_path / "run"
    events = []
    example = SimpleNamespace(id="task-a", question="A test question")
    monkeypatch.setattr(release_config, "ensure_hf_home", lambda: None)
    monkeypatch.setattr(loader, "open_snapshot", lambda: SimpleNamespace(digest="snapshot-digest"))
    monkeypatch.setattr(loader, "load_examples", lambda rows, snapshot: [
        SimpleNamespace(runtime=lambda: example)])

    class FakeRetriever:
        def __init__(self, *, cache_dir):
            assert cache_dir == out / "retrieval_cache"

        def preflight(self, query, runtime_example):
            assert query == example.question
            assert runtime_example is example
            events.append("preflight")
            if fail_preflight:
                raise retrieval.RetrievalError("offline retrieval unavailable")
            return {"result_count": 1}

    fake_runtime = ModuleType("hotpot_fullwiki.run_select_edit")

    def execute(manifest_path, output, *, usd_cap, workers):
        assert (manifest_path, output, usd_cap, workers) == (path, out, 1.0, 2)
        events.append("execute")
        return {"status": "sealed"}

    fake_runtime.run = execute
    monkeypatch.setitem(sys.modules, "hotpot_fullwiki.run_select_edit", fake_runtime)
    monkeypatch.setattr(retrieval, "BM25Retriever", FakeRetriever)
    if fail_preflight:
        with pytest.raises(retrieval.RetrievalError, match="unavailable"):
            hotpot.run(path, out, usd_cap=1.0, workers=2)
        assert events == ["preflight"]
    else:
        assert hotpot.run(path, out, usd_cap=1.0, workers=2) == {"status": "sealed"}
        assert events == ["preflight", "execute"]


def scoring_run(tmp_path):
    path = manifest_file(tmp_path)
    manifest = json.loads(path.read_text())
    out = tmp_path / "run"
    config = {"config_digest": "configuration", "manifest_digest": manifest["manifest_digest"]}
    write_json(out / "RUN_CONFIG.json", config)
    write_json(out / "SEALED.json", {**config, "expected_units": 2})
    write_json(out / "RUN_STATUS.json", {"status": "sealed"})
    return path, out, config


@pytest.mark.parametrize("mutation", ["missing_seal", "incomplete", "wrong_count", "wrong_config"])
def test_hotpot_score_rejects_unsealed_or_mismatched_run(tmp_path, monkeypatch, mutation):
    from hotpot_fullwiki import loader

    path, out, config = scoring_run(tmp_path)
    monkeypatch.setattr(loader, "load_examples", forbidden)
    if mutation == "missing_seal":
        (out / "SEALED.json").unlink()
    elif mutation == "incomplete":
        write_json(out / "RUN_STATUS.json", {"status": "incomplete"})
    elif mutation == "wrong_count":
        write_json(out / "SEALED.json", {**config, "expected_units": 1})
    else:
        write_json(out / "SEALED.json", {**config, "config_digest": "wrong", "expected_units": 2})
    with pytest.raises((RuntimeError, FileNotFoundError)):
        hotpot.score(path, out)
    assert not (out / "SCORE.json").exists()


def test_hotpot_score_uses_official_metrics_for_selected_candidates(tmp_path, monkeypatch):
    from inherit_mas import release_config
    from hotpot_fullwiki import loader

    path, out, config = scoring_run(tmp_path)
    fake_revision = ModuleType("hotpot_fullwiki.select_edit")
    fake_revision.CONDITION = "test-condition"
    monkeypatch.setitem(sys.modules, "hotpot_fullwiki.select_edit", fake_revision)
    monkeypatch.setattr(release_config, "ensure_hf_home", lambda: None)
    examples = [SimpleNamespace(id="task-a", answer="Paris", supporting_facts=[("France", 0)]),
                SimpleNamespace(id="task-b", answer="Rome", supporting_facts=[("Italy", 0)])]
    monkeypatch.setattr(loader, "load_examples", lambda rows: examples)
    for index, example in enumerate(examples):
        candidate = {"status": "complete", "prediction": {
            "answer": example.answer, "supporting_facts": example.supporting_facts}}
        selected = {**candidate, "candidate_index": 1, "prediction": {
            **candidate["prediction"], "answer": "Lyon" if index == 0 else "Rome"}}
        record = {"selected_candidate": 1, "candidates": [
            {**candidate, "candidate_index": 0}, selected]}
        write_json(out / "units" / "test-condition" / f"{example.id}.json", {
            "task_id": example.id, "config_digest": config["config_digest"],
            "status": "complete", "record": record})
    report = hotpot.score(path, out)
    assert report["n"] == 2
    assert report["metrics"] == {"answer_em": 0.5, "answer_f1": 0.5, "sp_em": 1.0,
                                 "sp_f1": 1.0, "joint_em": 0.5, "joint_f1": 0.5}
    assert report["usage"] == {"calls": 0, "tokens": 0, "estimated_usd": 0}
    assert json.loads((out / "SCORE.json").read_text()) == report
