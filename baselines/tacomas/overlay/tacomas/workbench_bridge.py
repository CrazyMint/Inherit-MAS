"""Canonical benchmark imports, independent of the native checkout location."""
from public_runner.workbench import _workbench_path

_workbench_path()
import wb_env as canonical  # noqa: E402,F401
