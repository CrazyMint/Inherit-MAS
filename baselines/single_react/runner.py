"""Public Single ReAct run and official postrun score entrypoints."""
from . import _runner


def run(manifest_path, out, *, usd_cap, workers, backbone="gpt-4o-mini"):
    return _runner.run("single-react", manifest_path, out, usd_cap=usd_cap,
                       workers=workers, backbone=backbone)


def score(manifest_path, out):
    return _runner.score("single-react", manifest_path, out)
