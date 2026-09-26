"""Public EvoAgent run and official postrun score entrypoints."""
from baselines.single_react import _runner


def run(manifest_path, out, *, usd_cap, workers, backbone="gpt-4o-mini"):
    return _runner.run("evoagent", manifest_path, out, usd_cap=usd_cap,
                       workers=workers, backbone=backbone)


def score(manifest_path, out):
    return _runner.score("evoagent", manifest_path, out)
