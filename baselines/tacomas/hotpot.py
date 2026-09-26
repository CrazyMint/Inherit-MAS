"""Structured HotpotQA TacoMAS integration used by the main comparison."""
from hotpot_fullwiki.baselines import runner
from . import hotpot_controller


def run(manifest_path, out, *, usd_cap, workers=4, backbone="gpt-4o-mini"):
    return runner.run(manifest_path, out, usd_cap=usd_cap, workers=workers,
                      backbone=backbone, method="tacomas_adapted", controller=hotpot_controller)


def score(manifest_path, out):
    return runner.score(manifest_path, out, method="tacomas_adapted")
