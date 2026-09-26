"""Original gold-free diagnostic judge and repair prompts."""
from __future__ import annotations

import json

JUDGE_CONTRACT = r'''{
  "quality_score": 0,
  "answer_likelihood": 0,
  "evidence_coverage": 0,
  "citation_grounding": 0,
  "critique": "specific skeptical audit",
  "keep": ["specific component"],
  "fix": ["specific missing or weak component"]
}'''


def judge_prompt(question: str, graph: dict | None, prediction: dict, trace: dict) -> tuple[str, str]:
    system = (
        "You are an independent, skeptical HotpotQA judge. You never see the gold answer or official supporting "
        "facts. Evaluate only the public question, retrieved observations, workflow, and proposed answer. Treat "
        "fluent unsupported answers as low quality. Check whether both hops are resolved, whether the answer follows "
        "from evidence, and whether every cited title/sentence was observed. Scores are 0-100. A score above 80 "
        "requires explicit evidence for both hops and no material uncertainty. Return JSON only."
    )
    user = json.dumps({"question": question, "graph": graph, "prediction": prediction,
                       "execution": trace, "contract": JUDGE_CONTRACT}, ensure_ascii=True)
    return system, user


def repair_prompt(original: str, error: str, contract: str) -> tuple[str, str]:
    return ("Repair one JSON response. Address the exact validator error. Return JSON only, no commentary.",
            f"Validator error: {error}\nContract:\n{contract}\nOriginal:\n{original}")
