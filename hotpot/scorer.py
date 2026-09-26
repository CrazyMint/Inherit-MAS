"""Official HotpotQA scorer adapter — answer EM/F1 + supporting-fact EM/F1.

Faithful re-implementation of the official `hotpot_evaluate_v1.py` metric
(SQuAD-style answer normalization + token-F1; supporting-fact set precision/
recall/F1/EM over (title, sent_id) pairs; joint EM/F1). No LLM judge. Deterministic.

Gold answers and supporting facts are scorer-only inputs, kept separate from
agent inputs by the loader's public projection.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

# Upstream evaluator provenance for the metric implementation.
OFFICIAL_EVALUATOR = {
    "repo": "https://github.com/hotpotqa/hotpot",
    "script": "hotpot_evaluate_v1.py",
    "pinned_commit": "fa3a36370899e1d85822de61e58c85ea19993154",
    "vendored_path": "hotpot/vendor/hotpot_evaluate_v1.py",
    "deviation": "import ujson->json only; grading logic verbatim",
    "parity_verified_by": "test_scorer.py differential subprocess tests",
}


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalization: lowercase, strip punctuation, articles, and
    redundant whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Token-level (F1, precision, recall) with the official yes/no/noanswer
    special-casing (a normalized answer that collapses to one of these must match
    exactly, else 0)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)
    if normalized_prediction in ("yes", "no", "noanswer") and \
            normalized_prediction != normalized_ground_truth:
        return ZERO
    if normalized_ground_truth in ("yes", "no", "noanswer") and \
            normalized_prediction != normalized_ground_truth:
        return ZERO
    pred_tokens = normalized_prediction.split()
    gt_tokens = normalized_ground_truth.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _sp_key(sf) -> Tuple[str, int]:
    """A supporting fact is a (title, sent_id) pair (list or tuple)."""
    return (str(sf[0]), int(sf[1]))


def supporting_fact_metrics(pred_sp: List, gold_sp: List) -> Dict[str, float]:
    """Set precision/recall/F1 + EM over (title, sent_id) supporting facts —
    Matches the official `update_sp`: precision/recall are 0.0
    (NOT 1.0) when their denominator is 0, so both-empty gives sp_f1=0.0, sp_em=1.0.
    """
    pred = {_sp_key(x) for x in pred_sp}
    gold = {_sp_key(x) for x in gold_sp}
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    em = 1.0 if (fp + fn) == 0 else 0.0
    return {"sp_em": em, "sp_f1": f1, "sp_precision": prec, "sp_recall": rec}


def score_example(pred_answer: str, gold_answer: str,
                  pred_sp: List, gold_sp: List) -> Dict[str, float]:
    """Answer EM/F1 + supporting-fact EM/F1 + joint EM/F1 for one example
    (official HotpotQA joint = product of answer and sp)."""
    em = 1.0 if exact_match_score(pred_answer, gold_answer) else 0.0
    a_f1, a_prec, a_rec = f1_score(pred_answer, gold_answer)
    sp = supporting_fact_metrics(pred_sp, gold_sp)
    joint_em = em * sp["sp_em"]
    joint_prec = a_prec * sp["sp_precision"]
    joint_rec = a_rec * sp["sp_recall"]
    joint_f1 = (2 * joint_prec * joint_rec / (joint_prec + joint_rec)) if (joint_prec + joint_rec) else 0.0
    return {"answer_em": em, "answer_f1": a_f1, "answer_precision": a_prec,
            "answer_recall": a_rec, **sp, "joint_em": joint_em, "joint_f1": joint_f1}


def aggregate(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over examples of each metric (official reports means × 100)."""
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}
