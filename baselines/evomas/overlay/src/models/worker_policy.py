"""Fail-closed worker routing policy, independent of provider SDKs."""
from __future__ import annotations

from collections.abc import Iterable


class ModelRolePolicyError(ValueError):
    """A task-executing model or palette violates the experiment policy."""


API_WORKER_MODELS = ("azure:gpt-4o-mini", "openai:gpt-4o-mini")
LOCAL_WORKER_MODELS = ("openai:qwen3-32b", "openai:Qwen/Qwen3-32B")
PRIVILEGED_ROLES = frozenset({"meta", "judge"})


def worker_family(model_id: str) -> str:
    if model_id in API_WORKER_MODELS or model_id == "gpt-4o-mini":
        return "gpt-4o-mini"
    if model_id in LOCAL_WORKER_MODELS or model_id in {"qwen3-32b", "Qwen/Qwen3-32B"}:
        return "qwen3-32b"
    raise ModelRolePolicyError(
        f"Unauthorized task-worker model {model_id!r}; GPT-5.4-mini is meta/judge only"
    )


def validate_worker_palette(model_list: Iterable[str] | None) -> tuple[str, ...]:
    if model_list is None or isinstance(model_list, (str, bytes)):
        raise ModelRolePolicyError("An explicit nonempty worker model palette is required")
    models = tuple(model_list)
    if not models:
        raise ModelRolePolicyError("An explicit nonempty worker model palette is required")
    families = {worker_family(model) for model in models}
    if len(families) != 1:
        raise ModelRolePolicyError("API and local worker regimes cannot be mixed")
    return models


def validate_model_role(model_id: str, *, role: str = "worker",
                        resolved_model: str | None = None) -> None:
    """Check logical identity and the actual provider request model, not a role label."""
    if role in PRIVILEGED_ROLES:
        return
    family = worker_family(model_id)
    if resolved_model is not None and worker_family(resolved_model) != family:
        raise ModelRolePolicyError("Resolved provider model differs from the worker regime")


def validate_worker_instance(model_id: str, model: object) -> None:
    """Reject provider/factory aliases whose constructed request model is unsafe."""
    validate_model_role(model_id)
    resolved = []
    for target in (model, getattr(model, "smolagents_model", None)):
        for name in ("model", "model_id"):
            value = getattr(target, name, None)
            if isinstance(value, str):
                resolved.append(value)
    if not resolved:
        raise ModelRolePolicyError("Worker provider model identity is unavailable")
    for value in resolved:
        validate_model_role(model_id, resolved_model=value)
