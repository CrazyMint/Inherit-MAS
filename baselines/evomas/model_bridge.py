"""Route native model calls without changing EvoMAS' controller or decoding."""
from __future__ import annotations

import os

from inherit_mas import release_config

provider_failures: list[str] = []


def install(backbone: str) -> None:
    provider_failures.clear()
    import src.models as package
    from src.models import model as native
    from src.models.worker_policy import PRIVILEGED_ROLES, worker_family

    class PublicModel(native.OpenAIModel):
        def __init__(self, logical: str, role: str, **kwargs):
            self.role = role
            self._logical_model = logical
            self.model = logical.split(":", 1)[-1]
            self.max_tokens = int(kwargs.get("max_tokens", 4096))
            self.temperature = kwargs.get("temperature", 0.7)
            self.top_p = kwargs.get("top_p", 1.0)
            self.reset_token_usage()
            self._last_input_tokens = self._last_output_tokens = 0
            if role not in PRIVILEGED_ROLES and worker_family(logical) != backbone:
                raise ValueError("native EvoMAS worker escaped selected backbone")
            if role in PRIVILEGED_ROLES and self.model != "gpt-5.4-mini":
                raise ValueError("native EvoMAS meta/judge must use GPT-5.4-mini")
            self._qwen = self.model == "qwen3-32b"
            if self._qwen:
                from public_runner.models import qwen_settings
                endpoint = qwen_settings()["urls"][0]
                self.provider_model = "qwen3-32b"
                self.client = release_config.chat_client(
                    endpoint=endpoint, key=os.environ.get("QWEN_API_KEY", "EMPTY"),
                    timeout=180, max_retries=0)
                self.temperature = 0
            else:
                credentials = release_config.credentials()
                entry = release_config.deployment(credentials, "5.4-mini" if self.model == "gpt-5.4-mini" else self.model)
                self.provider_model = entry.get("deployment", entry.get("model", self.model))
                self.client = release_config.chat_client(
                    endpoint=release_config.model_endpoint(credentials, entry),
                    key=release_config.api_key(entry["api_key_env"]), timeout=180, max_retries=0)
            self.smolagents_model = self._create_mock_smolagents_model()

        def _call_openai_api(self, messages):
            from src.utils.cost_ledger import LEDGER
            if self.role not in PRIVILEGED_ROLES and worker_family(self._logical_model) != backbone:
                raise ValueError("native EvoMAS worker model changed before request")
            request = {"model": self.provider_model, "messages": messages,
                       "temperature": self.temperature, "top_p": self.top_p}
            request["max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"] = self.max_tokens
            if self._qwen:
                request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                response = self.client.chat.completions.create(**request)
                if not response.choices or any(type(value) is not int or value < 0 for value in (
                        response.usage.prompt_tokens, response.usage.completion_tokens)):
                    raise RuntimeError("provider response lacks choices or exact token usage")
            except Exception as exc:
                provider_failures.append(type(exc).__name__)
                LEDGER.record(phase="provider_error", role=self.role, model_id=self._logical_model,
                              usage_known=False, error_type=type(exc).__name__)
                raise
            self._last_input_tokens = response.usage.prompt_tokens
            self._last_output_tokens = response.usage.completion_tokens
            self._cumulative_input_tokens += self._last_input_tokens
            self._cumulative_output_tokens += self._last_output_tokens
            if self.role not in PRIVILEGED_ROLES:
                context = LEDGER.context()
                LEDGER.record(phase="worker", role=context.get("agent_role") or "worker",
                              model_id=self._logical_model, input_tokens=self._last_input_tokens,
                              output_tokens=self._last_output_tokens,
                              cost_usd=0.0 if self._qwen else None,
                              provider_model=self.provider_model,
                              provider_response_model=getattr(response, "model", None))
            if self._qwen and getattr(response, "model", None) not in {"qwen3-32b", "Qwen/Qwen3-32B"}:
                provider_failures.append("unexpected_qwen_identity")
                raise RuntimeError("Qwen endpoint returned an unexpected model identity")
            return response.choices[0].message.content

    def get_model(model_id, *, role="worker", **kwargs):
        return PublicModel(model_id, role, **kwargs)

    native.get_model = get_model
    package.get_model = get_model
