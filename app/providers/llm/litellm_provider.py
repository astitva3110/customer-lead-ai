from app.config import Settings
from app.services.diagnostics.langsmith_tracing import add_token_usage, usage_payload
from app.services.diagnostics.recorder import current_trace


class LiteLLMProvider:
    """Single LLM adapter. Provider, model, and transport come from Settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.generation_model.strip())

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        text, usage = self.complete_with_usage(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._record_generation_tokens(usage)
        return text

    def complete_with_usage(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> tuple[str, dict]:
        if not self.is_configured:
            raise RuntimeError("generation model is not configured")
        import litellm

        litellm.drop_params = True
        kwargs: dict = {
            "model": self._settings.generation_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (
                self._settings.generation_temperature if temperature is None else temperature
            ),
            "timeout": self._settings.generation_timeout_seconds,
            "num_retries": self._settings.generation_num_retries,
        }
        tokens = self._settings.generation_max_tokens if max_tokens is None else max_tokens
        if tokens is not None:
            kwargs["max_tokens"] = tokens
        api_base = (self._settings.generation_api_base or "").strip()
        if api_base:
            kwargs["api_base"] = api_base
        api_key = (self._settings.generation_api_key or self._settings.openai_api_key or "").strip()
        if api_key:
            kwargs["api_key"] = api_key
        elif api_base:
            kwargs["api_key"] = "dummy"
        if self._settings.generation_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        extra_raw = (self._settings.generation_extra_body or "").strip()
        if extra_raw:
            import json

            extra_body = json.loads(extra_raw)
            if extra_body:
                kwargs["extra_body"] = extra_body
        if metadata:
            kwargs["metadata"] = metadata
        if tags:
            kwargs["tags"] = tags
        response = litellm.completion(**kwargs)
        usage = usage_payload(getattr(response, "usage", None))
        trace = current_trace()
        if trace is not None:
            add_token_usage(trace, usage)
        content = response.choices[0].message.content
        return content or "", usage

    def _record_generation_tokens(self, usage: dict) -> None:
        trace = current_trace()
        if trace is None or not usage:
            return
        generation = dict(trace.generation or {})
        if usage.get("input_tokens") is not None:
            generation["input_token_count"] = int(
                (generation.get("input_token_count") or 0) + usage["input_tokens"]
            )
        if usage.get("output_tokens") is not None:
            generation["output_token_count"] = int(
                (generation.get("output_token_count") or 0) + usage["output_tokens"]
            )
        trace.generation = generation
