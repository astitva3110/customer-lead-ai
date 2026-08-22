from app.config import Settings


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
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        return content or ""
