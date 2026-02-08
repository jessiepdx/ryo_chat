##########################################################################
#                                                                        #
#  This file (model_router.py) handles model selection and fallback      #
#  routing for Ollama-based inference calls.                             #
#                                                                        #
##########################################################################

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from ollama import AsyncClient


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


class ModelRouterError(Exception):
    """Base error for model router failures."""


class ModelResolutionError(ModelRouterError):
    """Raised when no valid model candidates can be resolved."""


class ModelExecutionError(ModelRouterError):
    """Raised when all candidate models fail execution."""

    def __init__(self, message: str, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass
class RouteMetadata:
    capability: str
    host: str
    requested_model: str | None
    configured_model: str | None
    allowed_models: list[str]
    candidate_models: list[str]
    selected_model: str | None = None
    attempted_models: list[str] = field(default_factory=list)
    fallback_count: int = 0
    available_models: list[str] | None = None
    list_failed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRouter:
    """Centralized model routing and fallback behavior for Ollama requests."""

    _capability_to_inference_key = {
        "analysis": "tool",
        "tool": "tool",
        "chat": "chat",
        "conversation": "chat",
        "dev_test": "chat",
        "embedding": "embedding",
        "generate": "generate",
        "multimodal": "multimodal",
    }

    def __init__(
        self,
        inference_config: dict | None = None,
        endpoint_override: str | None = None,
        default_host: str = DEFAULT_OLLAMA_HOST,
    ):
        self._inference_config = inference_config or {}
        self._endpoint_override = endpoint_override
        self._default_host = self._normalize_host(default_host)

    @staticmethod
    def _normalize_host(host: str | None) -> str | None:
        if host is None:
            return None
        return host.strip().rstrip("/")

    @staticmethod
    def _is_valid_host(host: str | None) -> bool:
        if not host:
            return False
        parsed = urlparse(host)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _inference_key(self, capability: str) -> str:
        return self._capability_to_inference_key.get(capability, capability)

    def _configured_inference(self, capability: str) -> dict[str, Any]:
        inference_key = self._inference_key(capability)
        data = self._inference_config.get(inference_key)
        return data if isinstance(data, dict) else {}

    def resolve_host(self, capability: str) -> str:
        override = self._normalize_host(self._endpoint_override)
        if self._is_valid_host(override):
            return override

        configured_host = self._normalize_host(self._configured_inference(capability).get("url"))
        if self._is_valid_host(configured_host):
            return configured_host

        return self._default_host

    def configured_model(self, capability: str) -> str | None:
        model = self._configured_inference(capability).get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        return None

    def candidate_models(
        self,
        capability: str,
        requested_model: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> list[str]:
        candidates: list[str] = []

        def add(value: str | None):
            if isinstance(value, str):
                value = value.strip()
            if value and value not in candidates:
                candidates.append(value)

        add(requested_model)
        if allowed_models:
            for allowed in allowed_models:
                add(allowed)
        add(self.configured_model(capability))

        return candidates

    async def list_available_models(self, host: str) -> list[str]:
        client = AsyncClient(host=host)
        model_list = await client.list()
        output: list[str] = []

        for entry in getattr(model_list, "models", []):
            if hasattr(entry, "model"):
                name = getattr(entry, "model")
            elif isinstance(entry, dict):
                name = entry.get("model")
            else:
                name = None

            if isinstance(name, str) and name not in output:
                output.append(name)

        return output

    async def resolve(
        self,
        capability: str,
        requested_model: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> RouteMetadata:
        candidates = self.candidate_models(
            capability=capability,
            requested_model=requested_model,
            allowed_models=allowed_models,
        )

        if not candidates:
            raise ModelResolutionError(
                f"No model candidates found for capability '{capability}'."
            )

        metadata = RouteMetadata(
            capability=capability,
            host=self.resolve_host(capability),
            requested_model=requested_model,
            configured_model=self.configured_model(capability),
            allowed_models=list(allowed_models or []),
            candidate_models=list(candidates),
        )

        try:
            available_models = await self.list_available_models(metadata.host)
            metadata.available_models = available_models
            for candidate in candidates:
                if candidate in available_models:
                    metadata.selected_model = candidate
                    return metadata

            # If model inventory is stale or filtered, try candidates anyway.
            metadata.selected_model = candidates[0]
            metadata.errors.append(
                "No candidate model matched advertised model list; trying candidates in priority order."
            )
            return metadata
        except Exception as error:
            metadata.list_failed = True
            metadata.selected_model = candidates[0]
            metadata.errors.append(f"Model inventory probe failed: {error}")
            return metadata

    @staticmethod
    async def _prime_stream(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        try:
            first_chunk = await anext(stream)
        except StopAsyncIteration:
            async def _empty() -> AsyncIterator[Any]:
                if False:
                    yield None

            return _empty()

        async def _forward() -> AsyncIterator[Any]:
            yield first_chunk
            async for chunk in stream:
                yield chunk

        return _forward()

    async def chat_with_fallback(
        self,
        capability: str,
        messages: list[Any],
        requested_model: str | None = None,
        allowed_models: list[str] | None = None,
        stream: bool = False,
        **chat_kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        metadata = await self.resolve(
            capability=capability,
            requested_model=requested_model,
            allowed_models=allowed_models,
        )

        candidates = list(metadata.candidate_models)
        if metadata.selected_model in candidates:
            candidates.remove(metadata.selected_model)
            candidates.insert(0, metadata.selected_model)

        for candidate in candidates:
            metadata.attempted_models.append(candidate)
            client = AsyncClient(host=metadata.host)
            try:
                if stream:
                    raw_stream = await client.chat(
                        model=candidate,
                        messages=messages,
                        stream=True,
                        **chat_kwargs,
                    )
                    response = await self._prime_stream(raw_stream)
                else:
                    response = await client.chat(
                        model=candidate,
                        messages=messages,
                        stream=False,
                        **chat_kwargs,
                    )

                metadata.selected_model = candidate
                metadata.fallback_count = max(0, len(metadata.attempted_models) - 1)
                return response, metadata.to_dict()
            except Exception as error:
                metadata.errors.append(f"{candidate}: {error}")

        raise ModelExecutionError(
            f"All candidate models failed for capability '{capability}'.",
            metadata=metadata.to_dict(),
        )
