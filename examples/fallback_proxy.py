from __future__ import annotations

from typing import Any, Callable, Optional

from openai import OpenAI
from api_model_proxy import APIModelProxy, PipelineResult


class FallbackProxy(APIModelProxy):
    """An :class:`~api_model_proxy.proxy.APIModelProxy` subclass that routes
    requests across multiple OpenAI-compatible backends with automatic
    fallback on failure.

    The first backend in the list is the primary.  If it returns an error
    response, the proxy tries the next backend, and so on.  If all
    backends fail, the last error is returned to the caller.

    Usage::

        backends = [
            ("https://api.openai.com/v1", "sk-...", "gpt-4o"),
            ("http://localhost:1234/v1", "", "local-model"),
        ]
        proxy = FallbackProxy(backends)

    Args:
        fallbacks: A list of ``(base_url, api_key, model)`` tuples.
            The first entry is the primary backend.
    """

    def __init__(self, fallbacks: list[tuple[str, str, str]]) -> None:
        clients = [
            OpenAI(base_url=url, api_key=key) for url, key, _ in fallbacks
        ]
        super().__init__(clients[0])
        self._clients = clients
        self._models = [m for _, _, m in fallbacks]
        self._healthy = [True] * len(clients)
        self._fallbacks = fallbacks

    # ------------------------------------------------------------------
    # Pipeline override — multi-backend fallback
    # ------------------------------------------------------------------

    def execute_request(
        self,
        body: dict,
        sdk_method: Callable[..., Any],
        serializer: Callable[[Any], dict] = lambda r: r.model_dump(),
    ) -> PipelineResult:
        for idx in range(len(self._clients)):
            if not self._healthy[idx]:
                continue

            self._client = self._clients[idx]
            result = super().execute_request(body, sdk_method, serializer)

            if "error" not in result.content:
                return result

            self._healthy[idx] = False

        # All backends exhausted — return the last result
        return result
