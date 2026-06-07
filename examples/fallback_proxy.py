from __future__ import annotations

from typing import List, Tuple

from openai import OpenAI
from api_model_proxy import APIModelProxy


class FallbackProxy(APIModelProxy):
    """An :class:`~api_model_proxy.proxy.APIModelProxy` subclass that routes
    requests across multiple OpenAI-compatible backends with circuit-breaker
    style fallback.

    When a backend returns an error response the proxy marks it as
    *unhealthy* and automatically routes subsequent requests to the
    next healthy fallback.  When all backends are unhealthy they are
    all reset on the next request.

    Usage::

        proxy = FallbackProxy([
            ("https://api.openai.com/v1",  "sk-...", "gpt-4o"),
            ("https://api.example.com/v1", "sk-...", "example-model"),
        ])
        proxy.deploy()

    Args:
        fallbacks: A list of ``(base_url, api_key, model)`` tuples.
            The first entry is treated as the primary backend.
    """

    def __init__(self, fallbacks: List[Tuple[str, str, str]]) -> None:
        self._fallback_list: List[dict] = []
        for base_url, api_key, model in fallbacks:
            client = OpenAI(api_key=api_key, base_url=base_url)
            self._fallback_list.append({
                "client": client,
                "model": model,
                "healthy": True,
            })

        if not self._fallback_list:
            raise ValueError("At least one fallback entry is required")

        super().__init__(self._fallback_list[0]["client"])
        self._current_index = 0

    def _preprocess_request(self, request: dict) -> dict:
        for i, fb in enumerate(self._fallback_list):
            if fb["healthy"]:
                self._current_index = i
                self._client = fb["client"]
                request["model"] = fb["model"]
                return request

        for fb in self._fallback_list:
            fb["healthy"] = True
        self._current_index = 0
        self._client = self._fallback_list[0]["client"]
        request["model"] = self._fallback_list[0]["model"]
        return request

    def _postprocess_response(self, response: dict) -> dict:
        if "error" in response:
            self._fallback_list[self._current_index]["healthy"] = False
        return response
