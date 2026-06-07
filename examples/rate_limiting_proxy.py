from __future__ import annotations

import time

from fastapi import HTTPException
from openai import OpenAI
from api_model_proxy import APIModelProxy


class RateLimitingProxy(APIModelProxy):
    """An :class:`~api_model_proxy.proxy.APIModelProxy` subclass that
    rate-limits incoming requests using a token-bucket algorithm.

    When the rate limit is exceeded the proxy raises
    :class:`fastapi.HTTPException` (``429 Too Many Requests``) which
    FastAPI converts into a proper JSON error response — no changes
    to the route handlers are required.

    Usage::

        proxy = RateLimitingProxy(client, rate=5.0, burst=10)

    Args:
        openai_client: An ``openai.OpenAI`` instance.
        rate: Token refill rate per second (default ``10.0``).
        burst: Maximum accumulated tokens (default ``20``).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        rate: float = 10.0,
        burst: int = 20,
    ) -> None:
        super().__init__(openai_client)
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def _preprocess_request(self, request: dict) -> dict:
        self._refill()
        if self._tokens < 1:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Rate limit exceeded. Please try again later.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        self._tokens -= 1
        return request

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def _postprocess_response(self, response: dict) -> dict:
        return response
