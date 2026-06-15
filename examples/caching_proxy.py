from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional

from openai import OpenAI
from api_model_proxy import APIModelProxy, PipelineResult


class CachingProxy(APIModelProxy):
    """An :class:`~api_model_proxy.proxy.APIModelProxy` subclass that caches
    inference responses in an in-memory LRU cache with TTL.

    Unlike the original hook-based approach, this uses the
    :meth:`~api_model_proxy.proxy.APIModelProxy.execute_request` override
    pattern, which **avoids the upstream API call entirely on cache hit**.

    Usage::

        proxy = CachingProxy(client, cache_size=500, ttl=120)

    Args:
        openai_client: An ``openai.OpenAI`` instance.
        cache_size: Maximum number of entries in the LRU cache (default 1000).
        ttl: Time-to-live in seconds for each entry (default 300).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        cache_size: int = 1000,
        ttl: int = 300,
    ) -> None:
        super().__init__(openai_client)
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._cache_size = cache_size
        self._ttl = ttl

    # ------------------------------------------------------------------
    # Pipeline override — true short-circuit
    # ------------------------------------------------------------------

    def execute_request(
        self,
        body: dict,
        sdk_method,
        serializer=lambda r: r.model_dump(),
    ) -> PipelineResult:
        key = self._make_key(body)
        cached = self._cache_get(key)
        if cached is not None:
            return PipelineResult(content=cached, status_code=200)

        result = super().execute_request(body, sdk_method, serializer)

        if "error" not in result.content:
            self._cache_set(key, result.content)

        return result

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[dict]:
        if key not in self._cache:
            return None
        timestamp, body = self._cache[key]
        if time.time() - timestamp < self._ttl:
            self._cache.move_to_end(key)
            return body
        del self._cache[key]
        return None

    def _cache_set(self, key: str, body: dict) -> None:
        self._cache[key] = (time.time(), body)
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _make_key(request: dict) -> str:
        raw = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()
