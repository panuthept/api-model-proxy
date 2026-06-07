from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional

from openai import OpenAI
from api_model_proxy import APIModelProxy


class CachingProxy(APIModelProxy):
    """An :class:`~api_model_proxy.proxy.APIModelProxy` subclass that caches
    inference responses in an in-memory LRU cache with TTL.

    On a cache hit the cached response is returned to the caller.  Note
    that the upstream API call is still dispatched (the current hook
    architecture does not support short-circuiting), but the cached
    response is returned instead of the fresh one.

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
        self._last_key: Optional[str] = None
        self._cache_hit_body: Optional[dict] = None

    def _preprocess_request(self, request: dict) -> dict:
        self._last_key = self._make_key(request)

        if self._last_key in self._cache:
            timestamp, body = self._cache[self._last_key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(self._last_key)
                self._cache_hit_body = body
                return request

        self._cache_hit_body = None
        return request

    def _postprocess_response(self, response: dict) -> dict:
        if self._cache_hit_body is not None and "error" not in response:
            return self._cache_hit_body

        if self._last_key is not None and "error" not in response:
            self._cache[self._last_key] = (time.time(), response)
            self._cache.move_to_end(self._last_key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return response

    @staticmethod
    def _make_key(request: dict) -> str:
        raw = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()
