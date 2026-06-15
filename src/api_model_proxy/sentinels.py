from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineResult:
    """The result of executing a request through the proxy pipeline.

    This is the only communication contract between
    :meth:`APIModelProxy.execute_request` and the route handlers.

    Route handlers inspect this to build the HTTP response —
    they never contain pipeline logic themselves.

    Attributes:
        content: The response body as a plain dict
            (may contain an ``"error"`` key on failure).
        status_code: HTTP status code (default ``200``).
        headers: Optional HTTP response headers.
    """

    content: dict
    status_code: int = 200
    headers: dict[str, str] | None = None
