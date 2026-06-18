from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Callable, TYPE_CHECKING

import uvicorn
from openai import OpenAIError

from .sentinels import PipelineResult

if TYPE_CHECKING:
    from openai import OpenAI


class APIModelProxy:
    """Base class for building OpenAI-compatible API proxies.

    Subclasses override :meth:`_preprocess_request` and/or
    :meth:`_postprocess_response` to intercept inference calls,
    or override :meth:`execute_request` to replace the entire pipeline.

    Example::

        class LoggingProxy(APIModelProxy):
            def _preprocess_request(self, request):
                print(f"Request: {request}")
                return request

            def _postprocess_response(self, response):
                print(f"Response: {response}")
                return response

        proxy = LoggingProxy(OpenAI())
        proxy.deploy(host="localhost", port=8000)
    """

    def __init__(self, openai_client: "OpenAI") -> None:
        self._client = openai_client

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def _preprocess_request(self, request: dict) -> dict:
        """Called before every inference request is forwarded to OpenAI.

        Args:
            request: The raw request body as a dict.

        Returns:
            The (possibly modified) request dict to forward upstream.
        """
        return request

    def _postprocess_response(self, response: dict) -> dict:
        """Called after every inference response is received from OpenAI.

        This is invoked for both successful responses **and** error
        responses, allowing subclasses to inspect or modify errors.
        If not overridden, the response is forwarded as-is.

        Args:
            response: The raw response body as a dict.

        Returns:
            The (possibly modified) response dict to return to the caller.
        """
        return response

    # ------------------------------------------------------------------
    # Pipeline — overridable as a single unit
    # ------------------------------------------------------------------

    def execute_request(
        self,
        body: dict,
        sdk_method: Callable[..., Any],
        serializer: Callable[[Any], dict] = lambda r: r.model_dump(),
    ) -> PipelineResult:
        """Execute the full request pipeline.

        The default implementation runs::

            body = _preprocess_request(body)
            try:
                raw = sdk_method(**body)
                response = serializer(raw)
            except OpenAIError as exc:
                response = _error_to_dict(exc)
                status = _status_code(exc)
            response = _postprocess_response(response)
            return PipelineResult(content=response, status_code=status)

        Override this method to replace the entire pipeline —
        for example to add caching, retry logic, circuit breakers,
        quality gates, or any custom orchestration — without
        modifying the route handlers.

        Args:
            body: The parsed request body dict.
            sdk_method: The OpenAI SDK method to call, e.g.
                ``proxy._client.chat.completions.create``.
                Called as ``sdk_method(**body)``.
            serializer: Converts the SDK result to a plain dict.
                Default: ``result.model_dump()``.
                Some endpoints (e.g. audio transcriptions) need a
                custom serializer because the SDK returns a string
                without a ``.model_dump()`` method.

        Returns:
            A :class:`PipelineResult` describing the HTTP response
            the route handler should return.
        """
        body = self._preprocess_request(body)
        status = 200

        try:
            raw = sdk_method(**body)
            response = serializer(raw)
        except OpenAIError as exc:
            response = _error_to_dict(exc)
            status = _status_code(exc)

        response = self._postprocess_response(response)

        return PipelineResult(content=response, status_code=status)

    def execute_streaming_request(
        self,
        body: dict,
        sdk_method: Callable[..., Any],
        serializer: Callable[[Any], dict] = lambda r: r.model_dump(),
    ) -> AsyncGenerator[bytes, None]:
        """Execute the streaming request pipeline.

        The default implementation runs::

            body = _preprocess_request(body)
            stream = sdk_method(**body, stream=True)
            for chunk in stream:
                chunk_dict = serializer(chunk)
                chunk_dict = _postprocess_response(chunk_dict)
                yield SSE-formatted bytes

        Override this method to replace the entire streaming pipeline
        — for example to add custom chunk transformation, backpressure
        handling, or alternative transport — without modifying the
        route handlers.

        Args:
            body: The parsed request body dict (after pre-processing).
            sdk_method: The OpenAI SDK method to call, e.g.
                ``proxy._client.chat.completions.create``.
                Called as ``sdk_method(**body, stream=True)``.
            serializer: Converts a raw SDK chunk object to a plain dict.
                Default: ``chunk.model_dump()``.

        Yields:
            SSE-formatted bytes for each chunk, terminating with
            ``data: [DONE]\n\n``.
        """
        from .streaming import _iterate_stream

        body = self._preprocess_request(body)
        stream = sdk_method(**body, stream=True)
        return _iterate_stream(
            stream=stream,
            postprocess=self._postprocess_response,
            serializer=serializer,
        )

    def deploy(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the proxy server.

        Args:
            host: Bind host (default ``"localhost"``).
            port: Bind port (default ``8000``).
        """
        from .server import create_app

        app = create_app(self)
        uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------
# Shared helpers (moved here from duplicated copies in route files)
# ------------------------------------------------------------------


def _error_to_dict(exc: OpenAIError) -> dict:
    """Serialise an :class:`openai.OpenAIError` to a plain dict.

    If the error has a JSON response body it is used directly;
    otherwise a generic error dict is constructed.
    """
    if hasattr(exc, "response") and exc.response is not None:
        try:
            return exc.response.json()
        except Exception:
            pass
    return {"error": {"message": str(exc), "type": type(exc).__name__}}


def _status_code(exc: OpenAIError) -> int:
    """Extract the HTTP status code from an :class:`openai.OpenAIError`."""
    if hasattr(exc, "status_code") and exc.status_code is not None:
        return int(exc.status_code)
    return 500
