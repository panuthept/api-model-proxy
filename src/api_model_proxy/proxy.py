from __future__ import annotations

from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from openai import OpenAI


class APIModelProxy:
    """Base class for building OpenAI-compatible API proxies.

    Subclasses override :meth:`_preprocess_request` and/or
    :meth:`_postprocess_response` to intercept inference calls.
    All other endpoints are forwarded transparently.

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
    # Server
    # ------------------------------------------------------------------

    def deploy(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the proxy server.

        Args:
            host: Bind host (default ``"localhost"``).
            port: Bind port (default ``8000``).
        """
        from .server import create_app

        app = create_app(self)
        uvicorn.run(app, host=host, port=port)
