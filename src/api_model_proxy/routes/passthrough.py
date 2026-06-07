from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter(tags=["Passthrough"])

# Headers that must not be forwarded from the downstream client to OpenAI
# (FastAPI/uvicorn sets its own transfer-encoding, host, etc.)
_HOP_BY_HOP_REQUEST = frozenset(
    [
        "host",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
        "content-length",  # httpx will recalculate
    ]
)

# Headers that must not be forwarded back from OpenAI to the downstream client
_HOP_BY_HOP_RESPONSE = frozenset(
    [
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    ]
)


@router.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def passthrough(request: Request, full_path: str) -> Response:
    """Catch-all transparent proxy for all non-inference endpoints.

    Forwards the request verbatim (method, path, headers, body) to the
    upstream OpenAI API using the API key and base URL from the configured
    ``openai.OpenAI`` client.  The response (including binary content and
    error bodies) is forwarded as-is.  No hooks are called.
    """
    proxy = request.app.state.proxy
    client = proxy._client

    # Build upstream URL: base_url already ends with '/' (e.g. https://api.openai.com/v1/)
    base_url = str(client.base_url).rstrip("/")
    upstream_url = f"{base_url}/{full_path}"

    # Preserve query string
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    # Build forwarded headers — inject Authorization from the configured client
    forwarded_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP_REQUEST
    }
    forwarded_headers["Authorization"] = f"Bearer {client.api_key}"

    # Include organisation header if configured
    org_id: str | None = getattr(client, "organization", None)
    if org_id:
        forwarded_headers["OpenAI-Organization"] = org_id

    body = await request.body()

    async with httpx.AsyncClient(timeout=600.0) as http:
        upstream_response = await http.request(
            method=request.method,
            url=upstream_url,
            headers=forwarded_headers,
            content=body,
        )

    response_headers = {
        k: v
        for k, v in upstream_response.headers.items()
        if k.lower() not in _HOP_BY_HOP_RESPONSE
    }

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )
