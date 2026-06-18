from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(tags=["Chat"])


@router.post("/chat/completions")
async def chat_completions(request: Request):
    """Proxy for ``POST /v1/chat/completions``.

    For non-streaming requests, delegates to
    :meth:`APIModelProxy.execute_request` and returns JSON.

    For streaming requests (``stream=True``), delegates to
    :meth:`APIModelProxy.execute_streaming_request` and returns
    a ``text/event-stream`` SSE response.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    if body.get("stream"):
        return StreamingResponse(
            proxy.execute_streaming_request(
                body=body,
                sdk_method=proxy._client.chat.completions.create,
            ),
            media_type="text/event-stream",
        )

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.chat.completions.create,
    )

    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )
