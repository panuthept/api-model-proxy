from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Responses"])

_STREAMING_NOT_SUPPORTED = {
    "error": {
        "message": (
            "Streaming is not yet supported by this proxy. "
            "It will be available in a future version."
        ),
        "type": "not_implemented",
        "code": "streaming_not_supported",
    }
}


@router.post("/responses")
async def create_response(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/responses`` (Responses API — inference).

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    # TODO(streaming): implement SSE streaming pass-through
    if body.get("stream"):
        return JSONResponse(status_code=501, content=_STREAMING_NOT_SUPPORTED)

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.responses.create,
    )

    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )
