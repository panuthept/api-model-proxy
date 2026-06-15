from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Embeddings"])


@router.post("/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/embeddings``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.embeddings.create,
    )

    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )
