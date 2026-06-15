from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Images"])


@router.post("/images/generations")
async def image_generations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/generations``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.images.generate,
    )

    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )


@router.post("/images/edits")
async def image_edits(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/edits``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    for field in ("image", "mask"):
        if field in body:
            upload = body[field]
            body[field] = (upload.filename, await upload.read(), upload.content_type)

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.images.edit,
    )

    return JSONResponse(content=result.content, status_code=result.status_code)


@router.post("/images/variations")
async def image_variations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/variations``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    if "image" in body:
        upload = body["image"]
        body["image"] = (upload.filename, await upload.read(), upload.content_type)

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.images.create_variation,
    )

    return JSONResponse(content=result.content, status_code=result.status_code)
