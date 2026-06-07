from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from openai import OpenAIError

router = APIRouter(tags=["Images"])


@router.post("/images/generations")
async def image_generations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/generations``."""
    proxy = request.app.state.proxy
    body: dict = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.images.generate(**body)
        response_dict = result.model_dump()
        response_dict = proxy._postprocess_response(response_dict)
        return JSONResponse(content=response_dict)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)


@router.post("/images/edits")
async def image_edits(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/edits``.

    Accepts multipart/form-data (image + optional mask + params).
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    for field in ("image", "mask"):
        if field in body:
            upload = body[field]
            body[field] = (upload.filename, await upload.read(), upload.content_type)

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.images.edit(**body)
        response_dict = result.model_dump()
        response_dict = proxy._postprocess_response(response_dict)
        return JSONResponse(content=response_dict)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)


@router.post("/images/variations")
async def image_variations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/images/variations``.

    Accepts multipart/form-data (image + params).
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    if "image" in body:
        upload = body["image"]
        body["image"] = (upload.filename, await upload.read(), upload.content_type)

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.images.create_variation(**body)
        response_dict = result.model_dump()
        response_dict = proxy._postprocess_response(response_dict)
        return JSONResponse(content=response_dict)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)


def _openai_error_to_dict(exc: OpenAIError) -> dict:
    if hasattr(exc, "response") and exc.response is not None:
        try:
            return exc.response.json()
        except Exception:
            return {"error": {"message": str(exc), "type": type(exc).__name__}}
    return {"error": {"message": str(exc), "type": type(exc).__name__}}


def _status_code(exc: OpenAIError) -> int:
    if hasattr(exc, "status_code") and exc.status_code is not None:
        return int(exc.status_code)
    return 500
