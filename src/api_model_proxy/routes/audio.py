from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from openai import OpenAIError

from api_model_proxy import _error_to_dict, _status_code

router = APIRouter(tags=["Audio"])


@router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/audio/transcriptions``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    if "file" in body:
        upload = body["file"]
        body["file"] = (upload.filename, await upload.read(), upload.content_type)

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.audio.transcriptions.create,
        serializer=lambda r: r.model_dump() if hasattr(r, "model_dump") else {"text": r},
    )

    return JSONResponse(content=result.content, status_code=result.status_code)


@router.post("/audio/translations")
async def audio_translations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/audio/translations``.

    Delegates the full request pipeline to
    :meth:`APIModelProxy.execute_request`.
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    if "file" in body:
        upload = body["file"]
        body["file"] = (upload.filename, await upload.read(), upload.content_type)

    result = proxy.execute_request(
        body=body,
        sdk_method=proxy._client.audio.translations.create,
        serializer=lambda r: r.model_dump() if hasattr(r, "model_dump") else {"text": r},
    )

    return JSONResponse(content=result.content, status_code=result.status_code)


@router.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    """Proxy for ``POST /v1/audio/speech``.

    Note: This endpoint returns raw binary audio, so it cannot use
    :meth:`APIModelProxy.execute_request` which expects a dict response.
    The pre/post hooks are still applied manually.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.speech.create(**body)
        audio_bytes = result.read()
        meta = {"bytes_length": len(audio_bytes)}
        proxy._postprocess_response(meta)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
        )
    except OpenAIError as exc:
        error_dict = _error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(content=error_dict, status_code=_status_code(exc))
