from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from openai import OpenAIError

router = APIRouter(tags=["Audio"])


@router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/audio/transcriptions``.

    Accepts multipart/form-data forwarded as keyword arguments to the
    OpenAI SDK.  Hooks fire on the JSON response body.
    """
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    # The 'file' field is an UploadFile; unwrap it for the SDK
    if "file" in body:
        upload = body["file"]
        body["file"] = (upload.filename, await upload.read(), upload.content_type)

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.transcriptions.create(**body)
        response_dict = result.model_dump() if hasattr(result, "model_dump") else {"text": result}
        response_dict = proxy._postprocess_response(response_dict)
        return JSONResponse(content=response_dict)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)


@router.post("/audio/translations")
async def audio_translations(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/audio/translations``."""
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    if "file" in body:
        upload = body["file"]
        body["file"] = (upload.filename, await upload.read(), upload.content_type)

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.translations.create(**body)
        response_dict = result.model_dump() if hasattr(result, "model_dump") else {"text": result}
        response_dict = proxy._postprocess_response(response_dict)
        return JSONResponse(content=response_dict)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)


@router.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    """Proxy for ``POST /v1/audio/speech`` (TTS).

    The request body is JSON; the response is raw audio bytes.
    Hooks receive the JSON body (request) and a dict with audio
    metadata (response) — the raw bytes are forwarded as-is.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.speech.create(**body)
        # result is an HttpxBinaryResponseContent; read bytes
        audio_bytes = result.read()
        # Give postprocess hook a lightweight dict (not the raw bytes)
        meta = {"bytes_length": len(audio_bytes)}
        proxy._postprocess_response(meta)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
        )
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
