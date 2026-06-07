from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from openai import OpenAIError

router = APIRouter(tags=["Embeddings"])


@router.post("/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Proxy for ``POST /v1/embeddings``.

    Calls :meth:`~api_model_proxy.proxy.APIModelProxy._preprocess_request`
    before forwarding and
    :meth:`~api_model_proxy.proxy.APIModelProxy._postprocess_response`
    on both success and error responses.
    """
    proxy = request.app.state.proxy
    body: dict = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.embeddings.create(**body)
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
