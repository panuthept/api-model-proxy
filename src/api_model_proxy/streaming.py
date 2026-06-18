from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Callable

from openai import OpenAIError


def _format_sse_chunk(data: dict) -> str:
    """Format a dict as a single SSE ``data:`` frame.

    Args:
        data: The chunk data as a plain dict.

    Returns:
        An SSE-formatted string ending with ``\\n\\n``.
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialise_stream_error(exc: OpenAIError) -> dict:
    """Serialise an :class:`openai.OpenAIError` into a chunk-compatible dict.

    If the error has a JSON response body it is used directly;
    otherwise a generic error dict is constructed.

    The resulting dict is compatible with the chunk type expected
    by downstream consumers (e.g., a ``ChatCompletionChunk``-like
    structure with an ``error`` key).
    """
    if hasattr(exc, "response") and exc.response is not None:
        try:
            return exc.response.json()
        except Exception:
            pass
    return {"error": {"message": str(exc), "type": type(exc).__name__}}


async def _iterate_stream(
    stream: Any,
    postprocess: Callable[[dict], dict],
    serializer: Callable[[Any], dict] = lambda r: r.model_dump(),
) -> AsyncGenerator[bytes, None]:
    """Iterate over an OpenAI sync stream and yield SSE-formatted bytes.

    Each chunk from the stream is serialised to a dict via *serializer*,
    then passed through *postprocess* (typically
    :meth:`APIModelProxy._postprocess_response`), and finally formatted
    as an SSE ``data:`` frame.

    If an :class:`openai.OpenAIError` occurs mid-stream, the error is
    serialised, passed through *postprocess*, and yielded as the next
    chunk before termination.

    The stream always terminates with ``data: [DONE]\\n\\n``.

    Args:
        stream: A sync iterable of SDK chunk objects (e.g. from
            ``client.chat.completions.create(stream=True, ...)``).
        postprocess: A callable that takes a chunk dict and returns a
            (possibly modified) chunk dict. This is called on every
            chunk and on error dicts.
        serializer: Converts a raw SDK chunk object to a plain dict.
            Default: ``chunk.model_dump()``.

    Yields:
        Bytes for each SSE frame, including the terminal ``[DONE]``.
    """
    try:
        for chunk in stream:
            chunk_dict = serializer(chunk)
            chunk_dict = postprocess(chunk_dict)
            yield _format_sse_chunk(chunk_dict).encode("utf-8")
    except OpenAIError as exc:
        error_dict = _serialise_stream_error(exc)
        error_dict = postprocess(error_dict)
        yield _format_sse_chunk(error_dict).encode("utf-8")

    yield b"data: [DONE]\n\n"
