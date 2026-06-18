from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError


class TestFormatSSEChunk:
    def test_formats_simple_dict(self):
        from api_model_proxy.streaming import _format_sse_chunk

        result = _format_sse_chunk({"key": "value"})
        assert result == 'data: {"key": "value"}\n\n'

    def test_formats_with_unicode(self):
        from api_model_proxy.streaming import _format_sse_chunk

        result = _format_sse_chunk({"content": "héllo wörld 🌍"})
        parsed = json.loads(result[5:].strip())
        assert parsed["content"] == "héllo wörld 🌍"

    def test_formats_with_nested_dict(self):
        from api_model_proxy.streaming import _format_sse_chunk

        data = {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}
        result = _format_sse_chunk(data)
        assert '"delta": {"content": "Hi"}' in result

    def test_formats_empty_dict(self):
        from api_model_proxy.streaming import _format_sse_chunk

        result = _format_sse_chunk({})
        assert result == "data: {}\n\n"

    def test_ensure_ascii_false_preserves_unicode(self):
        """SSE data should not escape non-ASCII characters."""
        from api_model_proxy.streaming import _format_sse_chunk

        result = _format_sse_chunk({"text": "日本語"})
        assert "日本語" in result
        assert "\\u" not in result


class TestSerialiseStreamError:
    def test_serialises_error_with_response_body(self):
        from api_model_proxy.streaming import _serialise_stream_error

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "error": {"message": "Rate limit", "type": "rate_limit_error"}
        }

        exc = OpenAIError("Rate limit")
        exc.response = mock_response

        result = _serialise_stream_error(exc)
        assert result == {"error": {"message": "Rate limit", "type": "rate_limit_error"}}

    def test_serialises_error_without_response(self):
        from api_model_proxy.streaming import _serialise_stream_error

        exc = OpenAIError("Something went wrong")

        result = _serialise_stream_error(exc)
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "OpenAIError"

    def test_serialises_subclass_error(self):
        from api_model_proxy.streaming import _serialise_stream_error
        from openai import APIStatusError

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "error": {"message": "Bad request", "type": "invalid_request_error"}
        }
        mock_response.status_code = 400
        mock_response.headers = {}

        exc = APIStatusError(
            message="Bad request",
            response=mock_response,
            body={"error": {"message": "Bad request", "type": "invalid_request_error"}},
        )

        result = _serialise_stream_error(exc)
        assert result["error"]["message"] == "Bad request"

    def test_fallback_on_json_parse_failure(self):
        from api_model_proxy.streaming import _serialise_stream_error

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError("Malformed JSON")

        exc = OpenAIError("parse error")
        exc.response = mock_response

        result = _serialise_stream_error(exc)
        assert result["error"]["message"] == "parse error"


@pytest.mark.asyncio
class TestIterateStream:
    """Tests for the _iterate_stream async generator."""

    async def _collect(self, gen):
        """Collect all bytes from an async generator into a single string."""
        return b"".join([chunk async for chunk in gen]).decode("utf-8")

    async def test_yields_sse_chunks_and_done(self):
        from api_model_proxy.streaming import _iterate_stream

        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"content": "Hello"}
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {"content": " world"}

        stream = [chunk1, chunk2]
        result = await self._collect(
            _iterate_stream(stream, postprocess=lambda d: d)
        )

        assert 'data: {"content": "Hello"}' in result
        assert 'data: {"content": " world"}' in result
        assert "data: [DONE]" in result

    async def test_applies_postprocess_to_each_chunk(self):
        from api_model_proxy.streaming import _iterate_stream

        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "original"}

        stream = [chunk]
        result = await self._collect(
            _iterate_stream(stream, postprocess=lambda d: {**d, "modified": True})
        )

        assert '"modified": true' in result

    async def test_applies_postprocess_to_successive_chunks(self):
        from api_model_proxy.streaming import _iterate_stream

        calls = []

        def tracking(d):
            calls.append(d["content"])
            return d

        chunks = []
        for text in ["a", "b", "c"]:
            m = MagicMock()
            m.model_dump.return_value = {"content": text}
            chunks.append(m)

        await self._collect(
            _iterate_stream(chunks, postprocess=tracking)
        )

        assert calls == ["a", "b", "c"]

    async def test_calls_serializer_for_each_chunk(self):
        from api_model_proxy.streaming import _iterate_stream

        chunk = MagicMock()
        raw_obj = object()

        def custom_serializer(obj):
            return {"custom": True, "obj": id(obj)}

        stream = [raw_obj]
        result = await self._collect(
            _iterate_stream(stream, postprocess=lambda d: d, serializer=custom_serializer)
        )

        assert '"custom": true' in result

    async def test_handles_empty_stream(self):
        from api_model_proxy.streaming import _iterate_stream

        result = await self._collect(
            _iterate_stream([], postprocess=lambda d: d)
        )

        assert result == "data: [DONE]\n\n"

    async def test_handles_mid_stream_openai_error(self):
        from api_model_proxy.streaming import _iterate_stream

        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "before error"}

        # A stream that yields one good chunk then raises
        def raising_stream():
            yield chunk
            raise OpenAIError("Connection reset")

        result = await self._collect(
            _iterate_stream(raising_stream(), postprocess=lambda d: d)
        )

        assert 'data: {"content": "before error"}' in result
        assert '"error"' in result
        assert '"Connection reset"' in result
        assert "data: [DONE]" in result

    async def test_mid_stream_error_also_passes_through_postprocess(self):
        from api_model_proxy.streaming import _iterate_stream

        calls = []

        def tracking(d):
            calls.append(d)
            return d

        def raising_stream():
            yield MagicMock(model_dump=MagicMock(return_value={"content": "ok"}))
            raise OpenAIError("Oops")

        await self._collect(
            _iterate_stream(raising_stream(), postprocess=tracking)
        )

        # postprocess called for both the success chunk and the error dict
        assert len(calls) == 2
        assert calls[0] == {"content": "ok"}
        assert "error" in calls[1]

    async def test_uses_default_serializer(self):
        from api_model_proxy.streaming import _iterate_stream

        chunk = MagicMock()
        chunk.model_dump.return_value = {"default": "serializer"}

        stream = [chunk]
        result = await self._collect(
            _iterate_stream(stream, postprocess=lambda d: d)
        )

        assert '"default": "serializer"' in result

    async def test_bytes_output(self):
        """Verify the generator yields bytes (not strings)."""
        from api_model_proxy.streaming import _iterate_stream

        chunk = MagicMock()
        chunk.model_dump.return_value = {"key": "val"}

        stream = [chunk]
        async for item in _iterate_stream(stream, postprocess=lambda d: d):
            assert isinstance(item, bytes)
            break
