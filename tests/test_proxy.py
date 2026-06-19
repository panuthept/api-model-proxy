from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from api_model_proxy import APIModelProxy, PipelineResult


class TestAPIModelProxyInit:
    def test_stores_client(self, mock_openai_client: MagicMock):
        proxy = APIModelProxy(mock_openai_client)
        assert proxy._client is mock_openai_client


class TestAPIModelProxyHooks:
    def test_preprocess_request_default_noop(self, proxy: APIModelProxy):
        request = {"model": "gpt-4", "messages": []}
        result = proxy._preprocess_request(request)
        assert result is request

    def test_postprocess_response_default_noop(self, proxy: APIModelProxy):
        response = {"choices": []}
        result = proxy._postprocess_response(response)
        assert result is response

    def test_subclass_can_override_both_hooks(self, mock_openai_client: MagicMock):
        class CustomProxy(APIModelProxy):
            def _preprocess_request(self, request):
                request["modified"] = True
                return request

            def _postprocess_response(self, response):
                response["modified"] = True
                return response

        proxy = CustomProxy(mock_openai_client)
        req = {"model": "gpt-4"}
        resp = {"choices": []}

        assert proxy._preprocess_request(req) == {"model": "gpt-4", "modified": True}
        assert proxy._postprocess_response(resp) == {"choices": [], "modified": True}

    def test_preprocess_can_raise(self, mock_openai_client: MagicMock):
        class BlockingProxy(APIModelProxy):
            def _preprocess_request(self, request):
                raise ValueError("blocked")

        proxy = BlockingProxy(mock_openai_client)
        with pytest.raises(ValueError, match="blocked"):
            proxy._preprocess_request({"model": "gpt-4"})


class TestAPIModelProxyDeploy:
    def test_deploy_creates_app_and_runs_uvicorn(self, mock_openai_client: MagicMock):
        proxy = APIModelProxy(mock_openai_client)

        with (
            patch("api_model_proxy.server.create_app") as mock_create_app,
            patch("api_model_proxy.proxy.uvicorn.run") as mock_uvicorn_run,
        ):
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app
            proxy.deploy(host="0.0.0.0", port=9000)

        mock_create_app.assert_called_once_with(proxy)
        mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=9000)

    def test_deploy_defaults(self, mock_openai_client: MagicMock):
        proxy = APIModelProxy(mock_openai_client)

        with (
            patch("api_model_proxy.server.create_app") as mock_create_app,
            patch("api_model_proxy.proxy.uvicorn.run") as mock_uvicorn_run,
        ):
            proxy.deploy()

        mock_create_app.assert_called_once_with(proxy)
        mock_uvicorn_run.assert_called_once_with(
            mock_create_app.return_value,
            host="localhost",
            port=8000,
        )


class TestExecuteRequest:
    """Tests for APIModelProxy.execute_request()."""

    def test_default_pipeline_calls_preprocess(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.return_value = MagicMock()
        mock_openai_client.chat.completions.create.return_value.model_dump.return_value = {"choices": []}

        calls = {"pre": 0}
        original = proxy._preprocess_request

        def tracking(request):
            calls["pre"] += 1
            return original(request)

        proxy._preprocess_request = tracking
        proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )
        assert calls["pre"] == 1

    def test_default_pipeline_calls_sdk(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        mock_sdk = mock_openai_client.chat.completions.create
        mock_sdk.return_value = MagicMock()
        mock_sdk.return_value.model_dump.return_value = {"choices": []}

        proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_sdk,
        )
        mock_sdk.assert_called_once_with(model="gpt-4", messages=[])

    def test_default_pipeline_calls_serializer(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        mock_sdk = mock_openai_client.chat.completions.create
        mock_raw = MagicMock()
        mock_sdk.return_value = mock_raw

        serializer = MagicMock(return_value={"custom": True})
        proxy.execute_request(
            body={"model": "gpt-4"},
            sdk_method=mock_sdk,
            serializer=serializer,
        )
        serializer.assert_called_once_with(mock_raw)

    def test_default_pipeline_calls_postprocess(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.return_value = MagicMock()
        mock_openai_client.chat.completions.create.return_value.model_dump.return_value = {"choices": []}

        calls = {"post": 0}
        original = proxy._postprocess_response

        def tracking(response):
            calls["post"] += 1
            return original(response)

        proxy._postprocess_response = tracking
        result = proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )
        assert calls["post"] == 1
        assert result.content == {"choices": []}

    def test_default_pipeline_handles_openai_error(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        from openai import APIStatusError
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit", "type": "rate_limit_error"}}

        mock_response.headers = MagicMock()

        mock_openai_client.chat.completions.create.side_effect = APIStatusError(
            message="Rate limit",
            response=mock_response,
            body={"error": {"message": "Rate limit", "type": "rate_limit_error"}},
        )

        result = proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )

        assert result.status_code == 429
        assert "error" in result.content

    def test_default_pipeline_passes_hooks_call_on_error(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        from openai import APIStatusError
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Bad request", "type": "invalid_request_error"}}

        mock_response.headers = MagicMock()

        mock_openai_client.chat.completions.create.side_effect = APIStatusError(
            message="Bad request",
            response=mock_response,
            body={"error": {"message": "Bad request", "type": "invalid_request_error"}},
        )

        calls = {"post": 0}
        original = proxy._postprocess_response

        def tracking(response):
            calls["post"] += 1
            return original(response)

        proxy._postprocess_response = tracking
        proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )

        assert calls["post"] == 1

    def test_returns_pipeline_result(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.return_value = MagicMock()
        mock_openai_client.chat.completions.create.return_value.model_dump.return_value = {"choices": []}

        result = proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )

        from api_model_proxy import PipelineResult
        assert isinstance(result, PipelineResult)
        assert result.content == {"choices": []}
        assert result.status_code == 200

    def test_subclass_can_override_pipeline(self, mock_openai_client: MagicMock):
        class ShortCircuitProxy(APIModelProxy):
            def execute_request(self, body, sdk_method, serializer=lambda r: r.model_dump()):
                return PipelineResult(content={"cached": True}, status_code=200)

        proxy = ShortCircuitProxy(mock_openai_client)
        result = proxy.execute_request(
            body={"model": "gpt-4"},
            sdk_method=mock_openai_client.chat.completions.create,
        )
        assert result.content == {"cached": True}
        assert result.status_code == 200
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_subclass_can_delegate_to_default(self, mock_openai_client: MagicMock):
        class LoggingProxy(APIModelProxy):
            def execute_request(self, body, sdk_method, serializer=lambda r: r.model_dump()):
                print(f"Request: {body}")
                result = super().execute_request(body, sdk_method, serializer)
                print(f"Response: {result.content}")
                return result

        mock_openai_client.chat.completions.create.return_value = MagicMock()
        mock_openai_client.chat.completions.create.return_value.model_dump.return_value = {"choices": []}

        proxy = LoggingProxy(mock_openai_client)
        result = proxy.execute_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )
        assert result.content == {"choices": []}
        assert result.status_code == 200


class TestExecuteStreamingRequest:
    """Tests for APIModelProxy.execute_streaming_request()."""

    @pytest.mark.asyncio
    async def test_returns_async_generator(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "Hello"}
        mock_openai_client.chat.completions.create.return_value = [chunk]

        gen = proxy.execute_streaming_request(
            body={"model": "gpt-4", "messages": [], "stream": True},
            sdk_method=mock_openai_client.chat.completions.create,
        )

        assert isinstance(gen, AsyncGenerator)

    @pytest.mark.asyncio
    async def test_calls_preprocess_request(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "Hello"}
        mock_openai_client.chat.completions.create.return_value = [chunk]

        calls = {"pre": 0}
        original = proxy._preprocess_request

        def tracking(request):
            calls["pre"] += 1
            return original(request)

        proxy._preprocess_request = tracking

        async for _ in proxy.execute_streaming_request(
            body={"model": "gpt-4", "messages": [], "stream": True},
            sdk_method=mock_openai_client.chat.completions.create,
        ):
            pass

        assert calls["pre"] == 1

    @pytest.mark.asyncio
    async def test_calls_sdk_with_stream_true(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "Hello"}
        mock_openai_client.chat.completions.create.return_value = [chunk]

        async for _ in proxy.execute_streaming_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        ):
            pass

        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[], stream=True
        )

    @pytest.mark.asyncio
    async def test_strips_stream_from_body(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        """stream flag should be removed from body to avoid duplicate kwarg."""
        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "Hello"}
        mock_openai_client.chat.completions.create.return_value = [chunk]

        async for _ in proxy.execute_streaming_request(
            body={"model": "gpt-4", "stream": True},
            sdk_method=mock_openai_client.chat.completions.create,
        ):
            pass

        # stream=True should only appear once (as explicit kwarg, not in body)
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", stream=True
        )

    @pytest.mark.asyncio
    async def test_yields_sse_bytes(self, proxy: APIModelProxy, mock_openai_client: MagicMock):
        chunk = MagicMock()
        chunk.model_dump.return_value = {"content": "Hello"}
        mock_openai_client.chat.completions.create.return_value = [chunk]

        gen = proxy.execute_streaming_request(
            body={"model": "gpt-4", "messages": []},
            sdk_method=mock_openai_client.chat.completions.create,
        )

        result = b"".join([item async for item in gen]).decode("utf-8")
        assert 'data: {"content": "Hello"}' in result
        assert "data: [DONE]" in result

    @pytest.mark.asyncio
    async def test_subclass_can_override_streaming(self, mock_openai_client: MagicMock):
        class CustomStreamProxy(APIModelProxy):
            async def custom_gen(self):
                yield b"custom stream data"

            def execute_streaming_request(self, body, sdk_method, serializer=lambda r: r.model_dump()):
                return self.custom_gen()

        proxy = CustomStreamProxy(mock_openai_client)
        gen = proxy.execute_streaming_request(
            body={"model": "gpt-4"},
            sdk_method=mock_openai_client.chat.completions.create,
        )
        result = b"".join([item async for item in gen]).decode("utf-8")
        assert result == "custom stream data"
        mock_openai_client.chat.completions.create.assert_not_called()
