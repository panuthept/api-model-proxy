from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api_model_proxy import APIModelProxy


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
