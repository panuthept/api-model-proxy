from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient


def _mock_async_client(upstream_response: httpx.Response) -> AsyncMock:
    mock_async_client = AsyncMock()
    mock_async_client.request = AsyncMock(return_value=upstream_response)
    return mock_async_client


class TestPassthrough:
    def test_forwards_request_to_upstream(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=200,
            json={"data": [{"id": "model-1"}]},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client_class.return_value.__aenter__.return_value = _mock_async_client(
                upstream_response
            )

            resp = client.get("/models")

        assert resp.status_code == 200
        assert resp.json() == {"data": [{"id": "model-1"}]}

    def test_sets_authorization_header(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=200,
            json={},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_client = _mock_async_client(upstream_response)
            mock_async_client_class.return_value.__aenter__.return_value = mock_client

            client.get("/models")

        _, kwargs = mock_client.request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    def test_preserves_query_string(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=200,
            json={},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_client = _mock_async_client(upstream_response)
            mock_async_client_class.return_value.__aenter__.return_value = mock_client

            client.get("/models?limit=10")

        _, kwargs = mock_client.request.call_args
        assert kwargs["url"] == "https://api.openai.com/v1/models?limit=10"

    def test_preserves_upstream_status_code(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=404,
            json={"error": {"message": "Not found"}},
            request=httpx.Request("GET", "https://api.openai.com/v1/nonexistent"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client_class.return_value.__aenter__.return_value = _mock_async_client(
                upstream_response
            )

            resp = client.get("/nonexistent")

        assert resp.status_code == 404
        assert resp.json() == {"error": {"message": "Not found"}}

    def test_post_request(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=200,
            json={"id": "file-xxx"},
            request=httpx.Request("POST", "https://api.openai.com/v1/files"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_client = _mock_async_client(upstream_response)
            mock_async_client_class.return_value.__aenter__.return_value = mock_client

            resp = client.post("/files", json={"purpose": "fine-tune"})

        assert resp.status_code == 200

        _, kwargs = mock_client.request.call_args
        assert kwargs["method"] == "POST"

    def test_uses_organization_header_when_set(
        self, client: TestClient, mock_openai_client: MagicMock
    ):
        mock_openai_client.organization = "org-123"

        upstream_response = httpx.Response(
            status_code=200,
            json={},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_client = _mock_async_client(upstream_response)
            mock_async_client_class.return_value.__aenter__.return_value = mock_client

            client.get("/models")

        _, kwargs = mock_client.request.call_args
        assert kwargs["headers"]["OpenAI-Organization"] == "org-123"

    def test_strips_hop_by_hop_headers(self, client: TestClient, mock_openai_client: MagicMock):
        upstream_response = httpx.Response(
            status_code=200,
            json={},
            headers={"content-type": "application/json", "transfer-encoding": "chunked"},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client_class.return_value.__aenter__.return_value = _mock_async_client(
                upstream_response
            )

            resp = client.get("/models")

        assert resp.status_code == 200
        assert "transfer-encoding" not in resp.headers
        assert resp.headers["content-type"] == "application/json"
