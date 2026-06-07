from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest
from openai import OpenAI

from api_model_proxy import APIModelProxy
from api_model_proxy.server import create_app


@pytest.fixture
def mock_openai_client() -> MagicMock:
    client = MagicMock(spec=OpenAI)

    type(client).base_url = PropertyMock(
        return_value="https://api.openai.com/v1/"
    )
    client.api_key = "sk-test-key"

    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_responses = MagicMock()
    mock_embeddings = MagicMock()
    mock_audio = MagicMock()
    mock_images = MagicMock()
    mock_moderations = MagicMock()

    client.chat = mock_chat
    client.completions = mock_completions
    client.responses = mock_responses
    client.embeddings = mock_embeddings
    client.audio = mock_audio
    client.images = mock_images
    client.moderations = mock_moderations

    return client


@pytest.fixture
def proxy(mock_openai_client: MagicMock) -> APIModelProxy:
    return APIModelProxy(mock_openai_client)


@pytest.fixture
def app(proxy: APIModelProxy):
    return create_app(proxy)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
