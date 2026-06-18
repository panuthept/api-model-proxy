from __future__ import annotations

from typing import Any
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


# ------------------------------------------------------------------
# Streaming mock fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_chat_stream_chunks() -> list[dict[str, Any]]:
    """Return a list of raw dicts representing chat completion stream chunks."""
    return [
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        },
    ]


@pytest.fixture
def mock_completion_stream_chunks() -> list[dict[str, Any]]:
    """Return a list of raw dicts representing legacy completion stream chunks."""
    return [
        {
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"text": "Hello", "index": 0, "finish_reason": None, "logprobs": None}
            ],
        },
        {
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"text": " world", "index": 0, "finish_reason": None, "logprobs": None}
            ],
        },
        {
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {"text": "", "index": 0, "finish_reason": "stop", "logprobs": None}
            ],
        },
    ]


@pytest.fixture
def mock_response_stream_chunks() -> list[dict[str, Any]]:
    """Return a list of raw dicts representing Responses API stream events."""
    return [
        {
            "type": "response.output_text.delta",
            "delta": "Hello",
        },
        {
            "type": "response.output_text.delta",
            "delta": " world",
        },
        {
            "type": "response.output_text.done",
            "delta": None,
        },
    ]


@pytest.fixture
def mock_chat_stream(mock_chat_stream_chunks: list[dict[str, Any]]) -> list[MagicMock]:
    """Return a list of MagicMock objects simulating a chat stream."""
    chunks = []
    for chunk_dict in mock_chat_stream_chunks:
        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = chunk_dict
        chunks.append(mock_chunk)
    return chunks


@pytest.fixture
def mock_completion_stream(
    mock_completion_stream_chunks: list[dict[str, Any]],
) -> list[MagicMock]:
    """Return a list of MagicMock objects simulating a completion stream."""
    chunks = []
    for chunk_dict in mock_completion_stream_chunks:
        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = chunk_dict
        chunks.append(mock_chunk)
    return chunks


@pytest.fixture
def mock_response_stream(
    mock_response_stream_chunks: list[dict[str, Any]],
) -> list[MagicMock]:
    """Return a list of MagicMock objects simulating a responses stream."""
    chunks = []
    for chunk_dict in mock_response_stream_chunks:
        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = chunk_dict
        chunks.append(mock_chunk)
    return chunks


@pytest.fixture
def mock_client_with_chat_stream(
    mock_openai_client: MagicMock, mock_chat_stream: list[MagicMock]
) -> MagicMock:
    """Configure the mock OpenAI client to return a chat stream on create(stream=True)."""
    mock_openai_client.chat.completions.create.return_value = mock_chat_stream
    return mock_openai_client


@pytest.fixture
def mock_client_with_completion_stream(
    mock_openai_client: MagicMock, mock_completion_stream: list[MagicMock]
) -> MagicMock:
    """Configure the mock OpenAI client to return a completion stream on create(stream=True)."""
    mock_openai_client.completions.create.return_value = mock_completion_stream
    return mock_openai_client


@pytest.fixture
def mock_client_with_response_stream(
    mock_openai_client: MagicMock, mock_response_stream: list[MagicMock]
) -> MagicMock:
    """Configure the mock OpenAI client to return a response stream on create(stream=True)."""
    mock_openai_client.responses.create.return_value = mock_response_stream
    return mock_openai_client


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
