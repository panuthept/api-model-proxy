from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from openai import APIError, APIStatusError, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


def _mock_chat_completion() -> dict:
    msg = ChatCompletionMessage(role="assistant", content="Hello!")
    choice = Choice(finish_reason="stop", index=0, message=msg)
    completion = ChatCompletion(
        id="chatcmpl-xxx",
        choices=[choice],
        created=1700000000,
        model="gpt-4",
        object="chat.completion",
    )
    return completion.model_dump()


def _mock_completion() -> dict:
    return {
        "id": "cmpl-xxx",
        "choices": [{"text": "Hello!", "finish_reason": "stop", "index": 0}],
        "created": 1700000000,
        "model": "gpt-4",
        "object": "text_completion",
    }


def _mock_embedding() -> dict:
    return {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "text-embedding-ada-002",
        "object": "list",
    }


def _mock_moderation() -> dict:
    return {
        "id": "modr-xxx",
        "model": "text-moderation-latest",
        "results": [
            {
                "flagged": False,
                "categories": {
                    "harassment": False,
                    "harassment/threatening": False,
                    "hate": False,
                    "hate/threatening": False,
                    "self-harm": False,
                    "self-harm/instructions": False,
                    "self-harm/intent": False,
                    "sexual": False,
                    "sexual/minors": False,
                    "violence": False,
                    "violence/graphic": False,
                    "illicit": False,
                    "illicit/violent": False,
                },
                "category_scores": {
                    "harassment": 0.0,
                    "harassment/threatening": 0.0,
                    "hate": 0.0,
                    "hate/threatening": 0.0,
                    "self-harm": 0.0,
                    "self-harm/instructions": 0.0,
                    "self-harm/intent": 0.0,
                    "sexual": 0.0,
                    "sexual/minors": 0.0,
                    "violence": 0.0,
                    "violence/graphic": 0.0,
                    "illicit": 0.0,
                    "illicit/violent": 0.0,
                },
                "category_applied_input_types": {
                    "harassment": ["text"],
                    "harassment/threatening": ["text"],
                    "hate": ["text"],
                    "hate/threatening": ["text"],
                    "self-harm": ["text"],
                    "self-harm/instructions": ["text"],
                    "self-harm/intent": ["text"],
                    "sexual": ["text"],
                    "sexual/minors": ["text"],
                    "violence": ["text"],
                    "violence/graphic": ["text"],
                    "illicit": ["text"],
                    "illicit/violent": ["text"],
                },
            }
        ],
    }


def _mock_response() -> dict:
    return {
        "id": "resp-xxx",
        "object": "response",
        "output": [{"type": "message", "content": [{"text": "Hello!"}]}],
    }


def _mock_image() -> dict:
    return {
        "created": 1700000000,
        "data": [{"url": "https://example.com/image.png"}],
    }


def _make_openai_error(status_code: int, message: str) -> OpenAIError:
    import json

    import httpx

    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        content=json.dumps({"error": {"message": message, "type": "invalid_request_error"}}),
    )
    return APIStatusError(
        message=message,
        response=response,
        body={"error": {"message": message, "type": "invalid_request_error"}},
    )


class TestChatCompletions:
    def test_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.return_value = ChatCompletion.model_validate(
            _mock_chat_completion()
        )

        resp = client.post("/chat/completions", json={"model": "gpt-4", "messages": []})

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-xxx"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[]
        )

    def test_success_v1_prefix(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.return_value = ChatCompletion.model_validate(
            _mock_chat_completion()
        )

        resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})

        assert resp.status_code == 200

    def test_streaming_not_supported(self, client: TestClient, mock_openai_client: MagicMock):
        resp = client.post(
            "/chat/completions",
            json={"model": "gpt-4", "messages": [], "stream": True},
        )

        assert resp.status_code == 501
        assert resp.json()["error"]["code"] == "streaming_not_supported"
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.chat.completions.create.side_effect = _make_openai_error(
            400, "Bad Request"
        )

        resp = client.post("/chat/completions", json={"model": "gpt-4", "messages": []})

        assert resp.status_code == 400
        assert "Bad Request" in resp.json()["error"]["message"]

    def test_hooks_are_called(self, client: TestClient, mock_openai_client: MagicMock, proxy):
        mock_openai_client.chat.completions.create.return_value = ChatCompletion.model_validate(
            _mock_chat_completion()
        )

        calls = {"pre": 0, "post": 0}

        original_pre = proxy._preprocess_request
        original_post = proxy._postprocess_response

        def tracking_pre(request):
            calls["pre"] += 1
            return original_pre(request)

        def tracking_post(response):
            calls["post"] += 1
            return original_post(response)

        proxy._preprocess_request = tracking_pre
        proxy._postprocess_response = tracking_post

        client.post("/chat/completions", json={"model": "gpt-4", "messages": []})

        assert calls["pre"] == 1
        assert calls["post"] == 1

    def test_hooks_are_called_on_error(self, client: TestClient, mock_openai_client: MagicMock, proxy):
        mock_openai_client.chat.completions.create.side_effect = _make_openai_error(
            400, "Bad Request"
        )

        calls = {"pre": 0, "post": 0}

        original_pre = proxy._preprocess_request
        original_post = proxy._postprocess_response

        def tracking_pre(request):
            calls["pre"] += 1
            return original_pre(request)

        def tracking_post(response):
            calls["post"] += 1
            return original_post(response)

        proxy._preprocess_request = tracking_pre
        proxy._postprocess_response = tracking_post

        client.post("/chat/completions", json={"model": "gpt-4", "messages": []})

        assert calls["pre"] == 1
        assert calls["post"] == 1


class TestCompletions:
    def test_success(self, client: TestClient, mock_openai_client: MagicMock):
        from openai.types.completion import Completion

        mock_openai_client.completions.create.return_value = Completion.model_validate(
            _mock_completion()
        )

        resp = client.post("/completions", json={"model": "gpt-4", "prompt": "Hello"})

        assert resp.status_code == 200
        assert resp.json()["id"] == "cmpl-xxx"

    def test_streaming_not_supported(self, client: TestClient, mock_openai_client: MagicMock):
        resp = client.post(
            "/completions",
            json={"model": "gpt-4", "prompt": "Hello", "stream": True},
        )

        assert resp.status_code == 501

    def test_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.completions.create.side_effect = _make_openai_error(
            429, "Rate limit exceeded"
        )

        resp = client.post("/completions", json={"model": "gpt-4", "prompt": "Hello"})

        assert resp.status_code == 429


class TestResponses:
    def test_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.responses.create.return_value = MagicMock(
            model_dump=MagicMock(return_value=_mock_response())
        )

        resp = client.post("/responses", json={"model": "gpt-4", "input": "Hello"})

        assert resp.status_code == 200
        assert resp.json()["id"] == "resp-xxx"

    def test_streaming_not_supported(self, client: TestClient, mock_openai_client: MagicMock):
        resp = client.post(
            "/responses",
            json={"model": "gpt-4", "input": "Hello", "stream": True},
        )

        assert resp.status_code == 501

    def test_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.responses.create.side_effect = _make_openai_error(
            500, "Internal error"
        )

        resp = client.post("/responses", json={"model": "gpt-4", "input": "Hello"})

        assert resp.status_code == 500


class TestEmbeddings:
    def test_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_result = MagicMock()
        mock_result.model_dump.return_value = _mock_embedding()
        mock_openai_client.embeddings.create.return_value = mock_result

        resp = client.post(
            "/embeddings",
            json={"model": "text-embedding-ada-002", "input": "Hello"},
        )

        assert resp.status_code == 200
        assert resp.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.embeddings.create.side_effect = _make_openai_error(
            400, "Invalid input"
        )

        resp = client.post(
            "/embeddings",
            json={"model": "text-embedding-ada-002", "input": ""},
        )

        assert resp.status_code == 400


class TestAudio:
    def test_transcriptions_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.audio.transcriptions.create.return_value = MagicMock(
            model_dump=MagicMock(return_value={"text": "Hello world"})
        )

        resp = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", b"fake-audio-data", "audio/mpeg")},
            data={"model": "whisper-1"},
        )

        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello world"

    def test_translations_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.audio.translations.create.return_value = MagicMock(
            model_dump=MagicMock(return_value={"text": "Bonjour le monde"})
        )

        resp = client.post(
            "/audio/translations",
            files={"file": ("test.mp3", b"fake-audio-data", "audio/mpeg")},
            data={"model": "whisper-1"},
        )

        assert resp.status_code == 200
        assert resp.json()["text"] == "Bonjour le monde"

    def test_speech_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_result = MagicMock()
        mock_result.read.return_value = b"fake-audio-bytes"
        mock_openai_client.audio.speech.create.return_value = mock_result

        resp = client.post(
            "/audio/speech",
            json={"model": "tts-1", "input": "Hello", "voice": "alloy"},
        )

        assert resp.status_code == 200
        assert resp.content == b"fake-audio-bytes"
        assert resp.headers["content-type"] == "audio/mpeg"

    def test_audio_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.audio.transcriptions.create.side_effect = _make_openai_error(
            400, "Audio error"
        )

        resp = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", b"fake", "audio/mpeg")},
            data={"model": "whisper-1"},
        )

        assert resp.status_code == 400


class TestImages:
    def test_generations_success(self, client: TestClient, mock_openai_client: MagicMock):
        from openai.types.images_response import ImagesResponse

        mock_openai_client.images.generate.return_value = ImagesResponse.model_validate(
            _mock_image()
        )

        resp = client.post(
            "/images/generations",
            json={"model": "dall-e-3", "prompt": "A cat"},
        )

        assert resp.status_code == 200
        assert resp.json()["data"][0]["url"] == "https://example.com/image.png"

    def test_edits_success(self, client: TestClient, mock_openai_client: MagicMock):
        from openai.types.images_response import ImagesResponse

        mock_openai_client.images.edit.return_value = ImagesResponse.model_validate(
            _mock_image()
        )

        resp = client.post(
            "/images/edits",
            files={
                "image": ("cat.png", b"fake-image-data", "image/png"),
            },
            data={"model": "dall-e-2", "prompt": "A cat with a hat"},
        )

        assert resp.status_code == 200

    def test_variations_success(self, client: TestClient, mock_openai_client: MagicMock):
        from openai.types.images_response import ImagesResponse

        mock_openai_client.images.create_variation.return_value = (
            ImagesResponse.model_validate(_mock_image())
        )

        resp = client.post(
            "/images/variations",
            files={"image": ("cat.png", b"fake-image-data", "image/png")},
            data={"model": "dall-e-2"},
        )

        assert resp.status_code == 200

    def test_images_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.images.generate.side_effect = _make_openai_error(
            400, "Content policy violation"
        )

        resp = client.post(
            "/images/generations",
            json={"model": "dall-e-3", "prompt": "something inappropriate"},
        )

        assert resp.status_code == 400


class TestModerations:
    def test_success(self, client: TestClient, mock_openai_client: MagicMock):
        mock_result = MagicMock()
        mock_result.model_dump.return_value = _mock_moderation()
        mock_openai_client.moderations.create.return_value = mock_result

        resp = client.post(
            "/moderations",
            json={"model": "text-moderation-latest", "input": "Hello"},
        )

        assert resp.status_code == 200
        assert resp.json()["results"][0]["flagged"] is False

    def test_openai_error(self, client: TestClient, mock_openai_client: MagicMock):
        mock_openai_client.moderations.create.side_effect = _make_openai_error(
            400, "Moderation error"
        )

        resp = client.post(
            "/moderations",
            json={"model": "text-moderation-latest", "input": ""},
        )

        assert resp.status_code == 400
