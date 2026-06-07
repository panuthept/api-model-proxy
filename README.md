# API Model Proxy

A lightweight, extensible proxy layer for OpenAI-compatible APIs. Drop it between your application and any OpenAI-compatible backend to intercept, log, modify, or filter requests and responses — without changing your existing client code.

## Features

- **OpenAI-compatible** — works as a drop-in with any `openai.OpenAI` client by setting `base_url`
- **Hook-based** — override `_preprocess_request` and `_postprocess_response` to customise behaviour
- **Full API coverage** — proxies all inference endpoints (chat, completions, responses, embeddings, audio, images, moderations) with hooks; all other endpoints (models, files, fine-tuning, batches, vector stores, etc.) are forwarded transparently
- **Error visibility** — `_postprocess_response` is called on both success and error responses, allowing subclasses to inspect or modify errors
- **FastAPI + uvicorn** — async, production-ready server with auto-generated docs at `/docs`

## Installation

```bash
git clone https://github.com/panuthept/api-model-proxy.git
cd api-model-proxy
python -m venv venv && source venv/bin/activate
pip install -e .
```

## Quick Start

### 1. Deploy the proxy

```python
from openai import OpenAI
from api_model_proxy import APIModelProxy

class LoggingProxy(APIModelProxy):
    def _preprocess_request(self, request):
        print(f"Request: {request}")
        return request

    def _postprocess_response(self, response):
        print(f"Response: {response}")
        return response

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",  # or any OpenAI-compatible endpoint
)
proxy = LoggingProxy(client)
proxy.deploy(host="localhost", port=8000)
```

### 2. Point your client at the proxy

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000", api_key="EMPTY")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)
```

No other changes needed — the proxy is fully transparent to the client.

## Customisation

Subclass `APIModelProxy` and override either or both hooks:

| Method | Called on | Default behaviour |
|---|---|---|
| `_preprocess_request(request: dict) -> dict` | Every inference request, before forwarding | No-op (return as-is) |
| `_postprocess_response(response: dict) -> dict` | Every inference response (success **and** error) | No-op (return as-is) |

### Example: request filtering

```python
class FilterProxy(APIModelProxy):
    _BLOCKED = {"gpt-4o", "gpt-4-turbo"}

    def _preprocess_request(self, request):
        if request.get("model") in self._BLOCKED:
            raise ValueError(f"Model {request['model']} is not allowed.")
        return request
```

### Example: response caching

```python
import hashlib, json

class CachingProxy(APIModelProxy):
    def __init__(self, client):
        super().__init__(client)
        self._cache = {}
        self._last_key = None

    def _preprocess_request(self, request):
        self._last_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return request

    def _postprocess_response(self, response):
        if self._last_key:
            self._cache[self._last_key] = response
        return response
```

## Endpoints

### Inference (hooks fire)

| Method | Path |
|---|---|
| POST | `/chat/completions` |
| POST | `/completions` |
| POST | `/responses` |
| POST | `/embeddings` |
| POST | `/audio/transcriptions` |
| POST | `/audio/translations` |
| POST | `/audio/speech` |
| POST | `/images/generations` |
| POST | `/images/edits` |
| POST | `/images/variations` |
| POST | `/moderations` |

All paths are also available with a `/v1/` prefix.

### Passthrough (no hooks, forwarded verbatim)

All other endpoints — models, files, fine-tuning, batches, vector stores, evals, organisation admin, and any future OpenAI endpoints — are forwarded transparently using the upstream client's base URL and API key.

## Streaming

Streaming (`stream=True`) is not yet supported. Requests with `stream=True` will receive a `501 Not Implemented` response. Streaming support is planned for the next version.

## Project Structure

```
src/api_model_proxy/
├── __init__.py          # exports APIModelProxy
├── proxy.py             # APIModelProxy base class
├── server.py            # FastAPI app factory
└── routes/
    ├── chat.py          # POST /chat/completions
    ├── completions.py   # POST /completions
    ├── responses.py     # POST /responses
    ├── embeddings.py    # POST /embeddings
    ├── audio.py         # POST /audio/*
    ├── images.py        # POST /images/*
    ├── moderations.py   # POST /moderations
    └── passthrough.py   # catch-all transparent proxy
```

## License

MIT
