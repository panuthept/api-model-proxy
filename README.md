# API Model Proxy

A lightweight, extensible proxy layer for OpenAI-compatible APIs. Drop it between your application and any OpenAI-compatible backend to intercept, log, modify, or filter requests and responses — without changing your existing client code.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-panuthept%2Fapi--model--proxy-blue?logo=github)](https://github.com/panuthept/api-model-proxy)

## Features

- **OpenAI-compatible** — works as a drop-in with any `openai.OpenAI` client by setting `base_url`
- **Hook-based** — override `_preprocess_request` and `_postprocess_response` to customise behaviour
- **Full API coverage** — proxies all inference endpoints (chat, completions, responses, embeddings, audio, images, moderations) with hooks; all other endpoints (models, files, fine-tuning, batches, vector stores, etc.) are forwarded transparently
- **Error visibility** — `_postprocess_response` is called on both success and error responses, allowing subclasses to inspect or modify errors
- **FastAPI + uvicorn** — async, production-ready server with auto-generated docs at `/docs`

## Use Cases

| Use case | Approach |
|---|---|
| **Request/response logging** | Override both hooks to log to stdout, a file, or an external service |
| **Model gating / filtering** | Raise in `_preprocess_request` to block disallowed models or prompt patterns |
| **Response caching** | Cache response dicts in `_postprocess_response` and serve from `_preprocess_request` |
| **Rate limiting / usage tracking** | Track request counts or token usage per user in `_postprocess_response` |
| **Prompt augmentation** | Inject system messages or rewrite user messages in `_preprocess_request` |
| **A/B testing** | Route requests to different model versions conditionally in `_preprocess_request` |

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

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "
from openai import OpenAI
from api_model_proxy import APIModelProxy

class LoggingProxy(APIModelProxy):
    def _preprocess_request(self, request):
        print(f'Request: {request}')
        return request
    def _postprocess_response(self, response):
        print(f'Response: {response}')
        return response

client = OpenAI(api_key='your-api-key', base_url='https://api.openai.com/v1')
LoggingProxy(client).deploy(host='0.0.0.0', port=8000)
"]
```

### docker-compose

```yaml
version: "3.9"
services:
  proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:?required}
```

Run with:

```bash
OPENAI_API_KEY=sk-... docker compose up
```

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

### More examples

See the [`examples/`](examples/) directory for ready-to-use proxy implementations:

- [`persistant_logging_proxy.py`](examples/persistant_logging_proxy.py) — logs every request/response to daily JSONL or YAML files
- [`caching_proxy.py`](examples/caching_proxy.py) — in-memory LRU response cache with configurable TTL
- [`fallback_proxy.py`](examples/fallback_proxy.py) — multi-backend circuit-breaker with automatic fallback
- [`rate_limiting_proxy.py`](examples/rate_limiting_proxy.py) — token-bucket rate limiter returning `429 Too Many Requests`

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

## Running Tests

The test suite uses `pytest`. Install the development dependencies and run:

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

The tests use a mocked `OpenAI` client so no API key or network access is required.

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
