# API Reference

## APIModelProxy

```python
class APIModelProxy(openai_client: OpenAI)
```

The base class for building OpenAI-compatible API proxies. Subclasses override the hook methods to intercept inference requests and responses.

### Constructor

```python
def __init__(self, openai_client: OpenAI) -> None
```

| Parameter | Type | Description |
|---|---|---|
| `openai_client` | `openai.OpenAI` | Configured OpenAI client instance. Its `base_url` and `api_key` are used for upstream requests. |

### Hooks

#### `_preprocess_request`

```python
def _preprocess_request(self, request: dict) -> dict
```

Called before every inference request is forwarded to the upstream API.

| | |
|---|---|
| **Args** | `request` — the raw request body parsed from JSON (or multipart form data) |
| **Returns** | The (possibly modified) request dict to forward upstream |
| **Default** | No-op (returns `request` unchanged) |
| **Raise** | Any exception to abort the request; the caller receives a `500 Internal Server Error` |

Called for all inference endpoints: chat, completions, responses, embeddings, audio, images, and moderations.

#### `_postprocess_response`

```python
def _postprocess_response(self, response: dict) -> dict
```

Called after every inference response is received from the upstream API.

| | |
|---|---|
| **Args** | `response` — the response body as a dict (on success) or the error body as a dict (on `OpenAIError`) |
| **Returns** | The (possibly modified) response dict to return to the caller |
| **Default** | No-op (returns `response` unchanged) |

This method is **always** called — on both `2xx` success responses and `4xx`/`5xx` error responses. Subclasses can inspect or modify error bodies the same way they handle success responses.

**Note for `/audio/speech`**: The response is binary audio bytes, not JSON. The hook receives a dict with a single key `{"bytes_length": <int>}` instead of the full response body. The audio bytes are forwarded as-is.

### Server

#### `deploy`

```python
def deploy(self, host: str = "localhost", port: int = 8000) -> None
```

Creates the FastAPI application via `create_app()` and starts the uvicorn server.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Bind address |
| `port` | `int` | `8000` | Bind port |

This call blocks until the server is shut down. Use `uvicorn.run()` under the hood, so the server supports graceful shutdown with `SIGINT`/`SIGTERM`.

---

## create_app

```python
def create_app(proxy: APIModelProxy) -> FastAPI
```

Factory function that builds and configures the FastAPI application.

| Parameter | Type | Description |
|---|---|---|
| `proxy` | `APIModelProxy` | The proxy instance whose hooks are called on inference requests. |

The returned `FastAPI` instance stores the proxy on `app.state.proxy`, making it accessible in route handlers via `request.app.state.proxy`.

**Router registration order:**

1. **Inference routers** (hooks fire) — registered under both `""` and `"/v1"` prefixes:
   - `/chat/completions`
   - `/completions`
   - `/responses`
   - `/embeddings`
   - `/audio/transcriptions`, `/audio/translations`, `/audio/speech`
   - `/images/generations`, `/images/edits`, `/images/variations`
   - `/moderations`
2. **Passthrough router** — registered last as a catch-all for `/{full_path:path}`

The application also provides auto-generated OpenAPI docs at `/docs` and `/redoc`.

---

## Route Handlers

### Inference Routes

All inference routes follow the same pattern:

1. Parse request body from JSON (or `multipart/form-data` for audio/images).
2. Call `proxy._preprocess_request(body)`.
3. Forward the (possibly modified) body to the corresponding OpenAI SDK method.
4. Serialise the SDK response via `.model_dump()`.
5. Call `proxy._postprocess_response(response_dict)`.
6. Return the (possibly modified) response as JSON.

On `OpenAIError` during step 3, the error is serialised to a dict and passed through `_postprocess_response` before being returned with the appropriate HTTP status code.

| Route | OpenAI SDK Method | Body Format |
|---|---|---|
| `POST /chat/completions` | `client.chat.completions.create()` | JSON |
| `POST /completions` | `client.completions.create()` | JSON |
| `POST /responses` | `client.responses.create()` | JSON |
| `POST /embeddings` | `client.embeddings.create()` | JSON |
| `POST /audio/transcriptions` | `client.audio.transcriptions.create()` | `multipart/form-data` |
| `POST /audio/translations` | `client.audio.translations.create()` | `multipart/form-data` |
| `POST /audio/speech` | `client.audio.speech.create()` | JSON (response is raw bytes) |
| `POST /images/generations` | `client.images.generate()` | JSON |
| `POST /images/edits` | `client.images.edit()` | `multipart/form-data` |
| `POST /images/variations` | `client.images.create_variation()` | `multipart/form-data` |
| `POST /moderations` | `client.moderations.create()` | JSON |

### Passthrough Route

```
ANY /{full_path:path}
```

A catch-all handler that forwards any request not matched by an inference route verbatim to the upstream API.

- The full request path (including query string) is forwarded to `{client.base_url}/{full_path}`.
- The `Authorization` header is set to `Bearer {client.api_key}`.
- The `OpenAI-Organization` header is forwarded if configured on the client.
- Hop-by-hop headers (`host`, `transfer-encoding`, `connection`, etc.) are stripped from both the request and response.
- No hooks are called.

This covers all non-inference endpoints: models, files, fine-tuning, batches, vector stores, evals, organisation admin, and any future OpenAI API additions.

---

## Error Handling

### OpenAI Error Serialisation

When an `OpenAIError` is raised by the SDK, it is serialised to a dict using the following logic:

1. If the exception has a `response` attribute with a JSON body, that body is returned as-is (preserving the upstream error structure).
2. Otherwise, a fallback dict is constructed: `{"error": {"message": str(exc), "type": type(exc).__name__}}`.

The serialised error dict is then passed through `_postprocess_response` before being returned to the caller.

### HTTP Status Codes

| Scenario | Status Code |
|---|---|
| Success | `200` |
| Streaming requested (unsupported) | `501` |
| OpenAI API error | Extracted from `OpenAIError.status_code` (falls back to `500`) |
| Pre-process hook raises | `500` |

---

## Streaming

**Not yet supported.** Any inference request with `"stream": true` in the body receives:

```json
{
  "error": {
    "message": "Streaming is not yet supported by this proxy. It will be available in a future version.",
    "type": "not_implemented",
    "code": "streaming_not_supported"
  }
}
```

with HTTP status `501`. The upstream SDK is not called.
