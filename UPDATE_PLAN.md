# Architecture Change: migrate pipeline logic from route handlers into `APIModelProxy.execute_request()`

> **Status:** Draft for review  
> **Target branch:** `main`  
> **Est. impact:** 13 files changed (1 new, 10 modified, 2 unchanged)

---

## Table of Contents

1. [Context & Problem](#1-context--problem)
2. [Design Principle](#2-design-principle)
3. [Contract: `PipelineResult`](#3-contract-pipelineresult)
4. [File-by-File Changes](#4-file-by-file-changes)
   - 4.1 [NEW: `src/api_model_proxy/sentinels.py`](#41-new-srcapi_model_proxysentinelspy)
   - 4.2 [MODIFY: `src/api_model_proxy/proxy.py`](#42-modify-srcapi_model_proxyproxypy)
   - 4.3 [MODIFY: `src/api_model_proxy/__init__.py`](#43-modify-srcapi_model_proxy__init__py)
   - 4.4 [MODIFY: Route files](#44-modify-route-files)
   - 4.5 [MODIFY: `passthrough.py`](#45-modify-passthroughpy)
   - 4.6 [MODIFY: `examples/caching_proxy.py`](#46-modify-examplescaching_proxypy)
   - 4.7 [MODIFY: Tests](#47-modify-tests)
5. [Route Handler Templates](#5-route-handler-templates)
6. [Backward Compatibility](#6-backward-compatibility)
7. [Migration Guide for Existing Proxy Subclasses](#7-migration-guide-for-existing-proxy-subclasses)
8. [Implementation Order](#8-implementation-order)

---

## 1. Context & Problem

Every inference route handler (`chat.py`, `completions.py`, `responses.py`, `embeddings.py`, `moderations.py`, `audio.py`, `images.py`) currently owns the full request→response pipeline inline:

```python
body = proxy._preprocess_request(body)
try:
    result = proxy._client.chat.completions.create(**body)
    response_dict = result.model_dump()
    response_dict = proxy._postprocess_response(response_dict)
    return JSONResponse(content=response_dict)
except OpenAIError as exc:
    error_dict = _openai_error_to_dict(exc)
    error_dict = proxy._postprocess_response(error_dict)
    return JSONResponse(status_code=_status_code(exc), content=error_dict)
```

This has three concrete problems:

| # | Problem | Evidence |
|---|---------|----------|
| 1 | **Duplicated boilerplate** — the `try/except OpenAIError`, `_openai_error_to_dict`, `_status_code`, and `result.model_dump()` pattern is copied verbatim across **7 route files**. | Identical ~30-line blocks in `chat.py`, `completions.py`, `responses.py`, `embeddings.py`, `moderations.py`, `audio.py`, `images.py`. Helpers are copy-pasted in each. |
| 2 | **No short-circuit support** — a subclass that wants caching or conditional bypass has no hook to intercept before the SDK call. The existing `_preprocess_request` and `_postprocess_response` hooks always bookend an SDK call. | `examples/caching_proxy.py` uses a hacky `_cache_hit_body` attribute to store a cached response in `_preprocess_request`, then return it in `_postprocess_response` — the upstream call is still dispatched unnecessarily. |
| 3 | **No retry / fallback / custom flow** — implementing retry logic, circuit-breaker fallback, or conditional routing requires workarounds (e.g., `FallbackProxy` swaps `self._client` in `_preprocess_request`, which works but couples routing logic to the hook API). | No mechanism exists to skip the SDK call or redirect to a different backend within a single invocation. |

### Scope

- **Inference** endpoints (chat, completions, responses, embeddings, moderations, audio, images) — these fire hooks and call the OpenAI SDK.
- **Passthrough** (catch-all) endpoints — no hooks fire, and no changes are needed (see §4.5).

---

## 2. Design Principle

The architecture change is guided by one principle:

> **Each layer owns exactly one concern, and the contract between them is `PipelineResult`.**

```
┌─────────────────────────────────────────────────┐
│                 Route Handler                    │
│  Owns: endpoint matching, body parsing,          │
│        response wrapping (JSONResponse/Response) │
└──────────────────────┬──────────────────────────┘
                       │ body, sdk_method, serializer
                       ▼
┌─────────────────────────────────────────────────┐
│           APIModelProxy.execute_request()         │
│  Owns: request→response pipeline (fully          │
│        overridable by subclasses)                │
└──────────────────────┬──────────────────────────┘
                       │ PipelineResult(content, status_code, headers)
                       ▼
┌─────────────────────────────────────────────────┐
│              Route Handler (resumes)              │
│  Unpacks PipelineResult → JSONResponse/Response  │
└─────────────────────────────────────────────────┘
```

**Key rules:**

- Route handlers parse the HTTP request, call `execute_request()`, and wrap the result. They do **not** catch `OpenAIError`, do **not** call `model_dump()`, do **not** call hooks directly.
- `execute_request()` runs the full pipeline: preprocess → SDK call → serialize → postprocess → error handling → `PipelineResult`. Subclasses override this for caching, retry, or any custom flow.
- `PipelineResult` is the **only** contract. No framework-prescribed patterns like `ImmediateResponse` or `RetryRequest` — just a dataclass with `content`, `status_code`, and `headers`.
- The `_preprocess_request` / `_postprocess_response` hooks still exist and are called by the default `execute_request()` implementation. Subclasses that only override hooks will continue to work unchanged.

---

## 3. Contract: `PipelineResult`

A lightweight dataclass that every `execute_request()` implementation must return.

```python
@dataclass
class PipelineResult:
    content: dict      # The JSON-serialisable response body (or error dict)
    status_code: int   # HTTP status code (200 for success, 4xx/5xx for errors)
    headers: dict | None = None  # Optional extra response headers
```

Route handlers unpack this to build the HTTP response:

- **JSON endpoints:** `JSONResponse(content=result.content, status_code=result.status_code, headers=result.headers)`
- **Speech endpoint (special):** returns raw binary `Response`, not `PipelineResult` (see §4.4).

---

## 4. File-by-File Changes

### 4.1 NEW: `src/api_model_proxy/sentinels.py`

Create a new module for shared data structures.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineResult:
    """The contract between :meth:`APIModelProxy.execute_request` and route
    handlers.

    Every ``execute_request()`` implementation must return a ``PipelineResult``.
    Route handlers unpack it to build the HTTP response.

    Args:
        content: The JSON-serialisable response body (success payload or error
            dict).
        status_code: HTTP status code (``200`` for success, ``4xx``/``5xx``
            for errors).
        headers: Optional extra response headers (e.g. ``X-Cache``).
    """
    content: dict
    status_code: int = 200
    headers: Optional[dict] = None
```

**Rationale for a separate file:** Keeps `proxy.py` focused on the proxy class. `PipelineResult` may also be imported by route files and tests; a dedicated module prevents circular imports.

---

### 4.2 MODIFY: `src/api_model_proxy/proxy.py`

#### 4.2.1 What stays

- `APIModelProxy.__init__` — unchanged  
- `APIModelProxy.deploy` — unchanged  
- `_preprocess_request` / `_postprocess_response` hooks — unchanged (still there, still called by default `execute_request()`)

#### 4.2.2 What's added

**Import `PipelineResult`:**

```python
from .sentinels import PipelineResult
```

**New method `execute_request()`:**

```python
def execute_request(
    self,
    body: dict,
    sdk_method: Callable[..., Any],
    serializer: Callable[[Any], dict] | None = None,
) -> PipelineResult:
    """Execute the full request→response pipeline.

    This is the default implementation. Subclasses may override it
    entirely to implement caching, retry, fallback, or any custom flow.

    Args:
        body: The request body (a JSON-serialisable dict).
        sdk_method: The bound OpenAI SDK method to call
            (e.g. ``self._client.chat.completions.create``).
        serializer: Optional callable to serialise the SDK result into a
            plain dict. Defaults to :meth:`_serialize_sdk_result`.

    Returns:
        A :class:`PipelineResult` with the response (or error) content,
        status code, and optional headers.
    """
    body = self._preprocess_request(body)

    try:
        raw = sdk_method(**body)
        response_dict = (serializer or self._serialize_sdk_result)(raw)
        response_dict = self._postprocess_response(response_dict)
        return PipelineResult(content=response_dict, status_code=200)
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        error_dict = self._postprocess_response(error_dict)
        return PipelineResult(
            content=error_dict,
            status_code=_status_code(exc),
        )
```

**New helper `_serialize_sdk_result()`:**

```python
@staticmethod
def _serialize_sdk_result(result: Any) -> dict:
    """Convert an SDK result object to a plain dict.

    The OpenAI SDK returns various types:
    - Objects with ``.model_dump()`` (most endpoints)
    - ``HttpxBinaryResponseContent`` (audio speech, handled specially)
    - Plain strings (e.g. audio transcriptions with certain models)
    """
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, str):
        return {"text": result}
    # Fallback — try to cast to dict or return as-is
    return dict(result) if hasattr(result, "__iter__") else {"result": str(result)}
```

**Moved helpers (previously duplicated in every route file):**

```python
def _openai_error_to_dict(exc: OpenAIError) -> dict:
    """Serialise an :class:`openai.OpenAIError` to a plain dict."""
    if hasattr(exc, "response") and exc.response is not None:
        try:
            return exc.response.json()
        except Exception:
            return {"error": {"message": str(exc), "type": type(exc).__name__}}
    return {"error": {"message": str(exc), "type": type(exc).__name__}}


def _status_code(exc: OpenAIError) -> int:
    """Extract the HTTP status code from an :class:`openai.OpenAIError`."""
    if hasattr(exc, "status_code") and exc.status_code is not None:
        return int(exc.status_code)
    return 500
```

These are module-level functions (not methods on the class) so they can be imported by tests and the audio/speech endpoint without needing a proxy instance.

#### 4.2.3 Diff summary

```diff
+ from .sentinels import PipelineResult
+ from openai import OpenAIError
+ from typing import Any, Callable

  class APIModelProxy:
      def __init__(self, openai_client):  ...  # UNCHANGED
      def _preprocess_request(self, request):  ...  # UNCHANGED
      def _postprocess_response(self, response):  ...  # UNCHANGED
      def deploy(self, ...):  ...  # UNCHANGED

+     def execute_request(self, body, sdk_method, serializer=None):
+         ...
+
+     @staticmethod
+     def _serialize_sdk_result(result):
+         ...

+ def _openai_error_to_dict(exc):  ...
+ def _status_code(exc):  ...
```

---

### 4.3 MODIFY: `src/api_model_proxy/__init__.py`

Add `PipelineResult` to the public API:

```python
from .proxy import APIModelProxy
from .sentinels import PipelineResult

__all__ = ["APIModelProxy", "PipelineResult"]
```

---

### 4.4 MODIFY: Route files

Seven route files are updated. The changes are mechanical and follow the same pattern for each.

#### Files modified

| File | Endpoints | Type |
|------|-----------|------|
| `routes/chat.py` | `POST /chat/completions` | JSON-IO + streaming rejection |
| `routes/completions.py` | `POST /completions` | JSON-IO + streaming rejection |
| `routes/responses.py` | `POST /responses` | JSON-IO + streaming rejection |
| `routes/embeddings.py` | `POST /embeddings` | JSON-IO (no streaming) |
| `routes/moderations.py` | `POST /moderations` | JSON-IO (no streaming) |
| `routes/images.py` | `POST /images/generations`, `/images/edits`, `/images/variations` | JSON-IO + multipart |
| `routes/audio.py` | `POST /audio/transcriptions`, `/audio/translations`, `/audio/speech` | Multipart + speech (special) |

#### 4.4.1 What's stripped from EVERY route file

1. `from openai import OpenAIError` — no longer needed
2. The entire `try/except OpenAIError` block
3. Calls to `result.model_dump()` — handled by `execute_request()` / `_serialize_sdk_result`
4. Direct calls to `proxy._preprocess_request(body)` and `proxy._postprocess_response(...)` — now called inside `execute_request()`
5. The module-level `_openai_error_to_dict` and `_status_code` functions (moved to `proxy.py`)

#### 4.4.2 What's kept in EVERY route file

1. Request parsing: `await request.json()` or `await request.form()`
2. Streaming rejection (`body.get("stream")` → 501) — for chat, completions, responses
3. Response wrapping: `JSONResponse(content=..., status_code=..., headers=...)`
4. Multipart file unwrapping (`UploadFile` → `(filename, bytes, content_type)`) — for audio/images
5. The audio/speech special case (returns raw `Response`, not `JSONResponse`)

#### 4.4.3 What's added to EVERY route file

A single call to `proxy.execute_request(body, sdk_method=..., serializer=...)`:

```python
# Before (every file):
body = proxy._preprocess_request(body)
try:
    result = proxy._client.chat.completions.create(**body)
    response_dict = result.model_dump()
    response_dict = proxy._postprocess_response(response_dict)
    return JSONResponse(content=response_dict)
except OpenAIError as exc:
    error_dict = _openai_error_to_dict(exc)
    error_dict = proxy._postprocess_response(error_dict)
    return JSONResponse(status_code=_status_code(exc), content=error_dict)


# After (every file):
result = proxy.execute_request(
    body,
    sdk_method=proxy._client.chat.completions.create,
    serializer=None,  # uses default _serialize_sdk_result
)
return JSONResponse(
    content=result.content,
    status_code=result.status_code,
    headers=result.headers,
)
```

#### 4.4.4 SDK method references per file

| Route file | `sdk_method` value |
|-----------|-------------------|
| `chat.py` | `proxy._client.chat.completions.create` |
| `completions.py` | `proxy._client.completions.create` |
| `responses.py` | `proxy._client.responses.create` |
| `embeddings.py` | `proxy._client.embeddings.create` |
| `moderations.py` | `proxy._client.moderations.create` |
| `images.py` — `/images/generations` | `proxy._client.images.generate` |
| `images.py` — `/images/edits` | `proxy._client.images.edit` |
| `images.py` — `/images/variations` | `proxy._client.images.create_variation` |
| `audio.py` — `/audio/transcriptions` | `proxy._client.audio.transcriptions.create` |
| `audio.py` — `/audio/translations` | `proxy._client.audio.translations.create` |

#### 4.4.5 The audio/speech endpoint (special case)

The TTS endpoint returns raw binary audio, not JSON. It cannot use `execute_request()` directly because the pipeline expects a `PipelineResult` with JSON content.

Instead, it uses:

```python
@router.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    proxy = request.app.state.proxy
    body = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.speech.create(**body)
        audio_bytes = result.read()
        meta = {"bytes_length": len(audio_bytes)}
        proxy._postprocess_response(meta)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except OpenAIError as exc:
        # Reuse the module-level error helpers from proxy.py
        error_dict = _openai_error_to_dict(exc)
        proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)
```

This imports `_openai_error_to_dict` and `_status_code` from `api_model_proxy.proxy` directly (they're module-level functions, not class methods). The hooks are still called for consistency. This is the **only** endpoint that does not use `execute_request()`.

---

### 4.5 MODIFY: `passthrough.py`

**No changes.** The passthrough router forwards requests verbatim without calling hooks or the pipeline. It's intentionally outside the hook system. This is documented with a comment (already present):

```python
# No hooks are called.
```

---

### 4.6 MODIFY: `examples/caching_proxy.py`

This is the key beneficiary of the new architecture. The hacky `_cache_hit_body` workaround is replaced by a clean `execute_request()` override.

**Before (current):**

```python
class CachingProxy(APIModelProxy):
    def __init__(self, openai_client, cache_size=1000, ttl=300):
        ...
        self._cache_hit_body = None  # <-- BAD: state leak

    def _preprocess_request(self, request):
        key = self._make_key(request)
        if key in self._cache and not expired(key):
            self._cache_hit_body = cached_response  # stash for later
            return request  # upstream call still dispatched!
        self._cache_hit_body = None
        return request

    def _postprocess_response(self, response):
        if self._cache_hit_body is not None:
            return self._cache_hit_body  # discard real response
        # ... store in cache
        return response
```

**After (new):**

```python
class CachingProxy(APIModelProxy):
    def __init__(self, openai_client, cache_size=1000, ttl=300):
        super().__init__(openai_client)
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._cache_size = cache_size
        self._ttl = ttl

    def execute_request(self, body, sdk_method, serializer=None):
        key = self._make_key(body)
        if key in self._cache:
            ts, cached = self._cache[key]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(key)
                return PipelineResult(content=cached, status_code=200)
            else:
                del self._cache[key]

        # Cache miss — run default pipeline
        result = super().execute_request(body, sdk_method, serializer)

        if result.status_code == 200:
            self._cache[key] = (time.time(), result.content)
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return result

    @staticmethod
    def _make_key(body: dict) -> str:
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()
```

Benefits:
- **No unnecessary upstream call** on cache hit — the real short-circuit.
- **No hacky `_cache_hit_body` attribute** — the state is local to `execute_request()`.
- **Clean delegation** to `super().execute_request()` for cache miss (which still calls hooks).
- **Headers can be added** — e.g., `PipelineResult(content=..., status_code=200, headers={"X-Cache": "hit"})`.

---

### 4.7 MODIFY: Tests

#### 4.7.1 `tests/test_proxy.py`

Add a new test class `TestExecuteRequest`:

```python
class TestExecuteRequest:
    """Tests for the new execute_request() default pipeline."""

    def test_success_returns_pipeline_result(self, proxy, mock_openai_client):
        # Setup: SDK method returns an object with model_dump()
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"id": "cmpl-xxx"}
        mock_openai_client.completions.create.return_value = mock_result

        result = proxy.execute_request(
            {"model": "gpt-4", "prompt": "Hello"},
            sdk_method=mock_openai_client.completions.create,
        )

        assert isinstance(result, PipelineResult)
        assert result.status_code == 200
        assert result.content == {"id": "cmpl-xxx"}

    def test_error_returns_error_pipeline_result(self, proxy, mock_openai_client):
        mock_openai_client.completions.create.side_effect = _make_openai_error(
            400, "Bad request"
        )

        result = proxy.execute_request(
            {"model": "gpt-4", "prompt": "Hello"},
            sdk_method=mock_openai_client.completions.create,
        )

        assert isinstance(result, PipelineResult)
        assert result.status_code == 400
        assert "error" in result.content

    def test_calls_preprocess_hook(self, proxy, mock_openai_client):
        calls = []
        original = proxy._preprocess_request
        proxy._preprocess_request = lambda b: (calls.append("pre") or original(b))

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {}
        mock_openai_client.completions.create.return_value = mock_result

        proxy.execute_request({}, mock_openai_client.completions.create)
        assert calls == ["pre"]

    def test_calls_postprocess_hook_on_success(self, proxy, mock_openai_client):
        calls = []
        original = proxy._postprocess_response
        proxy._postprocess_response = lambda r: (calls.append("post") or original(r))

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {}
        mock_openai_client.completions.create.return_value = mock_result

        proxy.execute_request({}, mock_openai_client.completions.create)
        assert calls == ["post"]

    def test_calls_postprocess_hook_on_error(self, proxy, mock_openai_client):
        calls = []
        original = proxy._postprocess_response
        proxy._postprocess_response = lambda r: (calls.append("post") or original(r))

        mock_openai_client.completions.create.side_effect = _make_openai_error(400, "err")

        proxy.execute_request({}, mock_openai_client.completions.create)
        assert calls == ["post"]

    def test_custom_serializer(self, proxy, mock_openai_client):
        def custom_serializer(result):
            return {"custom": "format"}

        mock_result = MagicMock()
        mock_openai_client.completions.create.return_value = mock_result

        result = proxy.execute_request(
            {},
            mock_openai_client.completions.create,
            serializer=custom_serializer,
        )

        assert result.content == {"custom": "format"}

    def test_subclass_can_override_execute_request(self, mock_openai_client):
        class CustomProxy(APIModelProxy):
            def execute_request(self, body, sdk_method, serializer=None):
                return PipelineResult(
                    content={"custom": True},
                    status_code=200,
                    headers={"X-Custom": "yes"},
                )

        proxy = CustomProxy(mock_openai_client)
        result = proxy.execute_request({}, lambda: None)
        assert result.content == {"custom": True}
        assert result.headers == {"X-Custom": "yes"}
```

#### 4.7.2 `tests/test_routes.py`

Update existing tests to work through `execute_request()`:

1. **Replace SDK mock patterns** — instead of mocking `proxy._client.X.Y.create` directly, tests can mock `proxy.execute_request` to return a `PipelineResult`.
2. **Hook tests remain valid** — but can also be tested via `execute_request()` mock assertions.
3. **Key changes per test class:**

   ```python
   # Before:
   mock_openai_client.chat.completions.create.return_value = ...
   resp = client.post("/chat/completions", json={...})
   
   # After (option 1 — mock the pipeline):
   proxy.execute_request = MagicMock(
       return_value=PipelineResult(content={"id": "chatcmpl-xxx", ...}, status_code=200)
   )
   resp = client.post("/chat/completions", json={...})
   
   # After (option 2 — let the real execute_request call the mocked SDK):
   # (tests that want to verify hook ordering or SDK interaction)
   mock_openai_client.chat.completions.create.return_value = ...
   resp = client.post("/chat/completions", json={...})
   ```

   Both approaches should work. The existing tests that mock SDK methods directly will continue to work because `execute_request()` calls the SDK method.

4. **New test: subclass overrides execute_request** — verify that a subclass that overrides `execute_request()` is called by the route handler.

5. **New test: custom headers** — verify that headers from `PipelineResult` are forwarded to `JSONResponse`.

6. **No changes to passthrough tests** — `tests/test_passthrough.py` is untouched.

#### 4.7.3 `tests/test_passthrough.py`

**No changes.** Passthrough routes don't use `execute_request()`.

---

## 5. Route Handler Templates

### 5.1 JSON-IO endpoint (standard)

```python
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Chat"])

_STREAMING_NOT_SUPPORTED = {
    "error": {
        "message": "Streaming is not yet supported by this proxy. ...",
        "type": "not_implemented",
        "code": "streaming_not_supported",
    }
}


@router.post("/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    body: dict = await request.json()

    # Streaming rejection (if applicable)
    if body.get("stream"):
        return JSONResponse(status_code=501, content=_STREAMING_NOT_SUPPORTED)

    result = proxy.execute_request(
        body,
        sdk_method=proxy._client.chat.completions.create,
    )
    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )
```

**Used by:** chat.py, completions.py, responses.py, embeddings.py, moderations.py, images.py (generations)

### 5.2 Multipart endpoint (file upload)

```python
@router.post("/images/edits")
async def image_edits(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    form = await request.form()
    body = dict(form)

    # Unwrap UploadFile → (filename, bytes, content_type)
    for field in ("image", "mask"):
        if field in body:
            upload = body[field]
            body[field] = (upload.filename, await upload.read(), upload.content_type)

    result = proxy.execute_request(body, sdk_method=proxy._client.images.edit)
    return JSONResponse(
        content=result.content,
        status_code=result.status_code,
        headers=result.headers,
    )
```

**Used by:** images.py (edits, variations), audio.py (transcriptions, translations)

### 5.3 Speech endpoint (special — raw binary)

```python
from openai import OpenAIError
from api_model_proxy.proxy import _openai_error_to_dict, _status_code


@router.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    proxy = request.app.state.proxy
    body: dict = await request.json()

    body = proxy._preprocess_request(body)

    try:
        result = proxy._client.audio.speech.create(**body)
        audio_bytes = result.read()
        proxy._postprocess_response({"bytes_length": len(audio_bytes)})
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except OpenAIError as exc:
        error_dict = _openai_error_to_dict(exc)
        proxy._postprocess_response(error_dict)
        return JSONResponse(status_code=_status_code(exc), content=error_dict)
```

**Used by:** audio.py (speech) — this is the **only** template that does NOT use `execute_request()`.

---

## 6. Backward Compatibility

### 6.1 What does NOT break

| Component | Status | Reason |
|-----------|--------|--------|
| `APIModelProxy.__init__()` | ✅ Unchanged | Same signature |
| `APIModelProxy.deploy()` | ✅ Unchanged | Same signature |
| `_preprocess_request()` hook | ✅ Unchanged | Still called by default `execute_request()` |
| `_postprocess_response()` hook | ✅ Unchanged | Still called by default `execute_request()` |
| Existing subclasses that only override hooks | ✅ Unchanged | Hooks still fire via `execute_request()` → calls them |
| `create_app()` in `server.py` | ✅ Unchanged | No routing changes |
| `passthrough.py` | ✅ Unchanged | No hooks, no pipeline |
| `tests/test_passthrough.py` | ✅ Unchanged | No changes |
| Example: `logging_proxy.py` | ✅ Unchanged | Only uses hooks, no override needed |
| Example: `rate_limiting_proxy.py` | ✅ Unchanged | Only uses `_preprocess_request` (raises `HTTPException` before pipeline) |
| Example: `fallback_proxy.py` | ✅ Unchanged | Only uses hooks (though could benefit from `execute_request()` override later) |

### 6.2 What DOES break (intentional, documented)

| Change | Impact | Mitigation |
|--------|--------|------------|
| Route files no longer export `_openai_error_to_dict` / `_status_code` | Anyone importing these from route files gets `ImportError` | Import from `api_model_proxy.proxy` instead (see §7) |
| `CachingProxy._cache_hit_body` workaround removed | Any subclass using this pattern must migrate to `execute_request()` | Documented in migration guide (§7) |

### 6.3 Why `fallback_proxy.py` and `rate_limiting_proxy.py` are unchanged

- **RateLimitingProxy** raises `HTTPException` in `_preprocess_request`, which FastAPI catches before the route handler reaches the SDK call. This still works perfectly — the exception is raised inside `execute_request()`, which doesn't catch `HTTPException` (only `OpenAIError`), so it propagates up to FastAPI's error handler.
- **FallbackProxy** swaps `self._client` in `_preprocess_request`. Since `execute_request()` calls `_preprocess_request()` first, then uses the (swapped) `self._client` for the SDK call, this still works. The example is left as-is for now; it could be migrated to override `execute_request()` for cleaner fallback logic, but that's optional.

---

## 7. Migration Guide for Existing Proxy Subclasses

### Scenario A: You only use `_preprocess_request` / `_postprocess_response` hooks

**No changes needed.** Your subclass continues to work exactly as before. The default `execute_request()` implementation calls both hooks.

```
class MyProxy(APIModelProxy):
    def _preprocess_request(self, body):
        body["extra"] = "data"
        return body
```

### Scenario B: You wanted caching/retry but couldn't short-circuit

**Override `execute_request()`** for full control:

```python
class MyCachingProxy(APIModelProxy):
    def execute_request(self, body, sdk_method, serializer=None):
        key = self._make_key(body)
        cached = self._cache_get(key)
        if cached:
            return PipelineResult(content=cached, status_code=200)

        result = super().execute_request(body, sdk_method, serializer)
        if result.status_code == 200:
            self._cache_set(key, result.content)
        return result
```

### Scenario C: You were importing `_openai_error_to_dict` or `_status_code` from route files

**Change your import:**

```python
# Before (will break):
from api_model_proxy.routes.chat import _openai_error_to_dict, _status_code

# After:
from api_model_proxy.proxy import _openai_error_to_dict, _status_code
```

### Scenario D: You were using `_cache_hit_body` workaround from examples/caching_proxy.py

**Replace your entire subclass** with the `execute_request()` override shown in §4.6. The `_cache_hit_body` pattern is removed; the new pattern is cleaner and avoids dispatching unnecessary upstream calls.

---

## 8. Implementation Order

The files should be implemented in this order to maintain a working codebase at every step:

| Step | File | What | Testable? |
|------|------|------|-----------|
| 1 | `sentinels.py` | NEW — `PipelineResult` dataclass | ✅ Unit-testable immediately |
| 2 | `proxy.py` | Add `execute_request()`, `_serialize_sdk_result`, moved helpers | ✅ Unit-testable via `test_proxy.py` |
| 3 | `__init__.py` | Export `PipelineResult` | ✅ |
| 4 | Route files (one by one) | Replace inline pipeline with `execute_request()` | ✅ Each route testable individually |
| 5 | `examples/caching_proxy.py` | Rewrite to override `execute_request()` | ⚠️ Manual verification |
| 6 | `tests/test_proxy.py` | Add `TestExecuteRequest` class | ✅ |
| 7 | `tests/test_routes.py` | Update tests (add pipeline mock tests) | ✅ |

**Recommended order for route file migration** (least risky first):

1. `embeddings.py` — simplest (no streaming, no multipart)
2. `moderations.py` — same as embeddings
3. `completions.py` — adds streaming rejection
4. `chat.py` — same pattern as completions
5. `responses.py` — same pattern as chat
6. `images.py` — adds multipart handling
7. `audio.py` — multipart + special speech endpoint

---

## Appendix: Full diff sketch for one route file (chat.py)

```diff
-from openai import OpenAIError
 from fastapi import APIRouter, Request
 from fastapi.responses import JSONResponse
 
@@ -22,24 +21,15 @@ async def chat_completions(request: Request) -> JSONResponse:
     body: dict = await request.json()
 
     if body.get("stream"):
         return JSONResponse(status_code=501, content=_STREAMING_NOT_SUPPORTED)
 
-    body = proxy._preprocess_request(body)
-
-    try:
-        result = proxy._client.chat.completions.create(**body)
-        response_dict = result.model_dump()
-        response_dict = proxy._postprocess_response(response_dict)
-        return JSONResponse(content=response_dict)
-    except OpenAIError as exc:
-        error_dict = _openai_error_to_dict(exc)
-        error_dict = proxy._postprocess_response(error_dict)
-        return JSONResponse(status_code=_status_code(exc), content=error_dict)
-
-
-def _openai_error_to_dict(exc: OpenAIError) -> dict:
-    ...
-
-
-def _status_code(exc: OpenAIError) -> int:
-    ...
+    result = proxy.execute_request(
+        body,
+        sdk_method=proxy._client.chat.completions.create,
+    )
+    return JSONResponse(
+        content=result.content,
+        status_code=result.status_code,
+        headers=result.headers,
+    )
```

The file shrinks from ~55 lines to ~25 lines — a ~55% reduction, and all the complexity moves into `proxy.py` where it's reusable and overridable.
