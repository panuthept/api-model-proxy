# Spec: Streaming Support for API Model Proxy

## Objective

Add streaming support (SSE) to the API Model Proxy for the three streaming-capable inference endpoints: chat completions, text completions, and the responses API. Currently, any request with `"stream": true` receives a `501 Not Implemented` error.

**Target users:** Developers using the API Model Proxy as a drop-in intermediary between their application and an OpenAI-compatible backend who need real-time token-by-token responses (e.g., chat UIs, live transcription, progressive rendering).

**Success criteria:**
- Clients can set `stream=True` on `/chat/completions`, `/completions`, and `/responses` (and their `/v1/` variants) and receive a valid SSE stream.
- The response stream matches OpenAI's SSE wire format so existing client libraries (e.g., `openai` Python SDK with `stream=True`) work transparently.
- `_preprocess_request` is called before the stream starts — the hook can modify or reject the request.
- `_postprocess_response` is called on **each chunk** in the stream — the hook can inspect, modify, or log individual chunks.
- Stream errors from the upstream API are serialised and sent as the next chunk in the stream (not as a separate HTTP error response), matching the "stream the error as a chunk, then close" strategy.
- Non-streaming requests continue to work exactly as before.
- Existing tests continue to pass; new tests cover streaming code paths.

## Related Specs

*(This is the first PRD for this project — no prior specs.)*

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | >= 3.8 |
| Web framework | FastAPI | >= 0.100.0 |
| Server | uvicorn | >= 0.23.0 |
| OpenAI SDK | openai | >= 1.0.0 |
| HTTP client | httpx | >= 0.24.0 |
| Testing | pytest + pytest-mock | latest |
| Linting / Formatting | Ruff | latest |
| SSE | `starlette.responses.StreamingResponse` | (bundled with FastAPI) |

**No new dependencies are required.** SSE streaming is built into Starlette/FastAPI via `StreamingResponse`.

## Commands

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-mock

# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_proxy.py -v
pytest tests/test_routes.py -v -k "stream"

# Lint and format
ruff check src/ tests/
ruff format src/ tests/ --check

# Format in-place
ruff format src/ tests/
```

## Project Structure

Changes are constrained to existing files; no new directories or major restructuring. The streaming logic will live in a shared module referenced by the route handlers.

```
src/api_model_proxy/
├── __init__.py          # Public API exports (add streaming helpers if public)
├── proxy.py             # APIModelProxy base class (add execute_streaming_request)
├── server.py            # FastAPI app factory (no changes expected)
├── sentinels.py         # PipelineResult dataclass (no changes expected)
├── streaming.py         # NEW — Shared streaming utilities (SSE formatting, chunk iteration)
└── routes/
    ├── chat.py          # Update: replace 501 with streaming code path
    ├── completions.py   # Update: replace 501 with streaming code path
    ├── responses.py     # Update: replace 501 with streaming code path
    ├── audio.py         # No change
    ├── images.py        # No change
    ├── embeddings.py    # No change
    ├── moderations.py   # No change
    └── passthrough.py   # No change

tests/
├── conftest.py          # Add streaming mock fixtures
├── test_proxy.py        # Add streaming pipeline tests
├── test_routes.py       # Add streaming test classes for chat, completions, responses
├── test_passthrough.py  # No change
├── test_server.py       # No change
└── test_streaming.py    # NEW — Streaming-specific unit tests (SSE formatting, hook chunk handling)
```

## Code Style

The project uses **Ruff** (default config) for linting and formatting. Key conventions observed in existing code:

- `from __future__ import annotations` at the top of every file.
- Type annotations on all function signatures.
- `MagicMock`-based mocking in tests (not `pytest-mock`'s `mocker` fixture).
- Dataclasses for data transfer objects (`PipelineResult`).
- Router functions receive `Request` and access proxy via `request.app.state.proxy`.

**Key streaming code snippet (illustrative style):**

```python
# src/api_model_proxy/streaming.py
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Callable

from openai import OpenAIError
from starlette.responses import StreamingResponse


def _format_sse_chunk(data: dict) -> str:
    """Format a dict as an SSE ``data:`` frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _iterate_stream(
    stream: Any,
    postprocess: Callable[[dict], dict],
    serializer: Callable[[Any], dict],
) -> AsyncGenerator[bytes, None]:
    """Iterate over an OpenAI stream, yield SSE-formatted chunks."""
    try:
        for chunk in stream:
            chunk_dict = serializer(chunk)
            chunk_dict = postprocess(chunk_dict)
            yield _format_sse_chunk(chunk_dict).encode("utf-8")
    except OpenAIError as exc:
        error_dict = _serialise_stream_error(exc)
        error_dict = postprocess(error_dict)
        yield _format_sse_chunk(error_dict).encode("utf-8")

    yield b"data: [DONE]\n\n"
```

**Naming conventions:**

| Convention | Rule |
|---|---|
| Files | `snake_case.py` |
| Classes | `PascalCase` |
| Functions/methods | `snake_case` (public) or `_snake_case` (private) |
| Constants | `UPPER_SNAKE_CASE` |
| Type vars | `PascalCase` prefixed with underscore if private |

## Streaming Wire Format

The proxy emits OpenAI-compatible SSE streams:

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0}]}\n\n
...
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n
data: [DONE]\n\n
```

The response `Content-Type` is `text/event-stream`.

## Testing Strategy

| Level | Framework | Location | Coverage Expectation |
|---|---|---|---|
| Unit (SSE formatting) | pytest | `tests/test_streaming.py` | All formatting edge cases (empty chunks, Unicode, error serialisation) |
| Unit (hook chunk handling) | pytest | `tests/test_streaming.py` | `_preprocess_request` called once per stream; `_postprocess_response` called per chunk |
| Integration (endpoints) | pytest + TestClient | `tests/test_routes.py` | Stream response is valid SSE, chunks have expected structure, `[DONE]` sent |
| Integration (error streams) | pytest + TestClient | `tests/test_routes.py` | Mid-stream errors are serialised as chunks, not aborting the connection |
| Regression | pytest | `tests/` (all) | All existing non-streaming tests must continue to pass |

**Mocking strategy:**
- The OpenAI SDK returns an iterable of `ChatCompletionChunk` (or equivalent) when `stream=True`.
- Mock `client.chat.completions.create(stream=True, ...)` to return a list of chunk objects.
- Use `MagicMock` with `spec=ChatCompletionChunk` or construct real `ChatCompletionChunk` objects via `model_validate` (matching the existing pattern in `test_routes.py`).

## Boundaries

### Always do
- Use `starlette.responses.StreamingResponse` for SSE responses (not raw `Response`).
- Call `_preprocess_request` before starting the stream — pass the modified body to the SDK.
- Call `_postprocess_response` on each individual chunk dict (after serialisation).
- Send `data: [DONE]\n\n` as the final frame to signal stream completion.
- Set `Content-Type: text/event-stream` on streaming responses.
- Wrap streaming SDK calls in try/except for `OpenAIError` — serialise to chunk on failure.
- Run full test suite before marking complete.
- Run Ruff linter on all changed files.

### Ask first
- Changing the `PipelineResult` dataclass or `execute_request` method signature.
- Adding new Python dependencies beyond what's in `setup.py`.
- Adding async streaming (AsyncOpenAI) support — keep it sync-only per spec.
- Modifying non-streaming route handler behavior.
- Adding streaming support for audio, images, or other non-primary endpoints.

### Never do
- Commit secrets, API keys, or credentials.
- Remove the 501 blocking check from routes before streaming is implemented.
- Break existing non-streaming functionality (all existing tests must pass).
- Leave `TODO(streaming)` comments in place without implementation.
- Remove or alter the `_STREAMING_NOT_SUPPORTED` error dict — keep it for any endpoint that is intentionally left non-streaming.
- Change the upstream client configuration or client-facing API.

## Implementation Architecture

```
Route handler (chat.py)
    │
    ├── body = await request.json()
    │
    ├── if body.get("stream"):
    │       │
    │       body = proxy._preprocess_request(body)   # once
    │       stream = client.create(**body, stream=True)
    │       │
    │       return StreamingResponse(
    │           proxy.execute_streaming_request(body, sdk_method, serializer),
    │           media_type="text/event-stream",
    │       )
    │
    └── else:
            result = proxy.execute_request(...)
            return JSONResponse(...)
```

## Open Questions

Nothing currently unresolved — all design decisions have been confirmed:

| Question | Decision |
|---|---|
| Which endpoints? | Chat, completions, and responses |
| Hook behavior on streaming? | `_preprocess_request` runs once before stream; `_postprocess_response` runs on each chunk |
| Can hooks modify chunk content? | Yes — modification allowed |
| Stream error handling? | Stream error as next chunk, then close |
| AsyncOpenAI support? | No — sync-only for now |
| Scope beyond streaming? | Streaming + update README/docs |

---

## Implementation Plan

### Major Components & Dependencies

```
tests/conftest.py (streaming fixtures)
        │
        ▼
src/api_model_proxy/streaming.py (NEW — SSE formatting, chunk iteration)
        │
        ▼
src/api_model_proxy/proxy.py (add execute_streaming_request method)
        │
        ▼
Routes: chat.py, completions.py, responses.py (replace 501 with streaming)
        │
        ▼
tests/test_streaming.py ──┐
                          ├──► Full test suite
tests/test_routes.py ─────┘
        │
        ▼
README.md + API_REFERENCE.md (docs)
```

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Different SDK chunk types per endpoint (chat vs completions vs responses) | Each route passes its own SDK method + serializer, matching non-streaming pattern |
| Throughput impact from `_postprocess_response` per chunk | Documented trade-off; subclasses can override for no-op |
| Chunk modification breaks SSE format | Hooks receive/return dicts; SSE formatting happens after hook returns — safe |
| Sync SDK stream in async FastAPI handler | Starlette's `StreamingResponse` handles sync iterables transparently |

### Task List

- [ ] **Task 1: Streaming mock fixtures** — Add `mock_chat_stream()`, `mock_completion_stream()`, `mock_response_stream()` fixtures to `tests/conftest.py` so test code can simulate streaming SDK responses.
  - *Acceptance:* Fixtures return iterables of SDK chunk objects with `.model_dump()` support.
  - *Verify:* `pytest tests/test_routes.py -k "stream"` (will fail until routes are updated, but fixtures import cleanly).
  - *Files:* `tests/conftest.py`

- [ ] **Task 2: Shared streaming utilities** — Create `src/api_model_proxy/streaming.py` with `_format_sse_chunk()`, `_serialise_stream_error()`, and the core async generator for iterating a stream with hooks.
  - *Acceptance:* Functions are importable, `_format_sse_chunk` produces valid SSE, error serialisation produces chunk-compatible dicts, the generator yields SSE bytes and terminates with `[DONE]`.
  - *Verify:* `pytest tests/test_streaming.py` passes.
  - *Files:* `src/api_model_proxy/streaming.py` (NEW)

- [ ] **Task 3: execute_streaming_request on APIModelProxy** — Add `execute_streaming_request(body, sdk_method, serializer)` method to `APIModelProxy` that runs `_preprocess_request`, starts the SDK stream, and returns an async generator of SSE bytes (running `_postprocess_response` per chunk).
  - *Acceptance:* New method is callable, invokes both hooks correctly, handles OpenAIError mid-stream, yields `[DONE]` at end.
  - *Verify:* `pytest tests/test_proxy.py -k "stream"` passes.
  - *Files:* `src/api_model_proxy/proxy.py`, `src/api_model_proxy/__init__.py`

- [ ] **Task 4: Chat completions streaming** — Update `chat.py` to replace the 501 block with a streaming code path that calls `proxy.execute_streaming_request(...)` and returns a `StreamingResponse`.
  - *Acceptance:* `POST /chat/completions` with `stream: true` returns `text/event-stream` with valid OpenAI-format SSE chunks.
  - *Verify:* `pytest tests/test_routes.py::TestChatCompletions::test_streaming` passes.
  - *Files:* `src/api_model_proxy/routes/chat.py`

- [ ] **Task 5: Completions streaming** — Same as Task 4 for `/completions`.
  - *Acceptance:* `POST /completions` with `stream: true` returns valid SSE.
  - *Verify:* `pytest tests/test_routes.py::TestCompletions::test_streaming` passes.
  - *Files:* `src/api_model_proxy/routes/completions.py`

- [ ] **Task 6: Responses API streaming** — Same as Task 4 for `/responses`.
  - *Acceptance:* `POST /responses` with `stream: true` returns valid SSE.
  - *Verify:* `pytest tests/test_routes.py::TestResponses::test_streaming` passes.
  - *Files:* `src/api_model_proxy/routes/responses.py`

- [ ] **Task 7: Streaming unit tests** — Write `tests/test_streaming.py` covering SSE formatting, error serialisation, the async generator with hook interactions, empty streams, and mid-stream errors.
  - *Acceptance:* All streaming utility functions have >90% line coverage.
  - *Verify:* `pytest tests/test_streaming.py -v` passes.
  - *Files:* `tests/test_streaming.py` (NEW)

- [ ] **Task 8: Streaming integration tests** — Add streaming test classes to `tests/test_routes.py` for each of the three endpoints, covering: successful stream, chunks have expected shape, `[DONE]` termination, error streams, hook calls on streams, `_preprocess_request` modification applied to stream request.
  - *Acceptance:* All streaming integration tests pass alongside existing non-streaming tests.
  - *Verify:* `pytest tests/test_routes.py -v` passes.
  - *Files:* `tests/test_routes.py`

- [ ] **Task 9: Update documentation** — Update `README.md` to remove the "Streaming not yet supported" caveat, add streaming usage examples, document hook behavior during streaming. Update `API_REFERENCE.md` with new method and SSE format notes.
  - *Acceptance:* README and API reference accurately describe streaming support.
  - *Verify:* `grep -i "not yet supported" README.md` returns no matches.
  - *Files:* `README.md`, `API_REFERENCE.md`

- [ ] **Task 10: Final verification** — Run full test suite and Ruff linter. Fix any issues.
  - *Acceptance:* `pytest tests/ -v` passes 100%. `ruff check src/ tests/` passes. `ruff format src/ tests/ --check` passes.
  - *Verify:* All three commands exit 0.
  - *Files:* (none — verification only)
