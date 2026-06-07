# Contributing

Thanks for your interest in contributing to API Model Proxy!

## Development Setup

```bash
git clone https://github.com/panuthept/api-model-proxy.git
cd api-model-proxy
python -m venv venv && source venv/bin/activate
pip install -e .
```

## Running Tests

The test suite uses `pytest` with a mocked `OpenAI` client — no API key needed.

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_proxy.py -v
```

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for code linting and formatting.

```bash
pip install ruff
ruff check src/ tests/
ruff format src/ tests/ --check
```

## Adding a New Inference Route

1. Create a new route file in `src/api_model_proxy/routes/` (follow the pattern in `chat.py` or `completions.py`).
2. Export the router in `src/api_model_proxy/routes/__init__.py`.
3. Register the router in `src/api_model_proxy/server.py` inside the `for prefix in ("", "/v1"):` loop.
4. Add tests in `tests/test_routes.py` following the existing class structure.
5. Run the full test suite to confirm nothing is broken.

## Pull Request Process

1. Ensure all existing tests pass and new tests cover your changes.
2. Run the linter and fix any issues.
3. Update the `README.md` if your change affects the public API, endpoints, or usage.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Open a pull request with a clear title and description of the change.

## Project Structure

```
src/api_model_proxy/
├── __init__.py          # Public API exports
├── proxy.py             # APIModelProxy base class
├── server.py            # FastAPI app factory
└── routes/              # Route handlers
    ├── __init__.py
    ├── chat.py
    ├── completions.py
    ├── responses.py
    ├── embeddings.py
    ├── audio.py
    ├── images.py
    ├── moderations.py
    └── passthrough.py   # Catch-all transparent proxy
tests/
├── conftest.py          # Fixtures (mock client, app, TestClient)
├── test_proxy.py        # APIModelProxy unit tests
├── test_routes.py       # Endpoint integration tests
├── test_passthrough.py  # Passthrough route tests
├── test_server.py       # App factory tests
```
