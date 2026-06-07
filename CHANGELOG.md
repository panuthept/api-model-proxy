# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-06-07

### Fixed

- `setup.py`: added `long_description` (reads `README.md`), `long_description_content_type`, `url`, `license`, `classifiers`, and `keywords` so PyPI renders a proper project description page

## [0.1.1] - 2026-06-07

### Added

- `PersistantLoggingProxy` example — logs every request/response to daily JSONL or YAML files
- `CachingProxy` example — in-memory LRU response cache with configurable TTL
- `FallbackProxy` example — multi-backend circuit-breaker across multiple OpenAI-compatible endpoints
- `RateLimitingProxy` example — token-bucket rate limiter with `429 Too Many Requests` responses
- Test suite (pytest with mocked `OpenAI` client)

### Changed

- Added `pyyaml` runtime dependency (used by `PersistantLoggingProxy`)

## [0.1.0] - 2026-06-07

### Added

- `APIModelProxy` base class with `_preprocess_request` and `_postprocess_response` hooks
- FastAPI server with async inference routes for all OpenAI API endpoints:
  - `/chat/completions`, `/completions`, `/responses`
  - `/embeddings`
  - `/audio/transcriptions`, `/audio/translations`, `/audio/speech`
  - `/images/generations`, `/images/edits`, `/images/variations`
  - `/moderations`
- Catch-all passthrough endpoint for all non-inference routes (models, files, fine-tuning, batches, vector stores, etc.)
- Endpoints available under both bare paths and `/v1/` prefix
- `deploy()` method binding to configurable host/port via uvicorn
- Error visibility — `_postprocess_response` fires on both success and error responses
- Example implementation: `PersistantLoggingProxy` (logs to daily JSONL/YAML files)
