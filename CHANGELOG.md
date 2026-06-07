# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Example implementations:
  - `CachingProxy` — in-memory LRU cache with TTL
  - `FallbackProxy` — multi-backend circuit-breaker across multiple OpenAI-compatible endpoints
  - `RateLimitingProxy` — token-bucket rate limiter with `429 Too Many Requests` responses

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
