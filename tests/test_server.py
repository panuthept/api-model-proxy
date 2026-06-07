from __future__ import annotations

from unittest.mock import MagicMock

from api_model_proxy.server import create_app


class TestCreateApp:
    def test_returns_fastapi_app(self, proxy: MagicMock):
        app = create_app(proxy)
        assert app.title == "API Model Proxy"
        assert app.version == "0.1.0"

    def test_stores_proxy_on_state(self, proxy: MagicMock):
        app = create_app(proxy)
        assert app.state.proxy is proxy

    def test_registers_routes(self, proxy: MagicMock):
        app = create_app(proxy)
        routes = {r.path for r in app.routes}

        assert "/chat/completions" in routes
        assert "/v1/chat/completions" in routes
        assert "/completions" in routes
        assert "/v1/completions" in routes
        assert "/responses" in routes
        assert "/v1/responses" in routes
        assert "/embeddings" in routes
        assert "/v1/embeddings" in routes
        assert "/audio/transcriptions" in routes
        assert "/audio/translations" in routes
        assert "/audio/speech" in routes
        assert "/v1/audio/speech" in routes
        assert "/images/generations" in routes
        assert "/v1/images/generations" in routes
        assert "/images/edits" in routes
        assert "/images/variations" in routes
        assert "/moderations" in routes
        assert "/v1/moderations" in routes
        assert "/{full_path:path}" in routes
