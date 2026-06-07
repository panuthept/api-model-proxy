from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from .proxy import APIModelProxy


def create_app(proxy: "APIModelProxy") -> FastAPI:
    """Create and configure the FastAPI application.

    All inference routers are registered first (specific paths),
    followed by the catch-all passthrough router last.

    Args:
        proxy: The :class:`~api_model_proxy.proxy.APIModelProxy` instance
               whose hooks will be called on inference requests.

    Returns:
        A configured :class:`fastapi.FastAPI` application ready to serve.
    """
    app = FastAPI(
        title="API Model Proxy",
        description="OpenAI-compatible proxy with pre/post-processing hooks.",
        version="0.1.0",
    )

    # Store proxy on app state so route handlers can access it via request.app.state.proxy
    app.state.proxy = proxy

    # ------------------------------------------------------------------
    # Inference routers (hooks fire on these)
    # Registered under both "" and "/v1" so the proxy works regardless
    # of whether the client sets base_url="http://host:port" or
    # base_url="http://host:port/v1".
    # ------------------------------------------------------------------
    from .routes.chat import router as chat_router
    from .routes.completions import router as completions_router
    from .routes.responses import router as responses_router
    from .routes.embeddings import router as embeddings_router
    from .routes.audio import router as audio_router
    from .routes.images import router as images_router
    from .routes.moderations import router as moderations_router

    for prefix in ("", "/v1"):
        app.include_router(chat_router, prefix=prefix)
        app.include_router(completions_router, prefix=prefix)
        app.include_router(responses_router, prefix=prefix)
        app.include_router(embeddings_router, prefix=prefix)
        app.include_router(audio_router, prefix=prefix)
        app.include_router(images_router, prefix=prefix)
        app.include_router(moderations_router, prefix=prefix)

    # ------------------------------------------------------------------
    # Passthrough router — must be registered LAST (catch-all)
    # ------------------------------------------------------------------
    from .routes.passthrough import router as passthrough_router

    app.include_router(passthrough_router)

    return app
