from .chat import router as chat_router
from .completions import router as completions_router
from .responses import router as responses_router
from .embeddings import router as embeddings_router
from .audio import router as audio_router
from .images import router as images_router
from .moderations import router as moderations_router
from .passthrough import router as passthrough_router

__all__ = [
    "chat_router",
    "completions_router",
    "responses_router",
    "embeddings_router",
    "audio_router",
    "images_router",
    "moderations_router",
    "passthrough_router",
]
