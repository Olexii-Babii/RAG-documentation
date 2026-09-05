from fastapi import Request

from services.embedding import EmbeddingService


def get_embedding_service(request: Request) -> EmbeddingService:
    return EmbeddingService(request.app.state.embedding_model)