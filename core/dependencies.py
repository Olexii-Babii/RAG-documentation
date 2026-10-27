from fastapi import Request

from services.embedding import EmbeddingService
from services.chunking import ChunkingService, HFTokenCounter


def get_embedding_service(request: Request) -> EmbeddingService:
    return EmbeddingService(request.app.state.embedding_model)


def get_chunking_service(request: Request) -> ChunkingService:
    token_counter = HFTokenCounter(request.app.state.embedding_model.tokenizer)
    return ChunkingService(token_counter)
