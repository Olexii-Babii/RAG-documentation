import httpx
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.db_async import get_db
from services.embedding import EmbeddingService
from services.chunking import ChunkingService, HFTokenCounter
from services.fetcher import DocumentFetcher
from services.ingestion import IngestionService
from services.parsing import DocumentParser
from services.sources import SourceService


def get_embedding_service(request: Request) -> EmbeddingService:
    return EmbeddingService(request.app.state.embedding_model)


def get_chunking_service(request: Request) -> ChunkingService:
    token_counter = HFTokenCounter(request.app.state.embedding_model.tokenizer)
    return ChunkingService(token_counter)


def get_fetch_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.fetch_client


def get_fetcher_service(fetcher_client: httpx.AsyncClient = Depends(get_fetch_client)) -> DocumentFetcher:
    return DocumentFetcher(fetcher_client)


def get_parser_service() -> DocumentParser:
    return DocumentParser()


def get_source_service(session: AsyncSession = Depends(get_db)) -> SourceService:
    return SourceService(session)


def get_ingestion_service(
        session: AsyncSession = Depends(get_db),
        embedder: EmbeddingService = Depends(get_embedding_service),
        chunker: ChunkingService = Depends(get_chunking_service),
        parser: DocumentParser = Depends(get_parser_service),
        fetcher: DocumentFetcher = Depends(get_fetcher_service),
) -> IngestionService:

    return IngestionService(
        session=session,
        embedder=embedder,
        chunker=chunker,
        parser=parser,
        fetcher=fetcher,
    )
