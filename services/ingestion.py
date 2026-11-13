import hashlib
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from models.document import DocumentStatusEnum, Document
from repositories.chunks import ChunkRepository, NewChunk
from repositories.documents import DocumentRepository
from repositories.sources import SourceRepository
from services.chunking import ChunkingService
from services.embedding import EmbeddingService
from services.fetcher import DocumentFetcher
from services.parsing import DocumentParser


class SourceNotFoundError(Exception):
    def __init__(self, source_id: uuid.UUID) -> None:
        self.source_id = source_id


class IngestionService:
    def __init__(
            self,
            session: AsyncSession,
            embedder: EmbeddingService,
            chunker: ChunkingService,
            parser: DocumentParser,
            fetcher: DocumentFetcher,
) -> None:
        self._session = session
        self._embedder = embedder
        self._chunker = chunker
        self._parser = parser
        self._fetcher = fetcher
        self._sources = SourceRepository(session)
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)

    async def ingest_document(self, source_id: uuid.UUID, url: str) -> Document:
        async with self._session.begin():
            if await self._sources.get_by_id(source_id) is None:
                raise SourceNotFoundError(source_id)

        html = await self._fetcher.fetch(url)
        parsed = self._parser.parse(html)
        content_hash = hashlib.sha256(
            "\n".join(b.text for b in parsed.blocks).encode()
        ).hexdigest()

        async with self._session.begin():
            existed =  await self._documents.get_by_content_hash(source_id, content_hash)

        if existed is not None:
            return existed

        chunk_data = self._chunker.chunk(parsed.blocks)
        vectors = await self._embedder.embed_batch([c.content for c in chunk_data])

        async with self._session.begin():
            document = await self._documents.add(
                source_id=source_id, url=url, title=parsed.title, content_hash=content_hash
            )
            new_chunks = [
                NewChunk(
                    content=c.content, chunk_index=c.chunk_index,
                    heading_path=c.heading_path, token_count=c.token_count, embedding=v,
                )
                for c, v in zip(chunk_data, vectors)
            ]
            await self._chunks.add_many(document.id, new_chunks)
            await self._documents.set_status(document.id, DocumentStatusEnum.INDEXED)

        return document