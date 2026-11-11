import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.chunk import Chunk
from models.document import Document


@dataclass(frozen=True)
class NewChunk:
    """Repo-local input shape for add_many. Deliberately not services.chunking.ChunkData -
    repositories must not depend on services (wrong layering direction); the caller
    (IngestionService) builds these from ChunkData + the embeddings it computed."""

    content: str
    chunk_index: int
    heading_path: str
    token_count: int
    embedding: list[float]


class ChunkRepository:
    """DB access for Chunk. No commits, no transactions (ADR-0002) -
    the caller's Unit of Work owns that."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_many(self, document_id: uuid.UUID, chunks: list[NewChunk]) -> list[Chunk]:
        rows = [
            Chunk(
                document_id=document_id,
                content=c.content,
                chunk_index=c.chunk_index,
                heading_path=c.heading_path,
                token_count=c.token_count,
                embedding=c.embedding,
            )
            for c in chunks
        ]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def search_similar(
        self,
        embedding: list[float],
        k: int = 10,
        source_ids: list[uuid.UUID] | None = None,
        ef_search: int = 100,
    ) -> list[tuple[Chunk, float]]:
        # SET LOCAL doesn't accept bind parameters (Postgres syntax rule) -
        # int() here is the injection guard, not a formality.
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

        distance = Chunk.embedding.cosine_distance(embedding)
        stmt = select(Chunk, distance.label("distance"))

        if source_ids:
            await self.session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            stmt = stmt.join(Document, Chunk.document_id == Document.id).where(
                Document.source_id.in_(source_ids)
            )

        stmt = stmt.order_by(distance).limit(k)

        result = await self.session.execute(stmt)
        return [(chunk, dist) for chunk, dist in result.all()]
