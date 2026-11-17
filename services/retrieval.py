import time
import uuid

from dataclasses import dataclass

from core.llm import LLMClient
from models.query import Query
from models.chunk import Chunk
from repositories.chunks import ChunkRepository
from repositories.queries import QueryRepository
from services.embedding import EmbeddingService


@dataclass(frozen=True)
class AnswerResult:
    query: Query
    chunks: list[Chunk]


class RetrievalService:
    def __init__(self, session, embedder: EmbeddingService, llm: LLMClient) -> None:
        self._session = session
        self._embedder_service = embedder
        self._llm = llm
        self._chunk_rep = ChunkRepository(session)
        self._query_rep = QueryRepository(session)

    async def answer_question(
            self,
            question: str,
            user_id: str | None = None,
            source_ids: list[uuid.UUID] | None = None,
            k: int = 10,
    ) -> AnswerResult:
        start = time.monotonic()

        question_embedding = await self._embedder_service.embed(question)

        async with self._session.begin():
            result = await self._chunk_rep.search_similar(question_embedding, k=k, source_ids=source_ids)
        chunks = [c for c, _ in result]

        answer = await self._llm.generate_answer(question, chunks)

        latency_ms = int((time.monotonic() - start) * 1000)

        async with self._session.begin():
            query = await self._query_rep.add(
                question=question,
                question_embedding=question_embedding,
                answer=answer,
                retrieved_chunk_ids=[c.id for c in chunks],
                latency_ms=latency_ms,
                user_id=user_id,
                cache_hit=False,
            )

        return AnswerResult(query, chunks)
