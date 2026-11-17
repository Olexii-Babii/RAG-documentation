import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from models import Query


class QueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
            self,
            question: str,
            question_embedding: list[float],
            answer: str | None,
            retrieved_chunk_ids: list[uuid.UUID],
            latency_ms: int,
            user_id: str | None = None,
            cache_hit: bool = False,
    ) -> Query:
        query = Query(
            question=question,
            question_embedding=question_embedding,
            answer=answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
            user_id=user_id,
            cache_hit=cache_hit,
        )
        self.session.add(query)
        await self.session.flush()

        return query

    async def get_by_id(self, query_id: uuid.UUID) -> Query | None:
        return await self.session.get(Query, query_id)
