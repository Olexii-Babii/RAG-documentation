import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from models.source import Source


class SourceRepository:
    """DB access for Source. No commits, no transactions (ADR-0002) -
    the caller's Unit of Work owns that."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, name: str, base_url: str | None = None) -> Source:
        source = Source(name=name, base_url=base_url)
        self.session.add(source)
        await self.session.flush()
        return source

    async def get_by_id(self, source_id: uuid.UUID) -> Source | None:
        return await self.session.get(Source, source_id)
