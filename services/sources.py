from sqlalchemy.ext.asyncio import AsyncSession

from models.source import Source
from repositories.sources import SourceRepository


class SourceService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._sources = SourceRepository(session)

    async def create_source(self, name: str, base_url: str | None) -> Source:
        async with self._session.begin():
            return await self._sources.add(name=name, base_url=base_url)
