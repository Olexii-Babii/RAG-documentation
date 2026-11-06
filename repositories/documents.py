import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.document import Document, DocumentStatusEnum


class DocumentRepository:
    """DB access for Document. No commits, no transactions (ADR-0002) -
    the caller's Unit of Work owns that."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
            self,
            source_id: uuid.UUID,
            url: str,
            title: str,
            content_hash: str,
    ) -> Document:
        document = Document(
            source_id=source_id,
            url=url,
            title=title,
            content_hash=content_hash,
            status=DocumentStatusEnum.PENDING,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_by_content_hash(
            self, source_id: uuid.UUID, content_hash: str
    ) -> Document | None:
        stmt = select(Document).where(
            Document.source_id == source_id,
            Document.content_hash == content_hash,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def set_status(self, document_id: uuid.UUID, status: DocumentStatusEnum) -> None:
        document = await self.session.get(Document, document_id)
        if document is None:
            return
        document.status = status
        await self.session.flush()
