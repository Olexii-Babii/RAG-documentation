import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from models.base import Base
from models.mixins import UUIDMixin, TimestampMixin

if TYPE_CHECKING:
    from models.source import Source
    from models.chunk import Chunk


class DocumentStatusEnum(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[DocumentStatusEnum] = mapped_column(
        SqlEnum(
            DocumentStatusEnum,
            native_enum=False,
            length=20,
            create_constraint=True,
            values_callable=lambda enum_cls: [m.value for m in enum_cls],
            name="document_status"
        ),
        nullable=False,
        default=DocumentStatusEnum.PENDING,
        server_default=DocumentStatusEnum.PENDING.value,
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False
    )
    source: Mapped["Source"] = relationship( back_populates="documents")

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_documents_source_content_hash"),
    )
