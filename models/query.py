import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, Boolean, String, Integer

from sqlalchemy.dialects.postgresql import ARRAY, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base
from models.mixins import UUIDMixin, TimestampMixin


class Query(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "queries"

    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(PgUUID(as_uuid=True)), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
