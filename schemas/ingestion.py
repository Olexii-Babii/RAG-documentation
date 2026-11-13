from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, HttpUrl

from models.document import DocumentStatusEnum


class SourceCreate(BaseModel):
    name: str
    base_url: str | None = None


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    base_url: str | None
    created_at: datetime


class DocumentCreate(BaseModel):
    url: HttpUrl


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_id: UUID
    title: str
    url: str
    status: DocumentStatusEnum
    content_hash: str
    indexed_at: datetime | None
    created_at: datetime