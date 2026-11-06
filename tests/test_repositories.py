import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.document import Document, DocumentStatusEnum
from models.source import Source
from repositories.documents import DocumentRepository
from repositories.sources import SourceRepository


async def test_source_add_assigns_id_and_timestamp(session):
    repo = SourceRepository(session)
    source = await repo.add(name="FastAPI docs", base_url="https://fastapi.tiangolo.com")

    assert source.id is not None
    assert source.created_at is not None
    assert source.name == "FastAPI docs"
    assert source.base_url == "https://fastapi.tiangolo.com"


async def test_source_add_without_base_url(session):
    repo = SourceRepository(session)
    source = await repo.add(name="No URL source")
    assert source.base_url is None


async def test_source_get_by_id_found(session):
    repo = SourceRepository(session)
    created = await repo.add(name="FastAPI docs")

    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "FastAPI docs"


async def test_source_get_by_id_missing_returns_none(session):
    repo = SourceRepository(session)
    assert await repo.get_by_id(uuid.uuid4()) is None


async def test_document_add_defaults_to_pending(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)

    document = await repo.add(
        source_id=source.id,
        url="https://fastapi.tiangolo.com/tutorial/query-params/",
        title="Query Parameters",
        content_hash="a" * 64,
    )

    assert document.id is not None
    assert document.status == DocumentStatusEnum.PENDING
    assert document.indexed_at is None
    assert document.source_id == source.id


async def test_document_get_by_id_found(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)
    created = await repo.add(
        source_id=source.id, url="https://x", title="X", content_hash="b" * 64
    )

    fetched = await repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.title == "X"


async def test_document_get_by_content_hash_found(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)
    await repo.add(source_id=source.id, url="https://x", title="X", content_hash="c" * 64)

    found = await repo.get_by_content_hash(source.id, "c" * 64)
    assert found is not None
    assert found.title == "X"


async def test_document_get_by_content_hash_missing_returns_none(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)

    assert await repo.get_by_content_hash(source.id, "d" * 64) is None


async def test_document_get_by_content_hash_scoped_to_source(session):
    """Same content_hash under a different source is a different document -
    this is exactly what the uq_documents_source_content_hash constraint
    (and this lookup) is scoped by."""
    source_a = await SourceRepository(session).add(name="Source A")
    source_b = await SourceRepository(session).add(name="Source B")
    repo = DocumentRepository(session)
    await repo.add(source_id=source_a.id, url="https://a", title="A", content_hash="e" * 64)

    assert await repo.get_by_content_hash(source_b.id, "e" * 64) is None
    assert await repo.get_by_content_hash(source_a.id, "e" * 64) is not None


async def test_document_duplicate_content_hash_in_same_source_rejected(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)
    await repo.add(source_id=source.id, url="https://x", title="X", content_hash="f" * 64)

    with pytest.raises(IntegrityError):
        await repo.add(source_id=source.id, url="https://y", title="Y", content_hash="f" * 64)


async def test_document_set_status_updates_existing(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)
    document = await repo.add(
        source_id=source.id, url="https://x", title="X", content_hash="g" * 64
    )

    await repo.set_status(document.id, DocumentStatusEnum.INDEXED)

    refreshed = await repo.get_by_id(document.id)
    assert refreshed.status == DocumentStatusEnum.INDEXED


async def test_document_set_status_on_missing_document_is_noop(session):
    repo = DocumentRepository(session)
    await repo.set_status(uuid.uuid4(), DocumentStatusEnum.INDEXED)  # must not raise


async def test_document_deleted_with_source_cascade(session):
    source = await SourceRepository(session).add(name="FastAPI docs")
    repo = DocumentRepository(session)
    document = await repo.add(
        source_id=source.id, url="https://x", title="X", content_hash="h" * 64
    )

    await session.delete(await session.get(Source, source.id))
    await session.flush()

    result = await session.execute(select(Document.id).where(Document.id == document.id))
    assert result.scalar_one_or_none() is None
