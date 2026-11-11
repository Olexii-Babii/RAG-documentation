import random
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.document import Document, DocumentStatusEnum
from models.source import Source
from repositories.chunks import ChunkRepository, NewChunk
from repositories.documents import DocumentRepository
from repositories.sources import SourceRepository

DIM = 384


def one_hot(index: int, scale: float = 1.0) -> list[float]:
    v = [0.0] * DIM
    v[index] = scale
    return v


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


async def _make_document(session, source_name: str = "FastAPI docs") -> Document:
    source = await SourceRepository(session).add(name=source_name)
    return await DocumentRepository(session).add(
        source_id=source.id, url="https://x", title="X", content_hash=uuid.uuid4().hex
    )


async def test_chunk_add_many_assigns_document_and_ids(session):
    document = await _make_document(session)
    repo = ChunkRepository(session)

    chunks = await repo.add_many(
        document.id,
        [
            NewChunk(content="first", chunk_index=0, heading_path="A", token_count=1, embedding=one_hot(0)),
            NewChunk(content="second", chunk_index=1, heading_path="A", token_count=1, embedding=one_hot(1)),
        ],
    )

    assert len(chunks) == 2
    assert all(c.id is not None for c in chunks)
    assert all(c.document_id == document.id for c in chunks)
    assert [c.content for c in chunks] == ["first", "second"]


async def test_chunk_add_many_duplicate_index_in_document_rejected(session):
    document = await _make_document(session)
    repo = ChunkRepository(session)

    with pytest.raises(IntegrityError):
        await repo.add_many(
            document.id,
            [
                NewChunk(content="a", chunk_index=0, heading_path="A", token_count=1, embedding=one_hot(0)),
                NewChunk(content="b", chunk_index=0, heading_path="A", token_count=1, embedding=one_hot(1)),
            ],
        )


async def test_chunk_search_similar_orders_by_cosine_distance(session):
    """query = one_hot(0). near/mid/far are built at increasing angles from
    it, so cosine distance - and therefore result order - is unambiguous."""
    document = await _make_document(session)
    repo = ChunkRepository(session)

    near = one_hot(0)
    near[1] = 0.1  # small angle from query
    mid = one_hot(0)
    mid[1] = 1.0  # 45 degrees from query
    far = one_hot(1)  # 90 degrees from query -> max cosine distance

    await repo.add_many(
        document.id,
        [
            NewChunk(content="far", chunk_index=0, heading_path="A", token_count=1, embedding=far),
            NewChunk(content="near", chunk_index=1, heading_path="A", token_count=1, embedding=near),
            NewChunk(content="mid", chunk_index=2, heading_path="A", token_count=1, embedding=mid),
        ],
    )

    # generous ef_search: this test asserts exact ordering among 3 vectors,
    # and HNSW is approximate - a wide-enough candidate window makes that
    # assertion reliable regardless of how much dead-tuple noise other tests
    # in the same run have left in the shared index (see ADR-0006 follow-up)
    results = await repo.search_similar(one_hot(0), k=10, ef_search=1000)

    assert [chunk.content for chunk, _ in results] == ["near", "mid", "far"]
    distances = [dist for _, dist in results]
    assert distances == sorted(distances)  # ascending: closest first


async def test_chunk_search_similar_respects_k(session):
    document = await _make_document(session)
    repo = ChunkRepository(session)
    await repo.add_many(
        document.id,
        [
            NewChunk(content=f"c{i}", chunk_index=i, heading_path="A", token_count=1, embedding=one_hot(i))
            for i in range(5)
        ],
    )

    results = await repo.search_similar(one_hot(0), k=2, ef_search=1000)
    assert len(results) == 2


async def test_chunk_search_similar_filters_by_source_ids(session):
    doc_a = await _make_document(session, source_name="Source A")
    doc_b = await _make_document(session, source_name="Source B")
    repo = ChunkRepository(session)

    await repo.add_many(
        doc_a.id,
        [NewChunk(content="from-a", chunk_index=0, heading_path="A", token_count=1, embedding=one_hot(0))],
    )
    await repo.add_many(
        doc_b.id,
        [NewChunk(content="from-b", chunk_index=0, heading_path="A", token_count=1, embedding=one_hot(0))],
    )

    results = await repo.search_similar(one_hot(0), k=10, source_ids=[doc_a.source_id])

    assert [chunk.content for chunk, _ in results] == ["from-a"]


async def test_chunk_search_similar_on_empty_table_returns_empty_list(session):
    repo = ChunkRepository(session)
    assert await repo.search_similar(one_hot(0), k=5) == []


async def test_chunk_search_similar_narrow_source_filter_still_finds_matches(session):
    """Regression for ADR-0006: HNSW finds the top-ef_search nearest chunks
    overall, THEN applies the source filter - a narrow/rare source can lose
    every candidate to the filter and return fewer than k (or zero) rows
    even though real matches exist. hnsw.iterative_scan=relaxed_order
    (set in search_similar whenever source_ids is given) widens the scan
    until it actually finds k matches instead of giving up early."""
    random.seed(0)

    def random_vec() -> list[float]:
        return [random.random() for _ in range(DIM)]

    common = await _make_document(session, source_name="Common")
    rare = await _make_document(session, source_name="Rare")
    repo = ChunkRepository(session)

    await repo.add_many(
        common.id,
        [
            NewChunk(content=f"c{i}", chunk_index=i, heading_path="A", token_count=1, embedding=random_vec())
            for i in range(300)
        ],
    )
    await repo.add_many(
        rare.id,
        [
            NewChunk(content=f"r{i}", chunk_index=i, heading_path="A", token_count=1, embedding=random_vec())
            for i in range(3)
        ],
    )

    results = await repo.search_similar(random_vec(), k=3, source_ids=[rare.source_id])

    assert len(results) == 3
    assert all(chunk.content.startswith("r") for chunk, _ in results)
