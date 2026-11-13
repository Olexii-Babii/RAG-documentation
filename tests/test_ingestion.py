import pathlib
import uuid

import pytest
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from models.chunk import Chunk
from models.document import DocumentStatusEnum
from repositories.sources import SourceRepository
from services.chunking import ChunkingService, HFTokenCounter
from services.embedding import EmbeddingService
from services.ingestion import IngestionService, SourceNotFoundError
from services.parsing import DocumentParser

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class FakeFetcher:
    """Returns canned HTML instead of hitting the network. Also records
    every URL it was asked for, so tests can assert fail-fast behaviour
    (fetch never attempted) without a real network mock."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        return self.html


@pytest.fixture(scope="module")
def model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def _make_service(session, model, html: str) -> IngestionService:
    fetcher = FakeFetcher(html)
    return IngestionService(
        session=session,
        embedder=EmbeddingService(model),
        chunker=ChunkingService(HFTokenCounter(model.tokenizer)),
        parser=DocumentParser(),
        fetcher=fetcher,
    )


async def _chunks_of(session, document_id) -> list[Chunk]:
    async with session.begin():
        result = await session.execute(select(Chunk).where(Chunk.document_id == document_id))
        return list(result.scalars().all())


async def test_ingest_document_happy_path(session, model):
    html = (FIXTURES / "query-params.html").read_text()
    service = _make_service(session, model, html)

    async with session.begin():
        source = await SourceRepository(session).add(name="FastAPI docs")

    document = await service.ingest_document(
        source.id, "https://fastapi.tiangolo.com/tutorial/query-params/"
    )

    assert document.status == DocumentStatusEnum.INDEXED
    assert document.title == "Query Parameters"
    assert document.source_id == source.id
    assert len(document.content_hash) == 64  # sha256 hex digest

    chunks = await _chunks_of(session, document.id)
    assert chunks  # at least one chunk
    assert all(c.embedding is not None and len(c.embedding) == 384 for c in chunks)
    assert [c.chunk_index for c in chunks] == sorted(c.chunk_index for c in chunks)


async def test_ingest_document_is_idempotent_for_same_content(session, model):
    html = (FIXTURES / "query-params.html").read_text()
    service = _make_service(session, model, html)

    async with session.begin():
        source = await SourceRepository(session).add(name="FastAPI docs")

    url = "https://fastapi.tiangolo.com/tutorial/query-params/"
    first = await service.ingest_document(source.id, url)
    chunks_after_first = await _chunks_of(session, first.id)

    second = await service.ingest_document(source.id, url)
    chunks_after_second = await _chunks_of(session, second.id)

    assert second.id == first.id  # same document, not a duplicate
    assert len(chunks_after_second) == len(chunks_after_first)  # chunks not re-created
    assert len(service._fetcher.calls) == 2  # it does fetch both times - can't know the
    # content is unchanged without re-fetching and re-hashing it


async def test_ingest_document_missing_source_raises_without_fetching(session, model):
    html = (FIXTURES / "query-params.html").read_text()
    service = _make_service(session, model, html)

    with pytest.raises(SourceNotFoundError):
        await service.ingest_document(uuid.uuid4(), "https://example.com")

    assert service._fetcher.calls == []  # fail-fast: never touched the network


async def test_ingest_document_two_pages_coexist_under_same_source(session, model):
    html_a = (FIXTURES / "query-params.html").read_text()
    html_b = (FIXTURES / "first-steps.html").read_text()

    async with session.begin():
        source = await SourceRepository(session).add(name="FastAPI docs")

    service_a = _make_service(session, model, html_a)
    service_b = _make_service(session, model, html_b)

    doc_a = await service_a.ingest_document(source.id, "https://fastapi.tiangolo.com/tutorial/query-params/")
    doc_b = await service_b.ingest_document(source.id, "https://fastapi.tiangolo.com/tutorial/first-steps/")

    assert doc_a.id != doc_b.id
    assert doc_a.content_hash != doc_b.content_hash
    assert doc_a.title == "Query Parameters"
    assert doc_b.title == "First Steps"

    chunks_a = await _chunks_of(session, doc_a.id)
    chunks_b = await _chunks_of(session, doc_b.id)
    assert chunks_a and chunks_b
    assert {c.document_id for c in chunks_a} == {doc_a.id}
    assert {c.document_id for c in chunks_b} == {doc_b.id}
