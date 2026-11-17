import uuid

import pytest
from sentence_transformers import SentenceTransformer

from models.chunk import Chunk
from repositories.chunks import ChunkRepository, NewChunk
from repositories.documents import DocumentRepository
from repositories.sources import SourceRepository
from services.embedding import EmbeddingService
from services.retrieval import RetrievalService


class FakeLLMClient:
    """Records what it was called with, returns a canned answer.

    Also enforces the ADR-0002 guarantee directly: asserts no DB
    transaction is open on the session while "generating" - if
    RetrievalService ever regresses to holding a transaction across the
    LLM call, every test using this fake fails loudly, not just one."""

    def __init__(self, session, answer: str = "canned answer") -> None:
        self._session = session
        self.answer = answer
        self.calls: list[tuple[str, list[Chunk]]] = []

    async def generate_answer(self, question: str, chunks: list[Chunk]) -> str:
        assert not self._session.in_transaction(), (
            "LLM call must not run inside an open transaction (ADR-0002)"
        )
        self.calls.append((question, chunks))
        return self.answer


@pytest.fixture(scope="module")
def model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


async def _seed_document(session, source_name: str = "FastAPI docs"):
    async with session.begin():
        source = await SourceRepository(session).add(name=source_name)
        document = await DocumentRepository(session).add(
            source_id=source.id, url="https://x", title="X", content_hash=uuid.uuid4().hex
        )
    return source, document


async def test_answer_question_happy_path(session, model):
    embedder = EmbeddingService(model)
    question = "How do I declare a query parameter?"
    question_vec = await embedder.embed(question)

    _, document = await _seed_document(session)
    async with session.begin():
        chunks = await ChunkRepository(session).add_many(
            document.id,
            [NewChunk(content="exact match chunk", chunk_index=0, heading_path="A",
                       token_count=1, embedding=question_vec)],
        )

    fake_llm = FakeLLMClient(session, answer="the answer")
    service = RetrievalService(session, embedder, fake_llm)

    result = await service.answer_question(question)

    assert result.query.answer == "the answer"
    assert result.query.question == question
    assert result.query.cache_hit is False
    assert result.query.latency_ms >= 0
    assert result.query.retrieved_chunk_ids == [chunks[0].id]
    assert [c.content for c in result.chunks] == ["exact match chunk"]

    assert len(fake_llm.calls) == 1
    called_question, called_chunks = fake_llm.calls[0]
    assert called_question == question
    assert [c.content for c in called_chunks] == ["exact match chunk"]


async def test_answer_question_respects_k(session, model):
    embedder = EmbeddingService(model)
    question = "question"
    question_vec = await embedder.embed(question)

    _, document = await _seed_document(session)
    async with session.begin():
        await ChunkRepository(session).add_many(
            document.id,
            [
                NewChunk(content=f"chunk {i}", chunk_index=i, heading_path="A",
                          token_count=1, embedding=question_vec)
                for i in range(5)
            ],
        )

    fake_llm = FakeLLMClient(session)
    service = RetrievalService(session, embedder, fake_llm)

    result = await service.answer_question(question, k=2)

    assert len(result.chunks) == 2
    assert len(result.query.retrieved_chunk_ids) == 2


async def test_answer_question_filters_by_source_ids(session, model):
    embedder = EmbeddingService(model)
    question = "question"
    question_vec = await embedder.embed(question)

    source_a, doc_a = await _seed_document(session, source_name="Source A")
    _, doc_b = await _seed_document(session, source_name="Source B")

    async with session.begin():
        await ChunkRepository(session).add_many(
            doc_a.id,
            [NewChunk(content="from a", chunk_index=0, heading_path="A", token_count=1, embedding=question_vec)],
        )
        await ChunkRepository(session).add_many(
            doc_b.id,
            [NewChunk(content="from b", chunk_index=0, heading_path="A", token_count=1, embedding=question_vec)],
        )

    fake_llm = FakeLLMClient(session)
    service = RetrievalService(session, embedder, fake_llm)

    result = await service.answer_question(question, source_ids=[source_a.id])

    assert [c.content for c in result.chunks] == ["from a"]


async def test_answer_question_with_no_chunks_still_persists(session, model):
    embedder = EmbeddingService(model)
    fake_llm = FakeLLMClient(session, answer="I don't know")
    service = RetrievalService(session, embedder, fake_llm)

    result = await service.answer_question("anything?", user_id="user-123")

    assert result.chunks == []
    assert result.query.retrieved_chunk_ids == []
    assert result.query.answer == "I don't know"
    assert result.query.user_id == "user-123"
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0][1] == []  # called with an empty chunk list, not skipped
