import anyio
import pytest
from sentence_transformers import SentenceTransformer

from services.embedding import EmbeddingService


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    return EmbeddingService(SentenceTransformer("all-MiniLM-L6-v2"))


def test_embed_returns_configured_dim(embedding_service):
    vec = anyio.run(embedding_service.embed, "test")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_embed_is_deterministic(embedding_service):
    vec1 = anyio.run(embedding_service.embed, "test")
    vec2 = anyio.run(embedding_service.embed, "test")
    assert vec1 == vec2


def test_embed_is_normalized(embedding_service):
    vec = anyio.run(embedding_service.embed, "test")
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_batch_matches_single(embedding_service):
    single = anyio.run(embedding_service.embed, "first")
    batch = anyio.run(embedding_service.embed_batch, ["first", "second"])
    assert batch[0] == pytest.approx(single, abs=1e-5)
    assert len(batch) == 2
    assert all(len(v) ==384 for v in batch)
