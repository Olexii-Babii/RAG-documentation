import pathlib

import pytest
from sentence_transformers import SentenceTransformer

from services.chunking import ChunkingService, HFTokenCounter
from services.parsing import DocumentBlock, DocumentParser

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class WordCounter:
    """Counts words instead of subword tokens: fast, deterministic, no model needed."""

    def count(self, text: str) -> int:
        return len(text.split())


def heading(text: str, path: tuple[str, ...]) -> DocumentBlock:
    return DocumentBlock(type="heading", text=text, heading_path=path, level=len(path))


def para(text: str, path: tuple[str, ...]) -> DocumentBlock:
    return DocumentBlock(type="paragraph", text=text, heading_path=path)


def raw_chunk(path: tuple[str, ...], text: str, tokens: int) -> tuple:
    return (path, [text], tokens)


@pytest.fixture
def service() -> ChunkingService:
    svc = ChunkingService(WordCounter())
    svc.min_tokens = 0  # disable merge pass unless a test opts in
    return svc


# --------------------------------------------------------------------------- #
# chunk() — packing and flush triggers
# --------------------------------------------------------------------------- #

def test_empty_blocks_returns_empty_list(service):
    assert service.chunk([]) == []


def test_heading_blocks_contribute_no_content(service):
    blocks = [heading("Title", ("Title",)), para("body text", ("Title",))]
    chunks = service.chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].content == "body text"
    assert "Title" not in chunks[0].content


def test_blocks_in_same_section_pack_together(service):
    blocks = [
        para("one two three", ("Doc", "Sec")),
        para("four five six", ("Doc", "Sec")),
    ]
    chunks = service.chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].content == "one two three\n\nfour five six"
    assert chunks[0].token_count == 6
    assert chunks[0].heading_path == "Doc > Sec"


def test_exceeding_target_flushes_new_chunk(service):
    service.target_tokens = 5
    blocks = [para("a b c d e", ("Doc",)), para("f g h", ("Doc",))]
    chunks = service.chunk(blocks)
    assert [c.content for c in chunks] == ["a b c d e", "f g h"]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_heading_path_change_flushes_even_under_target(service):
    service.target_tokens = 100
    blocks = [para("short a", ("Doc", "One")), para("short b", ("Doc", "Two"))]
    chunks = service.chunk(blocks)
    assert [c.heading_path for c in chunks] == ["Doc > One", "Doc > Two"]


def test_oversized_block_splits_into_multiple_chunks(service):
    service.max_tokens = 10
    service.target_tokens = 200
    long_text = "One two three. " * 10  # 30 "tokens" (words), way over max=10
    chunks = service.chunk([para(long_text, ("Doc",))])
    assert len(chunks) > 1
    assert all(c.token_count <= 10 for c in chunks)
    assert all(c.heading_path == "Doc" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# _split_oversized / _split_by_words
# --------------------------------------------------------------------------- #

def test_split_oversized_packs_sentences_greedily(service):
    service.max_tokens = 6
    text = "a b. c d. e f. g h."  # 4 sentences x 2 words = 8 words total
    pieces = service._split_oversized(text)
    assert all(len(p.split()) <= 6 for p in pieces)
    assert sum(len(p.split()) for p in pieces) == 8
    assert len(pieces) < 4  # proves sentences got packed, not one-piece-per-sentence


def test_split_oversized_falls_back_to_words_without_recursion(service):
    service.max_tokens = 5
    run_on = " ".join(f"w{i}" for i in range(23))  # one "sentence", no punctuation at all
    pieces = service._split_oversized(run_on)  # must not raise RecursionError
    assert all(len(p.split()) <= 5 for p in pieces)
    assert sum(len(p.split()) for p in pieces) == 23


# --------------------------------------------------------------------------- #
# _merge_small
# --------------------------------------------------------------------------- #

def test_merge_small_merges_single_small_chunk_backward():
    svc = ChunkingService(WordCounter())
    svc.min_tokens = 5
    chunks = [raw_chunk(("A",), "big", 10), raw_chunk(("A",), "small", 2)]
    merged = svc._merge_small(chunks)
    assert len(merged) == 1
    assert merged[0][1] == ["big", "small"]
    assert merged[0][2] == 12


def test_merge_small_merges_consecutive_small_chunks_without_losing_data():
    """Regression: two small chunks in a row used to silently drop the second."""
    svc = ChunkingService(WordCounter())
    svc.min_tokens = 5
    chunks = [
        raw_chunk(("A",), "big", 10),
        raw_chunk(("A",), "small-one", 2),
        raw_chunk(("A",), "small-two", 2),
    ]
    merged = svc._merge_small(chunks)
    assert len(merged) == 1
    assert merged[0][1] == ["big", "small-one", "small-two"]
    assert merged[0][2] == 14


def test_merge_small_leaves_leading_small_chunk_alone():
    svc = ChunkingService(WordCounter())
    svc.min_tokens = 5
    chunks = [raw_chunk(("A",), "small", 2), raw_chunk(("A",), "big", 10)]
    merged = svc._merge_small(chunks)
    assert len(merged) == 2  # no predecessor to merge into


def test_merge_small_path_becomes_common_prefix():
    svc = ChunkingService(WordCounter())
    svc.min_tokens = 5
    chunks = [
        raw_chunk(("Doc", "SectionA"), "big", 10),
        raw_chunk(("Doc", "SectionB"), "small", 2),
    ]
    merged = svc._merge_small(chunks)
    assert merged[0][0] == ("Doc",)


def test_new_path_common_prefix():
    svc = ChunkingService(WordCounter())
    assert svc._new_path(("A", "B", "C"), ("A", "B", "D")) == ("A", "B")
    assert svc._new_path(("A",), ("X",)) == ()
    assert svc._new_path(("A", "B"), ("A", "B")) == ("A", "B")


# --------------------------------------------------------------------------- #
# end-to-end: real parser output + real embedding-model tokenizer
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_counter() -> HFTokenCounter:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return HFTokenCounter(model.tokenizer)


@pytest.mark.parametrize("name", ["query-params", "first-steps", "python-types"])
def test_real_page_chunks_respect_token_limits(name, real_counter):
    html = (FIXTURES / f"{name}.html").read_text()
    doc = DocumentParser().parse(html)
    service = ChunkingService(real_counter)

    chunks = service.chunk(doc.blocks)

    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.content.strip()
        assert c.heading_path
        assert c.token_count <= service.max_tokens
