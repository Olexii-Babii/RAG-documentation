import re

from dataclasses import dataclass
from typing import Protocol

from services.parsing import DocumentBlock


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        pass


@dataclass(frozen=True)
class ChunkData:
    content: str
    chunk_index: int
    heading_path: str
    token_count: int


class HFTokenCounter:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text))


class ChunkingService:
    def __init__(self, block_counter: TokenCounter) -> None:
        self._counter = block_counter
        self.max_tokens = 256
        self.target_tokens = 200
        self.min_tokens = 24

    def chunk(self, blocks: list[DocumentBlock]) -> list[ChunkData]:
        raw: list[tuple[tuple[str, ...], list[str], int]] = []
        path, parts, tokens = None, [], 0

        def flush():
            nonlocal parts, tokens
            if parts:
                raw.append((path, parts, tokens))
            parts, tokens = [], 0

        for block in blocks:
            if block.type == "heading":
                continue

            if parts and block.heading_path != path:
                flush()

            block_tokens = self._counter.count(block.text)

            if block_tokens > self.max_tokens:
                flush()
                for piece in self._split_oversized(block.text):
                    raw.append((block.heading_path, [piece], self._counter.count(piece)))
                path = block.heading_path
                continue

            if parts and tokens + block_tokens > self.target_tokens:
                flush()

            path = block.heading_path
            parts.append(block.text)
            tokens += block_tokens

        flush()
        raw = self._merge_small(raw)

        return [
            ChunkData(
                content="\n\n".join(p),
                chunk_index=i,
                heading_path=" > ".join(pth),
                token_count=t,
            )
            for i, (pth, p, t) in enumerate(raw)
        ]

    def _split_oversized(self, text: str) -> list[str]:

        pieces: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sentence in _split_sentences(text):
            sentence_tokens = self._counter.count(sentence)

            if sentence_tokens > self.max_tokens:
                if buffer:
                    pieces.append(" ".join(buffer))
                    buffer, buffer_tokens = [], 0
                pieces.extend(self._split_by_words(sentence))
                continue

            if buffer and buffer_tokens + sentence_tokens > self.max_tokens:
                pieces.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0

            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            pieces.append(" ".join(buffer))

        return pieces

    def _split_by_words(self, text: str) -> list[str]:
        pieces: list[str] = []
        buffer: list[str] = []

        for word in text.split():
            candidate = buffer + [word]
            if buffer and self._counter.count(" ".join(candidate)) > self.max_tokens:
                pieces.append(" ".join(buffer))
                buffer = [word]
            else:
                buffer = candidate

        if buffer:
            pieces.append(" ".join(buffer))

        return pieces

    def _new_path(self, prev_path: tuple[str, ...], new_part: tuple[str, ...]) -> tuple[str, ...]:
        result = []
        for p1, p2 in zip(prev_path, new_part):
            if p1 == p2:
                result.append(p1)
            else:
                break
        return tuple(result)

    def _merge_small(self, chunks):
        merged: list[tuple[tuple[str, ...], list[str], int]] = []

        for path, parts, tokens in chunks:
            if merged and tokens < self.min_tokens:
                prev_path, prev_parts, prev_tokens = merged[-1]
                merged[-1] = (
                    self._new_path(prev_path, path),
                    prev_parts + parts,
                    prev_tokens + tokens,
                )
            else:
                merged.append((path, parts, tokens))

        return merged
