from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from core.llm import LLMClient, LLMUnavailableError

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(code: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError("boom", response=httpx2.Response(code, request=_REQUEST), body=None)


def _rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=_REQUEST), body=None)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_REQUEST)


def _fake_message(text: str) -> SimpleNamespace:
    # duck-typed stand-in for anthropic.types.Message - _extract_text only
    # ever reads .content -> blocks with .type / .text
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeMessages:
    """Stands in for client.messages. Returns `result` or raises `error`,
    and records every call for assertions on what was actually sent."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeAnthropicClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.messages = FakeMessages(result=result, error=error)


def _chunk(heading_path: str, content: str) -> SimpleNamespace:
    # duck-typed stand-in for models.chunk.Chunk - _build_prompt only reads
    # .heading_path / .content
    return SimpleNamespace(heading_path=heading_path, content=content)


# --------------------------------------------------------------------------- #
# _build_prompt - pure function, no client involved
# --------------------------------------------------------------------------- #

def test_build_prompt_includes_heading_path_and_question():
    client = LLMClient(FakeAnthropicClient())
    chunks = [
        _chunk("A > B", "first chunk text"),
        _chunk("A > C", "second chunk text"),
    ]

    prompt = client._build_prompt("What is X?", chunks)

    assert "[A > B]" in prompt
    assert "first chunk text" in prompt
    assert "[A > C]" in prompt
    assert "second chunk text" in prompt
    assert prompt.index("[A > B]") < prompt.index("[A > C]")  # order preserved
    assert prompt.endswith("Question: What is X?")


def test_build_prompt_with_no_chunks_still_has_question():
    client = LLMClient(FakeAnthropicClient())
    prompt = client._build_prompt("What is X?", [])
    assert prompt == "Context:\n\nQuestion: What is X?"


# --------------------------------------------------------------------------- #
# _extract_text
# --------------------------------------------------------------------------- #

def test_extract_text_finds_text_block_among_others():
    client = LLMClient(FakeAnthropicClient())
    response = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="the answer"),
    ])
    assert client._extract_text(response) == "the answer"


# --------------------------------------------------------------------------- #
# generate_answer - happy path and error translation
# --------------------------------------------------------------------------- #

async def test_generate_answer_returns_extracted_text():
    fake_client = FakeAnthropicClient(result=_fake_message("hello"))
    client = LLMClient(fake_client)

    answer = await client.generate_answer("question?", [_chunk("A", "ctx")])

    assert answer == "hello"
    assert len(fake_client.messages.calls) == 1
    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["messages"] == [{"role": "user", "content": client._build_prompt("question?", [_chunk("A", "ctx")])}]


@pytest.mark.parametrize("error_factory", [_rate_limit_error, _connection_error, lambda: _status_error(500)])
async def test_generate_answer_wraps_transient_errors(error_factory):
    client = LLMClient(FakeAnthropicClient(error=error_factory()))

    with pytest.raises(LLMUnavailableError):
        await client.generate_answer("q", [])


async def test_generate_answer_reraises_client_errors_unwrapped():
    client = LLMClient(FakeAnthropicClient(error=_status_error(400)))

    with pytest.raises(anthropic.APIStatusError):
        await client.generate_answer("q", [])
