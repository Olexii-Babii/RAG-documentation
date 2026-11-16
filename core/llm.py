import anthropic

from models.chunk import Chunk
from anthropic.types import Message

_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = (
    "You answer questions about technical documentation using ONLY the "
    "provided context chunks. Each chunk is labeled with its section path. "
    "Cite the section path(s) you used. If the context doesn't contain "
    "the answer, say so plainly instead of guessing."
)


class LLMUnavailableError(Exception):
    pass


class LLMClient:
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    async def generate_answer(self, question: str, chunks: list[Chunk]) -> str:
        prompt = self._build_prompt(question, chunks)

        try:
            response = await self._client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError:
            raise LLMUnavailableError("rate limited") from None
        except anthropic.APIConnectionError:
            raise LLMUnavailableError("connection failed") from None
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMUnavailableError(f"upstream error {e.status_code}") from None
            raise

        return self._extract_text(response)

    def _extract_text(self, response: Message) -> str:
        for content in response.content:
            if content.type == "text":
                return content.text

    def _build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        prepared_chunks = []
        for c in chunks:
            prepared_chunks.append(f"[{c.heading_path}]\n{c.content}\n")

        context = "".join(prepared_chunks)
        return f"Context:\n{context}\nQuestion: {question}"

    async def close(self) -> None:
        await self._client.close()
