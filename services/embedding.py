from anyio import to_thread
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    async def embed(self, text: str) -> list[float]:
        [vector] = await self.embed_batch([text])
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await to_thread.run_sync(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return arr.tolist()
