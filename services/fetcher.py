import httpx


class DocumentFetcher:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, url: str) -> str:
        response = await self._client.get(url)
        response.raise_for_status()
        return response.text