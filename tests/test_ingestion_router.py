import pathlib
import uuid

import httpx
import pytest

import main
from core.db_async import get_db
from core.dependencies import get_fetcher_service

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class FakeFetcher:
    """Same role as in test_ingestion.py: canned HTML instead of the
    network, plus a call log to prove fail-fast paths never fetch."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        return self.html


@pytest.fixture(scope="module")
async def client():
    async with main.lifespan(main.app):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
async def api_client(client, session):
    """Routes every get_db() in the app at this test's rollback-isolated
    session, instead of the real pool pointed at the dev database."""

    async def override_get_db():
        yield session

    main.app.dependency_overrides[get_db] = override_get_db
    yield client
    main.app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def set_fake_fetcher():
    """Call with HTML to make ingestion see that page instead of hitting
    the real network. Returns the fetcher so tests can inspect .calls."""

    def _set(html: str) -> FakeFetcher:
        fetcher = FakeFetcher(html)
        main.app.dependency_overrides[get_fetcher_service] = lambda: fetcher
        return fetcher

    yield _set
    main.app.dependency_overrides.pop(get_fetcher_service, None)


async def test_create_source_returns_201(api_client):
    response = await api_client.post(
        "/api/v1/sources", json={"name": "FastAPI docs", "base_url": "https://fastapi.tiangolo.com"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "FastAPI docs"
    assert body["base_url"] == "https://fastapi.tiangolo.com"
    assert uuid.UUID(body["id"])  # well-formed UUID


async def test_create_source_without_name_returns_422(api_client):
    response = await api_client.post("/api/v1/sources", json={"base_url": "https://x"})
    assert response.status_code == 422


async def test_create_document_happy_path(api_client, set_fake_fetcher):
    html = (FIXTURES / "query-params.html").read_text()
    set_fake_fetcher(html)

    source_response = await api_client.post("/api/v1/sources", json={"name": "FastAPI docs"})
    source = source_response.json()

    response = await api_client.post(
        f"/api/v1/sources/{source['id']}/documents",
        json={"url": "https://fastapi.tiangolo.com/tutorial/query-params/"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Query Parameters"
    assert body["status"] == "indexed"
    assert body["source_id"] == source["id"]
    assert len(body["content_hash"]) == 64


async def test_create_document_missing_source_returns_404_without_fetching(api_client, set_fake_fetcher):
    fetcher = set_fake_fetcher("<html></html>")
    missing_source_id = uuid.uuid4()

    response = await api_client.post(
        f"/api/v1/sources/{missing_source_id}/documents",
        json={"url": "https://example.com"},
    )

    assert response.status_code == 404
    assert str(missing_source_id) in response.json()["detail"]
    assert fetcher.calls == []  # fail-fast: never touched the fetcher


async def test_create_document_invalid_url_returns_422(api_client):
    source_response = await api_client.post("/api/v1/sources", json={"name": "FastAPI docs"})
    source = source_response.json()

    response = await api_client.post(
        f"/api/v1/sources/{source['id']}/documents",
        json={"url": "not-a-url"},
    )

    assert response.status_code == 422


async def test_get_document_returns_200(api_client, set_fake_fetcher):
    html = (FIXTURES / "query-params.html").read_text()
    set_fake_fetcher(html)

    source_response = await api_client.post("/api/v1/sources", json={"name": "FastAPI docs"})
    source = source_response.json()
    document_response = await api_client.post(
        f"/api/v1/sources/{source['id']}/documents",
        json={"url": "https://fastapi.tiangolo.com/tutorial/query-params/"},
    )
    document = document_response.json()

    response = await api_client.get(f"/api/v1/documents/{document['id']}")

    assert response.status_code == 200
    assert response.json()["title"] == "Query Parameters"


async def test_get_document_missing_returns_404(api_client):
    response = await api_client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404
