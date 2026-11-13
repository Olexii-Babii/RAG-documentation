from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy import text
from anyio import to_thread

from core.config import settings
from core.db_async import async_engine
from core.exceptions import source_not_found_handler
from routers.ingestion import router as ingestion_router
from sentence_transformers import SentenceTransformer
from services.ingestion import SourceNotFoundError

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    model = await to_thread.run_sync(SentenceTransformer, settings.EMBEDDING_MODEL)
    if model.get_embedding_dimension() != settings.EMBEDDING_DIM:
        raise RuntimeError("Embedding model dimension does not match settings")

    app.state.embedding_model = model
    app.state.fetch_client = httpx.AsyncClient()
    yield

    await app.state.fetch_client.aclose()
    await async_engine.dispose()


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(SourceNotFoundError, source_not_found_handler)
app.include_router(ingestion_router)


@app.get("/")
def read_root():
    return {"Hello": "World"}