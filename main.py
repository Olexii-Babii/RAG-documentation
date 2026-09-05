from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from anyio import to_thread

from core.config import settings
from core.db_async import async_engine
from sentence_transformers import SentenceTransformer

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    model = await to_thread.run_sync(SentenceTransformer, settings.EMBEDDING_MODEL)
    if model.get_embedding_dimension() != settings.EMBEDDING_DIM:
        raise RuntimeError("Embedding model dimension does not match settings")

    app.state.embedding_model = model
    yield

    await async_engine.dispose()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}