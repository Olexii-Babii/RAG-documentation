from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from core.db_async import async_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    yield

    await async_engine.dispose()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}