import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def _migrated_test_db() -> None:
    """Point Alembic at rag_system_test (see migrations/env.py) and bring it
    to head. The database itself must already exist - created once by hand:
        docker compose exec db psql -U admin -d rag_system \
            -c "CREATE DATABASE rag_system_test"
    """
    os.environ["ALEMBIC_TARGET_DB"] = settings.POSTGRES_TEST_DB
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(scope="session")
async def test_engine(_migrated_test_db: None):
    engine = create_async_engine(
        settings.create_db_url("postgresql+asyncpg", database=settings.POSTGRES_TEST_DB)
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(test_engine) -> AsyncSession:
    """One connection + one outer transaction per test, rolled back at
    teardown. Repositories never commit (ADR-0002), so nothing here ever
    needs to survive past a single test - rollback is enough, no cleanup."""
    async with test_engine.connect() as conn:
        await conn.begin()
        async_session = async_sessionmaker(bind=conn, expire_on_commit=False)()
        try:
            yield async_session
        finally:
            await async_session.close()
            await conn.rollback()
