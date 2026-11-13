from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.db_async import get_db
from core.dependencies import get_ingestion_service, get_source_service
from repositories.documents import DocumentRepository
from schemas.ingestion import DocumentCreate, DocumentRead, SourceCreate, SourceRead
from services.ingestion import IngestionService
from services.sources import SourceService

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/sources", response_model=SourceRead, status_code=status.HTTP_201_CREATED)
async def create_source(
    payload: SourceCreate,
    service: SourceService = Depends(get_source_service),
) -> SourceRead:
    source = await service.create_source(name=payload.name, base_url=payload.base_url)
    return SourceRead.model_validate(source)


@router.post(
    "/sources/{source_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    source_id: UUID,
    payload: DocumentCreate,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentRead:
    # 201, not the planned 202: ingestion is synchronous for now (phase 1) -
    # by the time we respond the document is already fully indexed. Switch
    # to 202 + a status-polling story once Celery lands (phase 2).
    document = await service.ingest_document(source_id, str(payload.url))
    return DocumentRead.model_validate(document)


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> DocumentRead:
    async with session.begin():
        document = await DocumentRepository(session).get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentRead.model_validate(document)
