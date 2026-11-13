from fastapi import Request
from fastapi.responses import JSONResponse

from services.ingestion import SourceNotFoundError


async def source_not_found_handler(request: Request, exc: SourceNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"Source {exc.source_id} not found"})
