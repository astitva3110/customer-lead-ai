import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.services.knowledge.ingestion_service import IngestionService
from app.kb.enums import DocumentType
from app.dependencies import get_ingestion_service, require_role
from app.domain.entities import User, UserRole
from app.schemas import DocumentProgress, DocumentStatusResponse, DocumentUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def run_background_ingestion(service: IngestionService, document_id: str) -> None:
    try:
        service.process_document(document_id)
    except Exception:
        logger.exception("background ingestion failed for document_id=%s", document_id)
        service.record_unexpected_failure(document_id)


@router.post("/documents", response_model=DocumentUploadResponse, status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Form("other"),
    language: str = Form("en"),
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    try:
        parsed_type = DocumentType(document_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid document_type: {document_type}") from exc

    original_name = Path(file.filename or "upload.pdf").name
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / original_name
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        record, is_duplicate = service.upload_document(
            temp_path,
            document_type=parsed_type,
            language=language,
        )

    if not record.document_id:
        raise HTTPException(status_code=400, detail=record.error or "Upload failed")
    if is_duplicate:
        return DocumentUploadResponse(
            document_id=record.document_id,
            status="DUPLICATE",
        )

    service.mark_processing(record.document_id)
    background_tasks.add_task(run_background_ingestion, service, record.document_id)
    return DocumentUploadResponse(document_id=record.document_id, status="PROCESSING")


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(
    document_id: str,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentStatusResponse:
    record = service.get_document_status(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status.value.upper(),
        error=record.error,
        progress=DocumentProgress(
            pages=record.pages,
            chunks=record.chunks,
            embedded=record.embedded,
        ),
    )
