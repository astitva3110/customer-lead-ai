"""Backward-compatible renderer entry point."""

from app.kb.enums import DocumentType, SourceType
from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers import render_retrieval_text as _render


def render_retrieval_text(
    *,
    title: str,
    structured_content: DocumentContent,
    canonical_url: str = "",
    document_type: DocumentType = DocumentType.OTHER,
    source_type: SourceType = SourceType.HTML,
) -> str:
    return _render(
        title=title,
        canonical_url=canonical_url,
        document_type=document_type,
        source_type=source_type,
        structured_content=structured_content,
    )
