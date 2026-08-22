"""Document-type-specific retrieval text rendering."""

from app.kb.enums import DocumentType, SourceType
from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import sanitize_retrieval_text
from app.kb.retrieval.renderers.default import render_default
from app.kb.retrieval.renderers.faq import render_faq
from app.kb.retrieval.renderers.pdf import render_pdf
from app.kb.retrieval.renderers.policy import render_policy
from app.kb.retrieval.renderers.product import render_product
from app.kb.retrieval.renderers.prospectus import maybe_render_prospectus


def render_retrieval_text(
    *,
    title: str,
    canonical_url: str,
    document_type: DocumentType,
    source_type: SourceType,
    structured_content: DocumentContent,
) -> str:
    """Render compact retrieval text using a document-type-specific template."""
    prospectus_text = maybe_render_prospectus(
        title=title,
        canonical_url=canonical_url,
        structured_content=structured_content,
    )
    if prospectus_text is not None:
        return sanitize_retrieval_text(prospectus_text)

    if document_type == DocumentType.POLICY:
        rendered = render_policy(title=title, structured_content=structured_content)
    elif document_type == DocumentType.FAQ:
        rendered = render_faq(title=title, structured_content=structured_content)
    elif document_type == DocumentType.PRODUCT:
        rendered = render_product(title=title, canonical_url=canonical_url, structured_content=structured_content)
    elif source_type in (SourceType.PDF, SourceType.SCANNED_PDF) or structured_content.pages:
        rendered = render_pdf(title=title, structured_content=structured_content)
    else:
        rendered = render_default(title=title, structured_content=structured_content)

    return sanitize_retrieval_text(rendered)
