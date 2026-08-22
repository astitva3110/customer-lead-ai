import re

from app.kb.enums import ExtractionMethod, ProcessingStatus, SourceType
from app.kb.models.canonical import CanonicalDocument, DocumentLink, utc_now
from app.kb.models.raw import CleanedArtifact, RawArtifact
from app.kb.models.structured_content import DocumentContent
from app.kb.infrastructure.repositories.canonical_repository import CanonicalRepository
from app.kb.structuring.links import extract_links
from app.kb.structuring.markdown import markdown_to_structured
from app.kb.structuring.pdf import pages_to_structured
from app.kb.url_normalizer import make_document_id


def _plain_text_from_structured(structured: DocumentContent) -> str:
    parts: list[str] = []

    def walk(nodes) -> None:
        for node in nodes:
            node_type = getattr(node, "type", None)
            if node_type == "paragraph":
                parts.append(node.text)
            elif node_type == "heading":
                parts.append(node.text)
            elif node_type == "list":
                parts.extend(node.items)
            elif node_type == "table":
                if node.headers:
                    parts.append(" | ".join(node.headers))
                for row in node.rows:
                    parts.append(" | ".join(row))
            elif node_type == "section":
                if node.heading:
                    parts.append(node.heading)
                walk(node.children)

    if structured.pages:
        for page in structured.pages:
            parts.append(f"Page {page.page_number}")
            walk(page.blocks)
    else:
        walk(structured.children)
    return "\n\n".join(part for part in parts if part.strip())


class DocumentBuilder:
    """Build canonical documents from CLEANED artifacts."""

    def __init__(self, canonical_repository: CanonicalRepository | None = None, *, canonical_store: CanonicalRepository | None = None) -> None:
        canonical_repository = canonical_repository or canonical_store
        self.canonical_repository = canonical_repository
        self.canonical_store = canonical_repository

    def build_from_cleaned(
        self,
        *,
        raw: RawArtifact,
        cleaned: CleanedArtifact,
        pages=None,
        version: int = 1,
        is_current: bool = True,
        processing_status: ProcessingStatus = ProcessingStatus.CLEANED,
    ) -> CanonicalDocument:
        now = utc_now()
        document_id = make_document_id(raw.website, raw.canonical_url)

        if raw.source_type == SourceType.HTML:
            structured = markdown_to_structured(cleaned.title, cleaned.content)
            plain_text = _plain_text_from_structured(structured) or cleaned.content
            links = extract_links(cleaned.content, raw.website)
        else:
            structured = pages_to_structured(cleaned.title, raw, pages or cleaned.pages)
            plain_text = cleaned.content
            links = []

        metadata = {
            "raw_content_hash": raw.content_hash,
            "cleaned_content_hash": cleaned.content_hash,
            **cleaned.metadata,
            **raw.metadata,
        }
        if raw.crawl_root_url:
            metadata["crawl_root_url"] = raw.crawl_root_url

        document = CanonicalDocument(
            document_id=document_id,
            website=raw.website,
            source_url=raw.url,
            canonical_url=raw.canonical_url,
            title=cleaned.title,
            source_type=raw.source_type,
            extraction_method=raw.extraction_method,
            language="en",
            version=version,
            is_current=is_current,
            content_hash=raw.content_hash,
            processing_status=processing_status,
            scraped_at=raw.scraped_at,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            structured_content=structured,
            plain_text=plain_text,
            links=links,
        )
        return document

    def build_from_raw(
        self,
        artifact: RawArtifact,
        *,
        version: int = 1,
        is_current: bool = True,
        processing_status: ProcessingStatus = ProcessingStatus.EXTRACTED,
    ) -> CanonicalDocument:
        """Legacy path without cleaning — prefer CleaningPipeline."""
        from app.kb.cleaning.registry import get_cleaner
        from app.kb.hashing import content_hash
        from app.kb.models.raw import CleanedArtifact

        cleaning = get_cleaner(artifact.website).clean(artifact)
        cleaned = CleanedArtifact(
            raw_content_hash=artifact.content_hash,
            url=artifact.url,
            canonical_url=artifact.canonical_url,
            website=artifact.website,
            title=artifact.title,
            source_type=artifact.source_type,
            extraction_method=artifact.extraction_method,
            content=cleaning.content,
            cleaned_at=cleaning.cleaned_at,
            content_hash=content_hash(cleaning.content),
            pages=cleaning.pages,
            ocr=cleaning.ocr or artifact.ocr,
            metadata={"cleaning": cleaning.cleaning_metadata()},
        )
        return self.build_from_cleaned(
            raw=artifact,
            cleaned=cleaned,
            pages=cleaning.pages,
            version=version,
            is_current=is_current,
            processing_status=processing_status,
        )

    def next_version(self, website: str, document_id: str, new_content_hash: str) -> int:
        if self.canonical_store is None:
            return 1
        current = self.canonical_store.read_current(website, document_id)
        if current is None:
            return 1
        if current.content_hash == new_content_hash:
            return current.version
        return current.version + 1
