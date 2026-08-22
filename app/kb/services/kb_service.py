"""Bridge crawled/scraped content into immutable RAW storage and canonical documents."""

from datetime import datetime, timezone

from app.config import settings
from app.kb.enums import ExtractionMethod, SourceType
from app.kb.hashing import content_hash
from app.kb.models.raw import RawArtifact
from app.kb.models.structured_content import PageContent, ParagraphNode
from app.kb.services.cleaning_pipeline import CleaningPipeline
from app.kb.services.document_builder import DocumentBuilder
from app.providers.pdf.pdf_extractor import PdfExtractResult
from app.kb.infrastructure.repositories.canonical_repository import CanonicalRepository
from app.kb.infrastructure.repositories.cleaned_repository import CleanedRepository
from app.kb.infrastructure.repositories.raw_repository import RawRepository, RawWriteResult
from app.kb.url_normalizer import extract_website, make_document_id, normalize_url


class KnowledgeBaseService:
    """KB service: RAW writes + cleaning pipeline + canonical representation."""

    def __init__(
        self,
        raw_repository: RawRepository | None = None,
        cleaned_repository: CleanedRepository | None = None,
        canonical_repository: CanonicalRepository | None = None,
        *,
        raw_store: RawRepository | None = None,
        cleaned_store: CleanedRepository | None = None,
        canonical_store: CanonicalRepository | None = None,
    ) -> None:
        raw_repository = raw_repository or raw_store or RawRepository(settings.raw_dir)
        cleaned_repository = cleaned_repository or cleaned_store or CleanedRepository(settings.cleaned_dir)
        canonical_repository = canonical_repository or canonical_store or CanonicalRepository(settings.canonical_dir)
        self.raw_repository = raw_repository
        self.cleaned_repository = cleaned_repository
        self.canonical_repository = canonical_repository
        self.raw_store = raw_repository
        self.cleaned_store = cleaned_repository
        self.canonical_store = canonical_repository
        self.builder = DocumentBuilder(self.canonical_repository)
        self.pipeline = CleaningPipeline(self.cleaned_repository, self.canonical_repository)

    def _resolve_version(self, website: str, document_id: str, body_hash: str) -> tuple[int, bool]:
        current = self.canonical_store.read_current(website, document_id)
        if current is None:
            return 1, True
        if current.content_hash == body_hash:
            return current.version, False
        return current.version + 1, True

    def _process_raw(self, artifact: RawArtifact, raw_result: RawWriteResult) -> None:
        if raw_result.duplicate:
            existing = self.canonical_store.read_current(artifact.website, make_document_id(artifact.website, artifact.canonical_url))
            if existing and existing.content_hash == artifact.content_hash:
                return
        self.pipeline.process(artifact)

    def ingest_html(
        self,
        url: str,
        title: str,
        markdown: str,
        *,
        crawl_root_url: str | None = None,
        scraped_at: datetime | None = None,
    ) -> tuple[RawWriteResult, bool]:
        website = extract_website(url)
        canonical_url = normalize_url(url)
        body_hash = content_hash(markdown)
        now = scraped_at or datetime.now(timezone.utc)

        artifact = RawArtifact(
            url=url,
            canonical_url=canonical_url,
            website=website,
            title=title,
            source_type=SourceType.HTML,
            extraction_method=ExtractionMethod.HTML_PARSER,
            content=markdown,
            scraped_at=now,
            content_hash=body_hash,
            crawl_root_url=crawl_root_url,
        )

        duplicate = self.raw_store.is_duplicate_content(website, canonical_url, body_hash)
        raw_result = self.raw_store.write(artifact)
        self._process_raw(raw_result.artifact, raw_result)
        return raw_result, not duplicate

    def ingest_pdf(
        self,
        url: str,
        result: PdfExtractResult,
        *,
        crawl_root_url: str | None = None,
        scraped_at: datetime | None = None,
    ) -> tuple[RawWriteResult, bool]:
        from app.kb.enums import ExtractionMethod, SourceType

        website = extract_website(url)
        canonical_url = normalize_url(url)
        now = scraped_at or datetime.now(timezone.utc)
        source_type = SourceType.SCANNED_PDF if result.extraction_method == ExtractionMethod.OCR else SourceType.PDF
        body_hash = content_hash(result.full_text)

        pages = [
            PageContent(
                page_number=page.page_number,
                ocr_confidence=page.ocr_confidence,
                blocks=page.blocks or [ParagraphNode(text=page.text, confidence=page.ocr_confidence)],
            )
            for page in result.pages
            if page.text.strip()
        ]

        artifact = RawArtifact(
            url=url,
            canonical_url=canonical_url,
            website=website,
            title=result.title,
            source_type=source_type,
            extraction_method=result.extraction_method,
            content=result.full_text,
            scraped_at=now,
            content_hash=body_hash,
            crawl_root_url=crawl_root_url,
            pages=pages or None,
            ocr=result.ocr,
        )

        duplicate = self.raw_store.is_duplicate_content(website, canonical_url, body_hash)
        raw_result = self.raw_store.write(artifact)
        self._process_raw(raw_result.artifact, raw_result)
        return raw_result, not duplicate

    def process_raw_artifact(self, artifact: RawArtifact):
        """Run cleaning pipeline on an existing RAW artifact."""
        return self.pipeline.process(artifact)
