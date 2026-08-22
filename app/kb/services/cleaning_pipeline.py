from app.kb.cleaning.registry import get_cleaner
from app.kb.enums import ProcessingStatus
from app.kb.hashing import content_hash
from app.kb.models.canonical import CanonicalDocument, utc_now
from app.kb.models.raw import CleanedArtifact, RawArtifact
from app.kb.services.document_builder import DocumentBuilder
from app.kb.infrastructure.repositories.canonical_repository import CanonicalRepository
from app.kb.infrastructure.repositories.cleaned_repository import CleanedRepository
from app.kb.url_normalizer import make_document_id
from app.kb.validation.document_validator import validate_document


class CleaningPipeline:
    """Convert RAW artifacts into CLEANED and CANONICAL documents."""

    def __init__(
        self,
        cleaned_repository: CleanedRepository,
        canonical_repository: CanonicalRepository | None = None,
        *,
        cleaned_store: CleanedRepository | None = None,
        canonical_store: CanonicalRepository | None = None,
    ) -> None:
        cleaned_repository = cleaned_repository or cleaned_store
        canonical_repository = canonical_repository or canonical_store
        self.cleaned_repository = cleaned_repository
        self.canonical_repository = canonical_repository
        self.cleaned_store = cleaned_repository
        self.canonical_store = canonical_repository
        self.builder = DocumentBuilder(canonical_repository)

    def process(self, raw: RawArtifact) -> tuple[CleanedArtifact, CanonicalDocument]:
        cleaner = get_cleaner(raw.website)
        cleaning = cleaner.clean(raw)

        cleaned_hash = content_hash(cleaning.content)
        cleaned = CleanedArtifact(
            raw_content_hash=raw.content_hash,
            url=raw.url,
            canonical_url=raw.canonical_url,
            website=raw.website,
            title=raw.title,
            source_type=raw.source_type,
            extraction_method=raw.extraction_method,
            content=cleaning.content,
            cleaned_at=cleaning.cleaned_at,
            content_hash=cleaned_hash,
            pages=cleaning.pages,
            ocr=cleaning.ocr or raw.ocr,
            metadata={
                "cleaning": cleaning.cleaning_metadata(),
            },
        )
        self.cleaned_store.write(cleaned)

        document_id = make_document_id(raw.website, raw.canonical_url)
        version = self.builder.next_version(raw.website, document_id, raw.content_hash)

        canonical = self.builder.build_from_cleaned(
            raw=raw,
            cleaned=cleaned,
            pages=cleaning.pages,
            version=version,
        )

        validation = validate_document(raw, cleaned, canonical)
        canonical.metadata["cleaning"] = cleaning.cleaning_metadata()
        canonical.metadata["validation"] = {
            "warnings": validation.warnings,
            "errors": validation.errors,
        }
        if validation.excluded and validation.exclusion_reason is not None:
            from app.kb.policy.domain_policy import build_exclusion_metadata

            canonical.metadata.update(build_exclusion_metadata(raw.website, validation.exclusion_reason))
        canonical.processing_status = validation.status
        canonical.updated_at = utc_now()

        if validation.errors:
            canonical.metadata["error_stage"] = "VALIDATION"
            canonical.metadata["error_message"] = "; ".join(validation.errors)

        if self.canonical_store is not None:
            self.canonical_store.write(canonical)

        return cleaned, canonical
