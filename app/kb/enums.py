from enum import StrEnum


class SourceType(StrEnum):
    HTML = "html"
    PDF = "pdf"
    SCANNED_PDF = "scanned_pdf"


class ExtractionMethod(StrEnum):
    HTML_PARSER = "html_parser"
    PDF_TEXT = "pdf_text"
    OCR = "ocr"
    NATIVE = "native"
    MIXED = "mixed"
    DOCX_NATIVE = "docx_native"
    DOC_NATIVE = "doc_native"


class DocumentUploadStatus(StrEnum):
    """Uploaded document lifecycle."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    VALIDATING = "validating"
    VALIDATED = "validated"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    STRUCTURED = "structured"
    INDEXED = "indexed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class ProcessingStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    EXTRACTED = "EXTRACTED"
    CLEANING = "CLEANING"
    CLEANED = "CLEANED"
    STRUCTURED = "STRUCTURED"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    EXCLUDED = "EXCLUDED"
    DEACTIVATED = "DEACTIVATED"


class WebsitePolicy(StrEnum):
    """Production KB website inclusion policy."""

    TARGET = "TARGET"
    NON_TARGET = "NON_TARGET"


class DocumentType(StrEnum):
    """Conservative document classification for retrieval preparation."""

    PRODUCT = "product"
    BROCHURE = "brochure"
    COMPANY = "company"
    POLICY = "policy"
    FAQ = "faq"
    BLOG = "blog"
    INVESTOR = "investor"
    NOTICE = "notice"
    PROSPECTUS = "prospectus"
    WEBPAGE = "webpage"
    OTHER = "other"


class RetrievalEligibilityStatus(StrEnum):
    """Retrieval-layer eligibility separate from canonical processing status."""

    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    REVIEW = "review"


class RetrievalExclusionReason(StrEnum):
    """Deterministic reasons for retrieval exclusion."""

    ACCOUNT_UI = "account/login UI"
    CART_CHECKOUT_UI = "cart/checkout UI"
    TRANSACTIONAL_UI = "transactional UI"
    THIRD_PARTY_APP_UI = "third-party application UI"
    TRACKING_APP_UI = "tracking application UI"
    SEARCH_UI = "search UI"
    COLLECTION_CATALOG_UI = "collection/catalog UI"
    SHOPIFY_SYSTEM_UI = "Shopify/system UI"
    INSUFFICIENT_CONTENT = "insufficient meaningful content"
    NAVIGATION_ONLY = "navigation-only page"
    OUT_OF_SCOPE = "out of production RAG scope"
    NOT_FOUND_PAGE = "404/not found page"


class RetrievalReviewReason(StrEnum):
    """Reasons a document requires human review before vector indexing."""

    OCR_QUALITY_CONCERN = "OCR quality may affect retrieval"
    AMBIGUOUS_RELEVANCE = "document relevance cannot be determined confidently"
    LOW_CONTENT_CONFIDENCE = "insufficient confidence in content usefulness"


class ExclusionReason(StrEnum):
    """Controlled reasons for excluding a document from the production KB."""

    NOT_FOUND_404 = "404 page"
    CART_CHECKOUT = "cart/checkout page"
    THIRD_PARTY_APP = "third-party app embed"
    NON_TARGET_WEBSITE = "non-target website"
    CRAWLER_ARTIFACT = "crawler artifact"
    INSUFFICIENT_CONTENT = "insufficient content"
    SPARSE_PDF_EXTRACTION = "sparse PDF text extraction"
    NAVIGATION_ONLY = "navigation-only page"
    EXTRACTION_FAILED = "extraction_failed"
    NOT_REQUIRED_FOR_PRODUCTION_KB = "not_required_for_production_kb"
