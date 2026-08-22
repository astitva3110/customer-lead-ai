from app.kb.cleaning.base import BaseCleaner
from app.kb.cleaning.earkart_com import EarkartComCleaner
from app.kb.cleaning.earkart_in import EarkartInCleaner
from app.kb.cleaning.generic import GenericCleaner
from app.kb.models.raw import RawArtifact


class FallbackCleaner(BaseCleaner):
    profile = "generic"
    profile_version = "1.0"

    def __init__(self) -> None:
        self.generic = GenericCleaner()

    def clean(self, artifact: RawArtifact):
        from app.kb.cleaning.base import CleaningResult

        if artifact.source_type.value == "html":
            text, removed = self.generic.clean_text(artifact.content)
            return CleaningResult(content=text, removed_elements=removed)
        if artifact.extraction_method.value == "ocr":
            from app.kb.cleaning.ocr import OcrCleaner

            return OcrCleaner().clean(artifact)
        from app.kb.cleaning.pdf import PdfCleaner

        return PdfCleaner().clean(artifact)


_CLEANERS: dict[str, BaseCleaner] = {
    "earkart.in": EarkartInCleaner(),
    "earkart.com": EarkartComCleaner(),
}


def get_cleaner(website: str) -> BaseCleaner:
    return _CLEANERS.get(website.lower(), FallbackCleaner())
