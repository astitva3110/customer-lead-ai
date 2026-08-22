from app.kb.models.raw import RawArtifact
from app.kb.models.structured_content import DocumentContent
from app.kb.structuring.markdown import markdown_to_structured


def pages_to_structured(title: str, artifact: RawArtifact, pages) -> DocumentContent:
    from app.kb.models.structured_content import PageContent

    page_models: list[PageContent] = []
    for page in pages or []:
        if page.blocks:
            page_models.append(
                PageContent(
                    page_number=page.page_number,
                    ocr_confidence=getattr(page, "ocr_confidence", None),
                    blocks=page.blocks,
                )
            )
        elif hasattr(page, "text"):
            from app.kb.models.structured_content import ParagraphNode

            page_models.append(
                PageContent(
                    page_number=page.page_number,
                    ocr_confidence=getattr(page, "ocr_confidence", None),
                    blocks=[ParagraphNode(text=page.text)],
                )
            )
    return DocumentContent(type="document", title=title, pages=page_models or None, ocr=artifact.ocr)
