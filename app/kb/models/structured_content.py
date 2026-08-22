from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ParagraphNode(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    text: str
    confidence: float | None = None


class HeadingNode(BaseModel):
    type: Literal["heading"] = "heading"
    text: str
    level: int = Field(ge=1, le=6)


class ListNode(BaseModel):
    type: Literal["list"] = "list"
    ordered: bool = False
    items: list[str]


class TableNode(BaseModel):
    type: Literal["table"] = "table"
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class SectionNode(BaseModel):
    type: Literal["section"] = "section"
    heading: str | None = None
    level: int = Field(default=1, ge=1, le=6)
    children: list["ContentNode"] = Field(default_factory=list)


ContentNode = Annotated[
    Union[ParagraphNode, HeadingNode, ListNode, TableNode, SectionNode],
    Field(discriminator="type"),
]

SectionNode.model_rebuild()


class OcrMetadata(BaseModel):
    engine: str = "tesseract"
    dpi: int = 200
    language: str = "eng"
    average_confidence: float | None = None


class PageContent(BaseModel):
    page_number: int = Field(ge=1)
    ocr_confidence: float | None = None
    blocks: list[ContentNode] = Field(default_factory=list)


class DocumentContent(BaseModel):
    type: Literal["document"] = "document"
    title: str | None = None
    children: list[ContentNode] = Field(default_factory=list)
    pages: list[PageContent] | None = None
    ocr: OcrMetadata | None = None


StructuredContent = DocumentContent
