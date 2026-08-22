"""V3 chunking engine — atomic knowledge units with 30–150 token target."""

from __future__ import annotations

from app.config import settings
from app.kb.chunking.semantic_units import build_semantic_units
from app.kb.chunking.tokenizer import CharacterEstimateTokenizer
from app.kb.chunking.v3.atomic_units import (
    AtomicUnit,
    atomize_semantic_unit,
    enforce_token_targets,
    flatten_semantic_units,
)
from app.kb.models.structured_content import DocumentContent
from app.kb.enums import DocumentType


class ChunkingEngineV3:
    def __init__(self) -> None:
        self.tokenizer = CharacterEstimateTokenizer(chars_per_token=settings.chunk_chars_per_token)
        self.target_min = settings.phase16_target_min_tokens
        self.target_max = settings.phase16_target_max_tokens
        self.hard_max = settings.chunk_max_tokens

    def build_atomic_units(
        self,
        *,
        structured_content: DocumentContent,
        document_type: DocumentType,
        title: str,
        canonical_url: str,
    ) -> list[AtomicUnit]:
        semantic_units = build_semantic_units(
            structured_content=structured_content,
            document_type=document_type,
            title=title,
            canonical_url=canonical_url,
        )
        flat = flatten_semantic_units(semantic_units)
        atoms: list[AtomicUnit] = []
        for unit in flat:
            atoms.extend(atomize_semantic_unit(unit))
        return enforce_token_targets(
            atoms,
            tokenizer=self.tokenizer,
            target_min=self.target_min,
            target_max=self.target_max,
            hard_max=self.hard_max,
        )
