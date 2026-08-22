"""V3 atomic knowledge-unit detection and splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.kb.chunking.semantic_units import SemanticUnit
from app.kb.chunking.tokenizer import CharacterEstimateTokenizer

HEARING_AID_TYPE_LINE = re.compile(
    r"^([A-Za-z\-/ ]+\([A-Z]{2,4}\))\s*[—–-]\s*(.+)$",
    re.MULTILINE,
)
FAQ_NUMBERED = re.compile(r"^(\d+\.\d+)\s+(.+\?)\s*(.+)$", re.MULTILINE | re.DOTALL)
FAQ_INLINE = re.compile(r"(\d+\.\d+)\s+([^?]+\?)\s+(.+?)(?=\d+\.\d+\s+[^?]+\?|$)", re.DOTALL)
BULLET_LINE = re.compile(r"^[-•*]\s+(.+)$", re.MULTILINE)
PRODUCT_HEADER = re.compile(r"^(\d+\.\d+(?:\s+[A-Za-z0-9+]+)?)\s*$", re.MULTILINE)
SPEC_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 /\-]+):\s*(.+)$", re.MULTILINE)
CLAUSE_LINE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$", re.MULTILINE)
OMNI_SECTION = re.compile(r"\bomni\b", re.I)
METADATA_PREAMBLE = re.compile(r"source pages:|snapshot date:|document type:", re.I)


@dataclass
class AtomicUnit:
    content: str
    section_path: list[str] = field(default_factory=list)
    page_number: int | None = None
    unit_type: str = "paragraph"
    source_block_path: list[str] = field(default_factory=list)
    parent_section: str | None = None
    subsection: str | None = None


def _derive_hierarchy(section_path: list[str]) -> tuple[str | None, str | None]:
    if not section_path:
        return None, None
    if len(section_path) == 1:
        return section_path[0], None
    return section_path[-2], section_path[-1]


def _with_hierarchy(unit: AtomicUnit) -> AtomicUnit:
    parent, subsection = _derive_hierarchy(unit.section_path)
    unit.parent_section = parent
    unit.subsection = subsection or (unit.section_path[-1] if unit.section_path else None)
    return unit


def _split_hearing_aid_types(unit: SemanticUnit) -> list[AtomicUnit]:
    matches = list(HEARING_AID_TYPE_LINE.finditer(unit.content))
    if len(matches) < 2:
        return []
    atoms: list[AtomicUnit] = []
    for match in matches:
        label = match.group(1).strip()
        description = match.group(2).strip()
        content = f"{label} — {description}"
        subsection = f"Types of Hearing Aids > {label.split('(')[0].strip()}"
        path = [*unit.section_path[:-1], subsection] if unit.section_path else [subsection]
        atoms.append(
            _with_hierarchy(
                AtomicUnit(
                    content=content,
                    section_path=path,
                    page_number=unit.page_number,
                    unit_type="hearing_aid_type",
                    source_block_path=[*unit.source_block_path, f"type:{label}"],
                )
            )
        )
    return atoms


def _split_bullets(unit: SemanticUnit, *, unit_type: str, min_items: int = 2) -> list[AtomicUnit]:
    lines = [line.strip() for line in unit.content.splitlines() if line.strip()]
    bullets = [match.group(1).strip() for line in lines if (match := BULLET_LINE.match(line))]
    if len(bullets) < min_items:
        return []
    atoms: list[AtomicUnit] = []
    for index, bullet in enumerate(bullets):
        atoms.append(
            _with_hierarchy(
                AtomicUnit(
                    content=bullet,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type=unit_type,
                    source_block_path=[*unit.source_block_path, f"bullet:{index}"],
                )
            )
        )
    return atoms


def _split_faq_pairs(unit: SemanticUnit) -> list[AtomicUnit]:
    if unit.unit_type == "faq_pair":
        return [
            _with_hierarchy(
                AtomicUnit(
                    content=unit.content,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="faq_pair",
                    source_block_path=unit.source_block_path,
                )
            )
        ]

    text = unit.content
    atoms: list[AtomicUnit] = []
    for match in FAQ_INLINE.finditer(text):
        number, question, answer = match.group(1), match.group(2).strip(), match.group(3).strip()
        if not answer:
            continue
        content = f"Question: {question}\n\nAnswer: {answer.strip()}"
        subsection = f"FAQ {number}"
        path = [*unit.section_path[:-1], subsection] if unit.section_path else [subsection]
        atoms.append(
            _with_hierarchy(
                AtomicUnit(
                    content=content,
                    section_path=path,
                    page_number=unit.page_number,
                    unit_type="faq_pair",
                    source_block_path=[*unit.source_block_path, f"faq:{number}"],
                )
            )
        )
    return atoms


def _split_product_blocks(unit: SemanticUnit) -> list[AtomicUnit]:
    text = unit.content.strip()
    if "Features:" not in text and "Why " not in text and len(text.split()) < 40:
        return []
    header_match = unit.section_path[-1] if unit.section_path else ""
    if re.search(r"\d+\.\d+", header_match):
        return [
            _with_hierarchy(
                AtomicUnit(
                    content=text,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="product_description",
                    source_block_path=unit.source_block_path,
                )
            )
        ]
    return []


def _split_policy_clauses(unit: SemanticUnit) -> list[AtomicUnit]:
    if unit.unit_type != "clause":
        return []
    return [
        _with_hierarchy(
            AtomicUnit(
                content=unit.content.strip(),
                section_path=unit.section_path,
                page_number=unit.page_number,
                unit_type="policy_clause",
                source_block_path=unit.source_block_path,
            )
        )
    ]


def _split_company_information(unit: SemanticUnit) -> list[AtomicUnit]:
    lowered = unit.content.lower()
    if "registered office" in lowered and "corporate office" in lowered:
        return [
            _with_hierarchy(
                AtomicUnit(
                    content=unit.content.strip(),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="company_information",
                    source_block_path=unit.source_block_path,
                )
            )
        ]
    if OMNI_SECTION.search(unit.content) and not METADATA_PREAMBLE.search(unit.content[:200]):
        return [
            _with_hierarchy(
                AtomicUnit(
                    content=unit.content.strip(),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="company_information",
                    source_block_path=unit.source_block_path,
                )
            )
        ]
    return []


def _split_contact_information(unit: SemanticUnit) -> list[AtomicUnit]:
    lowered = unit.content.lower()
    hints = ("customer support", "contact us", "reach us", "phone", "email", "@earkart")
    if any(hint in lowered for hint in hints):
        return [
            _with_hierarchy(
                AtomicUnit(
                    content=unit.content.strip(),
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="contact_information",
                    source_block_path=unit.source_block_path,
                )
            )
        ]
    return []


def _split_benefits(unit: SemanticUnit) -> list[AtomicUnit]:
    lowered = unit.content.lower()
    if "why choose" not in lowered and "what makes us better" not in lowered:
        return []
    lines = unit.content.splitlines()
    benefit_lines = [
        line.strip().lstrip("-•*").strip()
        for line in lines
        if line.strip().startswith(("-", "•", "*")) or "—" in line
    ]
    benefit_lines = [line for line in benefit_lines if len(line.split()) >= 4]
    if len(benefit_lines) < 2:
        return _split_bullets(unit, unit_type="product_feature", min_items=2)
    atoms: list[AtomicUnit] = []
    for index, line in enumerate(benefit_lines):
        atoms.append(
            _with_hierarchy(
                AtomicUnit(
                    content=line,
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="product_feature",
                    source_block_path=[*unit.source_block_path, f"benefit:{index}"],
                )
            )
        )
    return atoms


def _split_specifications(unit: SemanticUnit) -> list[AtomicUnit]:
    specs = list(SPEC_LINE.finditer(unit.content))
    if len(specs) < 2:
        return []
    atoms: list[AtomicUnit] = []
    for match in specs:
        name, value = match.group(1).strip(), match.group(2).strip()
        if len(value.split()) < 2:
            continue
        atoms.append(
            _with_hierarchy(
                AtomicUnit(
                    content=f"{name}: {value}",
                    section_path=unit.section_path,
                    page_number=unit.page_number,
                    unit_type="product_specification",
                    source_block_path=[*unit.source_block_path, f"spec:{name}"],
                )
            )
        )
    return atoms


def atomize_semantic_unit(unit: SemanticUnit) -> list[AtomicUnit]:
    if not unit.content.strip():
        return []
    if unit.unit_type == "heading":
        return []

    for splitter in (
        _split_hearing_aid_types,
        _split_faq_pairs,
        _split_company_information,
        _split_contact_information,
        _split_benefits,
        _split_specifications,
        _split_product_blocks,
        _split_policy_clauses,
    ):
        atoms = splitter(unit)
        if atoms:
            return atoms

    bullets = _split_bullets(unit, unit_type="product_feature", min_items=3)
    if bullets:
        return bullets

    return [
        _with_hierarchy(
            AtomicUnit(
                content=unit.content.strip(),
                section_path=unit.section_path,
                page_number=unit.page_number,
                unit_type=unit.unit_type if unit.unit_type != "semantic" else "paragraph",
                source_block_path=unit.source_block_path,
            )
        )
    ]


def flatten_semantic_units(units: list[SemanticUnit]) -> list[SemanticUnit]:
    flat: list[SemanticUnit] = []
    for unit in units:
        if unit.children:
            flat.extend(flatten_semantic_units(unit.children))
        elif unit.content.strip():
            flat.append(unit)
    return flat


def enforce_token_targets(
    atoms: list[AtomicUnit],
    *,
    tokenizer: CharacterEstimateTokenizer,
    target_min: int,
    target_max: int,
    hard_max: int,
) -> list[AtomicUnit]:
    """Split oversized atoms and drop heading-only fragments."""
    result: list[AtomicUnit] = []
    for atom in atoms:
        text = atom.content.strip()
        if not text or _is_heading_only(text, tokenizer):
            continue
        tokens = tokenizer.count(text)
        if tokens <= hard_max:
            result.append(atom)
            continue
        parts = _split_on_paragraphs(text, tokenizer, target_max, hard_max)
        for index, part in enumerate(parts):
            if _is_heading_only(part, tokenizer):
                continue
            result.append(
                AtomicUnit(
                    content=part,
                    section_path=atom.section_path,
                    page_number=atom.page_number,
                    unit_type=atom.unit_type,
                    source_block_path=[*atom.source_block_path, f"split:{index}"],
                    parent_section=atom.parent_section,
                    subsection=atom.subsection,
                )
            )
    return result


def _is_heading_only(text: str, tokenizer: CharacterEstimateTokenizer) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if tokenizer.count(stripped) <= 12 and not stripped.endswith((".", "?", "!")):
        if stripped == stripped.title() or CLAUSE_LINE.match(stripped):
            return True
    return False


def _split_on_paragraphs(
    text: str,
    tokenizer: CharacterEstimateTokenizer,
    target_max: int,
    hard_max: int,
) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if tokenizer.count(candidate) <= target_max:
            current = candidate
            continue
        if current:
            parts.append(current)
        if tokenizer.count(paragraph) <= hard_max:
            current = paragraph
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            chunk = ""
            for sentence in sentences:
                candidate = f"{chunk} {sentence}".strip() if chunk else sentence
                if tokenizer.count(candidate) <= target_max:
                    chunk = candidate
                else:
                    if chunk:
                        parts.append(chunk)
                    chunk = sentence
            current = chunk
    if current:
        parts.append(current)
    return parts or [text]
