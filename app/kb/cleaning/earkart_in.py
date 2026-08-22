import re

from app.kb.cleaning.base import BaseCleaner, CleaningResult
from app.kb.cleaning.generic import GenericCleaner
from app.kb.enums import ExtractionMethod, SourceType
from app.kb.models.raw import RawArtifact

_NAV_ITEM_RE = re.compile(r"^\s*[\*\-]\s+\[.*?\]\(https?://(?:www\.)?earkart\.in/", re.IGNORECASE)
_LOGO_HEADING_RE = re.compile(r"^#\s+\[\!\[.*\]\(.*\)\]\(.*\)\s*$")
_BECOME_PARTNER_RE = re.compile(r"^\[?\s*Become Our Partner\s*\]?\(.*\)\s*$", re.IGNORECASE)
_APPOINTMENT_RE = re.compile(r"^(####\s+Make An Appointment|Book Appointment)\s*$", re.IGNORECASE)
_HEADER_CONTACT_RE = re.compile(r"^\s*[\*\-]\s+(A-401 Sector 11|contact@earkart\.com|\[\s*\]\()", re.IGNORECASE)
_WHATSAPP_RE = re.compile(r"^!\[image\]\(.*whatsapp.*\)\s*Whatsapp\s*$", re.IGNORECASE)
_HELP_DESK_RE = re.compile(r"branding-resources\.s3\.|help-desk|helpdesk", re.IGNORECASE)

FOOTER_MARKERS = (
    "\n### Quick Links",
    "\n## Quick Links",
    "\n### Contact Information",
    "\n## Contact Information",
    "\nearKART is transforming the way India experiences hearing care",
    "\nDisclaimer:",
    "\nCopyright ©",
    "\nCIN: L74999DL2021PLC399313",
    "\nCall Today",
)

NAV_ONLY_LABELS = frozenset(
    {
        "home",
        "about us",
        "contact us",
        "hearing aids",
        "hearing loss",
        "blog",
        "press release",
        "earkart centers",
        "investor",
        "other products",
        "hearing aid test system",
    }
)


def _strip_footer(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    cut_at = len(text)
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
            removed.append("footer")
    if cut_at < len(text):
        text = text[:cut_at].strip()
    return text, sorted(set(removed))


def _is_nav_list_line(line: str) -> bool:
    stripped = line.strip()
    if not _NAV_ITEM_RE.match(stripped):
        return False
    label_match = re.search(r"\[([^\]]+)\]", stripped)
    if not label_match:
        return True
    label = label_match.group(1).strip().lower()
    if any(nav in label for nav in NAV_ONLY_LABELS):
        return True
    if "investor" in label or "hearing" in label or "radius" in label.lower():
        return True
    return stripped.count("[") >= 1 and stripped.startswith("*")


def _find_content_start(lines: list[str]) -> int:
    consecutive_nav = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_CONTACT_RE.match(stripped):
            consecutive_nav = 0
            continue
        if _BECOME_PARTNER_RE.match(stripped):
            consecutive_nav = 0
            continue
        if _LOGO_HEADING_RE.match(stripped):
            consecutive_nav = 0
            continue
        if _APPOINTMENT_RE.match(stripped):
            consecutive_nav = 0
            continue
        if _is_nav_list_line(stripped):
            consecutive_nav += 1
            continue
        if consecutive_nav >= 3:
            if stripped.startswith("## ") and "![" not in stripped:
                return index
            if stripped.startswith("# ") and "![" not in stripped:
                return index
            if len(stripped) > 30 and not stripped.startswith("*"):
                return index
        if stripped.startswith("## ") and "![" not in stripped and "quick links" not in stripped.lower():
            return index
        if stripped.startswith("# ") and "![" not in stripped and len(stripped) > 12:
            return index
    return 0


def _strip_header_nav(text: str) -> tuple[str, list[str]]:
    lines = text.split("\n")
    start = _find_content_start(lines)
    if start <= 0:
        return text, []
    removed_lines = lines[:start]
    removed = []
    if any(_is_nav_list_line(line) for line in removed_lines):
        removed.append("navigation")
    if any(_BECOME_PARTNER_RE.match(line.strip()) for line in removed_lines):
        removed.append("partner_cta")
    if any(_HEADER_CONTACT_RE.match(line.strip()) for line in removed_lines):
        removed.append("header_contact")
    if any(_APPOINTMENT_RE.match(line.strip()) for line in removed_lines):
        removed.append("appointment_widget")
    if any(_LOGO_HEADING_RE.match(line.strip()) for line in removed_lines):
        removed.append("logo_header")
    return "\n".join(lines[start:]).strip(), removed


def _strip_helpdesk(text: str) -> tuple[str, list[str]]:
    if not _HELP_DESK_RE.search(text):
        return text, []
    lines = [line for line in text.split("\n") if not _HELP_DESK_RE.search(line)]
    return "\n".join(lines).strip(), ["helpdesk_widget"]


class EarkartInCleaner(BaseCleaner):
    profile = "earkart.in"
    profile_version = "1.0"

    def __init__(self) -> None:
        self.generic = GenericCleaner()

    def clean(self, artifact: RawArtifact) -> CleaningResult:
        if artifact.source_type == SourceType.HTML:
            return self._clean_html(artifact)
        if artifact.extraction_method == ExtractionMethod.OCR:
            from app.kb.cleaning.ocr import OcrCleaner

            result = OcrCleaner(profile=self.profile).clean(artifact)
            result.profile = self.profile
            return result
        from app.kb.cleaning.pdf import PdfCleaner

        result = PdfCleaner(profile=self.profile).clean(artifact)
        result.profile = self.profile
        return result

    def _clean_html(self, artifact: RawArtifact) -> CleaningResult:
        removed: list[str] = []
        text, generic_removed = self.generic.clean_text(artifact.content)
        removed.extend(generic_removed)

        text, footer_removed = _strip_footer(text)
        removed.extend(footer_removed)

        text, nav_removed = _strip_header_nav(text)
        removed.extend(nav_removed)

        text, helpdesk_removed = _strip_helpdesk(text)
        removed.extend(helpdesk_removed)

        lines = []
        skip_appointment = False
        for line in text.split("\n"):
            if _APPOINTMENT_RE.match(line.strip()):
                skip_appointment = True
                removed.append("appointment_widget")
                continue
            if skip_appointment and not line.strip():
                skip_appointment = False
                continue
            if skip_appointment and line.strip().lower() == "book appointment":
                continue
            if _BECOME_PARTNER_RE.match(line.strip()):
                removed.append("partner_cta")
                continue
            if _WHATSAPP_RE.match(line.strip()):
                removed.append("whatsapp_widget")
                continue
            lines.append(line)
        text = self.generic.clean_text("\n".join(lines))[0]

        return CleaningResult(
            content=text,
            profile=self.profile,
            profile_version=self.profile_version,
            removed_elements=sorted(set(removed)),
        )
