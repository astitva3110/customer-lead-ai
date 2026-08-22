import re

from app.kb.cleaning.base import BaseCleaner, CleaningResult
from app.kb.cleaning.generic import GenericCleaner
from app.kb.enums import ExtractionMethod, SourceType
from app.kb.models.raw import RawArtifact

FOOTER_SECTIONS = (
    "\n## About",
    "\n## Help",
    "\n## Hearing Solutions",
    "\n## Policies",
    "\n## Get In Touch",
    "\n## Sign up for access",
    "\nPayment methods",
    "\n© 20",
    "\nDisclaimer:",
)

_CART_LINE_MARKERS = (
    "skip to content",
    "your cart is empty",
    "continue shopping",
    "have an account?",
    "to check out faster",
    "## your cart",
    "loading...",
    "## subtotal",
    "rs. 0.00",
    "taxes and shipping calculated at checkout",
    "check out",
)

_NAV_LABELS = frozenset({"home", "about us", "hearing solution", "omni", "hearing health", "tiny", "bluup"})

_JUDgeme_RE = re.compile(r"\[Judge\.me\].*$", re.MULTILINE)
_SHOPIFY_UI_RE = re.compile(
    r"Choosing a selection results in a full page refresh\.|Opens in a new window\.",
    re.IGNORECASE,
)
_VERIFIED_REVIEWS_RE = re.compile(r"\[\s*[\d.]+\s*out of 5 stars.*Verified\s*\]", re.IGNORECASE)
_SHARE_WIDGET_RE = re.compile(
    r"^(Share\s+Share|Link|Close share|Copy link)\s*$",
    re.IGNORECASE,
)
_SHARE_INLINE_RE = re.compile(r"Close share\s*Copy link|Share\s+Share", re.IGNORECASE)


def _strip_shopify_cart(text: str) -> tuple[str, bool]:
    lines = text.split("\n")
    output: list[str] = []
    skipping = False
    removed = False
    for line in lines:
        lower = line.strip().lower()
        if not skipping and ("skip to content" in lower or lower == "## your cart is empty"):
            skipping = True
            removed = True
            continue
        if skipping:
            if lower.startswith("# ") and "![!" not in lower:
                skipping = False
                output.append(line)
                continue
            if lower.startswith("## ") and "your cart" not in lower and "subtotal" not in lower:
                skipping = False
                output.append(line)
                continue
            if any(marker in lower for marker in _CART_LINE_MARKERS):
                continue
            if not line.strip():
                continue
            if "earkart india" in lower and "clinics" in lower:
                continue
            skipping = False
        output.append(line)
    return "\n".join(output), removed


def _is_shopify_nav_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("*"):
        return False
    lower = stripped.lower()
    return any(label in lower for label in _NAV_LABELS) or "[ log in]" in lower


def _strip_shopify_nav_lines(text: str) -> tuple[str, bool]:
    lines = text.split("\n")
    output: list[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if _is_shopify_nav_line(line):
            removed = True
            continue
        if lower in {"search", "cart", "hearing solution"}:
            removed = True
            continue
        if lower.startswith("[log in]") or lower.startswith("[ login") or "[ log in]" in lower:
            removed = True
            continue
        if stripped.startswith("[![cart]") or stripped.startswith("[ Book Appointment ]"):
            removed = True
            continue
        if stripped.startswith("* [ Facebook ]") or stripped.startswith("* [ Instagram ]") or stripped.startswith("* [ YouTube ]"):
            removed = True
            continue
        if stripped.startswith("[![earKART]") or stripped.startswith("[![earkart]"):
            removed = True
            continue
        output.append(line)
    return "\n".join(output), removed


def _strip_shopify_footer(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    cut = len(text)
    for marker in FOOTER_SECTIONS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
            removed.append("footer")
    if cut < len(text):
        text = text[:cut].strip()
    return text, sorted(set(removed))


class EarkartComCleaner(BaseCleaner):
    profile = "earkart.com"
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

        text, cart_removed = _strip_shopify_cart(text)
        if cart_removed:
            removed.append("shopify_cart")

        text, nav_removed = _strip_shopify_nav_lines(text)
        if nav_removed:
            removed.append("navigation")

        text, footer_removed = _strip_shopify_footer(text)
        removed.extend(footer_removed)

        if _JUDgeme_RE.search(text):
            text = _JUDgeme_RE.sub("", text)
            removed.append("judge_me_widget")

        if _SHOPIFY_UI_RE.search(text):
            text = _SHOPIFY_UI_RE.sub("", text)
            removed.append("shopify_ui")

        if _VERIFIED_REVIEWS_RE.search(text):
            text = _VERIFIED_REVIEWS_RE.sub("", text)
            removed.append("review_widget")

        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if _SHARE_WIDGET_RE.match(stripped) or _SHARE_INLINE_RE.search(stripped):
                removed.append("share_widget")
                continue
            lines.append(line)
        text = "\n".join(lines)

        text = self.generic.clean_text(text)[0]

        return CleaningResult(
            content=text,
            profile=self.profile,
            profile_version=self.profile_version,
            removed_elements=sorted(set(removed)),
        )
