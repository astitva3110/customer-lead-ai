from __future__ import annotations

import re

FILLER_PREFIX_RE = re.compile(
    r"^(?:"
    r"ok(?:ay)?|so|well|um+|uh+|please|actually|alright|right|"
    r"hi(?:ya)?|hello|hey|"
    r"how(?:'s| is| are) (?:it going|you|u|ya)(?: doing)?"
    r")[,!.?\s]+",
    re.IGNORECASE,
)

KNOW_ABOUT_RE = re.compile(
    r"^(?:"
    r"i\s+\w+\s+to\s+know\s+(?:more\s+)?about|"
    r"(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain)\s+(?:me\s+)?"
    r"(?:more\s+)?about|"
    r"tell\s+me\s+(?:more\s+)?about|"
    r"what\s+(?:can\s+you\s+)?tell\s+me\s+about"
    r")\s+(?P<topic>.+?)\s*$",
    re.IGNORECASE,
)
TOPIC_TRAILING_RE = re.compile(
    r"\s+(?:first|instead|please|then|now|though)\s*$",
    re.IGNORECASE,
)
KNOWLEDGE_PATTERNS = (
    r"\bwhat is\b",
    r"\bwhat's\b",
    r"\bwhat are\b",
    r"\bwhat should\b",
    r"\bwhat does\b",
    r"\bwhat can i\b",
    r"\bwhat about\b",
    r"\bwho\s+(?:is|are|was|were|founded)\b",
    r"\bwhere\s+(?:is|are|can|do|does|was)\b",
    r"\bwhen\s+(?:is|are|do|does|can|was|will)\b",
    r"\bwhy\b",
    r"\bbenefits?\b",
    r"\bhow long\b",
    r"\bhow does\b",
    r"\bhow do i\b",
    r"\bhow much\b",
    r"\bhow about\b",
    r"\bcost\b",
    r"\bfeatures?\b",
    r"\bwarranty\b",
    r"\bbattery\b",
    r"\bmrp\b",
    r"\bprice\b",
    r"\bpricing\b",
    r"\bspecs?\b",
    r"\bdelivery\b",
    r"\bdifference\b",
    r"\bpolic(?:y|ies)\b",
    r"\brefund\b",
    r"\breturn policy\b",
    r"\bprivacy\b",
    r"\bfounded\b",
    r"\bfounder\b",
    r"\bceo\b",
    r"\btroubleshoot",
    r"tell(?:\s+me)?(?:\s+more)?\s+about",
    r"(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain)",
    r"\bexplain\b",
    r"\bdetails?\b",
    r"\binformation\b",
    r"\bbluetooth\b",
)
INTERROGATIVE_FACT_RE = re.compile(
    r"^(?:please\s+)?(?:also\s+)?(?:and\s+)?"
    r"(?:who|what|where|when|why|which|how|"
    r"is|are|was|were|does|do|did|can|could|would|should)\b",
    re.IGNORECASE,
)


def strip_query_fillers(message: str) -> str:
    """Remove stacked greetings and small talk. Does not invent a new question."""
    text = (message or "").strip()
    if not text:
        return ""
    while True:
        updated = FILLER_PREFIX_RE.sub("", text, count=1).strip()
        updated = updated.lstrip("?!,. ").strip()
        if updated == text:
            return text
        text = updated
    return text


def canonicalize_knowledge_query(message: str) -> str:
    """Keep the knowledge-bearing portion. Conversational wrappers become 'What is …?'."""
    original = (message or "").strip()
    text = strip_query_fillers(original)
    if not text:
        return original
    match = KNOW_ABOUT_RE.match(text)
    if not match:
        return text
    topic = re.sub(r"\s+", " ", (match.group("topic") or "").strip(" ?.!,;:"))
    topic = TOPIC_TRAILING_RE.sub("", topic).strip(" ?.!,;:")
    if len(topic) < 2:
        return text
    return f"What is {topic}?"


def looks_like_knowledge_request(message: str) -> bool:
    text = strip_query_fillers(message or "")
    if not text:
        return False
    if KNOW_ABOUT_RE.match(text):
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in KNOWLEDGE_PATTERNS)


def looks_like_informational_question(message: str) -> bool:
    """Factual or informational questions that must be answered from RAG, not GENERAL."""
    text = strip_query_fillers(message or "")
    if not text:
        return False
    if looks_like_knowledge_request(text):
        return True
    return bool(INTERROGATIVE_FACT_RE.match(text))
