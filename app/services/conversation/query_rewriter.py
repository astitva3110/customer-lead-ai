from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from app.helpers.conversation_turn import wants_more_product_info
from app.helpers.query_normalize import canonicalize_knowledge_query
from app.services.conversation.models import ConversationGoal, ConversationState

PRONOUN_RE = re.compile(r"\b(it|its|this|that|the product)\b", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(
    r"\b(?:price|mrp|cost|warranty|battery|features?|specs?|channels?|how much)\b",
    re.IGNORECASE,
)
COMPANY_RE = re.compile(r"\bearkart\b|\bcompany\b|\bclinic\b", re.IGNORECASE)
CATALOG_SCOPE_RE = re.compile(
    r"\b(?:all|every|each)\s+products?\b"
    r"|\bproducts?\s+(?:that\s+)?(?:you|u)\s+have\b"
    r"|\bthat\s+(?:you|u)\s+have\b"
    r"|\b(?:catalog|pricelist|price\s*list)\b"
    r"|\bavailable\s+products?\b",
    re.IGNORECASE,
)
BLUUP_PLUS_RE = re.compile(r"\bbluup\s*\+", re.IGNORECASE)
PHONE_LIKE_RE = re.compile(r"^\+?\d[\d\s\-()]{6,}\d$")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

KNOWN_PRODUCTS = (
    "Radius M16",
    "Radius 16",
    "Radius 8",
    "Radius",
    "TINY",
    "FAME SP",
    "FAME P",
    "FAME 2T",
    "FAME",
    "FORT ULTRA POWER",
    "FORT",
    "OMNI",
    "Bluup+",
    "Bluup",
)

PRODUCT_ALIASES = {
    "tinny": "TINY",
    "tiney": "TINY",
    "tiiny": "TINY",
    "tinyy": "TINY",
    "blup": "Bluup",
    "bluep": "Bluup",
    "bluupp": "Bluup",
    "bluuppp": "Bluup",
    "omny": "OMNI",
}

SPELLING_FIXES = {
    "prodcut": "product",
    "warrenty": "warranty",
    "waranty": "warranty",
    "warrantly": "warranty",
    "batery": "battery",
    "battary": "battery",
    "featurs": "features",
    "infomation": "information",
    "informaton": "information",
    "whant": "want",
}

INFORMAL_FIXES = {
    "knw": "know",
    "abt": "about",
    "wanna": "want to",
    "gonna": "going to",
}

REWRITE_CONFIDENCE_FLOOR = 0.85


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def changed(self) -> bool:
        return (self.rewritten_query or "").strip() != (self.original_query or "").strip()


def routing_query(state: ConversationState) -> str:
    return (state.query_rewritten or state.user_message or "").strip()


def extract_product(message: str) -> str:
    folded = _fold_bluup_plus(message or "")
    lowered = folded.lower()
    for name in KNOWN_PRODUCTS:
        if name.lower() in lowered:
            return name
    return ""


def apply_named_product(state: ConversationState) -> None:
    message = routing_query(state) or (state.user_message or "")
    if re.search(r"\bdifference\b|\bcompare\b|\bvs\.?\b", message, flags=re.IGNORECASE):
        return
    named = extract_product(message)
    if not named:
        return
    if state.lead_collection_active and state.awaiting_field:
        return
    explicit_switch = bool(
        re.search(
            r"\b(?:actually|instead|changed my mind|rather|i want|i'll buy|i will buy|buy|purchase|better|mean)\b",
            message,
            flags=re.IGNORECASE,
        )
    )
    if not state.product or explicit_switch:
        state.product = named
    elif state.conversation_goal == ConversationGoal.SUPPORT and named:
        state.product = named


def needs_rewrite(message: str, product: str) -> bool:
    if not product:
        return False
    if is_catalog_scope_query(message):
        return False
    if not PRONOUN_RE.search(message or ""):
        return False
    return product.lower() not in (message or "").lower()


def rewrite_query(message: str, product: str) -> str:
    text = message.strip()
    text = re.sub(
        rf"\bits\s+(.+?)(\?|$)",
        rf"the \1 of {product}\2",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bthe product\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bit\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthat\b", product, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


class QueryRewriter:
    def normalize(self, state: ConversationState) -> RewriteResult:
        original = (state.user_message or "").strip()
        result = normalize_for_routing(original, product=state.product or "")
        state.trace = dict(state.trace or {})
        state.trace["query_rewrite"] = result.to_dict()
        if result.changed and result.confidence >= REWRITE_CONFIDENCE_FLOOR:
            state.query_rewritten = result.rewritten_query
            if result.entities and not state.product:
                state.product = result.entities[0]
        else:
            state.query_rewritten = ""
            result = RewriteResult(
                original_query=original,
                rewritten_query=original,
                confidence=1.0 if not result.changed else result.confidence,
                entities=result.entities,
            )
            state.trace["query_rewrite"] = result.to_dict()
        return result

    def apply(self, state: ConversationState) -> ConversationState:
        original = (state.user_message or "").strip()
        source = (state.query_rewritten or original).strip()
        trace = state.trace or {}
        catalog_scope = is_catalog_scope_query(original)
        resolved = "" if catalog_scope else str(trace.get("resolved_query") or "").strip()
        if resolved and resolved.lower() not in {original.lower(), source.lower()}:
            state.query_rewritten = resolved
            return state
        if resolved and resolved.lower() != original.lower():
            state.query_rewritten = resolved
            return state
        sub_questions = [] if catalog_scope else (trace.get("sub_questions") or [])
        if sub_questions:
            first = str(sub_questions[0]).strip()
            if first and first.lower() != original.lower():
                state.query_rewritten = first
                return state
        prepared = confirmation_knowledge_query(state) or canonicalize_knowledge_query(source)
        if needs_rewrite(prepared, state.product):
            prepared = rewrite_query(prepared, state.product)
        elif should_bind_product(prepared, state.product):
            prepared = bind_product_query(prepared, state.product)
        if prepared and prepared != original:
            state.query_rewritten = prepared
        elif source != original:
            state.query_rewritten = source
        else:
            state.query_rewritten = ""
        return state


def normalize_for_routing(message: str, *, product: str = "") -> RewriteResult:
    original = (message or "").strip()
    if not original or PHONE_LIKE_RE.match(original):
        return RewriteResult(original_query=original, rewritten_query=original, confidence=1.0)
    text = _fold_bluup_plus(original)
    confidence = 1.0
    entities: list[str] = []

    informal, informal_conf = _replace_map(text, INFORMAL_FIXES)
    if informal != text:
        text = informal
        confidence = min(confidence, informal_conf)

    spelling, spelling_conf = _replace_map(text, SPELLING_FIXES)
    if spelling != text:
        text = spelling
        confidence = min(confidence, spelling_conf)

    aliased, alias_conf, found = _replace_product_aliases(text)
    if aliased != text:
        text = aliased
        confidence = min(confidence, alias_conf)
        entities.extend(found)

    named = extract_product(text)
    if named:
        text = _restore_product_casing(text, named)
        if named not in entities:
            entities.append(named)

    if product and needs_rewrite(text, product):
        text = rewrite_query(text, product)
        confidence = min(confidence, 0.92)
        if product not in entities:
            entities.append(product)
    elif product and should_bind_product(text, product):
        text = bind_product_query(text, product)
        confidence = min(confidence, 0.92)
        if product not in entities:
            entities.append(product)

    text = _possessive_attribute(text)
    text = _light_sentence_case(text, original)
    text = re.sub(r"\s+", " ", text).strip()
    if text == original:
        return RewriteResult(original_query=original, rewritten_query=original, confidence=1.0, entities=entities)
    return RewriteResult(
        original_query=original,
        rewritten_query=text,
        confidence=confidence,
        entities=entities,
    )


def confirmation_knowledge_query(state: ConversationState) -> str:
    product = state.product or ""
    if product and wants_more_product_info(state.user_message or ""):
        return f"What is {product}?"
    return ""


def is_catalog_scope_query(message: str) -> bool:
    return bool(CATALOG_SCOPE_RE.search(message or ""))


def _fold_bluup_plus(text: str) -> str:
    return BLUUP_PLUS_RE.sub("Bluup+", text)


def should_bind_product(message: str, product: str) -> bool:
    if not product or not message:
        return False
    if is_catalog_scope_query(message):
        return False
    if product.lower() in message.lower():
        return False
    if COMPANY_RE.search(message):
        return False
    named = extract_product(message)
    if named and named.lower() != product.lower():
        return False
    return bool(ATTRIBUTE_RE.search(message))


def bind_product_query(message: str, product: str) -> str:
    text = message.rstrip(" ?.!")
    return f"{text} of {product}?"


def _replace_map(text: str, mapping: dict[str, str]) -> tuple[str, float]:
    updated = text
    changed = False
    for source, target in mapping.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        if pattern.search(updated):
            updated = pattern.sub(target, updated)
            changed = True
    return updated, (0.95 if changed else 1.0)


def _replace_product_aliases(text: str) -> tuple[str, float, list[str]]:
    entities: list[str] = []
    updated = text
    confidence = 1.0

    def _alias(match: re.Match[str]) -> str:
        nonlocal confidence
        token = match.group(0)
        key = token.lower()
        if key in PRODUCT_ALIASES:
            name = PRODUCT_ALIASES[key]
            if name not in entities:
                entities.append(name)
            confidence = min(confidence, 0.96)
            return name
        fuzzy = _fuzzy_product(token)
        if fuzzy:
            if fuzzy not in entities:
                entities.append(fuzzy)
            confidence = min(confidence, 0.9)
            return fuzzy
        return token

    updated = TOKEN_RE.sub(_alias, updated)
    return updated, confidence, entities


def _fuzzy_product(token: str) -> str:
    if len(token) < 4:
        return ""
    lowered = token.lower()
    if lowered in {item.lower() for item in KNOWN_PRODUCTS}:
        return ""
    best_name = ""
    best_ratio = 0.0
    for name in KNOWN_PRODUCTS:
        candidate = name.split()[0]
        if len(candidate) < 4:
            continue
        ratio = SequenceMatcher(None, lowered, candidate.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name if " " not in name else candidate
    if best_ratio >= 0.86:
        return best_name
    return ""


def _restore_product_casing(text: str, product: str) -> str:
    return re.sub(rf"\b{re.escape(product)}\b", product, text, flags=re.IGNORECASE)


def _possessive_attribute(text: str) -> str:
    products = "|".join(re.escape(name) for name in KNOWN_PRODUCTS)
    pattern = re.compile(
        rf"\b(?P<pre>what is)\s+(?P<product>{products})\s+(?P<attr>warranty|price|battery|mrp|cost)\b(?P<mark>\??)",
        re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        product = extract_product(match.group("product")) or match.group("product")
        attr = match.group("attr").lower()
        mark = match.group("mark") or "?"
        return f"What is {product}'s {attr}{mark}"

    return pattern.sub(_replace, text)


def _light_sentence_case(text: str, original: str) -> str:
    if not text:
        return text
    if text == original:
        return text
    first = text[0]
    if first.isalpha() and first.islower():
        return first.upper() + text[1:]
    return text
