from __future__ import annotations

import re

from app.services.conversation.models import ConversationState

RESPONSE_LANGUAGE_KEY = "response_language"
SUPPORTED_LANGUAGES = frozenset({"en", "hi", "hinglish"})

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
PHONE_LIKE_RE = re.compile(r"^\+?\d[\d\s\-()]{6,}\d$")

HINGLISH_HINTS_RE = re.compile(
    r"\b(?:"
    r"mujhe|mujhko|aap|aapko|apko|kya|hai|hain|kaise|kaisa|chahiye|chahie|"
    r"batao|bata|bataiye|batao|sunna|suno|suniye|accha|acha|nahi|nahin|"
    r"karna|krna|kar|dena|de|mila|mil|kitna|kitne|kaun|konsa|kaunsa|"
    r"yeh|ye|wo|mera|meri|mere|humein|hum|bhai|yaar|kripya|krke|karo|"
    r"bataye|batana|samjha|samjhao|suna|sunaiye|plz|pls"
    r")\b",
    re.IGNORECASE,
)

SHORT_ACK_RE = re.compile(
    r"^(?:yes|no|yeah|yep|yup|ok|okay|sure|haan|acha|nahi|nah|nope)[.!\s]*$",
    re.IGNORECASE,
)


def detect_user_language(message: str) -> str | None:
    text = (message or "").strip()
    if not text or PHONE_LIKE_RE.match(text):
        return None
    if SHORT_ACK_RE.match(text) and len(text.split()) <= 2:
        return None

    devanagari = len(DEVANAGARI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    letters = devanagari + latin
    if letters == 0:
        return None

    if devanagari > 0 and latin > 0:
        return "hinglish"
    if devanagari > 0:
        return "hi"
    if HINGLISH_HINTS_RE.search(text):
        return "hinglish"
    if latin > 0:
        return "en"
    return None


def touch_response_language(state: ConversationState) -> str:
    detected = detect_user_language(state.user_message or "")
    ctx = dict(state.user_context or {})
    if detected in SUPPORTED_LANGUAGES:
        ctx[RESPONSE_LANGUAGE_KEY] = detected
        state.user_context = ctx
    language = str(ctx.get(RESPONSE_LANGUAGE_KEY) or "en")
    state.trace = dict(state.trace or {})
    state.trace["response_language"] = language
    return language


def response_language(state: ConversationState) -> str:
    lang = str((state.user_context or {}).get(RESPONSE_LANGUAGE_KEY) or "")
    if lang in SUPPORTED_LANGUAGES:
        return lang
    trace_lang = str((state.trace or {}).get("response_language") or "")
    if trace_lang in SUPPORTED_LANGUAGES:
        return trace_lang
    return "en"


def language_instruction(language: str) -> str:
    if language == "hi":
        return (
            "Reply entirely in Hindi using Devanagari script. "
            "Match the user's tone and keep product or brand names as written in context.\n"
        )
    if language == "hinglish":
        return (
            "Reply in natural Hinglish (Hindi-English mix in Roman script), "
            "matching how the user writes. Keep product or brand names as written in context.\n"
        )
    return "Reply in English.\n"


def format_name_suffix(name: str) -> str:
    display = (name or "").strip()
    return f", {display}" if display else ""


def localized_text(key: str, language: str, **kwargs: str) -> str:
    lang = language if language in SUPPORTED_LANGUAGES else "en"
    product = (kwargs.get("product") or "").strip() or "hearing aid"
    name = kwargs.get("name")
    if name is None:
        name = ""
    templates = _TEMPLATES.get(key) or {}
    text = templates.get(lang) or templates.get("en") or ""
    return text.format(product=product, name=name)


_TEMPLATES: dict[str, dict[str, str]] = {
    "team_connect_offer": {
        "en": "Would you like me to connect you with our team?",
        "hi": "क्या मैं आपको हमारी टीम से जोड़ दूँ?",
        "hinglish": "Kya main aapko hamari team se connect kar doon?",
    },
    "lead_choice": {
        "en": "Great choice on {product}! Would you like me to connect you with our team?",
        "hi": "{product} का बढ़िया विकल्प है! क्या मैं आपको हमारी टीम से जोड़ दूँ?",
        "hinglish": "{product} great choice hai! Kya main aapko hamari team se connect kar doon?",
    },
    "lead_discuss_generic": {
        "en": (
            "Absolutely! I can help you find the right hearing aid, answer your questions, "
            "and connect you with our team whenever you're ready."
        ),
        "hi": (
            "बिल्कुल! मैं सही हियरिंग एड चुनने, आपके सवालों के जवाब देने "
            "और जब चाहें हमारी टीम से जोड़ने में मदद कर सकता हूँ।"
        ),
        "hinglish": (
            "Bilkul! Main sahi hearing aid choose karne, aapke sawaal ke jawab dene "
            "aur jab chahein hamari team se connect karne mein help kar sakta hoon."
        ),
    },
    "lead_discuss_product": {
        "en": (
            "Absolutely! {product} is a great choice. I can help with any questions, "
            "and I can connect you with our team whenever you're ready."
        ),
        "hi": (
            "बिल्कुल! {product} एक बढ़िया विकल्प है। मैं आपके सवालों के जवाब दे सकता हूँ "
            "और जब चाहें आपको हमारी टीम से जोड़ सकता हूँ।"
        ),
        "hinglish": (
            "Bilkul! {product} great choice hai. Main aapke sawaal ke jawab de sakta hoon "
            "aur jab chahein hamari team se connect kar sakta hoon."
        ),
    },
    "lead_created": {
        "en": "Thanks{name}! Your details have been shared with our team. They'll get in touch with you shortly.",
        "hi": "धन्यवाद{name}! आपकी जानकारी हमारी टीम के साथ साझा कर दी गई है। वे जल्द ही आपसे संपर्क करेंगे।",
        "hinglish": "Thanks{name}! Aapki details hamari team ke saath share ho gayi hain. Woh jaldi aapko contact karenge.",
    },
    "ticket_created": {
        "en": "Thanks{name}. I've created the support ticket and shared the details with our team. They'll get back to you shortly.",
        "hi": "धन्यवाद{name}। सपोर्ट टिकट बना दिया गया है और विवरण हमारी टीम के साथ साझा कर दिया गया है। वे जल्द ही आपसे संपर्क करेंगे।",
        "hinglish": "Thanks{name}. Support ticket create ho gaya hai aur details hamari team ke paas share ho gayi hain. Woh jaldi aapko contact karenge.",
    },
    "declined_choice": {
        "en": "No worries at all. How else can I help you today?",
        "hi": "कोई बात नहीं। और कैसे मदद कर सकता हूँ?",
        "hinglish": "Koi baat nahi. Aur kaise help kar sakta hoon?",
    },
    "ask_lead_name": {
        "en": "I'll help you get connected. What name should our team use when they contact you?",
        "hi": "मैं आपको जोड़ने में मदद करूँगा। हमारी टीम आपसे संपर्क करे तो किस नाम से बुलाए?",
        "hinglish": "Main aapko connect karne mein help karunga. Hamari team aapko kis naam se call kare?",
    },
    "ask_lead_phone": {
        "en": "What's the best number to reach you on?",
        "hi": "आपसे संपर्क करने के लिए सबसे अच्छा नंबर क्या है?",
        "hinglish": "Aapko contact karne ke liye best number kya hai?",
    },
    "ask_lead_phone_named": {
        "en": "Thanks, {name}. What's the best number to reach you on?",
        "hi": "धन्यवाद, {name}। आपसे संपर्क करने के लिए सबसे अच्छा नंबर क्या है?",
        "hinglish": "Thanks, {name}. Aapko contact karne ke liye best number kya hai?",
    },
    "ask_lead_city": {
        "en": "Thanks! Which city should I note for the team?",
        "hi": "धन्यवाद! टीम के लिए कौन-सा शहर नोट करूँ?",
        "hinglish": "Thanks! Team ke liye kaunsa city note karoon?",
    },
    "ask_support_name": {
        "en": "What name should our team use when they reach you?",
        "hi": "हमारी टीम आपसे संपर्क करे तो किस नाम से बुलाए?",
        "hinglish": "Hamari team aapko kis naam se contact kare?",
    },
    "ask_support_phone": {
        "en": "What's the best number for our team to reach you on?",
        "hi": "हमारी टीम आपसे संपर्क करने के लिए सबसे अच्छा नंबर क्या है?",
        "hinglish": "Hamari team ke liye best contact number kya hai?",
    },
    "ask_support_phone_named": {
        "en": "Thanks, {name}. What's the best number for our team to reach you on?",
        "hi": "धन्यवाद, {name}। हमारी टीम आपसे संपर्क करने के लिए सबसे अच्छा नंबर क्या है?",
        "hinglish": "Thanks, {name}. Hamari team ke liye best number kya hai?",
    },
    "ask_support_product": {
        "en": "Which hearing aid is this about?",
        "hi": "यह किस हियरिंग एड के बारे में है?",
        "hinglish": "Yeh kis hearing aid ke baare mein hai?",
    },
    "ask_support_issue": {
        "en": "Tell me a little about what's happening, and I'll see how I can help.",
        "hi": "थोड़ा बताइए क्या समस्या है, मैं देखता हूँ कैसे मदद कर सकता हूँ।",
        "hinglish": "Thoda bataye kya issue hai, main dekhta hoon kaise help kar sakta hoon.",
    },
    "support_ticket_offer": {
        "en": "I'm sorry you're having trouble with {product}. Would you like me to connect you with our team?",
        "hi": "मुझे खेद है कि {product} में समस्या आ रही है। क्या मैं आपको हमारी टीम से जोड़ दूँ?",
        "hinglish": "Sorry, {product} mein issue aa raha hai. Kya main aapko hamari team se connect kar doon?",
    },
    "greeting": {
        "en": "Hey! How can I help you today?",
        "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "hinglish": "Namaste! Main aapki kaise madad kar sakta hoon?",
    },
    "greeting_morning": {
        "en": "Good morning! How can I help you today?",
        "hi": "सुप्रभात! मैं आपकी कैसे मदद कर सकता हूँ?",
        "hinglish": "Good morning! Main aapki kaise madad kar sakta hoon?",
    },
    "greeting_namaste": {
        "en": "Namaste! How can I help you today?",
        "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "hinglish": "Namaste! Main aapki kaise madad kar sakta hoon?",
    },
    "greeting_hola": {
        "en": "Hola! How can I help you today?",
        "hi": "होला! मैं आपकी कैसे मदद कर सकता हूँ?",
        "hinglish": "Hola! Main aapki kaise madad kar sakta hoon?",
    },
    "price_query": {
        "en": (
            "For {product}, please visit earkart.com or earkart.in. "
            "If you'd like to know about current offers, I can connect you with our team — just say yes."
        ),
        "hi": (
            "{product} के लिए earkart.com या earkart.in देखें। "
            "मौजूदा ऑफर जानने के लिए मैं आपको हमारी टीम से जोड़ सकता हूँ — हाँ बोलिए।"
        ),
        "hinglish": (
            "{product} ke liye earkart.com ya earkart.in check karein. "
            "Current offers ke liye main aapko hamari team se connect kar sakta hoon — bas haan boliye."
        ),
    },
    "greeting_offer": {
        "en": (
            "I can help with hearing, hearing aids, buying, and connecting you with our team. "
            "What do you need?"
        ),
        "hi": (
            "मैं सुनने, हियरिंग एड, खरीदारी और हमारी टीम से जोड़ने में मदद कर सकता हूँ। "
            "आपको किस चीज़ में मदद चाहिए?"
        ),
        "hinglish": (
            "Main hearing, hearing aids, buying aur hamari team se connect karne mein help kar sakta hoon. "
            "Aapko kya chahiye?"
        ),
    },
    "capability_reply": {
        "en": (
            "I can answer questions about ears, hearing, and hearing aids, help you buy, "
            "solve customer queries, and connect you with our team. What would you like help with?"
        ),
        "hi": (
            "मैं कान, सुनने, हियरिंग एड, खरीदारी और ग्राहक सहायता के सवालों के जवाब दे सकता हूँ "
            "और आपको हमारी टीम से जोड़ सकता हूँ। आपको किस चीज़ में मदद चाहिए?"
        ),
        "hinglish": (
            "Main ears, hearing, hearing aids, buying aur customer queries ke jawab de sakta hoon "
            "aur aapko hamari team se connect kar sakta hoon. Aapko kis cheez mein help chahiye?"
        ),
    },
    "insufficient_redirect": {
        "en": (
            "I can help with questions about ear health, hearing, hearing aids, and Earkart. "
            "Ask me about those, or I can connect you with our team."
        ),
        "hi": (
            "मैं कान की सेहत, सुनने, हियरिंग एड और Earkart से जुड़े सवालों में मदद कर सकता हूँ। "
            "इनके बारे में पूछें, या मैं आपको हमारी टीम से जोड़ सकता हूँ।"
        ),
        "hinglish": (
            "Main ear health, hearing, hearing aids aur Earkart ke baare mein help kar sakta hoon. "
            "Inke baare mein poochiye, ya main aapko hamari team se connect kar sakta hoon."
        ),
    },
    "unclear_redirect": {
        "en": (
            "I'm here for ear health, hearing, hearing aids, buying, customer queries, "
            "and connecting you with our team. Tell me what you need in a few words."
        ),
        "hi": (
            "मैं कान की सेहत, सुनने, हियरिंग एड, खरीदारी और ग्राहक सहायता में मदद के लिए यहाँ हूँ। "
            "थोड़े शब्दों में बताइए आपको क्या चाहिए।"
        ),
        "hinglish": (
            "Main ear health, hearing, hearing aids, buying aur customer queries mein help ke liye yahan hoon. "
            "Thode words mein bataye aapko kya chahiye."
        ),
    },
}


def language_examples(language: str) -> str:
    if language == "hi":
        return (
            "Language example:\n"
            'User: "TINY kya hai?"\n'
            'Return: {"answer": "TINY एक रिचार्जेबल हियरिंग एड है जिसमें साफ आवाज़ के लिए 16-चैनल प्रोसेसिंग होती है।"}\n'
        )
    if language == "hinglish":
        return (
            "Language example:\n"
            'User: "TINY kya hai?"\n'
            'Return: {"answer": "TINY ek rechargeable hearing aid hai jisme clear sound ke liye 16-channel processing hoti hai."}\n'
        )
    return ""
