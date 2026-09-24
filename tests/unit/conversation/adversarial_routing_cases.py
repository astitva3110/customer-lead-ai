"""Golden adversarial routing cases for Earkart chatbot.

Each case uses trap words that can mislead regex/phrase routing (problem, hearing,
listening, aid, issue, call, etc.) while the true user goal belongs to a
different route.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.conversation.models import ChatMode


@dataclass(frozen=True)
class AdversarialRoutingCase:
    id: str
    message: str
    expected_mode: ChatMode
    trap_words: tuple[str, ...]
    why_tricky: str
    wrong_regex_guess: ChatMode
    """What naive keyword/regex routing often picks."""
    phrase_router_known_gap: bool = False
    """True when phrase router (no LLM) is known to misroute this case today."""
    semantic_intent: str = ""
    semantic_action: str = ""
    expected_turn_intent: str = ""
    """When set, also assert current_turn_intent (e.g. GENERAL steering)."""
    expect_retrieve: bool | None = None
    """When set, assert trace['should_retrieve']."""
    expect_semantic_used: bool = True
    """False when phrase router or skip rules handle the case without LLM."""


def _case(
    case_id: str,
    message: str,
    expected: ChatMode,
    trap_words: tuple[str, ...],
    why_tricky: str,
    wrong_guess: ChatMode,
    *,
    phrase_gap: bool = False,
    intent: str = "",
    action: str = "",
) -> AdversarialRoutingCase:
    intent_map = {
        ChatMode.LEAD: ("lead", "start_lead"),
        ChatMode.KNOWLEDGE: ("knowledge", "answer_knowledge"),
        ChatMode.SUPPORT: ("support", "start_support"),
        ChatMode.REJECTED: ("other", "reject"),
    }
    default_intent, default_action = intent_map.get(expected, ("conversation", "conversation"))
    return AdversarialRoutingCase(
        id=case_id,
        message=message,
        expected_mode=expected,
        trap_words=trap_words,
        why_tricky=why_tricky,
        wrong_regex_guess=wrong_guess,
        phrase_router_known_gap=phrase_gap,
        semantic_intent=intent or default_intent,
        semantic_action=action or default_action,
    )


ADVERSARIAL_ROUTING_CASES: tuple[AdversarialRoutingCase, ...] = (
    _case(
        "listening_call_problem",
        "i have a problem of listening to the call",
        ChatMode.KNOWLEDGE,
        ("problem", "listening", "call"),
        "Contains 'problem' and 'listening' like a device fault, but user describes "
        "difficulty hearing phone calls — a hearing-health consultation, not support.",
        ChatMode.SUPPORT,
    ),
    AdversarialRoutingCase(
        id="cannot_hear_phone_calls",
        message="i cannot hear properly on phone calls anymore",
        expected_mode=ChatMode.KNOWLEDGE,
        trap_words=("cannot", "hear", "phone", "calls"),
        why_tricky="Sounds like a phone/device issue; user is describing personal hearing difficulty.",
        wrong_regex_guess=ChatMode.SUPPORT,
        semantic_intent="knowledge",
        semantic_action="answer_knowledge",
        expect_semantic_used=False,
    ),
    _case(
        "hearing_loss_consultation",
        "i have hearing loss of 45% which hearing aid should i need",
        ChatMode.KNOWLEDGE,
        ("hearing", "loss", "hearing aid"),
        "Mentions hearing aid but asks which product fits — product guidance, not purchase.",
        ChatMode.LEAD,
    ),
    _case(
        "hinglish_acquisition_typo",
        "hi mujhe heaing aid chaiye thi",
        ChatMode.LEAD,
        ("hi", "heaing", "aid", "chaiye"),
        "Typo + greeting prefix; regex may miss acquisition or treat as casual chat.",
        ChatMode.KNOWLEDGE,
    ),
    _case(
        "hinglish_acquisition_clean",
        "mujhe hearing aid chaiye",
        ChatMode.LEAD,
        ("hearing", "aid", "chaiye"),
        "Sounds like a hearing health problem; user wants to obtain a device.",
        ChatMode.SUPPORT,
    ),
    _case(
        "hindi_acquisition",
        "मुझे हियरिंग एड चाहिए",
        ChatMode.LEAD,
        ("हियरिंग", "एड", "चाहिए"),
        "Devanagari purchase intent; Latin-script regex will not match.",
        ChatMode.KNOWLEDGE,
    ),
    _case(
        "device_no_sound_english",
        "i have a problem with my hearing aid, no sound",
        ChatMode.SUPPORT,
        ("problem", "hearing aid", "no sound"),
        "'problem' suggests health concern; 'no sound' is a device fault on owned hardware.",
        ChatMode.KNOWLEDGE,
        phrase_gap=True,
    ),
    _case(
        "device_issue_hinglish",
        "mera hearing aid ka issue hai awaz nahi aa rahi",
        ChatMode.SUPPORT,
        ("issue", "hearing aid", "awaz", "nahi"),
        "Hinglish device fault without English 'not working' phrase — regex may miss support.",
        ChatMode.KNOWLEDGE,
        phrase_gap=True,
    ),
    _case(
        "device_stopped_working",
        "my TINY stopped working yesterday",
        ChatMode.SUPPORT,
        ("TINY", "stopped working"),
        "Product name plus fault; should not become a product-info question.",
        ChatMode.KNOWLEDGE,
    ),
    _case(
        "battery_problem_info_question",
        "what is the problem with TINY battery life",
        ChatMode.KNOWLEDGE,
        ("problem", "TINY", "battery"),
        "Asks about battery using the word 'problem' — informational, not a ticket.",
        ChatMode.SUPPORT,
    ),
    _case(
        "buy_because_old_broke",
        "i want to buy a new hearing aid because my old one broke",
        ChatMode.LEAD,
        ("buy", "broke", "hearing aid"),
        "Mentions a broken device but primary goal is purchasing a new one.",
        ChatMode.SUPPORT,
    ),
    _case(
        "connect_sales_hearing_problem",
        "connect me to sales, i have hearing problem",
        ChatMode.LEAD,
        ("connect", "sales", "hearing", "problem"),
        "Contains hearing-health wording but explicit sales connection request.",
        ChatMode.KNOWLEDGE,
    ),
    _case(
        "hearing_aid_problems_info",
        "tell me what problems hearing aids solve",
        ChatMode.KNOWLEDGE,
        ("problems", "hearing aids"),
        "Educational question about hearing aids — not reporting a device fault.",
        ChatMode.SUPPORT,
    ),
    AdversarialRoutingCase(
        id="sunai_problem_hindi",
        message="mujhe sunai problem hai",
        expected_mode=ChatMode.KNOWLEDGE,
        trap_words=("sunai", "problem"),
        why_tricky="Personal hearing difficulty in Hindi — consultation, not device support.",
        wrong_regex_guess=ChatMode.SUPPORT,
        semantic_intent="knowledge",
        semantic_action="answer_knowledge",
        expect_semantic_used=False,
    ),
    _case(
        "explicit_support_ticket",
        "open a support ticket, my hearing aid is broken",
        ChatMode.SUPPORT,
        ("support", "ticket", "broken", "hearing aid"),
        "Explicit ticket language must stay support even with product mention.",
        ChatMode.KNOWLEDGE,
    ),
    AdversarialRoutingCase(
        id="capability_question",
        message="kya kar sakte ho",
        expected_mode=ChatMode.KNOWLEDGE,
        trap_words=("kya", "kar"),
        why_tricky="Capability/steering question — not knowledge retrieval.",
        wrong_regex_guess=ChatMode.KNOWLEDGE,
        semantic_intent="conversation",
        semantic_action="conversation",
        expected_turn_intent="GENERAL",
        expect_retrieve=False,
        expect_semantic_used=False,
    ),
    _case(
        "mixed_buy_and_warranty",
        "i want to buy TINY, what is its warranty?",
        ChatMode.KNOWLEDGE,
        ("buy", "warranty", "TINY"),
        "Mixed buy + factual question — answer knowledge first, keep sales interest.",
        ChatMode.LEAD,
        intent="knowledge",
        action="answer_knowledge",
    ),
)
