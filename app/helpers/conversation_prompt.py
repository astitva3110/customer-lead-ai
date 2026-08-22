from __future__ import annotations

import json
import re

from app.services.conversation.models import ConversationGoal, ConversationState, TurnIntent
from app.helpers.conversation_reply import conversational_fallback
from app.helpers.conversation_turn import is_greeting_only, last_assistant_text

PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

CONVERSATION_SYSTEM_PROMPT = """You are a warm, concise Earkart hearing-care representative.
Respond like a human sales/support specialist helping a person, not like a form or state machine.

Before every reply, consider the user's latest message and the recent conversation.
Respond to what they are actually saying now.

Rules:
- Acknowledge the current message first. Answer it before asking anything.
- Use conversation history for short replies such as yes/no.
- Do not repeat your previous assistant message.
- If the user only greeted you, greet them back briefly and offer to help. Do not introduce the company unless they asked about it.
- If they greeted you and asked a question, greet them briefly, then answer that question. Do not start with a company overview.
- Greet only when the user greeted you on this turn. Do not greet every turn.
- If they volunteered a name, acknowledge it once naturally (for example "Nice to meet you, Rahul!"). Do not ask for the name again.
- Do not use the user's name in every reply. Use it only when it feels natural.
- Do not ask for information already known (name, city, product, issue, phone).
- Do not collect phone, name, or city unless the situation says the user asked to be contacted and a detail is missing.
- If the situation says the user just said they want to buy, acknowledge the purchase intent naturally and make sales assistance available without forcing a choice. Do not ask for phone, name, or city yet. Do not list features, price, and warranty as a menu.
- After the user chooses a direction or asks a question, follow that. Do not repeat the learning-or-sales-team offer.
- If they volunteered a name, city, or other detail, acknowledge it naturally. Do not ask for it again.
- If the user asked a question, change products, changed their mind, or said something unrelated, deal with that. Do not pull them back to a checklist.
- If the user described a problem, be briefly empathetic and help. Do not start a name/phone questionnaire unless they asked for a ticket.
- Never say or imply that the user is progressing through a form, steps, or remaining fields.
- Never mention internal labels such as LEAD, SUPPORT, awaiting_phone, awaiting_city, workflow, or missing fields.
- Never claim a lead or support ticket was created unless a successful tool result is supplied.
- If a successful contact or ticket result is supplied, thank them and close. Do not ask whether they need anything else.
- Never invent company or product facts (price, warranty, specs, battery, availability, delivery, medical claims). If no knowledge context is supplied, do not state those facts.
- Do not always end with a question. Often acknowledge, answer, or offer a next step. A complete sentence with no follow-up is fine.
- Do not use repetitive closers such as "Would you like to know more?", "Do you need any other help?", or "How can I assist you further?" unless they are genuinely useful.
- Keep replies short, professional, and varied.

Return JSON only: {"answer": "<reply>"}
"""


def redact_phone(text: str) -> str:
    return PHONE_RE.sub("[phone]", text or "")


def _situation(state: ConversationState) -> str:
    trace = state.trace or {}
    capabilities = trace.get("capabilities") or []
    collecting_lead = "LEAD_INFORMATION_COLLECTION" in capabilities or "CREATE_LEAD" in capabilities
    collecting_support = (
        "SUPPORT_INFORMATION_COLLECTION" in capabilities or "CREATE_SUPPORT_TICKET" in capabilities
    )
    if collecting_lead:
        missing = _missing_contact_labels(state)
        if missing:
            return (
                "The user asked to be contacted. Ask naturally only for their "
                f"{missing[0]}. Do not mention other details or remaining steps."
            )
        return "The user asked to be contacted. You already have the details you need."
    if collecting_support:
        missing = _missing_support_labels(state)
        if missing:
            return (
                "The user asked for a support ticket. Ask only for their "
                f"{missing[0]}. Do not mention other details or remaining steps."
            )
        return "The user asked for a support ticket. You already have the details you need."
    if is_greeting_only(state.user_message or ""):
        if state.product or state.conversation_goal in {
            ConversationGoal.LEAD,
            ConversationGoal.SALES,
            ConversationGoal.SUPPORT,
        }:
            return (
                "The user greeted you again. Greet them back briefly. "
                "Do not restart the conversation or describe the company."
            )
        return (
            "The user greeted you. Greet them back briefly and offer to help. "
            "Do not describe the company unless they asked about it."
        )
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}:
        product = state.product or "a hearing aid"
        if state.current_turn_intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES}:
            return (
                f"The user just said they want to buy {product}. "
                "Acknowledge that purchase intent naturally and keep sales assistance available without forcing a choice. "
                "Do not ask for phone, name, or city. Do not list features, price, and warranty as a menu."
            )
        if state.current_turn_intent == TurnIntent.CONTEXT_UPDATE:
            return (
                "The user shared personal information. Acknowledge it naturally. "
                "If they gave a name, greet them by name. Do not request contact details."
            )
        return (
            f"The user is considering a purchase of {product}. "
            "Respond only to this message. Do not repeat an offer to learn more or connect with sales. "
            "Do not request contact details."
        )
    if state.conversation_goal == ConversationGoal.SUPPORT:
        if state.current_turn_intent == TurnIntent.SUPPORT_INTENT:
            return (
                "The user described a product issue. Be briefly empathetic and help with what they said. "
                "Do not start a ticket questionnaire unless they asked for a ticket."
            )
        return (
            "The user has a product issue. Help with this message only. "
            "Do not repeat an earlier offer to open a ticket or collect details."
        )
    return "Respond to the current message."


def _missing_contact_labels(state: ConversationState) -> list[str]:
    missing: list[str] = []
    if not state.phone:
        missing.append("phone number")
    if not state.user_name:
        missing.append("name")
    if not state.city:
        missing.append("city")
    return missing


def _missing_support_labels(state: ConversationState) -> list[str]:
    missing: list[str] = []
    if not state.user_name:
        missing.append("name")
    if not state.product:
        missing.append("product")
    if not state.phone:
        missing.append("phone number")
    if not state.support_issue:
        missing.append("issue")
    return missing


def build_conversation_user_prompt(state: ConversationState) -> str:
    history = []
    for item in (state.conversation_history or [])[-6:]:
        history.append(
            {
                "role": item.get("role") or "",
                "content": redact_phone((item.get("content") or "")[:400]),
            }
        )
    tool = (state.trace or {}).get("tool_called") or ""
    tool_result = ""
    if tool == "create_lead" and not state.error:
        tool_result = "contact_request_recorded"
    elif tool == "create_support_ticket" and not state.error:
        tool_result = "support_request_recorded"
    elif state.error in {"lead_create_failed", "ticket_create_failed"}:
        tool_result = "request_failed"
    payload = {
        "situation": _situation(state),
        "product": state.product or "",
        "name_known": bool(state.user_name),
        "phone_known": bool(state.phone),
        "city_known": bool(state.city),
        "issue_known": bool(state.support_issue),
        "user_context": state.user_context or {},
        "tool_result": tool_result,
        "last_assistant": redact_phone(last_assistant_text(state))[:400],
        "recent_history": history,
        "current_message": redact_phone(state.user_message or ""),
    }
    if state.user_name:
        payload["name"] = state.user_name
    if not payload["tool_result"]:
        payload.pop("tool_result")
    if not payload["user_context"]:
        payload.pop("user_context")
    return json.dumps(payload, ensure_ascii=False)


def parse_conversation_answer(raw: str, state: ConversationState) -> str:
    from app.helpers.generation_json import extract_json_object

    payload = extract_json_object(raw) or {}
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return conversational_fallback(state)
    return answer.strip()
