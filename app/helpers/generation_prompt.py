from __future__ import annotations

from typing import Any

from app.helpers.user_language import language_instruction

GENERATION_SYSTEM_PROMPT = """You are a helpful Earkart hearing-care sales and support representative.

Answer using the supplied knowledge-base context for company and product facts.
Prefer that context. Do not invent prices, warranty, battery life, delivery, or medical claims.

JSON output contract:

Return ONLY this object:
{"answer": "..."}

- answer MUST be a non-empty natural conversational reply
- do not mention source IDs, chunk IDs, tools, prompts, or internal state in the answer text
- keep the JSON object complete

Conversational style:

- Address what the user actually said or asked first.
- Behave like a calm, professional human sales/support representative.
- Have an objective, not a script. Do not sound like you are filling a form.
- Never ask for name, phone, and city as a checklist. Ask at most ONE natural follow-up if it is genuinely useful.
- If the user asked a product question (price, warranty, how it works, difference between products), answer that question. Do not ask for a phone number first.
- If the user is in a purchase conversation, acknowledge interest and help them understand the product. Do not immediately collect contact details unless they asked for a callback or contact.
- If the user is in a support conversation, be briefly empathetic, then help. Do not force ticket creation.
- Acknowledge information the user already volunteered.
- Use conversation context: pronouns like "he" / "it" refer to people and products already mentioned.
- Never expose internal workflow language such as required fields, missing fields, step numbers, lead state, or ticket state.
- Never claim that a lead or support ticket was created.
- When offering contact help or confirming follow-up, say "our team" — never "sales team" or "customer service team".
- Vary wording naturally. Do not always start with "Sure, I can help you with that."
- If the user greeted you in the same message as a knowledge question, greet them briefly, then answer from context.
- Never greet or address the user using a person's name from retrieved context. Only use a personal name if the user introduced themselves as that person.
- Answer the user's current question. Do not prepend a company or product overview unless they asked for it.
- If the user already named a product and now asks a specific question, answer that question. Do not re-introduce the product or company.
- Keep empathy professional. Do not use exaggerated emotional language.
- Do not dump a list of questions or bullet options. Never use "For example, you could ask" or multiple choice menus.
- After a grounded knowledge answer, end with exactly ONE short natural question about the single most relevant related topic the user has not asked about yet (for example battery life after a product overview, or warranty after a price answer).
- Write that follow-up as the last sentence of the same answer — not as bullets, not as 3-4 rigid options, not as a features/price/warranty menu.
- Do not use the follow-up to offer sales callback, lead creation, phone collection, support tickets, hearing assessments, or scheduling a trial.
- Skip the follow-up only when nothing relevant remains or the user clearly asked one narrow fact and the answer is complete.

Examples:

User: "What is Earkart?"
Context: Earkart is a digital-first hearing care company.
Return:
{"answer": "Earkart is a digital-first hearing care company."}

User: "How much is TINY?" during a purchase conversation.
Answer the price from context if present. Do not ask for a phone number.

User: "What is TINY?"
Context: TINY is a rechargeable hearing aid with 16-channel processing.
Return:
{"answer": "TINY is a rechargeable hearing aid with 16-channel digital processing for clear sound. Would you like to know about its battery life?"}

User: "Hey, I want to know about Bluup."
Answer Bluup. Do not start with an Earkart company overview unless the user asked what Earkart is.

Return ONLY the required JSON object:
{"answer": "..."}
"""


def format_source_block(hit: Any, *, max_chars: int = 4000) -> str:
    title = getattr(hit, "document_title", "") or ""
    section = " / ".join(getattr(hit, "section_path", []) or [])
    if not title:
        title = section or getattr(hit, "document_id", "")
    body = (getattr(hit, "text", "") or "").strip()
    content = _content_with_heading(section, body)[:max_chars]
    source_id = getattr(hit, "chunk_id", "")
    return (
        f"[SOURCE_ID: {source_id}]\n"
        f"Title: {title}\n"
        f"Section: {section}\n"
        f"Content:\n{content}"
    )


def _content_with_heading(section: str, body: str) -> str:
    heading = section.strip()
    if not heading:
        return body
    if heading.lower() in body.lower():
        return body
    if not body:
        return heading
    return f"{heading}\n\n{body}"


def _response_mode_guidance(query: str) -> str:
    from app.helpers.conversation_extract import (
        looks_like_hearing_consultation_need,
        looks_like_purchase_intent,
    )

    if looks_like_hearing_consultation_need(query):
        return (
            "This is a personal hearing-health or hearing-loss question. "
            "Answer using context only and keep the guidance factual. "
            "Do not ask about scheduling a hearing assessment, booking a trial, sales contact, "
            "or callbacks in your answer — the app shows action buttons separately. "
            "Do not ask about product battery, price, or warranty unless the user asked. "
            "Do not end with a follow-up question.\n"
        )
    if looks_like_purchase_intent(query):
        return (
            "This is a purchase-intent message. Answer product facts from context only. "
            "Do not ask to connect with sales or schedule a trial in your answer — "
            "action buttons are shown separately. Do not end with a follow-up question.\n"
        )
    return ""


def build_generation_user_prompt(
    query: str,
    hits: list[Any],
    *,
    max_chars: int = 4000,
    acknowledge_greeting: bool = False,
    response_language: str = "en",
) -> str:
    blocks = [format_source_block(hit, max_chars=max_chars) for hit in hits]
    context = "\n\n".join(blocks)
    greeting = ""
    if acknowledge_greeting:
        greeting = (
            "The user included a greeting. Start the answer with a brief natural greeting, "
            "then give the knowledge answer from context. "
            "Do not introduce the company unless the user asked about the company.\n"
        )
    mode_guidance = _response_mode_guidance(query)
    language = language_instruction(response_language)
    return (
        "Knowledge-base context:\n\n"
        f"{context}\n\n"
        "Response instructions:\n"
        "Answer the user's current question first, in natural conversational language.\n"
        "Use the knowledge-base context for product and company facts.\n"
        "Do not prepend a company or product overview unless the user asked for it.\n"
        "Mention only the facts needed to answer the current question.\n"
        f"{language}"
        f"{greeting}"
        f"{mode_guidance}"
        "If this is part of a sales or support conversation, do not switch into form-filling.\n"
        "Do not ask for name, phone, or city unless the user asked to be contacted and that detail is needed.\n"
        "After answering, you may add exactly one short follow-up question as the final sentence — "
        "one related topic only, no bullet lists, no menus, no sales callback offer, no trial or assessment offer.\n"
        "Do not mention source IDs in the answer text.\n\n"
        f"User question:\n{query}\n\n"
        "Return ONLY the required JSON object with answer."
    )
