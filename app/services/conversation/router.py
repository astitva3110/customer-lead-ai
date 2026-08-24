from __future__ import annotations

import re

from app.services.conversation.models import (
    ChatMode,
    ConversationGoal,
    ConversationState,
    LeadStage,
    LeadStatus,
    LeadWorkflow,
    TicketStatus,
    TurnIntent,
)
from app.services.conversation.query_rewriter import extract_product
from app.interfaces.providers.llm import LLMProvider
from app.helpers.conversation_extract import (
    extract_city,
    extract_issue,
    extract_name,
    extract_phone_from_text,
    extract_user_context,
    looks_like_contact_request,
    looks_like_ticket_request,
    merge_user_context,
)
from app.helpers.phone import apply_phone_to_state, default_phone_region, extract_and_validate_phone
from app.helpers.conversation_turn import (
    is_acknowledgement_only,
    is_greeting_only,
    is_phone_refusal,
    is_short_no,
    is_short_yes,
    last_assistant_text,
    offered_callback,
    offered_product_information,
    strip_greeting_prefix,
    wants_more_product_info,
    is_tell_more,
)
from app.helpers.turn_understanding import (
    TURN_UNDERSTANDING_SYSTEM,
    TurnUnderstanding,
    parse_turn_understanding,
)
from app.helpers.conversation_history import compact_recent_history
from app.helpers.state_manager import (
    STATE_MANAGER_SYSTEM,
    apply_state_manager_result,
    build_state_manager_prompt,
    conversation_status_label,
    current_status_label,
    parse_state_manager_result,
    populate_deterministic_state_trace,
    turn_intent_from_status,
)
from app.helpers.query_normalize import looks_like_knowledge_request
from app.services.diagnostics.recorder import current_trace, record_turn_understanding

LEAD_PATTERNS = (
    r"contact me",
    r"call me",
    r"callback",
    r"get in touch",
    r"arrange (?:a )?(?:callback|purchase|order)",
    r"\bquote\b",
    r"\bsales\b",
)
PURCHASE_PATTERNS = (
    r"want to buy",
    r"want to purchase",
    r"want to order",
    r"like to (?:buy|purchase|order)",
    r"interested in buying",
    r"can i buy",
    r"can i purchase",
    r"how (?:can|do) i (?:buy|purchase|order)",
    r"want this one",
    r"want to get",
    r"i'll buy",
    r"i will buy",
    r"let'?s buy",
    r"\border\b",
)
SUPPORT_PATTERNS = (
    r"not working",
    r"stopped working",
    r"isn't working",
    r"isnt working",
    r"\bbroken\b",
    r"\brepair\b",
    r"\bticket\b",
    r"\bsupport\b",
    r"hearing aid is not",
    r"hearing aid isn't",
    r"low sound",
    r"low volume",
)
PHONE_LIKE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")
WANT_PRODUCT_RE = re.compile(
    r"\b(?:i want|i'd like|i['’]?m interested in|i am interested in)\b",
    re.IGNORECASE,
)
KNOW_EXCEPTION_RE = re.compile(r"\b(?:know|understand|learn|hear about)\b", re.IGNORECASE)


def _matches(message: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)


def _actionable_lead(message: str, product: str, knowledge: bool) -> bool:
    if _matches(message, PURCHASE_PATTERNS):
        return True
    if knowledge:
        return False
    if product and WANT_PRODUCT_RE.search(message) and not KNOW_EXCEPTION_RE.search(message):
        return True
    return False


def _sales_pitch_query(product: str) -> str:
    name = product or "the product"
    return f"What is {name}?"


class ChatRouter:
    """Classifies the current turn. Does not lock the conversation into a form."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    def route(self, state: ConversationState) -> ConversationState:
        state.explicit_action = ""
        understanding = self._understand(state)
        self._apply_understanding(state, understanding)
        if self._should_refine(state, understanding):
            refined = self._refine_with_llm(state)
            if refined and refined.confidence >= understanding.confidence:
                self._apply_understanding(state, refined, merge=True)
                understanding = refined
        if self._needs_state_manager(state, understanding):
            managed = self._run_state_manager(state)
            if managed:
                self._apply_state_manager(state, managed)
                understanding = self._understanding_from_state_manager(state, managed, understanding)
        else:
            populate_deterministic_state_trace(
                state,
                needs_rag=bool((state.trace or {}).get("should_retrieve")),
            )
        self._sync_lead_stage(state)
        state.trace = dict(state.trace or {})
        state.trace["turn_understanding"] = understanding.to_dict()
        state.trace["conversation_status"] = conversation_status_label(state)
        state.trace["current_status"] = current_status_label(state)
        state.trace["active_product"] = state.product or state.trace.get("active_product") or ""
        state.trace["product"] = state.product or ""
        state.trace["lead_stage"] = state.lead_stage or LeadStage.NOT_STARTED
        state.trace["lead_status"] = state.lead_stage or LeadStage.NOT_STARTED
        state.trace["lead_collection_active"] = bool(state.lead_collection_active)
        state.trace["history_turn_count"] = len(state.conversation_history or [])
        if current_trace():
            method = "llm" if state.trace.get("state_manager_llm") or state.trace.get("turn_understanding_llm") else "deterministic"
            record_turn_understanding(understanding.to_dict(), method=method)
        return state

    def _understand(self, state: ConversationState) -> TurnUnderstanding:
        raw_message = state.user_message or ""
        message = strip_greeting_prefix(raw_message)
        last_assistant = last_assistant_text(state)
        if is_greeting_only(raw_message):
            return TurnUnderstanding(
                turn_intent=TurnIntent.GENERAL,
                needs_rag=False,
                confidence=0.95,
            )
        if is_short_yes(raw_message) and last_assistant:
            if offered_callback(last_assistant):
                return TurnUnderstanding(
                    turn_intent=TurnIntent.ACTION,
                    needs_rag=False,
                    lead_intent=True,
                    explicit_action="create_lead",
                    confidence=0.93,
                )
            if offered_product_information(last_assistant):
                return TurnUnderstanding(
                    turn_intent=TurnIntent.KNOWLEDGE,
                    needs_rag=True,
                    confidence=0.9,
                )
        if is_acknowledgement_only(raw_message):
            return TurnUnderstanding(
                turn_intent=TurnIntent.GENERAL,
                needs_rag=False,
                confidence=0.92,
            )
        if wants_more_product_info(raw_message) and last_assistant:
            info_offer = offered_product_information(last_assistant)
            callback_offer = offered_callback(last_assistant)
            collecting = bool(state.awaiting_field) and not info_offer and not callback_offer
            if not collecting:
                if (
                    info_offer
                    or is_tell_more(raw_message)
                    or (
                        state.product
                        and state.conversation_goal
                        in {ConversationGoal.LEAD, ConversationGoal.SALES, ConversationGoal.SUPPORT}
                    )
                ):
                    return TurnUnderstanding(
                        turn_intent=TurnIntent.KNOWLEDGE,
                        needs_rag=True,
                        confidence=0.92,
                    )
                return TurnUnderstanding(turn_intent=TurnIntent.CONFIRMATION, needs_rag=False, confidence=0.85)
        if is_short_no(raw_message) or is_phone_refusal(raw_message):
            return TurnUnderstanding(turn_intent=TurnIntent.CONFIRMATION, needs_rag=False, confidence=0.9)
        product = extract_product(message)
        purchase_phrase = _matches(message, PURCHASE_PATTERNS) or _actionable_lead(message, product, False)
        contact = looks_like_contact_request(message)
        lead_phrase = purchase_phrase or contact or _matches(message, LEAD_PATTERNS)
        support_phrase = _matches(message, SUPPORT_PATTERNS)
        knowledge_phrase = looks_like_knowledge_request(message)
        if re.search(r"how (?:can|do) i (?:buy|purchase|order)", message, flags=re.IGNORECASE):
            knowledge_phrase = bool(
                re.search(
                    r"\b(?:what is|what's|what are|why|benefits?|warranty|battery|price|delivery|difference)\b",
                    message,
                    flags=re.IGNORECASE,
                )
            )
        if knowledge_phrase and re.search(r"\b(?:why|benefits?)\b", message, flags=re.IGNORECASE):
            if not re.search(r"want to (?:buy|purchase|order)|call me|callback|contact me", message, flags=re.IGNORECASE):
                purchase_phrase = False
                lead_phrase = contact
        ticket = looks_like_ticket_request(message)
        name = extract_name(message)
        city = extract_city(message)
        phone = extract_phone_from_text(message, default_phone_region(state))
        issue = extract_issue(message)
        context = extract_user_context(message)
        providing = bool(state.awaiting_field) and not (knowledge_phrase or lead_phrase or support_phrase)
        if PHONE_LIKE.search(message) and state.mode in {ChatMode.LEAD, ChatMode.SUPPORT} and not knowledge_phrase:
            providing = True
        mixed = knowledge_phrase and (purchase_phrase or support_phrase or contact)
        needs_rag = bool(knowledge_phrase)
        action = None
        if contact:
            action = "create_lead"
        if ticket:
            action = "create_ticket"
        if mixed:
            intent = TurnIntent.MIXED
        elif providing:
            intent = TurnIntent.PROVIDE_INFORMATION
        elif knowledge_phrase:
            intent = TurnIntent.KNOWLEDGE
        elif purchase_phrase or contact:
            intent = TurnIntent.LEAD_INTENT
        elif support_phrase:
            intent = TurnIntent.SUPPORT_INTENT
        elif name or city or phone or context:
            intent = TurnIntent.CONTEXT_UPDATE
        elif state.mode:
            intent = TurnIntent.GENERAL
        else:
            intent = TurnIntent.KNOWLEDGE
            needs_rag = True
        confidence = 0.9 if intent not in {TurnIntent.GENERAL, TurnIntent.UNKNOWN} else 0.6
        if intent == TurnIntent.GENERAL and state.mode:
            confidence = 0.9
        return TurnUnderstanding(
            turn_intent=intent,
            needs_rag=needs_rag,
            lead_intent=purchase_phrase or contact,
            support_intent=support_phrase or ticket,
            information_updates={
                "name": name or None,
                "phone": phone[0] if phone else None,
                "city": city or None,
                "product": product or None,
                "issue": issue or None,
            },
            user_context_updates=context,
            explicit_action=action,
            confidence=confidence,
        )

    def _apply_understanding(
        self,
        state: ConversationState,
        understanding: TurnUnderstanding,
        *,
        merge: bool = False,
    ) -> None:
        updates = understanding.information_updates
        if updates.get("name") and (merge or not state.user_name):
            state.user_name = updates["name"] or state.user_name
        if updates.get("city") and (merge or not state.city):
            state.city = updates["city"] or state.city
            if state.city:
                state.city_country = state.city_country or "IN"
        if updates.get("phone") and not state.phone:
            result = extract_and_validate_phone(updates["phone"] or "", default_phone_region(state))
            if result.valid:
                apply_phone_to_state(state, result)
        if updates.get("issue") and not state.support_issue:
            state.support_issue = updates["issue"] or ""
        if understanding.user_context_updates:
            state.user_context = merge_user_context(state.user_context, understanding.user_context_updates)
        if understanding.lead_intent:
            state.lead_intent = True
            state.sales_interest = True
            if not understanding.support_intent:
                state.support_intent = False
        if understanding.support_intent and not understanding.lead_intent:
            state.support_intent = True
        if understanding.explicit_action:
            state.explicit_action = understanding.explicit_action
            if understanding.explicit_action == "create_lead":
                state.lead_collection_active = True

        state.current_turn_intent = understanding.turn_intent
        state.trace = dict(state.trace or {})
        state.trace["should_retrieve"] = bool(understanding.needs_rag)
        state.trace["should_create_lead"] = False
        state.trace["should_create_ticket"] = False
        state.trace["detected_product"] = updates.get("product") or extract_product(state.user_message)
        state.trace["information_provided"] = understanding.turn_intent == TurnIntent.PROVIDE_INFORMATION
        state.trace["ask_missing"] = False
        state.trace["needs_natural_reply"] = False
        capabilities: list[str] = []
        if understanding.needs_rag:
            capabilities.append("RAG")
        if understanding.explicit_action == "create_lead":
            capabilities.append("CREATE_LEAD")
        if understanding.explicit_action == "create_ticket":
            capabilities.append("CREATE_SUPPORT_TICKET")
        if (
            understanding.turn_intent
            in {
                TurnIntent.CONFIRMATION,
                TurnIntent.GENERAL,
                TurnIntent.CONTEXT_UPDATE,
                TurnIntent.LEAD_INTENT,
                TurnIntent.SUPPORT_INTENT,
            }
            and not understanding.needs_rag
            and not understanding.explicit_action
        ):
            capabilities.append("CONVERSATION_ONLY")
        state.trace["capabilities"] = capabilities

        intent = understanding.turn_intent
        if intent in {TurnIntent.CONFIRMATION, TurnIntent.ACTION}:
            if intent == TurnIntent.ACTION and understanding.explicit_action == "create_lead":
                state.conversation_goal = ConversationGoal.SALES
                state.lead_intent = True
                state.lead_collection_active = True
                state.intent = ChatMode.LEAD
                state.mode = ChatMode.LEAD
                return
            if intent == TurnIntent.ACTION and understanding.explicit_action == "create_ticket":
                state.conversation_goal = ConversationGoal.SUPPORT
                state.support_intent = True
                state.intent = ChatMode.SUPPORT
                state.mode = ChatMode.SUPPORT
                return
            if state.conversation_goal == ConversationGoal.SUPPORT or state.mode == ChatMode.SUPPORT:
                state.intent = ChatMode.SUPPORT
                state.mode = ChatMode.SUPPORT
            elif state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES} or state.mode == ChatMode.LEAD:
                state.intent = ChatMode.LEAD
                state.mode = ChatMode.LEAD
            else:
                state.intent = ChatMode.KNOWLEDGE
                state.mode = ChatMode.KNOWLEDGE
            state.trace["should_retrieve"] = False
            state.trace["needs_natural_reply"] = True
            if is_short_no(state.user_message or "") or is_phone_refusal(state.user_message or ""):
                state.awaiting_field = ""
            return

        if intent in {TurnIntent.PROVIDE_INFORMATION, TurnIntent.CONTEXT_UPDATE, TurnIntent.GENERAL}:
            if is_greeting_only(state.user_message or "") and not state.mode:
                state.intent = ChatMode.KNOWLEDGE
                state.mode = ChatMode.KNOWLEDGE
                state.trace["should_retrieve"] = False
                state.trace["needs_natural_reply"] = True
                return
            if state.conversation_goal == ConversationGoal.SUPPORT or state.mode == ChatMode.SUPPORT:
                state.intent = ChatMode.SUPPORT
                state.mode = ChatMode.SUPPORT
            elif state.conversation_goal == ConversationGoal.LEAD or state.mode == ChatMode.LEAD or state.lead_intent:
                state.intent = ChatMode.LEAD
                state.mode = ChatMode.LEAD
                if not state.conversation_goal or state.conversation_goal == ConversationGoal.NONE:
                    state.conversation_goal = ConversationGoal.SALES
                    state.lead_intent = True
            else:
                state.intent = ChatMode.KNOWLEDGE
                state.mode = ChatMode.KNOWLEDGE
                state.trace["should_retrieve"] = False
                state.trace["needs_natural_reply"] = True
            return

        if intent in {TurnIntent.LEAD_INTENT, TurnIntent.MIXED} and understanding.lead_intent and not understanding.support_intent:
            state.conversation_goal = ConversationGoal.SALES
            state.sales_interest = True
            state.lead_intent = True
            if understanding.explicit_action == "create_lead" or state.lead_collection_active:
                state.lead_collection_active = True
                state.lead_status = (
                    state.lead_status if state.lead_status != LeadStatus.IDLE else LeadStatus.COLLECTING
                )
                state.lead_workflow = (
                    state.lead_workflow
                    if state.lead_workflow not in {LeadWorkflow.NONE, ""}
                    else LeadWorkflow.DISCUSSING_PRODUCT
                )
                if understanding.needs_rag:
                    state.return_mode = ChatMode.LEAD
                    state.intent = ChatMode.KNOWLEDGE
                    state.mode = ChatMode.KNOWLEDGE
                else:
                    state.intent = ChatMode.LEAD
                    state.mode = ChatMode.LEAD
                    state.return_mode = ""
                return
            state.lead_collection_active = False
            state.lead_status = LeadStatus.IDLE
            state.lead_workflow = LeadWorkflow.DISCUSSING_PRODUCT
            if understanding.needs_rag:
                state.return_mode = ChatMode.LEAD
                state.intent = ChatMode.KNOWLEDGE
                state.mode = ChatMode.KNOWLEDGE
                return
            pitch_query = _sales_pitch_query(state.product)
            state.trace["should_retrieve"] = True
            state.trace["sub_questions"] = [pitch_query]
            state.trace["resolved_query"] = pitch_query
            state.trace["next_action"] = "SALES_PITCH_AND_OFFER_CONTACT"
            state.trace["needs_natural_reply"] = True
            state.trace["capabilities"] = ["RAG", "CONVERSATION_ONLY"]
            state.return_mode = ChatMode.LEAD
            state.intent = ChatMode.KNOWLEDGE
            state.mode = ChatMode.KNOWLEDGE
            return

        if intent in {TurnIntent.SUPPORT_INTENT, TurnIntent.MIXED} and understanding.support_intent:
            state.conversation_goal = ConversationGoal.SUPPORT
            if state.ticket_status == TicketStatus.IDLE:
                state.ticket_status = TicketStatus.COLLECTING
            if understanding.needs_rag:
                state.return_mode = ChatMode.SUPPORT
                state.intent = ChatMode.KNOWLEDGE
                state.mode = ChatMode.KNOWLEDGE
            else:
                state.intent = ChatMode.SUPPORT
                state.mode = ChatMode.SUPPORT
                state.return_mode = ""
            return

        if intent == TurnIntent.MIXED and understanding.needs_rag:
            if state.conversation_goal == ConversationGoal.SUPPORT or understanding.support_intent:
                state.return_mode = ChatMode.SUPPORT
                state.conversation_goal = ConversationGoal.SUPPORT
            elif state.conversation_goal == ConversationGoal.LEAD or understanding.lead_intent:
                state.return_mode = ChatMode.LEAD
                state.conversation_goal = ConversationGoal.LEAD
            state.intent = ChatMode.KNOWLEDGE
            state.mode = ChatMode.KNOWLEDGE
            return

        if understanding.needs_rag or intent == TurnIntent.KNOWLEDGE:
            if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES, ConversationGoal.SUPPORT}:
                goal = state.conversation_goal
            elif state.mode in {ChatMode.LEAD, ChatMode.SUPPORT}:
                goal = state.mode
                state.conversation_goal = goal
            else:
                goal = ""
            if goal in {ChatMode.SUPPORT, ConversationGoal.SUPPORT} and state.ticket_status != TicketStatus.CREATED:
                state.return_mode = ChatMode.SUPPORT
            elif goal in {ChatMode.LEAD, ConversationGoal.LEAD, ConversationGoal.SALES} and state.lead_status != LeadStatus.CREATED:
                state.return_mode = ChatMode.LEAD
            if not state.conversation_goal or state.conversation_goal == ConversationGoal.NONE:
                state.conversation_goal = ConversationGoal.KNOWLEDGE
                state.sales_interest = bool(state.sales_interest and goal in {ChatMode.LEAD, ConversationGoal.SALES})
            state.intent = ChatMode.KNOWLEDGE
            state.mode = ChatMode.KNOWLEDGE
            if (state.lead_collection_active or state.awaiting_field) and state.conversation_goal in {
                ConversationGoal.LEAD,
                ConversationGoal.SALES,
            }:
                state.trace["resume_lead_after_knowledge"] = True
                state.trace["next_action"] = "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"
            elif goal == ConversationGoal.SUPPORT and (
                state.awaiting_field or state.explicit_action == "create_ticket"
            ):
                state.trace["resume_support_after_knowledge"] = True
                state.trace["next_action"] = "ANSWER_KNOWLEDGE_THEN_RESUME_SUPPORT"
            return

        if state.mode:
            state.intent = state.mode
            return
        state.conversation_goal = ConversationGoal.KNOWLEDGE
        state.intent = ChatMode.KNOWLEDGE
        state.mode = ChatMode.KNOWLEDGE

    def _should_refine(self, state: ConversationState, understanding: TurnUnderstanding) -> bool:
        if not self._llm or not getattr(self._llm, "is_configured", False):
            return False
        if understanding.turn_intent != TurnIntent.MIXED:
            return False
        if understanding.user_context_updates:
            return False
        return len((state.user_message or "").split()) >= 18

    def _refine_with_llm(self, state: ConversationState) -> TurnUnderstanding | None:
        if not self._llm:
            return None
        history = compact_recent_history(state.conversation_history)
        user = (
            f"conversation_goal={state.conversation_goal}\n"
            f"product={state.product}\n"
            f"user_context={state.user_context}\n"
            f"history={history}\n"
            f"message={state.user_message}"
        )
        try:
            raw = self._llm.complete(TURN_UNDERSTANDING_SYSTEM, user, temperature=0.0, max_tokens=256)
        except Exception:
            return None
        parsed = parse_turn_understanding(raw)
        if parsed:
            state.trace = dict(state.trace or {})
            state.trace["llm_call_count"] = int(state.trace.get("llm_call_count") or 0) + 1
            state.trace["turn_understanding_llm"] = True
        return parsed

    def _needs_state_manager(self, state: ConversationState, understanding: TurnUnderstanding) -> bool:
        if not self._llm or not getattr(self._llm, "is_configured", False):
            return False
        message = (state.user_message or "").strip()
        if is_greeting_only(message):
            return False
        if understanding.turn_intent == TurnIntent.PROVIDE_INFORMATION and state.awaiting_field:
            return False
        if understanding.turn_intent == TurnIntent.CONTEXT_UPDATE and understanding.confidence >= 0.9:
            return False
        if understanding.confidence >= 0.88:
            return False
        if understanding.turn_intent in {
            TurnIntent.KNOWLEDGE,
            TurnIntent.LEAD_INTENT,
            TurnIntent.SUPPORT_INTENT,
            TurnIntent.MIXED,
        }:
            return False
        words = message.split()
        return len(words) <= 4 and bool(state.conversation_history)

    def _run_state_manager(self, state: ConversationState):
        if not self._llm:
            return None
        try:
            raw = self._llm.complete(
                STATE_MANAGER_SYSTEM,
                build_state_manager_prompt(state),
                temperature=0.0,
                max_tokens=512,
            )
        except Exception:
            return None
        parsed = parse_state_manager_result(raw)
        if parsed:
            state.trace = dict(state.trace or {})
            state.trace["llm_call_count"] = int(state.trace.get("llm_call_count") or 0) + 1
            state.trace["state_manager_llm"] = True
            state.trace["state_manager"] = parsed.to_dict()
        return parsed

    def _apply_state_manager(self, state: ConversationState, result) -> None:
        apply_state_manager_result(state, result)
        intent = turn_intent_from_status(result.current_status)
        needs_rag = result.current_status == "KNOWLEDGE" or result.next_action in {
            "ANSWER_KNOWLEDGE",
            "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD",
            "ANSWER_AND_RESUME_LEAD",
        }
        understanding = TurnUnderstanding(
            turn_intent=intent,
            needs_rag=needs_rag,
            lead_intent=result.conversation_status == "SALE" or result.sales_interest,
            support_intent=result.conversation_status == "SUPPORT",
            explicit_action=state.explicit_action or None,
            confidence=result.confidence,
        )
        self._apply_understanding(state, understanding, merge=True)

    def _understanding_from_state_manager(
        self,
        state: ConversationState,
        result,
        previous: TurnUnderstanding,
    ) -> TurnUnderstanding:
        return TurnUnderstanding(
            turn_intent=state.current_turn_intent or previous.turn_intent,
            needs_rag=bool(state.trace.get("should_retrieve")),
            lead_intent=state.lead_intent,
            support_intent=state.support_intent,
            information_updates=previous.information_updates,
            user_context_updates=previous.user_context_updates,
            explicit_action=state.explicit_action or None,
            confidence=max(previous.confidence, result.confidence),
        )

    def _sync_lead_stage(self, state: ConversationState) -> None:
        if state.lead_status == LeadStatus.CREATED:
            state.lead_stage = LeadStage.COMPLETED
            return
        if state.lead_collection_active or state.explicit_action == "create_lead" or state.awaiting_field:
            state.lead_stage = LeadStage.IN_PROGRESS
            return
        if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES} or state.sales_interest:
            if state.user_name or state.phone or state.city:
                state.lead_stage = LeadStage.IN_PROGRESS
            elif state.lead_stage != LeadStage.COMPLETED:
                state.lead_stage = LeadStage.NOT_STARTED

