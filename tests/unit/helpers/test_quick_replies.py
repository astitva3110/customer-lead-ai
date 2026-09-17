from app.helpers.conversation_extract import looks_like_hearing_consultation_need, looks_like_trial_request
from app.helpers.conversation_reply import price_query_reply
from app.helpers.quick_replies import (
    CONSULTATION_TRIAL_LABEL,
    CONSULTATION_TRIAL_MESSAGE,
    PENDING_LEAD_OFFER,
    PENDING_PRODUCT_INTEREST,
    attach_lead_offer_buttons,
    attach_product_interest_quick_replies,
    build_product_interest_replies,
    lead_choice_prompt,
    resolve_inbound_choice,
    quick_replies_from_trace,
    set_lead_choice_offer,
    should_attach_product_interest,
)
from app.services.conversation.models import ConversationGoal, ConversationState
from tests.unit.conversation.fakes import make_orchestrator


def test_resolve_lead_yes_no_ids() -> None:
    assert resolve_inbound_choice("lead_yes") == {"action": "lead_yes"}
    assert resolve_inbound_choice("lead_no") == {"action": "lead_no"}
    assert resolve_inbound_choice("ticket_yes") == {"action": "ticket_yes"}


def test_followup_id_maps_to_full_question() -> None:
    resolved = resolve_inbound_choice("followup_return")
    assert resolved == {"action": "followup", "message": "What is the return policy?"}


def test_attach_lead_offer_buttons_skips_duplicate_for_price_reply() -> None:
    state = ConversationState(response=price_query_reply("Bluup+"))
    attach_lead_offer_buttons(state)
    assert "great choice" not in state.response.lower()
    assert state.response.count("our team") == 1
    assert quick_replies_from_trace(state)
    assert state.trace.get("pending_choice") == PENDING_LEAD_OFFER


def test_set_lead_choice_offer_attaches_yes_no_for_lead_flow() -> None:
    state = ConversationState(response="I can connect you with sales.")
    set_lead_choice_offer(state, product="TINY")
    replies = quick_replies_from_trace(state)
    assert len(replies) == 2
    assert replies[0]["id"] == "lead_yes"
    assert state.trace.get("pending_choice") == PENDING_LEAD_OFFER


def test_lead_choice_prompt_uses_sales_team_language() -> None:
    prompt = lead_choice_prompt("TINY")
    assert "connect you with our team" in prompt.lower()
    assert "create a lead" not in prompt.lower()


def test_build_product_interest_replies_for_named_product() -> None:
    replies = build_product_interest_replies("TINY")
    assert replies[0]["label"] == "I want TINY"
    assert resolve_inbound_choice(replies[0]["id"]) == {
        "action": "followup",
        "message": "I want TINY",
    }


def test_attach_product_interest_quick_replies_after_product_question() -> None:
    state = ConversationState(
        user_message="What is TINY?",
        product="TINY",
        response="TINY is a rechargeable hearing aid.",
        trace={"should_retrieve": True, "retrieval_used": True},
    )
    attach_product_interest_quick_replies(state)
    replies = quick_replies_from_trace(state)
    assert replies[0]["label"] == "I want TINY"
    assert state.trace.get("pending_choice") == PENDING_PRODUCT_INTEREST


def test_should_not_attach_product_interest_for_policy_questions() -> None:
    state = ConversationState(user_message="What are the terms and conditions?")
    assert not should_attach_product_interest(state)


def test_hearing_consultation_need_detects_loss_and_selection_questions() -> None:
    assert looks_like_hearing_consultation_need("I have hearing loss")
    assert looks_like_hearing_consultation_need("Which hearing aid should I buy?")
    assert looks_like_hearing_consultation_need("What hearing aid do I need?")
    assert not looks_like_hearing_consultation_need("What is hearing loss?")


def test_build_product_interest_replies_includes_trial_for_consultation() -> None:
    replies = build_product_interest_replies("", include_trial=True)
    labels = [item["label"] for item in replies]
    assert "I want a hearing aid" in labels
    assert CONSULTATION_TRIAL_LABEL in labels
    trial = next(item for item in replies if item["label"] == CONSULTATION_TRIAL_LABEL)
    assert resolve_inbound_choice(trial["id"]) == {
        "action": "followup",
        "message": CONSULTATION_TRIAL_MESSAGE,
    }


def test_attach_consultation_quick_replies_after_hearing_loss_question() -> None:
    state = ConversationState(
        user_message="I have hearing loss, which hearing aid should I buy?",
        response="A hearing test helps determine the right aid for your needs.",
        trace={"should_retrieve": True, "retrieval_used": True},
    )
    attach_product_interest_quick_replies(state)
    replies = quick_replies_from_trace(state)
    labels = [item["label"] for item in replies]
    assert CONSULTATION_TRIAL_LABEL in labels
    assert "I want a hearing aid" in labels


def test_i_want_a_hearing_aid_routes_to_sales_not_support() -> None:
    import json

    from tests.unit.conversation.test_semantic_router import ScriptedRouterLLM

    support_payload = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    orchestrator, *_ = make_orchestrator(llm=ScriptedRouterLLM([support_payload]))
    cid = "hearing-aid-interest"
    orchestrator.handle(cid, "i have 50% hearing loss what should i do")
    second = orchestrator.handle(cid, "I want a hearing aid")
    assert second.conversation_goal != ConversationGoal.SUPPORT
    assert second.mode != "SUPPORT"
    assert second.support_intent is False
    assert "customer service team" not in (second.response or "").lower()


def test_trial_request_starts_lead_collection() -> None:
    orchestrator, _, _, lead_tool, *_ = make_orchestrator()
    cid = "trial-lead"
    first = orchestrator.handle(cid, "I have hearing loss, which hearing aid should I buy?")
    assert any(item["label"] == CONSULTATION_TRIAL_LABEL for item in quick_replies_from_trace(first))
    second = orchestrator.handle(cid, CONSULTATION_TRIAL_MESSAGE)
    assert second.explicit_action == "create_lead" or second.lead_collection_active
    assert second.awaiting_field in {"name", "phone", "phone_country", "city"}
    assert looks_like_trial_request(CONSULTATION_TRIAL_MESSAGE)
    assert lead_tool.leads == []


def test_offered_lead_choice_detects_connect_with_our_team_wording() -> None:
    from app.helpers.quick_replies import offered_lead_choice

    text = (
        "Nice to meet you, Astitva! I'm glad you're interested in buying a hearing aid. "
        "Would you like to connect with our team to discuss further?"
    )
    assert offered_lead_choice(text)


def test_attach_lead_offer_after_sales_pitch_avoids_llm_team_offer_wording() -> None:
    from app.helpers.quick_replies import (
        PENDING_LEAD_OFFER,
        attach_lead_offer_after_sales_pitch,
        quick_replies_from_trace,
    )

    state = ConversationState(
        response=(
            "Nice to meet you, Astitva! I'm glad you're interested in buying a hearing aid. "
            "Would you like to connect with our team to discuss further?"
        ),
        product="",
        trace={"sales_pitch_from_rag": True, "next_action": "SALES_PITCH_AND_OFFER_CONTACT"},
    )
    attach_lead_offer_after_sales_pitch(state)
    assert state.response.count("connect") == 1
    assert "Great choice" not in state.response
    assert quick_replies_from_trace(state)
    assert state.trace.get("pending_choice") == PENDING_LEAD_OFFER


def test_attach_lead_offer_after_sales_pitch_avoids_duplicate_paragraph() -> None:
    from app.helpers.quick_replies import (
        PENDING_LEAD_OFFER,
        attach_lead_offer_after_sales_pitch,
        quick_replies_from_trace,
    )

    state = ConversationState(
        response=lead_choice_prompt("TINY"),
        product="TINY",
        trace={"sales_pitch_from_rag": True},
    )
    attach_lead_offer_after_sales_pitch(state)
    assert state.response.count("connect you with our team") == 1
    assert quick_replies_from_trace(state)
    assert state.trace.get("pending_choice") == PENDING_LEAD_OFFER


def test_apply_declined_choice_resets_lead_mode() -> None:
    from app.helpers.quick_replies import apply_declined_choice
    from app.services.conversation.models import ChatMode

    state = ConversationState(
        mode=ChatMode.LEAD,
        intent=ChatMode.LEAD,
        conversation_goal=ConversationGoal.SALES,
        lead_intent=True,
        trace={"sales_pitch_from_rag": True, "next_action": "SALES_PITCH_AND_OFFER_CONTACT"},
    )
    apply_declined_choice(state)
    assert state.mode == ChatMode.KNOWLEDGE
    assert "no worries" in state.response.lower()
    assert "sales_pitch_from_rag" not in (state.trace or {})


def test_rag_answers_without_pending_choice_do_not_surface_buttons() -> None:
    state = ConversationState(
        response="TINY is compact. Would you like to know about its battery life?",
        trace={"knowledge_followup": True},
    )
    assert quick_replies_from_trace(state) == []
