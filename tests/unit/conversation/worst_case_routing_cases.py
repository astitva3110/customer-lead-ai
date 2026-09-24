"""Worst-case / messy real-world routing queries for Jev router evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.conversation.models import ChatMode


@dataclass(frozen=True)
class WorstCaseRoutingCase:
    id: str
    message: str
    expected: str  # LEAD | KNOWLEDGE | SUPPORT | GENERAL
    category: str


WORST_CASE_ROUTING_CASES: tuple[WorstCaseRoutingCase, ...] = (
    # --- LEAD: buy / want / connect sales ---
    WorstCaseRoutingCase("lead_01", "mujhe hearing aid chaiye", "LEAD", "hinglish_acquisition"),
    WorstCaseRoutingCase("lead_02", "hi mujhe heaing aid chaiye thi", "LEAD", "hinglish_typo"),
    WorstCaseRoutingCase("lead_03", "मुझे हियरिंग एड चाहिए", "LEAD", "hindi_acquisition"),
    WorstCaseRoutingCase("lead_04", "mujhe hearing chaiye", "LEAD", "hinglish_shorthand"),
    WorstCaseRoutingCase("lead_05", "i want to buy heaing aid", "LEAD", "english_typo"),
    WorstCaseRoutingCase("lead_06", "I need TINY hearing aid", "LEAD", "product_buy"),
    WorstCaseRoutingCase("lead_07", "connect me to sales team", "LEAD", "explicit_sales"),
    WorstCaseRoutingCase("lead_08", "ok connect me to the sales team", "LEAD", "sales_confirm"),
    WorstCaseRoutingCase("lead_09", "please create a lead for Radius", "LEAD", "explicit_lead"),
    WorstCaseRoutingCase("lead_10", "i want to buy a new hearing aid because my old one broke", "LEAD", "buy_despite_broken"),
    WorstCaseRoutingCase("lead_11", "mujhe ek hearing aid lena hai", "LEAD", "hindi_lena"),
    WorstCaseRoutingCase("lead_12", "call me back i want hearing aid", "LEAD", "callback_buy"),
    WorstCaseRoutingCase("lead_13", "book demo for TINY", "LEAD", "demo_request"),
    WorstCaseRoutingCase("lead_14", "mujhe radius m16 chahiye", "LEAD", "product_hinglish"),
    # --- KNOWLEDGE: hearing health / consultation / product info ---
    WorstCaseRoutingCase("know_01", "i have hearing loss of 45% which hearing aid should i need", "KNOWLEDGE", "consultation"),
    WorstCaseRoutingCase("know_02", "mujhe sunai problem hai", "KNOWLEDGE", "hindi_hearing_concern"),
    WorstCaseRoutingCase("know_03", "mujhe awaaz nhi aate hia", "KNOWLEDGE", "hinglish_cannot_hear"),
    WorstCaseRoutingCase("know_04", "mujhe aawa nhi rahi hai call par", "KNOWLEDGE", "typo_call_hearing"),
    WorstCaseRoutingCase("know_05", "i cannot hear properly on phone calls anymore", "KNOWLEDGE", "phone_hearing"),
    WorstCaseRoutingCase("know_06", "i have a problem of listening to the call", "KNOWLEDGE", "listening_trap"),
    WorstCaseRoutingCase("know_07", "tell me what problems hearing aids solve", "KNOWLEDGE", "educational_problem_word"),
    WorstCaseRoutingCase("know_08", "what is the problem with TINY battery life", "KNOWLEDGE", "info_problem_word"),
    WorstCaseRoutingCase("know_09", "What is Radius M16?", "KNOWLEDGE", "product_info"),
    WorstCaseRoutingCase("know_10", "TINY ka battery kitna chalta hai", "KNOWLEDGE", "hindi_battery"),
    WorstCaseRoutingCase("know_11", "how much is TINY?", "KNOWLEDGE", "price_info"),
    WorstCaseRoutingCase("know_12", "what is tinny warrenty?", "KNOWLEDGE", "typo_warranty"),
    WorstCaseRoutingCase("know_13", "meri sunai kam ho gayi hai", "KNOWLEDGE", "hindi_hearing_loss"),
    WorstCaseRoutingCase("know_14", "phone par baat sunne mein problem hai", "KNOWLEDGE", "hindi_phone_hearing"),
    WorstCaseRoutingCase("know_15", "i want to buy TINY, what is its warranty?", "KNOWLEDGE", "mixed_buy_warranty"),
    WorstCaseRoutingCase("know_16", "kya hearing aid se sunai badhti hai", "KNOWLEDGE", "hindi_info_question"),
    WorstCaseRoutingCase("know_17", "who is Rohit Misra?", "KNOWLEDGE", "person_info"),
    WorstCaseRoutingCase("know_18", "sunai nahi aa rahi properly", "KNOWLEDGE", "hinglish_hearing"),
    # --- SUPPORT: device faults / tickets ---
    WorstCaseRoutingCase("sup_01", "my hearing aid is not working", "SUPPORT", "device_not_working"),
    WorstCaseRoutingCase("sup_02", "my TINY stopped working", "SUPPORT", "product_stopped"),
    WorstCaseRoutingCase("sup_03", "mera hearing aid ka issue hai awaz nahi aa rahi", "SUPPORT", "hinglish_device"),
    WorstCaseRoutingCase("sup_04", "i have a problem with my hearing aid, no sound", "SUPPORT", "problem_no_sound"),
    WorstCaseRoutingCase("sup_05", "hearing aid sound bahut kam hai", "SUPPORT", "hinglish_low_volume"),
    WorstCaseRoutingCase("sup_06", "open a support ticket, my hearing aid is broken", "SUPPORT", "explicit_ticket"),
    WorstCaseRoutingCase("sup_07", "TINY me awaz nahi aa rahi", "SUPPORT", "hindi_device_no_sound"),
    WorstCaseRoutingCase("sup_08", "Radius M16 not charging", "SUPPORT", "charging_issue"),
    WorstCaseRoutingCase("sup_09", "mera tiny kharab ho gaya", "SUPPORT", "hindi_broken"),
    WorstCaseRoutingCase("sup_10", "hearing aid repair chahiye", "SUPPORT", "repair_request"),
    WorstCaseRoutingCase("sup_11", "device band ho gaya hai", "SUPPORT", "hindi_device_off"),
    WorstCaseRoutingCase("sup_12", "bluup me sound nahi aa rahi", "SUPPORT", "product_hinglish_fault"),
    # --- GENERAL: greeting / capability / unclear ---
    WorstCaseRoutingCase("gen_01", "kya kar sakte ho", "GENERAL", "capability_hindi"),
    WorstCaseRoutingCase("gen_02", "what can you do?", "GENERAL", "capability_en"),
    WorstCaseRoutingCase("gen_03", "hi", "GENERAL", "greeting"),
    WorstCaseRoutingCase("gen_04", "namaste", "GENERAL", "greeting_hindi"),
    WorstCaseRoutingCase("gen_05", "thanks", "GENERAL", "thanks"),
    WorstCaseRoutingCase("gen_06", "bolo", "GENERAL", "unclear_hindi"),
    WorstCaseRoutingCase("gen_07", "sir", "GENERAL", "unclear_sir"),
    WorstCaseRoutingCase("gen_08", "Traaas hore", "GENERAL", "gibberish"),
    # --- TRAP: mixed / ambiguous (guardrails should save) ---
    WorstCaseRoutingCase("trap_01", "connect me to sales, i have hearing problem", "LEAD", "sales_plus_hearing"),
    WorstCaseRoutingCase("trap_02", "mujhe hearing aid chaiye lekin pehle warranty batao", "KNOWLEDGE", "buy_then_info"),
    WorstCaseRoutingCase("trap_03", "problem hai sunne mein hearing aid ke baare mein batao", "KNOWLEDGE", "problem_plus_info"),
    WorstCaseRoutingCase("trap_04", "i have hearing problem need aid", "LEAD", "problem_need_aid"),
    WorstCaseRoutingCase("trap_05", "call par sunai nahi deti hearing aid chahiye", "LEAD", "call_hearing_want"),
)
