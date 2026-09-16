from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class HearingSymptomTopic(str, Enum):
    NONE = "none"
    HEARING_DIFFICULTY = "hearing_difficulty"
    DEVICE_NO_SOUND = "device_no_sound"
    DEVICE_LOW_VOLUME = "device_low_volume"
    DEVICE_FAULT = "device_fault"


@dataclass(frozen=True)
class HearingSymptomAnalysis:
    topic: HearingSymptomTopic
    canonical_en: str
    device_mentioned: bool
    matched: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic.value,
            "canonical_en": self.canonical_en,
            "device_mentioned": self.device_mentioned,
            "matched": self.matched,
        }


_NEGATION = r"(?:nahi|nhi|nahin|nah|ni|नहीं|नही)"
_SOUND = r"(?:awaz|aawaz|avaaz|awaaz|sound|sunai|sunaai|sunna|voice|आवाज|सुनाई|सुन)"
_DEVICE = (
    r"(?:hearing aids?|hearing machine|hearing device|device|machine|"
    r"tiny|bluup|radius|earkart|कान\s*की\s*मशीन|हियरिंग\s*एड)"
)
_PERSONAL_SUBJECT = r"(?:mujhe|mujhko|muje|meko|mere|mera|main|me|i|my)"

_PERSONAL_HEARING_RE = re.compile(
    rf"(?:"
    rf"\b{_PERSONAL_SUBJECT}\b.{{0,35}}{_SOUND}.{{0,25}}{_NEGATION}"
    rf"|\b{_PERSONAL_SUBJECT}\b.{{0,35}}{_NEGATION}.{{0,25}}{_SOUND}"
    rf"|{_SOUND}.{{0,25}}{_NEGATION}.{{0,25}}(?:aa|aati|aate|aata|aara?h[aei]|ati|ate|deta|deti|milti|milte|milta|rahi|rhe|rahe)"
    rf"|(?:kuch|properly|clearly)?\s*{_SOUND}\s+{_NEGATION}"
    rf"|{_SOUND}\s+(?:kam|low|weak)"
    rf"|sunai\s+(?:kam|low|weak)"
    rf"|(?:can't|cannot|can not)\s+hear\b"
    rf"|(?:difficulty|trouble|problem)\s+(?:hearing|listening)\b"
    rf"|(?:hearing|sunne)\s+(?:problem|issue|loss|kam)\b"
    rf")",
    re.IGNORECASE,
)

_DEVICE_SOUND_ISSUE_RE = re.compile(
    rf"(?:"
    rf"\b{_DEVICE}\b.{{0,45}}{_SOUND}.{{0,25}}{_NEGATION}"
    rf"|\b{_DEVICE}\b.{{0,45}}{_NEGATION}.{{0,25}}{_SOUND}"
    rf"|{_SOUND}.{{0,25}}{_NEGATION}.{{0,45}}\b{_DEVICE}\b"
    rf"|{_NEGATION}.{{0,25}}{_SOUND}.{{0,45}}\b{_DEVICE}\b"
    rf"|no sound|low sound|low volume|very low"
    rf")",
    re.IGNORECASE,
)

_DEVICE_FAULT_RE = re.compile(
    r"\b(?:not working|stopped working|broken|repair|faulty|defective|kharab|खराब)\b",
    re.IGNORECASE,
)


def analyze_hearing_symptom(message: str) -> HearingSymptomAnalysis:
    text = (message or "").strip()
    if not text:
        return HearingSymptomAnalysis(HearingSymptomTopic.NONE, text, False, False)

    device_mentioned = bool(re.search(rf"\b{_DEVICE}\b", text, flags=re.IGNORECASE))
    device_sound = bool(_DEVICE_SOUND_ISSUE_RE.search(text))
    device_fault = device_mentioned and bool(_DEVICE_FAULT_RE.search(text))
    personal_hearing = bool(_PERSONAL_HEARING_RE.search(text))

    if device_sound or (device_mentioned and personal_hearing):
        if re.search(rf"(?:{_SOUND}|volume|sunai)\s+(?:kam|low|weak)", text, flags=re.IGNORECASE):
            return HearingSymptomAnalysis(
                HearingSymptomTopic.DEVICE_LOW_VOLUME,
                "hearing aid low sound or volume",
                True,
                True,
            )
        if device_fault:
            return HearingSymptomAnalysis(
                HearingSymptomTopic.DEVICE_FAULT,
                "hearing aid is not working",
                True,
                True,
            )
        return HearingSymptomAnalysis(
            HearingSymptomTopic.DEVICE_NO_SOUND,
            "hearing aid has no sound",
            True,
            True,
        )

    if personal_hearing:
        if re.search(rf"(?:{_SOUND}|sunai)\s+(?:kam|low|weak)", text, flags=re.IGNORECASE):
            return HearingSymptomAnalysis(
                HearingSymptomTopic.HEARING_DIFFICULTY,
                "difficulty hearing or low hearing",
                False,
                True,
            )
        return HearingSymptomAnalysis(
            HearingSymptomTopic.HEARING_DIFFICULTY,
            "difficulty hearing or cannot hear sound",
            False,
            True,
        )

    return HearingSymptomAnalysis(HearingSymptomTopic.NONE, text, device_mentioned, False)


def canonical_hearing_query(message: str) -> str:
    analysis = analyze_hearing_symptom(message)
    if analysis.matched:
        return analysis.canonical_en
    return (message or "").strip()


def _is_informational_hearing_question(message: str) -> bool:
    text = (message or "").strip()
    if not re.search(
        r"^\s*(?:what|who|where|when|why|how|tell me about|explain)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    return not re.search(
        r"\b(?:i|my|me|our|mujhe|mujhko|muje|mera|mere)\b",
        text,
        flags=re.IGNORECASE,
    )


def is_personal_hearing_difficulty(message: str) -> bool:
    if _is_informational_hearing_question(message):
        return False
    analysis = analyze_hearing_symptom(message)
    return analysis.matched and analysis.topic == HearingSymptomTopic.HEARING_DIFFICULTY


def is_device_hearing_symptom(message: str) -> bool:
    analysis = analyze_hearing_symptom(message)
    return analysis.matched and analysis.topic in {
        HearingSymptomTopic.DEVICE_NO_SOUND,
        HearingSymptomTopic.DEVICE_LOW_VOLUME,
        HearingSymptomTopic.DEVICE_FAULT,
    }
