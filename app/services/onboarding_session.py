from typing import Any

from app.services.gemini_service import ChatMessage, OnboardingExtractedField
from app.services.onboarding_fields import ONBOARDING_FIELDS


def field_keys() -> list[str]:
    return [field["key"] for field in ONBOARDING_FIELDS]


def normalize_collected_fields(collected_fields: dict[str, str] | None) -> dict[str, str]:
    if not collected_fields:
        return {}

    allowed_keys = set(field_keys())
    normalized: dict[str, str] = {}
    for key, value in collected_fields.items():
        if key not in allowed_keys:
            continue
        cleaned = value.strip()
        if cleaned:
            normalized[key] = cleaned
    return normalized


def merge_collected_fields(
    existing_fields: dict[str, str] | None,
    extracted_fields: list[OnboardingExtractedField],
) -> dict[str, str]:
    merged = normalize_collected_fields(existing_fields)
    allowed_keys = set(field_keys())

    for item in extracted_fields:
        if item.key not in allowed_keys:
            continue
        cleaned = item.value.strip()
        if cleaned:
            merged[item.key] = cleaned
    return merged


def get_missing_fields(collected_fields: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for field in ONBOARDING_FIELDS:
        if not collected_fields.get(field["key"]):
            missing.append(
                {
                    "key": field["key"],
                    "label": field["label"],
                    "description": field["description"],
                    "question": field["question"],
                }
            )
    return missing


def build_fallback_question(missing_fields: list[dict[str, str]]) -> str:
    if not missing_fields:
        return "기본 정보 확인을 마쳤습니다."
    return missing_fields[0]["question"]


def count_user_turns(chat_history: list[ChatMessage], current_user_prompt: str | None) -> int:
    prior_turns = sum(1 for message in chat_history if message.role == "user")
    return prior_turns + (1 if current_user_prompt and current_user_prompt.strip() else 0)


def build_onboarding_context(
    collected_fields: dict[str, str],
    missing_fields: list[dict[str, str]],
    current_turn: int,
    max_turns: int,
) -> dict[str, Any]:
    return {
        "required_fields": ONBOARDING_FIELDS,
        "collected_fields": collected_fields,
        "missing_fields": missing_fields,
        "current_turn": current_turn,
        "max_turns": max_turns,
        "all_fields_collected": len(missing_fields) == 0,
        "is_last_allowed_turn": current_turn >= max_turns,
    }
