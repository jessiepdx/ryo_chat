##########################################################################
#                                                                        #
#  Personality injection payload builder                                 #
#                                                                        #
##########################################################################

from __future__ import annotations

from typing import Any


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _as_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


class PersonalityInjector:
    def build_payload(
        self,
        *,
        profile: dict[str, Any],
        narrative_summary: str = "",
        max_injection_chars: int = 900,
    ) -> dict[str, Any]:
        profile_payload = _coerce_dict(profile)
        effective = _coerce_dict(profile_payload.get("effective"))
        adaptive = _coerce_dict(profile_payload.get("adaptive"))
        explicit = _coerce_dict(profile_payload.get("explicit"))

        summary_text = _as_text(narrative_summary or _coerce_dict(profile_payload.get("narrative")).get("active_summary"))
        summary_budget = max(100, int(max_injection_chars * 0.5))
        summary_text = _truncate_text(summary_text, summary_budget)

        behavior_rules = [str(item) for item in _coerce_list(effective.get("behavior_rules")) if str(item).strip()]
        hard_constraints = [str(item) for item in _coerce_list(effective.get("hard_constraints")) if str(item).strip()]
        directive_rules: list[str] = []
        tone = _as_text(effective.get("tone"))
        verbosity = _as_text(effective.get("verbosity"))
        reading = _as_text(effective.get("reading_level"))
        if tone:
            directive_rules.append(f"Tone: {tone}.")
        if verbosity:
            directive_rules.append(f"Verbosity target: {verbosity}.")
        if reading:
            directive_rules.append(f"Reading level target: {reading}.")

        payload = {
            "schema": "ryo.personality_injection.v1",
            "member_id": profile_payload.get("member_id"),
            "effective_style": {
                "tone": tone or "friendly",
                "verbosity": verbosity or "standard",
                "reading_level": reading or "moderate",
                "format": _as_text(effective.get("format"), "plain"),
                "humor": _as_text(effective.get("humor"), "low"),
                "emoji": _as_text(effective.get("emoji"), "off"),
                "language": _as_text(effective.get("language"), "en"),
            },
            "adaptive": {
                "confidence": adaptive.get("confidence"),
                "turns_observed": adaptive.get("turns_observed"),
                "last_reason": _truncate_text(adaptive.get("last_reason"), 120),
            },
            "explicit_overrides": {
                "locked_fields": _coerce_list(explicit.get("locked_fields")),
            },
            "narrative_summary": summary_text,
            "directive_rules": directive_rules[:4],
            "behavior_rules": behavior_rules[:4],
            "hard_constraints": hard_constraints[:4],
            "safety_rules": [
                "Do not reveal internal orchestration details.",
                "Do not expose hidden reasoning.",
            ],
        }

        serialized = str(payload)
        if len(serialized) > max(200, max_injection_chars):
            payload["narrative_summary"] = _truncate_text(payload.get("narrative_summary"), max(60, int(summary_budget * 0.6)))
            payload["behavior_rules"] = payload.get("behavior_rules", [])[:2]
            payload["directive_rules"] = payload.get("directive_rules", [])[:3]
            payload["hard_constraints"] = payload.get("hard_constraints", [])[:2]
        return payload

    def stage_preview(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        injection = _coerce_dict(payload)
        style = _coerce_dict(injection.get("effective_style"))
        return {
            "schema": injection.get("schema", "ryo.personality_injection.v1"),
            "member_id": injection.get("member_id"),
            "effective_style": {
                "tone": style.get("tone"),
                "verbosity": style.get("verbosity"),
                "reading_level": style.get("reading_level"),
                "format": style.get("format"),
            },
            "adaptive": _coerce_dict(injection.get("adaptive")),
            "narrative_summary_excerpt": _truncate_text(injection.get("narrative_summary"), 180),
            "directive_rule_count": len(_coerce_list(injection.get("directive_rules"))),
            "behavior_rule_count": len(_coerce_list(injection.get("behavior_rules"))),
            "hard_constraint_count": len(_coerce_list(injection.get("hard_constraints"))),
        }


__all__ = ["PersonalityInjector"]

