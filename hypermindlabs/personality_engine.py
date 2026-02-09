##########################################################################
#                                                                        #
#  Personality adaptation engine                                         #
#                                                                        #
##########################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any


_VERBOSITY_LEVELS = ("brief", "standard", "detailed", "aspie", "exhaustive", "savant")
_READING_LEVELS = ("simple", "moderate", "advanced")
_TONE_LEVELS = ("friendly", "annoyed", "sarcastic", "aggressive", "vile")
_FORMAT_LEVELS = ("plain", "markdown_light")
_EMOJI_LEVELS = ("off", "minimal", "normal")
_HUMOR_LEVELS = ("low", "medium", "high")
_TONE_ALIASES = {
    "friendly": "friendly",
    "annoyed": "annoyed",
    "sarcastic": "sarcastic",
    "aggressive": "aggressive",
    "vile": "vile",
    # Legacy/alternate tone labels mapped into the new ladder.
    "professional": "friendly",
    "neutral": "friendly",
    "energetic": "friendly",
    "direct": "annoyed",
    "assertive": "annoyed",
    "harsh": "aggressive",
    "angry": "aggressive",
    "toxic": "vile",
}
_PROFANITY_TOKENS = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "damn",
    "bullshit",
    "cunt",
    "bastard",
}
_INSULT_TOKENS = {
    "idiot",
    "stupid",
    "moron",
    "dumb",
    "clown",
    "trash",
    "garbage",
    "loser",
}
_SARCASM_MARKERS = (
    "yeah right",
    "sure buddy",
    "as if",
    "/s",
    "great job genius",
)


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(fallback)
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return bool(fallback)


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_enum(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
    normalized = _as_text(value, fallback).lower()
    return normalized if normalized in allowed else fallback


def _normalize_tone(value: Any, fallback: str = "friendly") -> str:
    fallback_normalized = _TONE_ALIASES.get(_as_text(fallback, "friendly").lower(), "friendly")
    candidate = _as_text(value, fallback_normalized).lower()
    mapped = _TONE_ALIASES.get(candidate, candidate)
    return mapped if mapped in _TONE_LEVELS else fallback_normalized


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", _as_text(text))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _level_index(levels: tuple[str, ...], label: str) -> int:
    normalized = _as_text(label).lower()
    try:
        return levels.index(normalized)
    except ValueError:
        return 0


def _step_towards(levels: tuple[str, ...], current: str, target: str, max_step: int) -> str:
    current_idx = _level_index(levels, current)
    target_idx = _level_index(levels, target)
    if current_idx == target_idx:
        return levels[current_idx]
    step = max(1, int(max_step))
    if target_idx > current_idx:
        return levels[min(target_idx, current_idx + step)]
    return levels[max(target_idx, current_idx - step)]


def _tone_from_hostility_score(score: float) -> str:
    value = _clamp(float(score), 0.0, 1.0)
    if value >= 0.86:
        return "vile"
    if value >= 0.66:
        return "aggressive"
    if value >= 0.46:
        return "sarcastic"
    if value >= 0.24:
        return "annoyed"
    return "friendly"


@dataclass
class PersonalityRuntimeConfig:
    enabled: bool = True
    adaptive_enabled: bool = True
    narrative_enabled: bool = True
    max_injection_chars: int = 900
    narrative_summary_max_chars: int = 360
    narrative_source_history_limit: int = 8
    rollup_turn_threshold: int = 8
    rollup_char_threshold: int = 2200
    max_active_chunks: int = 6
    adaptation_min_turns: int = 4
    adaptation_max_step_per_window: int = 1
    adaptation_window_turns: int = 4
    verbosity_short_token_threshold: int = 10
    verbosity_long_token_threshold: int = 36
    default_tone: str = "friendly"
    default_verbosity: str = "brief"
    default_reading_level: str = "moderate"
    default_format: str = "plain"
    default_humor: str = "low"
    default_emoji: str = "off"
    default_language: str = "en"

    @classmethod
    def from_runtime(cls, runtime_payload: dict[str, Any] | None = None) -> "PersonalityRuntimeConfig":
        payload = runtime_payload if isinstance(runtime_payload, dict) else {}
        return cls(
            enabled=_as_bool(payload.get("enabled"), True),
            adaptive_enabled=_as_bool(payload.get("adaptive_enabled"), True),
            narrative_enabled=_as_bool(payload.get("narrative_enabled"), True),
            max_injection_chars=max(200, _as_int(payload.get("max_injection_chars"), 900)),
            narrative_summary_max_chars=max(120, _as_int(payload.get("narrative_summary_max_chars"), 360)),
            narrative_source_history_limit=max(3, _as_int(payload.get("narrative_source_history_limit"), 8)),
            rollup_turn_threshold=max(2, _as_int(payload.get("rollup_turn_threshold"), 8)),
            rollup_char_threshold=max(300, _as_int(payload.get("rollup_char_threshold"), 2200)),
            max_active_chunks=max(1, _as_int(payload.get("max_active_chunks"), 6)),
            adaptation_min_turns=max(1, _as_int(payload.get("adaptation_min_turns"), 4)),
            adaptation_max_step_per_window=max(1, _as_int(payload.get("adaptation_max_step_per_window"), 1)),
            adaptation_window_turns=max(1, _as_int(payload.get("adaptation_window_turns"), 4)),
            verbosity_short_token_threshold=max(3, _as_int(payload.get("verbosity_short_token_threshold"), 10)),
            verbosity_long_token_threshold=max(8, _as_int(payload.get("verbosity_long_token_threshold"), 36)),
            default_tone=_normalize_tone(payload.get("default_tone"), "friendly"),
            default_verbosity=_normalize_enum(payload.get("default_verbosity"), _VERBOSITY_LEVELS, "brief"),
            default_reading_level=_normalize_enum(payload.get("default_reading_level"), _READING_LEVELS, "moderate"),
            default_format=_normalize_enum(payload.get("default_format"), _FORMAT_LEVELS, "plain"),
            default_humor=_normalize_enum(payload.get("default_humor"), _HUMOR_LEVELS, "low"),
            default_emoji=_normalize_enum(payload.get("default_emoji"), _EMOJI_LEVELS, "off"),
            default_language=_as_text(payload.get("default_language"), "en")[:12] or "en",
        )


class PersonalityEngine:
    def _default_profile(self, member_id: int, config: PersonalityRuntimeConfig) -> dict[str, Any]:
        return {
            "schema": "ryo.personality_profile.v1",
            "member_id": int(member_id),
            "explicit": {
                "tone": config.default_tone,
                "verbosity": config.default_verbosity,
                "format": config.default_format,
                "reading_level": config.default_reading_level,
                "humor": config.default_humor,
                "emoji": config.default_emoji,
                "language": config.default_language,
                "hard_constraints": [],
                "locked_fields": [],
            },
            "adaptive": {
                "tone": config.default_tone,
                "verbosity": config.default_verbosity,
                "reading_level": config.default_reading_level,
                "confidence": 0.0,
                "turns_observed": 0,
                "last_reason": "initial_defaults",
                "verbosity_score": 0.25,
                "reading_score": 0.45,
                "last_adapt_turn": 0,
            },
            "narrative": {
                "active_summary": "",
                "chunk_count": 0,
                "last_chunk_index": 0,
                "last_rollup_at": "",
                "turns_since_rollup": 0,
                "chars_since_rollup": 0,
            },
            "effective": {
                "tone": config.default_tone,
                "verbosity": config.default_verbosity,
                "format": config.default_format,
                "reading_level": config.default_reading_level,
                "humor": config.default_humor,
                "emoji": config.default_emoji,
                "language": config.default_language,
                "hard_constraints": [],
                "behavior_rules": [],
            },
        }

    def _merge_locked_fields(self, explicit: dict[str, Any], locked_fields: list[Any]) -> list[str]:
        merged: list[str] = []
        for source in (_coerce_list(explicit.get("locked_fields")), locked_fields):
            for item in source:
                field = _as_text(item).lower()
                if not field or field in merged:
                    continue
                merged.append(field)
        return merged

    def _compute_effective(self, profile: dict[str, Any], config: PersonalityRuntimeConfig) -> dict[str, Any]:
        explicit = _coerce_dict(profile.get("explicit"))
        adaptive = _coerce_dict(profile.get("adaptive"))
        locked_fields = self._merge_locked_fields(explicit, _coerce_list(profile.get("locked_fields")))
        profile["locked_fields"] = locked_fields
        explicit["locked_fields"] = list(locked_fields)

        def _pick_style(field: str, default_value: str, normalizer) -> str:
            explicit_value = explicit.get(field)
            adaptive_value = adaptive.get(field)
            if field in locked_fields:
                return normalizer(explicit_value, default_value)
            if adaptive_value is not None and _as_text(adaptive_value):
                return normalizer(adaptive_value, default_value)
            return normalizer(explicit_value, default_value)

        effective: dict[str, Any] = {
            "tone": _pick_style("tone", config.default_tone, _normalize_tone),
            "verbosity": _pick_style(
                "verbosity",
                config.default_verbosity,
                lambda value, fallback: _normalize_enum(value, _VERBOSITY_LEVELS, fallback),
            ),
            "format": _normalize_enum(explicit.get("format"), _FORMAT_LEVELS, config.default_format),
            "reading_level": _pick_style(
                "reading_level",
                config.default_reading_level,
                lambda value, fallback: _normalize_enum(value, _READING_LEVELS, fallback),
            ),
            "humor": _normalize_enum(explicit.get("humor"), _HUMOR_LEVELS, config.default_humor),
            "emoji": _normalize_enum(explicit.get("emoji"), _EMOJI_LEVELS, config.default_emoji),
            "language": _as_text(explicit.get("language"), config.default_language)[:12] or config.default_language,
            "hard_constraints": [str(item) for item in _coerce_list(explicit.get("hard_constraints")) if str(item).strip()],
        }
        behavior_rules: list[str] = []
        verbosity = effective.get("verbosity")
        if verbosity == "brief":
            behavior_rules.append("Prefer concise replies unless user asks for depth.")
        elif verbosity == "detailed":
            behavior_rules.append("Provide fuller explanations with clear structure.")
        elif verbosity == "aspie":
            behavior_rules.append("Use highly technical detail, explicit assumptions, and precision-first wording.")
        elif verbosity == "exhaustive":
            behavior_rules.append("Cover all major facets, edge cases, and tradeoffs in a structured response.")
        elif verbosity == "savant":
            behavior_rules.append("Deliver maximal depth with rigorous stepwise reasoning and dense technical context.")
        if effective.get("reading_level") == "simple":
            behavior_rules.append("Use simple wording and shorter sentences.")
        if effective.get("emoji") == "off":
            behavior_rules.append("Avoid emoji in normal responses.")
        effective["behavior_rules"] = behavior_rules[:4]
        profile["effective"] = effective
        profile["explicit"] = explicit
        profile["adaptive"] = adaptive
        return profile

    def resolve_profile(
        self,
        *,
        member_id: int,
        stored_profile: dict[str, Any] | None,
        runtime_config: PersonalityRuntimeConfig,
        latest_narrative_summary: str = "",
        narrative_chunk_count: int = 0,
        last_chunk_index: int = 0,
    ) -> dict[str, Any]:
        profile = self._default_profile(member_id, runtime_config)
        row = stored_profile if isinstance(stored_profile, dict) else {}

        explicit = _coerce_dict(row.get("explicit_directive_json"))
        adaptive_state = _coerce_dict(row.get("adaptive_state_json"))
        effective_profile = _coerce_dict(row.get("effective_profile_json"))
        locked_fields = _coerce_list(row.get("locked_fields_json"))

        profile["explicit"].update(explicit)
        profile["adaptive"].update(adaptive_state)
        if isinstance(effective_profile, dict) and effective_profile:
            profile["effective"].update(effective_profile)

        narrative = _coerce_dict(profile.get("narrative"))
        narrative.update(_coerce_dict(adaptive_state.get("narrative")))
        if latest_narrative_summary:
            narrative["active_summary"] = _as_text(latest_narrative_summary)
        narrative["chunk_count"] = max(_as_int(narrative.get("chunk_count"), 0), int(narrative_chunk_count))
        narrative["last_chunk_index"] = max(_as_int(narrative.get("last_chunk_index"), 0), int(last_chunk_index))
        profile["narrative"] = narrative
        profile["locked_fields"] = locked_fields
        profile["profile_version"] = max(1, _as_int(row.get("profile_version"), 1))
        profile["updated_at"] = _as_text(row.get("updated_at"))
        return self._compute_effective(profile, runtime_config)

    def profile_for_storage(self, profile: dict[str, Any]) -> dict[str, Any]:
        payload = _coerce_dict(profile)
        adaptive = _coerce_dict(payload.get("adaptive"))
        narrative = _coerce_dict(payload.get("narrative"))
        adaptive["narrative"] = narrative
        return {
            "explicit_directive_json": _coerce_dict(payload.get("explicit")),
            "adaptive_state_json": adaptive,
            "effective_profile_json": _coerce_dict(payload.get("effective")),
            "locked_fields_json": _coerce_list(payload.get("locked_fields")),
        }

    def _signals(
        self,
        *,
        user_message: str,
        analysis_payload: dict[str, Any] | None = None,
        runtime_config: PersonalityRuntimeConfig,
    ) -> dict[str, Any]:
        text = _as_text(user_message)
        tokens = _tokenize(text)
        token_count = len(tokens)
        char_count = len(text)
        avg_token_len = (sum(len(token) for token in tokens) / max(1, token_count)) if token_count else 0.0
        punctuation_count = len(re.findall(r"[!?.,;:]", text))
        punctuation_density = punctuation_count / max(1, char_count)
        letters = re.findall(r"[A-Za-z]", text)
        uppercase_letters = [char for char in letters if char.isupper()]
        uppercase_ratio = len(uppercase_letters) / max(1, len(letters))
        lowered_text = text.lower()
        lowered_tokens = [token.lower() for token in tokens]
        profanity_hits = sum(1 for token in lowered_tokens if token in _PROFANITY_TOKENS)
        insult_hits = sum(1 for token in lowered_tokens if token in _INSULT_TOKENS)
        sarcasm_hits = sum(1 for marker in _SARCASM_MARKERS if marker in lowered_text)

        short_threshold = max(2, runtime_config.verbosity_short_token_threshold)
        long_threshold = max(short_threshold + 1, runtime_config.verbosity_long_token_threshold)
        verbosity_score = _clamp((token_count - short_threshold) / max(1, long_threshold - short_threshold), 0.0, 1.0)
        reading_score = _clamp(((avg_token_len - 3.0) / 5.0) + (punctuation_density * 1.8), 0.0, 1.0)
        hostility_score = 0.0
        hostility_score += min(0.7, (profanity_hits * 0.2) + (insult_hits * 0.25))
        hostility_score += min(0.2, text.count("!") * 0.04)
        hostility_score += max(0.0, (uppercase_ratio - 0.45) * 1.1)
        if sarcasm_hits > 0:
            hostility_score = max(hostility_score, 0.48)
        hostility_score = _clamp(hostility_score, 0.0, 1.0)
        tone_target = _tone_from_hostility_score(hostility_score)

        analysis = _coerce_dict(analysis_payload)
        response_style = _coerce_dict(analysis.get("response_style"))
        length_hint = _as_text(response_style.get("length")).lower()
        if length_hint in {"very_short", "short", "concise"}:
            verbosity_score = min(verbosity_score, 0.25)
        elif length_hint in {"detailed", "long", "medium"}:
            verbosity_score = max(verbosity_score, 0.78)
        elif length_hint in {"exhaustive", "savant", "very_long", "deep"}:
            verbosity_score = max(verbosity_score, 0.95)

        complexity_hint = "moderate"
        if reading_score < 0.33:
            complexity_hint = "simple"
        elif reading_score >= 0.72:
            complexity_hint = "advanced"
        return {
            "token_count": token_count,
            "char_count": char_count,
            "avg_token_len": round(avg_token_len, 3),
            "punctuation_density": round(punctuation_density, 4),
            "verbosity_score_target": round(verbosity_score, 4),
            "reading_score_target": round(reading_score, 4),
            "hostility_score_target": round(hostility_score, 4),
            "tone_target": tone_target,
            "complexity_hint": complexity_hint,
            "length_hint": length_hint or "unknown",
        }

    def _score_to_verbosity(self, score: float) -> str:
        if score < 0.2:
            return "brief"
        if score < 0.4:
            return "standard"
        if score < 0.58:
            return "detailed"
        if score < 0.74:
            return "aspie"
        if score < 0.88:
            return "exhaustive"
        return "savant"

    def _score_to_reading_level(self, score: float) -> str:
        if score < 0.35:
            return "simple"
        if score < 0.72:
            return "moderate"
        return "advanced"

    def adapt_after_turn(
        self,
        *,
        profile: dict[str, Any],
        user_message: str,
        assistant_message: str,
        analysis_payload: dict[str, Any] | None,
        runtime_config: PersonalityRuntimeConfig,
    ) -> dict[str, Any]:
        working = _coerce_dict(profile)
        working.setdefault("explicit", {})
        working.setdefault("adaptive", {})
        working.setdefault("narrative", {})
        before_effective = _coerce_dict(working.get("effective"))
        before_snapshot = {
            "adaptive": _coerce_dict(working.get("adaptive")),
            "effective": before_effective,
            "narrative": _coerce_dict(working.get("narrative")),
        }

        explicit = _coerce_dict(working.get("explicit"))
        adaptive = _coerce_dict(working.get("adaptive"))
        narrative = _coerce_dict(working.get("narrative"))
        locked_fields = self._merge_locked_fields(explicit, _coerce_list(working.get("locked_fields")))
        working["locked_fields"] = locked_fields

        signals = self._signals(
            user_message=user_message,
            analysis_payload=analysis_payload,
            runtime_config=runtime_config,
        )

        turns_observed = _as_int(adaptive.get("turns_observed"), 0) + 1
        adaptive["turns_observed"] = turns_observed
        window_turns = max(1, runtime_config.adaptation_window_turns)
        alpha = 1.0 / float(window_turns)
        verbosity_score = _clamp(
            (_as_float(adaptive.get("verbosity_score"), 0.25) * (1.0 - alpha))
            + (_as_float(signals.get("verbosity_score_target"), 0.25) * alpha),
            0.0,
            1.0,
        )
        reading_score = _clamp(
            (_as_float(adaptive.get("reading_score"), 0.45) * (1.0 - alpha))
            + (_as_float(signals.get("reading_score_target"), 0.45) * alpha),
            0.0,
            1.0,
        )
        adaptive["verbosity_score"] = round(verbosity_score, 4)
        adaptive["reading_score"] = round(reading_score, 4)

        candidate_verbosity = self._score_to_verbosity(verbosity_score)
        candidate_reading = self._score_to_reading_level(reading_score)
        candidate_tone = _normalize_tone(signals.get("tone_target"), runtime_config.default_tone)
        current_verbosity = _normalize_enum(adaptive.get("verbosity"), _VERBOSITY_LEVELS, runtime_config.default_verbosity)
        current_reading = _normalize_enum(adaptive.get("reading_level"), _READING_LEVELS, runtime_config.default_reading_level)
        current_tone = _normalize_tone(
            adaptive.get("tone"),
            _normalize_tone(explicit.get("tone"), runtime_config.default_tone),
        )
        last_adapt_turn = _as_int(adaptive.get("last_adapt_turn"), 0)
        can_shift = (turns_observed - last_adapt_turn) >= window_turns

        changed_fields: list[str] = []
        if runtime_config.adaptive_enabled and turns_observed >= runtime_config.adaptation_min_turns and can_shift:
            if "tone" not in locked_fields:
                next_tone = _step_towards(
                    _TONE_LEVELS,
                    current_tone,
                    candidate_tone,
                    runtime_config.adaptation_max_step_per_window,
                )
                if next_tone != current_tone:
                    adaptive["tone"] = next_tone
                    changed_fields.append("tone")
            if "verbosity" not in locked_fields:
                next_verbosity = _step_towards(
                    _VERBOSITY_LEVELS,
                    current_verbosity,
                    candidate_verbosity,
                    runtime_config.adaptation_max_step_per_window,
                )
                if next_verbosity != current_verbosity:
                    adaptive["verbosity"] = next_verbosity
                    changed_fields.append("verbosity")
            if "reading_level" not in locked_fields:
                next_reading = _step_towards(
                    _READING_LEVELS,
                    current_reading,
                    candidate_reading,
                    runtime_config.adaptation_max_step_per_window,
                )
                if next_reading != current_reading:
                    adaptive["reading_level"] = next_reading
                    changed_fields.append("reading_level")
            if changed_fields:
                adaptive["last_adapt_turn"] = turns_observed

        adaptive["tone"] = _normalize_tone(adaptive.get("tone"), _normalize_tone(explicit.get("tone"), runtime_config.default_tone))
        confidence = _clamp(float(turns_observed) / float(max(1, runtime_config.adaptation_min_turns * 2)), 0.0, 1.0)
        adaptive["confidence"] = round(confidence, 3)
        adaptive["last_reason"] = "adaptive_signal_update"

        turns_since_rollup = _as_int(narrative.get("turns_since_rollup"), 0) + 1
        chars_since_rollup = _as_int(narrative.get("chars_since_rollup"), 0) + len(_as_text(user_message)) + len(_as_text(assistant_message))
        narrative["turns_since_rollup"] = turns_since_rollup
        narrative["chars_since_rollup"] = chars_since_rollup

        working["adaptive"] = adaptive
        working["narrative"] = narrative
        working = self._compute_effective(working, runtime_config)

        after_snapshot = {
            "adaptive": _coerce_dict(working.get("adaptive")),
            "effective": _coerce_dict(working.get("effective")),
            "narrative": _coerce_dict(working.get("narrative")),
        }
        reason_code = "adaptive_update" if changed_fields else "adaptive_observation"
        reason_detail = (
            "Adjusted profile fields: " + ", ".join(changed_fields)
            if changed_fields
            else "Updated adaptation signal buffers without discrete style shift."
        )
        return {
            "profile": working,
            "changed_fields": changed_fields,
            "signals": signals,
            "reason_code": reason_code,
            "reason_detail": reason_detail,
            "before": before_snapshot,
            "after": after_snapshot,
        }

    def apply_analysis_style(
        self,
        *,
        analysis_payload: dict[str, Any] | None,
        profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = _coerce_dict(analysis_payload)
        persona = _coerce_dict(profile)
        effective = _coerce_dict(persona.get("effective"))
        response_style = _coerce_dict(payload.get("response_style"))
        tone = _as_text(effective.get("tone"))
        verbosity = _as_text(effective.get("verbosity"))

        if tone:
            response_style["tone"] = tone
        if verbosity:
            response_style["verbosity"] = verbosity
        if verbosity == "brief":
            response_style["length"] = "concise"
        elif verbosity in {"detailed", "aspie", "exhaustive", "savant"}:
            response_style["length"] = "detailed"
        elif not _as_text(response_style.get("length")):
            response_style["length"] = "medium"
        payload["response_style"] = response_style
        return payload

    def apply_explicit_directive(
        self,
        *,
        profile: dict[str, Any],
        directive: dict[str, Any] | None,
        runtime_config: PersonalityRuntimeConfig,
    ) -> dict[str, Any]:
        payload = _coerce_dict(profile)
        explicit = _coerce_dict(payload.get("explicit"))
        incoming = _coerce_dict(directive)
        if not incoming:
            return payload

        if "tone" in incoming:
            explicit["tone"] = _normalize_tone(incoming.get("tone"), runtime_config.default_tone)

        for field, allowed, fallback in (
            ("verbosity", _VERBOSITY_LEVELS, runtime_config.default_verbosity),
            ("format", _FORMAT_LEVELS, runtime_config.default_format),
            ("reading_level", _READING_LEVELS, runtime_config.default_reading_level),
            ("humor", _HUMOR_LEVELS, runtime_config.default_humor),
            ("emoji", _EMOJI_LEVELS, runtime_config.default_emoji),
        ):
            if field not in incoming:
                continue
            explicit[field] = _normalize_enum(incoming.get(field), allowed, fallback)
        if "language" in incoming:
            explicit["language"] = _as_text(incoming.get("language"), runtime_config.default_language)[:12] or runtime_config.default_language
        if "hard_constraints" in incoming:
            explicit["hard_constraints"] = [
                str(item)
                for item in _coerce_list(incoming.get("hard_constraints"))
                if str(item).strip()
            ][:8]
        if "locked_fields" in incoming:
            explicit["locked_fields"] = [
                _as_text(item).lower()
                for item in _coerce_list(incoming.get("locked_fields"))
                if _as_text(item).strip()
            ][:12]

        payload["explicit"] = explicit
        return self._compute_effective(payload, runtime_config)


__all__ = [
    "PersonalityEngine",
    "PersonalityRuntimeConfig",
]
