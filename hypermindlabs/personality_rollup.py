##########################################################################
#                                                                        #
#  Narrative rollup helper                                               #
#                                                                        #
##########################################################################

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any


def _as_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class NarrativeRollupEngine:
    def should_rollup(self, *, profile: dict[str, Any], turn_threshold: int, char_threshold: int) -> bool:
        payload = _coerce_dict(profile)
        narrative = _coerce_dict(payload.get("narrative"))
        turns_since_rollup = _as_int(narrative.get("turns_since_rollup"), 0)
        chars_since_rollup = _as_int(narrative.get("chars_since_rollup"), 0)
        if turns_since_rollup >= max(1, int(turn_threshold)):
            return True
        if chars_since_rollup >= max(200, int(char_threshold)):
            return True
        return False

    def _history_excerpt_lines(
        self,
        *,
        history_messages: list[dict[str, Any]],
        max_messages: int,
        max_chars_per_line: int = 160,
    ) -> list[str]:
        records = history_messages[-max_messages:] if max_messages > 0 else history_messages
        lines: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            role = "assistant" if record.get("member_id") is None else "user"
            prefix = "Assistant" if role == "assistant" else "User"
            text = _truncate_text(record.get("message_text"), max_chars_per_line)
            if not text:
                continue
            lines.append(f"{prefix}: {text}")
        return lines

    def _extract_key_terms(self, text: str, max_terms: int = 6) -> list[str]:
        words = re.findall(r"[A-Za-z0-9']+", _as_text(text).lower())
        counts: dict[str, int] = {}
        for word in words:
            if len(word) <= 3:
                continue
            counts[word] = counts.get(word, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [term for term, _ in ordered[:max_terms]]

    def build_rollup(
        self,
        *,
        profile: dict[str, Any],
        history_messages: list[dict[str, Any]],
        user_message: str,
        assistant_message: str,
        analysis_payload: dict[str, Any] | None,
        max_source_messages: int,
        max_summary_chars: int,
    ) -> dict[str, Any]:
        payload = _coerce_dict(profile)
        narrative = _coerce_dict(payload.get("narrative"))
        analysis = _coerce_dict(analysis_payload)

        lines = self._history_excerpt_lines(
            history_messages=history_messages if isinstance(history_messages, list) else [],
            max_messages=max(2, max_source_messages),
            max_chars_per_line=150,
        )
        if _as_text(user_message):
            lines.append(f"User: {_truncate_text(user_message, 180)}")
        if _as_text(assistant_message):
            lines.append(f"Assistant: {_truncate_text(assistant_message, 220)}")

        topic = _as_text(analysis.get("topic"), "general")
        intent = _as_text(analysis.get("intent"), "respond")
        preamble = f"Topic={topic}; intent={intent}. "
        combined = preamble + " | ".join(lines[-max(2, max_source_messages):])
        summary_text = _truncate_text(combined, max(120, max_summary_chars))

        source_char_count = sum(len(line) for line in lines) + len(preamble)
        summary_char_count = max(1, len(summary_text))
        compression_ratio = round(float(source_char_count) / float(summary_char_count), 3)

        next_chunk_index = _as_int(narrative.get("last_chunk_index"), 0) + 1
        chunk_payload = {
            "topic": topic,
            "intent": intent,
            "key_terms": self._extract_key_terms(" ".join(lines[-6:])),
            "line_count": len(lines),
            "source_char_count": source_char_count,
            "summary_char_count": summary_char_count,
            "created_at": _utc_now_iso(),
        }
        return {
            "chunk_index": next_chunk_index,
            "summary_text": summary_text,
            "summary_json": chunk_payload,
            "compression_ratio": compression_ratio,
            "line_count": len(lines),
            "source_char_count": source_char_count,
            "summary_char_count": summary_char_count,
        }

    def apply_rollup_to_profile(
        self,
        *,
        profile: dict[str, Any],
        rollup_result: dict[str, Any],
    ) -> dict[str, Any]:
        payload = _coerce_dict(profile)
        narrative = _coerce_dict(payload.get("narrative"))
        narrative["active_summary"] = _as_text(rollup_result.get("summary_text"))
        narrative["chunk_count"] = max(_as_int(narrative.get("chunk_count"), 0), _as_int(rollup_result.get("chunk_index"), 0))
        narrative["last_chunk_index"] = _as_int(rollup_result.get("chunk_index"), 0)
        narrative["last_rollup_at"] = _utc_now_iso()
        narrative["turns_since_rollup"] = 0
        narrative["chars_since_rollup"] = 0
        payload["narrative"] = narrative
        return payload

    def narrative_preview(self, profile: dict[str, Any]) -> dict[str, Any]:
        narrative = _coerce_dict(_coerce_dict(profile).get("narrative"))
        return {
            "chunk_count": _as_int(narrative.get("chunk_count"), 0),
            "last_chunk_index": _as_int(narrative.get("last_chunk_index"), 0),
            "last_rollup_at": _as_text(narrative.get("last_rollup_at")),
            "turns_since_rollup": _as_int(narrative.get("turns_since_rollup"), 0),
            "chars_since_rollup": _as_int(narrative.get("chars_since_rollup"), 0),
            "active_summary_excerpt": _truncate_text(narrative.get("active_summary"), 180),
        }


__all__ = ["NarrativeRollupEngine"]

