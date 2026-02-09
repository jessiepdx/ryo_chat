##########################################################################
#                                                                        #
#  Temporal context helpers for orchestrator message grounding.          #
#                                                                        #
##########################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:  # pragma: no cover - fallback for minimal Python builds
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]


def _resolve_timezone(name: str | None) -> tuple[Any, str]:
    tz_name = str(name or "UTC").strip() or "UTC"
    if tz_name.upper() == "UTC" or ZoneInfo is None:
        return timezone.utc, "UTC"
    try:
        return ZoneInfo(tz_name), tz_name
    except ZoneInfoNotFoundError:
        return timezone.utc, "UTC"


def _parse_datetime_text(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def coerce_datetime_utc(value: Any, assume_tz: Any = timezone.utc) -> datetime | None:
    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_datetime_text(value)
    else:
        parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=assume_tz)
    return parsed.astimezone(timezone.utc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_excerpt(text: Any, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_history_messages(
    history_messages: Iterable[Mapping[str, Any]] | None,
    now_utc: datetime,
    assume_tz: Any,
    history_limit: int,
    excerpt_max_chars: int,
) -> list[dict[str, Any]]:
    if history_limit <= 0:
        return []

    normalized: list[dict[str, Any]] = []
    for record in history_messages or []:
        if not isinstance(record, Mapping):
            continue
        history_ts = coerce_datetime_utc(record.get("message_timestamp"), assume_tz=assume_tz)
        if history_ts is None:
            continue

        member_id = record.get("member_id")
        role = "assistant" if member_id is None else "user"
        age_seconds = max(0, int((now_utc - history_ts).total_seconds()))

        normalized.append(
            {
                "history_id": record.get("history_id"),
                "message_id": record.get("message_id"),
                "member_id": member_id,
                "role": role,
                "timestamp_utc": _iso_utc(history_ts),
                "age_seconds": age_seconds,
                "excerpt": _compact_excerpt(record.get("message_text"), excerpt_max_chars),
            }
        )

    normalized.sort(
        key=lambda item: (
            str(item.get("timestamp_utc", "")),
            _safe_int(item.get("history_id"), 0),
            _safe_int(item.get("message_id"), 0),
        )
    )

    return normalized[-history_limit:]


def build_temporal_context(
    *,
    platform: str | None,
    chat_type: str | None,
    chat_host_id: Any,
    topic_id: Any,
    timezone_name: str | None,
    now_utc: datetime | None = None,
    inbound_sent_at: Any = None,
    inbound_received_at: Any = None,
    history_messages: Iterable[Mapping[str, Any]] | None = None,
    history_limit: int = 20,
    excerpt_max_chars: int = 160,
) -> dict[str, Any]:
    tzinfo, resolved_timezone = _resolve_timezone(timezone_name)
    current_utc = coerce_datetime_utc(now_utc, assume_tz=timezone.utc) or datetime.now(timezone.utc)
    current_utc = current_utc.replace(microsecond=0)
    current_local = current_utc.astimezone(tzinfo).replace(microsecond=0)

    inbound_sent_utc = coerce_datetime_utc(inbound_sent_at, assume_tz=tzinfo)
    inbound_received_utc = coerce_datetime_utc(inbound_received_at, assume_tz=timezone.utc) or current_utc

    history_entries = _normalize_history_messages(
        history_messages=history_messages,
        now_utc=current_utc,
        assume_tz=tzinfo,
        history_limit=max(0, int(history_limit)),
        excerpt_max_chars=max(0, int(excerpt_max_chars)),
    )

    return {
        "schema": "ryo.temporal_context.v1",
        "clock": {
            "now_utc": _iso_utc(current_utc),
            "now_local": current_local.isoformat(),
            "timezone": resolved_timezone,
            "unix_epoch": int(current_utc.timestamp()),
        },
        "inbound": {
            "source": str(platform or "unknown"),
            "message_sent_at_utc": None if inbound_sent_utc is None else _iso_utc(inbound_sent_utc),
            "message_received_at_utc": _iso_utc(inbound_received_utc),
        },
        "timeline": {
            "chat_type": chat_type,
            "chat_host_id": chat_host_id,
            "topic_id": topic_id,
            "recent": history_entries,
        },
    }

