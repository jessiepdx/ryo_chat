##########################################################################
#                                                                        #
#  Progressive history recall engine                                     #
#                                                                        #
#  Provides a bounded, multi-round deepening strategy to locate          #
#  relevant historical chat entries while keeping context compact.       #
#                                                                        #
##########################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any


def _as_text(value: Any, fallback: str = "") -> str:
    cleaned = str(value if value is not None else "").strip()
    return cleaned if cleaned else fallback


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_datetime_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = _as_text(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _utc_iso(value: Any) -> str | None:
    dt = _coerce_datetime_utc(value)
    if dt is None:
        return None
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9']+", _as_text(value).lower()) if token]


def _quoted_targets(value: str) -> list[str]:
    targets: list[str] = []
    for match in re.findall(r"\"([^\"]+)\"|'([^']+)'", _as_text(value)):
        # re.findall returns tuples because of alternation groups.
        if isinstance(match, tuple):
            text = _as_text(match[0] or match[1]).lower()
        else:
            text = _as_text(match).lower()
        if len(text) >= 3 and text not in targets:
            targets.append(text)
    return targets


def _build_query_targets(query_text: str) -> tuple[set[str], list[str], str]:
    query = _as_text(query_text).lower()
    tokens = set(_tokenize(query))
    phrases = _quoted_targets(query)
    if not phrases and len(tokens) <= 8 and len(query) <= 120:
        # For short queries, use the whole lower-cased query as a weak phrase hint.
        if len(query) >= 4:
            phrases.append(query)
    return tokens, phrases, query


def _lexical_overlap(query_tokens: set[str], candidate_text: str) -> float:
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(candidate_text))
    if not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return float(overlap) / float(max(1, len(query_tokens)))


def _contains_target_phrase(candidate_text: str, query_phrase_hints: list[str]) -> bool:
    lowered = _as_text(candidate_text).lower()
    if not lowered:
        return False
    for phrase in query_phrase_hints:
        if phrase and phrase in lowered:
            return True
    return False


def _candidate_role(record: dict[str, Any]) -> str:
    return "assistant" if record.get("member_id") is None else "user"


@dataclass
class ProgressiveHistoryExplorerConfig:
    enabled: bool = True
    max_rounds: int = 5
    round_windows_hours: tuple[int, ...] = (12, 48, 168, 720)
    semantic_limit_start: int = 3
    semantic_limit_step: int = 2
    timeline_limit_start: int = 24
    timeline_limit_step: int = 24
    context_radius: int = 2
    match_threshold: float = 0.42
    max_selected: int = 8
    max_message_chars: int = 220

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any] | None = None) -> "ProgressiveHistoryExplorerConfig":
        payload = runtime if isinstance(runtime, dict) else {}

        def bool_value(key: str, default: bool) -> bool:
            value = payload.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
            return bool(default)

        def int_value(key: str, default: int, *, minimum: int = 0, maximum: int = 10_000) -> int:
            parsed = _safe_int(payload.get(key), default)
            if parsed < minimum:
                return minimum
            if parsed > maximum:
                return maximum
            return parsed

        def float_value(key: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
            parsed = _safe_float(payload.get(key), default)
            if parsed < minimum:
                return minimum
            if parsed > maximum:
                return maximum
            return parsed

        raw_windows = payload.get("round_windows_hours")
        windows: list[int] = []
        if isinstance(raw_windows, list):
            source = raw_windows
        elif isinstance(raw_windows, str):
            source = [item.strip() for item in raw_windows.split(",")]
        else:
            source = list(cls.round_windows_hours)
        for item in source:
            value = _safe_int(item, 0)
            if value <= 0:
                continue
            if value in windows:
                continue
            windows.append(value)
        if not windows:
            windows = list(cls.round_windows_hours)

        max_rounds = int_value("max_rounds", cls.max_rounds, minimum=1, maximum=12)
        if len(windows) > max_rounds:
            windows = windows[:max_rounds]
        if len(windows) < max_rounds:
            last_window = windows[-1] if windows else 12
            while len(windows) < max_rounds:
                last_window = min(last_window * 2, 24 * 365 * 3)
                windows.append(last_window)

        return cls(
            enabled=bool_value("enabled", True),
            max_rounds=max_rounds,
            round_windows_hours=tuple(windows[:max_rounds]),
            semantic_limit_start=int_value("semantic_limit_start", cls.semantic_limit_start, minimum=1, maximum=64),
            semantic_limit_step=int_value("semantic_limit_step", cls.semantic_limit_step, minimum=0, maximum=32),
            timeline_limit_start=int_value("timeline_limit_start", cls.timeline_limit_start, minimum=1, maximum=500),
            timeline_limit_step=int_value("timeline_limit_step", cls.timeline_limit_step, minimum=0, maximum=500),
            context_radius=int_value("context_radius", cls.context_radius, minimum=1, maximum=8),
            match_threshold=float_value("match_threshold", cls.match_threshold, minimum=0.05, maximum=0.98),
            max_selected=int_value("max_selected", cls.max_selected, minimum=1, maximum=24),
            max_message_chars=int_value("max_message_chars", cls.max_message_chars, minimum=40, maximum=1000),
        )


class ProgressiveHistoryExplorer:
    """
    Iterative deepening recall strategy for chat history:
    - Expands time window in rounds
    - Blends semantic retrieval + lexical matching
    - Returns compact stacked context around the best target hit
    """

    def __init__(self, chat_history_manager: Any, config: ProgressiveHistoryExplorerConfig):
        self._chat_history = chat_history_manager
        self._config = config

    def _score_record(
        self,
        *,
        query_tokens: set[str],
        query_phrase_hints: list[str],
        query_lower: str,
        record: dict[str, Any],
        semantic_distance: float | None = None,
    ) -> dict[str, Any]:
        message_text = _as_text(record.get("message_text"))
        lexical = _lexical_overlap(query_tokens, message_text)
        phrase_hit = _contains_target_phrase(message_text, query_phrase_hints)
        raw_contains_query = bool(query_lower and len(query_lower) >= 4 and query_lower in message_text.lower())
        semantic_score = 0.0
        if semantic_distance is not None and semantic_distance >= 0:
            semantic_score = max(0.0, 1.0 / (1.0 + semantic_distance))

        score = (semantic_score * 0.52) + (lexical * 0.48)
        if phrase_hit:
            score += 0.25
        elif raw_contains_query:
            score += 0.15

        return {
            "history_id": record.get("history_id"),
            "message_id": record.get("message_id"),
            "role": _candidate_role(record),
            "timestamp_utc": _utc_iso(record.get("message_timestamp")),
            "message_text": message_text,
            "score": round(score, 4),
            "signals": {
                "lexical_overlap": round(lexical, 4),
                "semantic_score": round(semantic_score, 4),
                "phrase_hit": bool(phrase_hit),
                "contains_query": bool(raw_contains_query),
            },
        }

    def _merge_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[Any, dict[str, Any]] = {}
        for entry in candidates:
            history_id = entry.get("history_id")
            if history_id is None:
                continue
            current = deduped.get(history_id)
            if not isinstance(current, dict):
                deduped[history_id] = dict(entry)
                continue
            if _safe_float(entry.get("score"), 0.0) > _safe_float(current.get("score"), 0.0):
                deduped[history_id] = dict(entry)
        ordered = list(deduped.values())
        ordered.sort(key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)
        return ordered

    def _stack_context_around_target(
        self,
        *,
        timeline: list[dict[str, Any]],
        target_history_id: Any,
        fallback_entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        if not timeline:
            compact = []
            for index, entry in enumerate(fallback_entries[: self._config.max_selected]):
                compact.append(
                    {
                        "index": index,
                        "history_id": entry.get("history_id"),
                        "message_id": entry.get("message_id"),
                        "role": _as_text(entry.get("role"), "user"),
                        "timestamp_utc": entry.get("timestamp_utc"),
                        "message_text": _as_text(entry.get("message_text"))[: self._config.max_message_chars],
                        "score": entry.get("score"),
                        "signals": entry.get("signals"),
                    }
                )
            return compact, 0

        timeline_index_map: dict[Any, int] = {}
        for idx, row in enumerate(timeline):
            history_id = row.get("history_id")
            if history_id is None:
                continue
            timeline_index_map[history_id] = idx

        if target_history_id not in timeline_index_map:
            # Fall back to compact candidate list if target isn't inside fetched timeline.
            compact = []
            for index, entry in enumerate(fallback_entries[: self._config.max_selected]):
                compact.append(
                    {
                        "index": index,
                        "history_id": entry.get("history_id"),
                        "message_id": entry.get("message_id"),
                        "role": _as_text(entry.get("role"), "user"),
                        "timestamp_utc": entry.get("timestamp_utc"),
                        "message_text": _as_text(entry.get("message_text"))[: self._config.max_message_chars],
                        "score": entry.get("score"),
                        "signals": entry.get("signals"),
                    }
                )
            return compact, 0

        target_idx = timeline_index_map[target_history_id]
        start = max(0, target_idx - self._config.context_radius)
        end = min(len(timeline), target_idx + self._config.context_radius + 1)
        context_rows = timeline[start:end]
        context_index = target_idx - start

        stacked: list[dict[str, Any]] = []
        for idx, row in enumerate(context_rows):
            stacked.append(
                {
                    "index": idx,
                    "history_id": row.get("history_id"),
                    "message_id": row.get("message_id"),
                    "role": _candidate_role(row),
                    "timestamp_utc": _utc_iso(row.get("message_timestamp")),
                    "message_text": _as_text(row.get("message_text"))[: self._config.max_message_chars],
                    "score": None,
                    "signals": {},
                }
            )
        return stacked[: self._config.max_selected], context_index

    def explore(
        self,
        *,
        query_text: str,
        chat_host_id: Any,
        chat_type: str,
        platform: str,
        topic_id: Any = None,
        history_recall_requested: bool = False,
        switched: bool = False,
        allow_history_search: bool = True,
    ) -> dict[str, Any]:
        query = _as_text(query_text)
        if not self._config.enabled:
            return {
                "schema": "ryo.progressive_history_recall.v1",
                "enabled": False,
                "found": False,
                "reason": "disabled",
                "rounds": [],
                "selected": [],
                "selected_count": 0,
            }
        if not query or chat_host_id is None or not _as_text(chat_type) or not _as_text(platform):
            return {
                "schema": "ryo.progressive_history_recall.v1",
                "enabled": True,
                "found": False,
                "reason": "insufficient_scope_or_query",
                "rounds": [],
                "selected": [],
                "selected_count": 0,
            }
        if not allow_history_search:
            return {
                "schema": "ryo.progressive_history_recall.v1",
                "enabled": True,
                "found": False,
                "reason": "history_search_not_allowed",
                "rounds": [],
                "selected": [],
                "selected_count": 0,
            }

        query_tokens, phrase_hints, query_lower = _build_query_targets(query)
        rounds: list[dict[str, Any]] = []
        best_entries: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None
        best_timeline: list[dict[str, Any]] = []
        found = False
        found_round = 0

        # Lower threshold slightly for explicit historical recall asks.
        target_threshold = self._config.match_threshold
        if history_recall_requested:
            target_threshold = max(0.2, target_threshold * 0.82)

        for round_index, window_hours in enumerate(self._config.round_windows_hours[: self._config.max_rounds], start=1):
            semantic_limit = self._config.semantic_limit_start + (round_index - 1) * self._config.semantic_limit_step
            timeline_limit = self._config.timeline_limit_start + (round_index - 1) * self._config.timeline_limit_step
            semantic_limit = max(1, semantic_limit)
            timeline_limit = max(4, timeline_limit)

            scope_topic = bool(topic_id) and (round_index == 1 or (round_index == 2 and switched))
            scoped_topic_id = topic_id if scope_topic else None
            semantic_results = []
            timeline_results = []
            round_candidates: list[dict[str, Any]] = []

            try:
                semantic_results = self._chat_history.searchChatHistory(
                    text=query,
                    limit=semantic_limit,
                    chatHostID=chat_host_id,
                    chatType=chat_type,
                    platform=platform,
                    topicID=scoped_topic_id,
                    scopeTopic=scope_topic,
                    timeInHours=window_hours,
                ) or []
            except Exception:  # noqa: BLE001
                semantic_results = []

            for row in semantic_results:
                record = row if isinstance(row, dict) else {}
                round_candidates.append(
                    self._score_record(
                        query_tokens=query_tokens,
                        query_phrase_hints=phrase_hints,
                        query_lower=query_lower,
                        record=record,
                        semantic_distance=_safe_float(record.get("distance"), 1.0),
                    )
                )

            try:
                timeline_results = self._chat_history.getChatHistory(
                    chatHostID=chat_host_id,
                    chatType=chat_type,
                    platform=platform,
                    topicID=scoped_topic_id,
                    timeInHours=window_hours,
                    limit=timeline_limit,
                ) or []
            except Exception:  # noqa: BLE001
                timeline_results = []

            for row in timeline_results:
                record = row if isinstance(row, dict) else {}
                lexical = _lexical_overlap(query_tokens, _as_text(record.get("message_text")))
                phrase_hit = _contains_target_phrase(_as_text(record.get("message_text")), phrase_hints)
                contains_query = bool(query_lower and len(query_lower) >= 4 and query_lower in _as_text(record.get("message_text")).lower())
                # Keep lexical-only timeline additions bounded to meaningful candidates.
                if lexical < 0.12 and not phrase_hit and not contains_query:
                    continue
                round_candidates.append(
                    self._score_record(
                        query_tokens=query_tokens,
                        query_phrase_hints=phrase_hints,
                        query_lower=query_lower,
                        record=record,
                        semantic_distance=None,
                    )
                )

            deduped = self._merge_candidates(round_candidates)
            best_in_round = deduped[0] if deduped else None
            round_found = False
            if isinstance(best_in_round, dict):
                best_score = _safe_float(best_in_round.get("score"), 0.0)
                signals = best_in_round.get("signals")
                signals_map = signals if isinstance(signals, dict) else {}
                round_found = (
                    best_score >= target_threshold
                    and (
                        bool(signals_map.get("phrase_hit"))
                        or bool(signals_map.get("contains_query"))
                        or _safe_float(signals_map.get("lexical_overlap"), 0.0) >= 0.2
                        or _safe_float(signals_map.get("semantic_score"), 0.0) >= 0.62
                    )
                )

            rounds.append(
                {
                    "round": round_index,
                    "time_window_hours": int(window_hours),
                    "scope_topic": bool(scope_topic),
                    "semantic_limit": int(semantic_limit),
                    "timeline_limit": int(timeline_limit),
                    "semantic_hits": len(semantic_results),
                    "timeline_hits": len(timeline_results),
                    "candidate_count": len(deduped),
                    "best_history_id": None if not best_in_round else best_in_round.get("history_id"),
                    "best_score": None if not best_in_round else best_in_round.get("score"),
                    "found": bool(round_found),
                }
            )

            if deduped:
                if best_entry is None or _safe_float(deduped[0].get("score"), 0.0) > _safe_float(best_entry.get("score"), 0.0):
                    best_entry = deduped[0]
                    best_entries = deduped
                    best_timeline = list(timeline_results)

            if round_found:
                found = True
                found_round = round_index
                break

        if best_entry is None:
            return {
                "schema": "ryo.progressive_history_recall.v1",
                "enabled": True,
                "found": False,
                "reason": "no_candidates",
                "rounds": rounds,
                "selected": [],
                "selected_count": 0,
            }

        target_history_id = best_entry.get("history_id")
        target_score = _safe_float(best_entry.get("score"), 0.0)
        if not found and history_recall_requested and target_score >= max(0.22, target_threshold * 0.72):
            # For explicit recall intent, accept the best available candidate after bounded deepening.
            found = True
            if found_round <= 0:
                found_round = len(rounds)

        stacked_context, context_index = self._stack_context_around_target(
            timeline=best_timeline,
            target_history_id=target_history_id,
            fallback_entries=best_entries,
        )

        selected: list[dict[str, Any]] = []
        for row in stacked_context[: self._config.max_selected]:
            selected.append(
                {
                    "history_id": row.get("history_id"),
                    "message_id": row.get("message_id"),
                    "role": row.get("role"),
                    "timestamp_utc": row.get("timestamp_utc"),
                    "message_text": _as_text(row.get("message_text"))[: self._config.max_message_chars],
                    "score": row.get("score"),
                    "signals": row.get("signals") if isinstance(row.get("signals"), dict) else {},
                }
            )

        decision_reason = (
            "target_found_with_progressive_rounds"
            if found
            else "best_effort_progressive_recall_no_confident_target"
        )
        if history_recall_requested and found:
            decision_reason = "explicit_history_recall_with_progressive_match"

        return {
            "schema": "ryo.progressive_history_recall.v1",
            "enabled": True,
            "found": bool(found),
            "found_round": int(found_round) if found_round > 0 else None,
            "decision_reason": decision_reason,
            "target_history_id": target_history_id,
            "target_message_id": best_entry.get("message_id"),
            "target_context_index": int(context_index),
            "target_score": round(target_score, 4),
            "query": query,
            "rounds": rounds,
            "selected": selected,
            "selected_count": len(selected),
            "stacked_context": stacked_context[: self._config.max_selected],
        }
