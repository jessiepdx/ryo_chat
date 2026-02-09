##########################################################################
#                                                                        #
#  This file (agents.py) contains the agents modules for Hypermind Labs  #
#                                                                        #
#  Created by:  Jessie W                                                 #
#  Github: jessiepdx                                                     #
#  Contributors:                                                         #
#      Robit                                                             #
#  Created: February 1st, 2025                                           #
#  Modified: April 3rd, 2025                                             #
#                                                                        #
##########################################################################


###########
# IMPORTS #
###########

from datetime import datetime, timedelta, timezone
import inspect
import json
import logging
import re
import requests
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.model_router import ModelExecutionError, ModelRouter
from hypermindlabs.policy_manager import PolicyManager, PolicyValidationError
from hypermindlabs.temporal_context import (
    build_temporal_context,
    coerce_datetime_utc,
    utc_now_iso,
)
from hypermindlabs.tool_registry import (
    ToolRegistryStore,
    build_tool_specs,
    model_tool_definitions,
    normalize_custom_tool_payload,
    register_runtime_tools,
)
from hypermindlabs.tool_sandbox import (
    ToolSandboxEnforcer,
    ToolSandboxPolicyStore,
    merge_sandbox_policies,
    normalize_tool_sandbox_policy,
)
from hypermindlabs.tool_runtime import ToolRuntime
from hypermindlabs.utils import (
    ChatHistoryManager, 
    ConfigManager, 
    ConsoleColors, 
    KnowledgeManager, 
    UsageManager, 
    MemberManager
)
from ollama import AsyncClient, ChatResponse, Message
from pydantic import BaseModel

# Tweepy logic eventual goes into the Twitter / X UI code
import tweepy
import tweepy.asynchronous



###########
# GLOBALS #
###########

chatHistory = ChatHistoryManager()
config = ConfigManager()
knowledge = KnowledgeManager()
members = MemberManager()
usage = UsageManager()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")


def _runtime_int(path: str, default: int) -> int:
    return config.runtimeInt(path, default)


def _runtime_float(path: str, default: float) -> float:
    return config.runtimeFloat(path, default)


def _runtime_value(path: str, default: Any = None) -> Any:
    return config.runtimeValue(path, default)


def _runtime_bool(path: str, default: bool) -> bool:
    return config.runtimeBool(path, default)


def _fallback_stream(message_text: str) -> AsyncIterator[Any]:
    async def _stream():
        yield SimpleNamespace(
            message=SimpleNamespace(content=message_text),
            done=True,
            total_duration=0,
            load_duration=0,
            prompt_eval_count=0,
            prompt_eval_duration=0,
            eval_count=0,
            eval_duration=0,
        )

    return _stream()


def _routing_from_error(error: Exception) -> dict[str, Any]:
    if isinstance(error, ModelExecutionError):
        metadata = getattr(error, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
    return {"errors": [str(error)], "status": "failed_all_candidates"}


_ALLOWED_TOOL_HINTS = {"braveSearch", "chatHistorySearch", "knowledgeSearch", "skipTools"}
_DIAGNOSTIC_REQUEST_PATTERNS = (
    r"\bdebug\b",
    r"\btrace\b",
    r"\binternal\b",
    r"\borchestrat(?:e|ion)\b",
    r"\bstage(?:s)?\b",
    r"\btool call(?:s)?\b",
    r"\bshow .*reasoning\b",
)
_META_LEAK_PATTERNS = (
    r"\bone agent of many agents\b",
    r"\bfuture agents?\b",
    r"\bmessage analysis\b",
    r"\bknown context\b",
    r"\btool results?\b",
    r"\binternal (?:state|notes|thoughts|reasoning)\b",
    r"\bchain[- ]of[- ]thought\b",
)
_TOPIC_SHIFT_PATTERNS = (
    r"\bnew topic\b",
    r"\bchange (?:the )?topic\b",
    r"\bswitch (?:the )?topic\b",
    r"\bdifferent topic\b",
    r"\blet'?s talk about\b",
    r"\blet'?s discuss\b",
    r"\bmove on\b",
    r"\bchange subject\b",
    r"\bstop talking about\b",
)
_HISTORY_RECALL_PATTERNS = (
    r"\bremember\b",
    r"\brecall\b",
    r"\bearlier\b",
    r"\bprevious(?:ly)?\b",
    r"\blast time\b",
    r"\bwhat did i (?:ask|say)\b",
    r"\byou said\b",
    r"\bwhen did i\b",
    r"\bfrom before\b",
)
_TOPIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "have",
    "been",
    "were",
    "your",
    "you",
    "please",
    "just",
    "really",
    "very",
    "then",
    "than",
    "them",
    "they",
    "their",
    "there",
    "here",
    "more",
    "some",
    "want",
    "need",
    "tell",
    "like",
    "give",
    "show",
    "help",
    "make",
    "look",
    "lets",
    "let",
    "talk",
    "topic",
    "change",
    "switch",
    "stop",
    "new",
}


def _parse_json_like(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(value: Any, fallback: str = "") -> str:
    cleaned = str(value if value is not None else "").strip()
    return cleaned if cleaned else fallback


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return fallback


def _as_string_list(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, list):
        source = value
    elif isinstance(value, str):
        source = [value]
    else:
        return output
    for item in source:
        cleaned = _as_text(item)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_message_list(value: Any) -> list[Message]:
    output: list[Message] = []
    if not isinstance(value, list):
        return output

    for item in value:
        role = ""
        content = ""
        if isinstance(item, dict):
            role = _as_text(item.get("role"))
            content = _as_text(item.get("content"))
        else:
            role = _as_text(getattr(item, "role", ""))
            content = _as_text(getattr(item, "content", ""))

        if role and content:
            output.append(Message(role=role, content=content))
    return output


def _stream_log_snippet(value: Any, max_chars: int = 160) -> str:
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _stream_human_preview(value: Any, max_chars: int = 220) -> str:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[-max_chars:]
    return "..." + text[-(max_chars - 3) :]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _duration_seconds_from_ns(value: Any) -> float:
    nanoseconds = _safe_int(value, 0)
    if nanoseconds <= 0:
        return 0.0
    return float(nanoseconds) / 1_000_000_000.0


def _tokens_per_second(tokens: int, duration_seconds: float) -> float | None:
    if tokens <= 0 or duration_seconds <= 0.0:
        return None
    return float(tokens) / float(duration_seconds)


def _ollama_stream_stats_summary(stats: dict[str, Any] | None) -> dict[str, Any]:
    payload = _coerce_dict(stats)
    prompt_tokens = max(0, _safe_int(payload.get("prompt_eval_count"), 0))
    completion_tokens = max(0, _safe_int(payload.get("eval_count"), 0))
    total_tokens = prompt_tokens + completion_tokens

    prompt_seconds = _duration_seconds_from_ns(payload.get("prompt_eval_duration"))
    completion_seconds = _duration_seconds_from_ns(payload.get("eval_duration"))
    total_eval_seconds = prompt_seconds + completion_seconds
    total_duration_seconds = _duration_seconds_from_ns(payload.get("total_duration"))
    load_duration_seconds = _duration_seconds_from_ns(payload.get("load_duration"))

    prompt_tps = _tokens_per_second(prompt_tokens, prompt_seconds)
    completion_tps = _tokens_per_second(completion_tokens, completion_seconds)
    total_tps = _tokens_per_second(total_tokens, total_eval_seconds)

    summary: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_seconds": round(prompt_seconds, 4),
        "completion_seconds": round(completion_seconds, 4),
        "total_eval_seconds": round(total_eval_seconds, 4),
        "total_duration_seconds": round(total_duration_seconds, 4),
        "load_duration_seconds": round(load_duration_seconds, 4),
    }
    if prompt_tps is not None:
        summary["prompt_tokens_per_second"] = round(prompt_tps, 2)
    if completion_tps is not None:
        summary["completion_tokens_per_second"] = round(completion_tps, 2)
    if total_tps is not None:
        summary["total_tokens_per_second"] = round(total_tps, 2)
    return summary


def _stage_meta_log_summary(meta: dict[str, Any] | None) -> dict[str, Any]:
    payload = _coerce_dict(meta)
    summary: dict[str, Any] = {}
    for key in ("model", "selected_model", "tool_calls", "requested_tool_calls", "executed_tool_calls"):
        if key in payload:
            summary[key] = payload.get(key)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens_per_second"):
        if key in payload:
            summary[key] = payload.get(key)

    json_payload = payload.get("json")
    if isinstance(json_payload, dict):
        summary["json_keys"] = list(json_payload.keys())[:8]
        if "selected_count" in json_payload:
            summary["selected_count"] = json_payload.get("selected_count")
    return summary


def _known_context_stage_preview(tool_results: dict[str, Any] | None) -> dict[str, Any]:
    payload = _coerce_dict(tool_results)
    temporal = _coerce_dict(payload.get("temporal_context"))
    timeline = _coerce_dict(temporal.get("timeline"))
    recent_raw = timeline.get("recent")
    recent = recent_raw if isinstance(recent_raw, list) else []

    latest_user_excerpt = ""
    latest_assistant_excerpt = ""
    for entry in reversed(recent):
        if not isinstance(entry, dict):
            continue
        role = _as_text(entry.get("role")).lower()
        excerpt = _as_text(entry.get("excerpt"))
        if not excerpt:
            continue
        if not latest_user_excerpt and role == "user":
            latest_user_excerpt = excerpt
        elif not latest_assistant_excerpt and role == "assistant":
            latest_assistant_excerpt = excerpt
        if latest_user_excerpt and latest_assistant_excerpt:
            break

    preview = {
        "user_interface": payload.get("user_interface"),
        "member_first_name": payload.get("member_first_name"),
        "telegram_username": payload.get("telegram_username"),
        "chat_type": payload.get("chat_type"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "temporal_context": {
            "clock": _coerce_dict(temporal.get("clock")),
            "inbound": _coerce_dict(temporal.get("inbound")),
            "timeline": {
                "chat_type": timeline.get("chat_type"),
                "chat_host_id": timeline.get("chat_host_id"),
                "topic_id": timeline.get("topic_id"),
                "recent_count": len(recent),
                "latest_user_excerpt": latest_user_excerpt,
                "latest_assistant_excerpt": latest_assistant_excerpt,
                "history_trimmed_for_small_talk": bool(timeline.get("history_trimmed_for_small_talk")),
            },
        },
        "topic_transition": _coerce_dict(payload.get("topic_transition")),
        "memory_circuit": {
            "strategy": _coerce_dict(payload.get("memory_circuit")).get("strategy"),
            "active_topic": _coerce_dict(payload.get("memory_circuit")).get("active_topic"),
            "history_search_allowed": _coerce_dict(payload.get("memory_circuit")).get("history_search_allowed"),
            "history_recall_requested": _coerce_dict(payload.get("memory_circuit")).get("history_recall_requested"),
            "small_talk_turn": _coerce_dict(payload.get("memory_circuit")).get("small_talk_turn"),
            "routing_note": _coerce_dict(payload.get("memory_circuit")).get("routing_note"),
        },
    }
    return preview


def _memory_selection_stage_preview(memory_selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = _coerce_dict(memory_selection)
    selected_raw = selection.get("selected")
    selected = selected_raw if isinstance(selected_raw, list) else []
    selected_preview = []
    for item in selected:
        entry = _coerce_dict(item)
        selected_preview.append(
            {
                "history_id": entry.get("history_id"),
                "message_id": entry.get("message_id"),
                "role": _as_text(entry.get("role"), "user"),
                "timestamp_utc": entry.get("timestamp_utc"),
                "score": entry.get("score"),
                "signals": _coerce_dict(entry.get("signals")),
                "message_text": _as_text(entry.get("message_text")),
            }
        )
    return {
        "schema": selection.get("schema", "ryo.memory_recall.v1"),
        "mode": selection.get("mode"),
        "recall_scope": selection.get("recall_scope"),
        "history_search_allowed": selection.get("history_search_allowed"),
        "decision_reason": selection.get("decision_reason"),
        "selection_threshold": selection.get("selection_threshold"),
        "candidates_considered": selection.get("candidates_considered"),
        "selected_count": selection.get("selected_count", len(selected_preview)),
        "selected": selected_preview,
    }


def _history_recall_requested(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in _HISTORY_RECALL_PATTERNS)


def _tokenize_topic_text(text: str) -> list[str]:
    lowered = _as_text(text).lower()
    if not lowered:
        return []
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9']+", lowered):
        cleaned = token.strip("'")
        if not cleaned:
            continue
        if cleaned in _TOPIC_STOPWORDS:
            continue
        if len(cleaned) < 3 and not cleaned.isdigit():
            continue
        tokens.append(cleaned)
    return tokens


def _topic_keywords(text: str, *, max_terms: int) -> list[str]:
    if max_terms <= 0:
        return []
    tokens = _tokenize_topic_text(text)
    if not tokens:
        return []

    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        if token not in first_index:
            first_index[token] = idx

    ordered = sorted(
        counts.keys(),
        key=lambda token: (-counts[token], first_index.get(token, 10_000)),
    )
    return ordered[:max_terms]


def _topic_label(text: str, *, max_terms: int, fallback: str = "general") -> str:
    labels = _topic_keywords(text, max_terms=max(1, max_terms))
    if not labels:
        return fallback
    return ", ".join(labels)


def _is_small_talk_message(text: str) -> bool:
    cleaned = _as_text(text)
    if not cleaned:
        return True
    stripped = cleaned.strip()
    if not stripped:
        return True

    token_pattern = r"[A-Za-z0-9']+"
    tokens = [token.strip("'") for token in re.findall(token_pattern, stripped) if token.strip("'")]
    token_count = len(tokens)
    char_count = len(stripped)
    if token_count == 0:
        return True
    # Structural brevity fallback only (no phrase/keyword matching).
    if char_count <= 24 and token_count <= 4:
        return True
    if char_count <= 48 and token_count <= 7 and ("\n" not in stripped):
        return True
    return False


def _last_user_message_text(history_messages: list[dict[str, Any]] | None) -> str:
    for record in reversed(history_messages or []):
        if not isinstance(record, dict):
            continue
        if record.get("member_id") is None:
            continue
        message_text = _as_text(record.get("message_text"))
        if message_text:
            return message_text
    return ""


def _detect_topic_transition(
    *,
    current_message: str,
    history_messages: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not _runtime_bool("topic_shift.enabled", True):
        return {
            "switched": False,
            "from_topic": "general",
            "to_topic": "general",
            "summary": "Topic shift detection is disabled by runtime settings.",
            "reason": "disabled",
            "confidence": "low",
            "jaccard_similarity": 1.0,
            "history_recall_requested": _history_recall_requested(current_message),
        }

    keyword_terms = max(1, _runtime_int("topic_shift.keyword_terms", 4))
    min_token_count = max(1, _runtime_int("topic_shift.min_token_count", 3))
    jaccard_threshold = max(0.0, min(1.0, _runtime_float("topic_shift.jaccard_switch_threshold", 0.18)))

    current_text = _as_text(current_message)
    previous_text = _last_user_message_text(history_messages)
    history_recall = _history_recall_requested(current_text)
    small_talk_turn = _is_small_talk_message(current_text)

    current_tokens = set(_tokenize_topic_text(current_text))
    previous_tokens = set(_tokenize_topic_text(previous_text))

    if previous_tokens or current_tokens:
        union_size = max(1, len(previous_tokens | current_tokens))
        overlap_size = len(previous_tokens & current_tokens)
        jaccard_similarity = float(overlap_size) / float(union_size)
    else:
        jaccard_similarity = 1.0

    explicit_switch = any(re.search(pattern, current_text.lower()) for pattern in _TOPIC_SHIFT_PATTERNS)
    lexical_switch = (
        len(current_tokens) >= min_token_count
        and len(previous_tokens) >= min_token_count
        and jaccard_similarity <= jaccard_threshold
        and not small_talk_turn
    )

    from_topic = _topic_label(previous_text, max_terms=keyword_terms, fallback="general")
    to_topic = _topic_label(current_text, max_terms=keyword_terms, fallback=from_topic or "general")
    if not to_topic:
        to_topic = "general"
    if not from_topic:
        from_topic = "general"

    switched = False
    reason = "no_signal"
    confidence = "low"
    if history_recall:
        reason = "history_recall_requested"
    elif small_talk_turn:
        to_topic = "general"
        reason = "small_talk_neutral"
        confidence = "high"
    elif not previous_text:
        reason = "no_prior_user_message"
    elif explicit_switch and to_topic != from_topic:
        switched = True
        reason = "explicit_switch_signal"
        confidence = "high"
    elif lexical_switch and to_topic != from_topic:
        switched = True
        reason = "lexical_divergence"
        confidence = "medium"

    if switched:
        summary = (
            f"The user has switched the topic from '{from_topic}' to '{to_topic}'. "
            f"Prioritize '{to_topic}' as the point of attention unless they explicitly ask to recall '{from_topic}'."
        )
    else:
        summary = (
            "Latest message is a short social turn; keep context lightweight and avoid reviving stale topics."
            if reason == "small_talk_neutral"
            else f"Continue focus on topic '{to_topic}'"
            if to_topic != "general"
            else "No strong topic anchor detected in the latest user message."
        )

    return {
        "switched": switched,
        "from_topic": from_topic,
        "to_topic": to_topic,
        "summary": summary,
        "reason": reason,
        "confidence": confidence,
        "jaccard_similarity": round(jaccard_similarity, 4),
        "history_recall_requested": history_recall,
        "small_talk_turn": small_talk_turn,
    }


def _build_memory_circuit(
    *,
    current_message: str,
    history_messages: list[dict[str, Any]] | None,
    topic_transition: dict[str, Any] | None,
) -> dict[str, Any]:
    transition = _coerce_dict(topic_transition)
    keyword_terms = max(1, _runtime_int("topic_shift.keyword_terms", 4))
    recent_user_messages_limit = max(1, _runtime_int("topic_shift.recent_user_messages", 6))
    deweight_history_on_switch = _runtime_bool("topic_shift.deweight_history_search_on_switch", True)

    recent_user_messages: list[str] = []
    for record in reversed(history_messages or []):
        if not isinstance(record, dict):
            continue
        if record.get("member_id") is None:
            continue
        excerpt = _as_text(record.get("message_text"))
        if not excerpt:
            continue
        recent_user_messages.append(excerpt)
        if len(recent_user_messages) >= recent_user_messages_limit:
            break
    recent_user_messages.reverse()

    recent_user_topics: list[str] = []
    for entry in recent_user_messages:
        topic_label = _topic_label(entry, max_terms=keyword_terms, fallback="")
        if topic_label and topic_label not in recent_user_topics:
            recent_user_topics.append(topic_label)

    active_topic = _as_text(transition.get("to_topic"))
    if not active_topic or active_topic == "general":
        active_topic = _topic_label(current_message, max_terms=keyword_terms, fallback="general")
    if (not active_topic or active_topic == "general") and recent_user_topics:
        active_topic = recent_user_topics[-1]

    switched = _as_bool(transition.get("switched"), False)
    history_recall_requested = _as_bool(transition.get("history_recall_requested"), False) or _history_recall_requested(current_message)
    small_talk_turn = _as_bool(transition.get("small_talk_turn"), False) or _is_small_talk_message(current_message)
    history_search_allowed = bool(history_recall_requested or not (switched and deweight_history_on_switch))
    if small_talk_turn and not history_recall_requested:
        active_topic = "general"
        history_search_allowed = False

    routing_note = (
        "Short social turn detected; skip heavy history recall unless user explicitly asks for it."
        if small_talk_turn and not history_recall_requested
        else f"Favor new topic '{active_topic}' and suppress semantic history search unless explicitly requested."
        if switched and not history_search_allowed
        else f"Memory can include recent context for topic '{active_topic}'."
    )

    return {
        "schema": "ryo.memory_circuit.v1",
        "strategy": "recency_topic_anchor_v1",
        "active_topic": active_topic,
        "recent_user_topics": recent_user_topics,
        "topic_transition": transition,
        "history_recall_requested": history_recall_requested,
        "history_search_allowed": history_search_allowed,
        "small_talk_turn": small_talk_turn,
        "routing_note": routing_note,
    }


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _normalize_brevity_mode(value: Any, fallback: str = "standard") -> str:
    mode = _as_text(value, fallback).strip().lower()
    if mode in {"brief", "brief_social", "short_social", "social"}:
        return "brief_social"
    return "standard"


def _history_record_role(record: dict[str, Any]) -> str:
    return "assistant" if record.get("member_id") is None else "user"


def _build_semantic_boost_map(
    *,
    query_text: str,
    chat_host_id: Any,
    chat_type: Any,
    platform: Any,
    topic_id: Any,
    history_search_allowed: bool,
) -> dict[Any, float]:
    if not history_search_allowed:
        return {}
    if not _runtime_bool("retrieval.memory_use_semantic", True):
        return {}

    candidate_limit = max(1, _runtime_int("retrieval.memory_semantic_candidates", 4))
    time_window = max(1, _runtime_int("retrieval.chat_history_window_hours", 12))
    try:
        semanticResults = chatHistory.searchChatHistory(
            text=query_text,
            limit=candidate_limit,
            chatHostID=chat_host_id,
            chatType=_as_text(chat_type),
            platform=_as_text(platform),
            topicID=topic_id,
            scopeTopic=True,
            timeInHours=time_window,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Semantic memory selector failed; falling back to lexical-only scoring: {error}")
        return {}

    boostMap: dict[Any, float] = {}
    for result in semanticResults or []:
        if not isinstance(result, dict):
            continue
        historyID = result.get("history_id")
        distance = _coerce_float(result.get("distance"), 1.0)
        boostMap[historyID] = max(0.0, 1.0 / (1.0 + max(0.0, distance)))
    return boostMap


def _select_memory_recall(
    *,
    current_message: str,
    history_messages: list[dict[str, Any]] | None,
    analysis_payload: dict[str, Any] | None,
    topic_transition: dict[str, Any] | None,
    memory_circuit: dict[str, Any] | None,
    chat_host_id: Any,
    chat_type: Any,
    platform: Any,
    topic_id: Any,
) -> dict[str, Any]:
    payload = _coerce_dict(analysis_payload)
    transition = _coerce_dict(topic_transition)
    circuit = _coerce_dict(memory_circuit)
    directive = _coerce_dict(payload.get("memory_directive"))

    mode = _as_text(directive.get("mode"), "continue_topic")
    recall_scope = _as_text(directive.get("recall_scope"), "hybrid").lower()
    if recall_scope not in {"none", "recent", "semantic", "hybrid"}:
        recall_scope = "hybrid"
    history_search_allowed = _as_bool(
        directive.get("history_search_allowed"),
        fallback=_as_bool(circuit.get("history_search_allowed"), True),
    )
    max_items = max(
        0,
        min(
            12,
            int(
                directive.get(
                    "max_items",
                    _runtime_int("retrieval.memory_recall_limit", 4),
                )
            ),
        ),
    )
    min_score = max(0.0, _coerce_float(directive.get("min_score"), _runtime_float("retrieval.memory_min_score", 0.65)))

    records = history_messages if isinstance(history_messages, list) else []
    candidates_considered = min(len(records), max(1, _runtime_int("retrieval.memory_candidate_limit", 30)))
    records = records[-candidates_considered:]

    history_recall_requested = _as_bool(circuit.get("history_recall_requested"), False)
    small_talk_turn = _as_bool(circuit.get("small_talk_turn"), False)
    switched = _as_bool(transition.get("switched"), False)

    if max_items <= 0 or not records:
        reason = "max_items_zero" if max_items <= 0 else "no_history_candidates"
        return {
            "schema": "ryo.memory_recall.v1",
            "mode": mode,
            "recall_scope": recall_scope,
            "history_search_allowed": history_search_allowed,
            "selected": [],
            "selected_count": 0,
            "candidates_considered": len(records),
            "decision_reason": reason,
            "topic_transition": transition,
            "memory_circuit": {
                "active_topic": circuit.get("active_topic"),
                "routing_note": circuit.get("routing_note"),
                "small_talk_turn": small_talk_turn,
            },
        }

    if small_talk_turn and not history_recall_requested:
        return {
            "schema": "ryo.memory_recall.v1",
            "mode": "lightweight_social",
            "recall_scope": "none",
            "history_search_allowed": False,
            "selected": [],
            "selected_count": 0,
            "candidates_considered": len(records),
            "decision_reason": "small_talk_turn_without_recall_request",
            "topic_transition": transition,
            "memory_circuit": {
                "active_topic": circuit.get("active_topic"),
                "routing_note": circuit.get("routing_note"),
                "small_talk_turn": small_talk_turn,
            },
        }

    if recall_scope == "none":
        history_search_allowed = False

    query_text = _as_text(current_message)
    query_tokens = set(_tokenize_topic_text(query_text))
    semantic_boost = {}
    if recall_scope in {"semantic", "hybrid"}:
        semantic_boost = _build_semantic_boost_map(
            query_text=query_text,
            chat_host_id=chat_host_id,
            chat_type=chat_type,
            platform=platform,
            topic_id=topic_id,
            history_search_allowed=history_search_allowed,
        )

    scored: list[dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)
    for record in records:
        if not isinstance(record, dict):
            continue
        message_text = _as_text(record.get("message_text"))
        if not message_text:
            continue
        role = _history_record_role(record)
        candidate_tokens = set(_tokenize_topic_text(message_text))
        lexical_overlap = 0.0
        if query_tokens:
            lexical_overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
        recency_seconds = 0
        record_ts = coerce_datetime_utc(record.get("message_timestamp"), assume_tz=timezone.utc)
        if record_ts is not None:
            recency_seconds = max(0, int((now_utc - record_ts).total_seconds()))
        recency_weight = 1.0 / (1.0 + (float(recency_seconds) / 3600.0))
        semantic_weight = _coerce_float(semantic_boost.get(record.get("history_id"), 0.0), 0.0)

        if recall_scope == "recent":
            semantic_weight = 0.0
        if recall_scope == "semantic":
            lexical_overlap = lexical_overlap * 0.6

        score = (lexical_overlap * 2.2) + (semantic_weight * 1.4) + (recency_weight * 0.5)
        if role == "user":
            score += 0.08
        if switched and not history_recall_requested and lexical_overlap < 0.2:
            score -= 0.35

        scored.append(
            {
                "history_id": record.get("history_id"),
                "message_id": record.get("message_id"),
                "role": role,
                "timestamp_utc": None if record_ts is None else record_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "message_text": message_text,
                "score": round(score, 4),
                "signals": {
                    "lexical_overlap": round(lexical_overlap, 4),
                    "semantic_weight": round(semantic_weight, 4),
                    "recency_weight": round(recency_weight, 4),
                    "recency_seconds": recency_seconds,
                },
            }
        )

    scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    if history_recall_requested:
        selected = scored[:max_items]
    else:
        selected = [item for item in scored if _coerce_float(item.get("score"), 0.0) >= min_score][:max_items]

    return {
        "schema": "ryo.memory_recall.v1",
        "mode": mode,
        "recall_scope": recall_scope,
        "history_search_allowed": history_search_allowed,
        "selected": selected,
        "selected_count": len(selected),
        "candidates_considered": len(scored),
        "decision_reason": (
            "history_recall_requested"
            if history_recall_requested
            else "scored_selection"
        ),
        "selection_threshold": min_score,
        "topic_transition": transition,
        "memory_circuit": {
            "active_topic": circuit.get("active_topic"),
            "routing_note": circuit.get("routing_note"),
            "small_talk_turn": small_talk_turn,
        },
    }


def _normalize_analysis_payload(
    raw_analysis_text: str,
    known_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = _parse_json_like(raw_analysis_text)
    payload = parsed if isinstance(parsed, dict) else {}
    tool_results = payload.get("tool_results")
    if isinstance(tool_results, dict):
        payload = tool_results

    context_data = known_context if isinstance(known_context, dict) else {}
    context_topic_transition = _coerce_dict(context_data.get("topic_transition"))
    context_memory_circuit = _coerce_dict(context_data.get("memory_circuit"))
    payload_topic_transition = _coerce_dict(payload.get("topic_transition"))
    payload_memory_directive = _coerce_dict(payload.get("memory_directive"))

    transition_switched = _as_bool(
        payload_topic_transition.get("switched"),
        fallback=_as_bool(context_topic_transition.get("switched"), False),
    )
    transition_from_topic = _as_text(
        payload_topic_transition.get("from_topic"),
        _as_text(context_topic_transition.get("from_topic"), "general"),
    )
    transition_to_topic = _as_text(
        payload_topic_transition.get("to_topic"),
        _as_text(context_topic_transition.get("to_topic"), _as_text(payload.get("topic"), "general")),
    )
    transition_reason = _as_text(
        payload_topic_transition.get("reason"),
        _as_text(context_topic_transition.get("reason"), "none"),
    )
    transition_confidence = _as_text(
        payload_topic_transition.get("confidence"),
        _as_text(context_topic_transition.get("confidence"), "low"),
    )
    transition_summary = _as_text(
        payload_topic_transition.get("summary"),
        _as_text(context_topic_transition.get("summary")),
    )
    transition_history_recall = _as_bool(
        payload_topic_transition.get("history_recall_requested"),
        fallback=_as_bool(context_topic_transition.get("history_recall_requested"), False),
    )

    context_summary = _as_text(payload.get("context_summary"))
    if not context_summary:
        member_name = _as_text(context_data.get("member_first_name"), "member")
        interface_name = _as_text(context_data.get("user_interface"), "unknown")
        chat_type = _as_text(context_data.get("chat_type"), "member")
        context_summary = (
            f"Interface: {interface_name}; chat type: {chat_type}; "
            f"latest message from: {member_name}."
        )
    if transition_switched and transition_summary:
        context_summary = f"{transition_summary} {context_summary}".strip()
    maxContextSummaryChars = max(60, _runtime_int("orchestrator.analysis_context_summary_max_chars", 220))
    if len(context_summary) > maxContextSummaryChars:
        if maxContextSummaryChars <= 3:
            context_summary = context_summary[:maxContextSummaryChars]
        else:
            context_summary = context_summary[: maxContextSummaryChars - 3].rstrip() + "..."

    style_raw = payload.get("response_style")
    style = style_raw if isinstance(style_raw, dict) else {}
    tone = _as_text(style.get("tone"), "friendly")
    length = _as_text(style.get("length"), "concise")
    if length not in {"very_short", "short", "concise", "medium", "detailed"}:
        length = "concise"

    tool_hints = []
    for hint in _as_string_list(payload.get("tool_hints")):
        if hint in _ALLOWED_TOOL_HINTS and hint not in tool_hints:
            tool_hints.append(hint)

    history_search_allowed = _as_bool(
        payload_memory_directive.get("history_search_allowed"),
        fallback=_as_bool(context_memory_circuit.get("history_search_allowed"), True),
    )
    if transition_switched and not history_search_allowed:
        tool_hints = [hint for hint in tool_hints if hint != "chatHistorySearch"]

    risk_flags = _as_string_list(payload.get("risk_flags")) or ["none"]
    topic = _as_text(payload.get("topic"), transition_to_topic or "general")
    if transition_switched and transition_to_topic:
        topic = transition_to_topic
    intent = _as_text(payload.get("intent"), "answer_user")

    memory_mode = _as_text(
        payload_memory_directive.get("mode"),
        "focus_new_topic" if transition_switched else "continue_topic",
    )
    memory_reason = _as_text(
        payload_memory_directive.get("reason"),
        (
            "Topic switch detected; de-prioritize stale topic memory unless user asks for recall."
            if transition_switched and not history_search_allowed
            else "No topic switch override; keep normal recency-weighted memory."
        ),
    )
    payload_brevity_directive = _coerce_dict(payload.get("brevity_directive"))
    brevity_mode = _normalize_brevity_mode(
        payload_brevity_directive.get("mode"),
        "standard",
    )
    brevity_reason = _as_text(
        payload_brevity_directive.get("reason"),
        "analysis_reasoned_briefness" if brevity_mode == "brief_social" else "analysis_reasoned_standard",
    )

    return {
        "topic": topic,
        "intent": intent,
        "needs_tools": _as_bool(payload.get("needs_tools"), fallback=False),
        "tool_hints": tool_hints,
        "risk_flags": risk_flags,
        "topic_transition": {
            "switched": transition_switched,
            "from_topic": transition_from_topic,
            "to_topic": transition_to_topic,
            "summary": transition_summary,
            "reason": transition_reason,
            "confidence": transition_confidence,
            "history_recall_requested": transition_history_recall,
        },
        "memory_directive": {
            "mode": memory_mode,
            "history_search_allowed": history_search_allowed,
            "reason": memory_reason,
        },
        "brevity_directive": {
            "mode": brevity_mode,
            "reason": brevity_reason,
        },
        "response_style": {
            "tone": tone,
            "length": length,
        },
        "context_summary": context_summary,
    }


def _user_requested_diagnostics(message_text: str) -> bool:
    text = _as_text(message_text).lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _DIAGNOSTIC_REQUEST_PATTERNS)


def _line_has_meta_leak(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered) for pattern in _META_LEAK_PATTERNS)


def _sanitize_final_response(
    text: str,
    *,
    user_message: str = "",
    allow_internal_diagnostics: bool = False,
) -> str:
    raw = str(text or "")
    if not raw.strip():
        return "I could not generate a complete reply this turn. Please try again."
    if allow_internal_diagnostics or _user_requested_diagnostics(user_message):
        return raw.strip()

    lines = [line for line in raw.splitlines() if line.strip()]
    filtered = [line for line in lines if not _line_has_meta_leak(line)]
    cleaned = "\n".join(filtered).strip()
    if cleaned:
        return cleaned

    # If every line was meta/internal, return a safe user-facing fallback.
    return "I can help with that. Could you restate what you want me to focus on?"



################
# ORCHESTRATOR #
################

class ConversationOrchestrator:

    def __init__(self, message: str, memberID: int, context: dict=None, messageID: int=None, options: dict=None):
        self._messages: list[Message] = []
        self._analysisStats: dict[str, Any] = {}
        self._devStats: dict[str, Any] = {}
        self._chatResponseMessage = ""

        self._message = message
        self._messageID = messageID
        self._context = context if isinstance(context, dict) else {}
        self._options = options if isinstance(options, dict) else {}
        stageCallback = self._options.get("stage_callback")
        self._stage_callback = stageCallback if callable(stageCallback) else None
        self._transientSession = _as_bool(
            self._options.get("transient_session"),
            fallback=_as_bool(self._context.get("guest_mode"), False),
        )

        # Resolve member context. Transient sessions avoid DB-backed identity lookup.
        if self._transientSession:
            self._memberData = {
                "first_name": _as_text(self._context.get("member_first_name"), "Guest"),
                "username": _as_text(self._context.get("telegram_username"), "guest"),
            }
        else:
            resolvedMember = members.getMemberByID(memberID)
            self._memberData = (
                resolvedMember
                if isinstance(resolvedMember, dict)
                else {"first_name": "Member", "username": ""}
            )

        # Check the context and get a short collection of recent chat history into the orchestrator's messages list
        self._chatHostID = self._context.get("chat_host_id")
        self._chatType = self._context.get("chat_type")
        self._communityID = self._context.get("community_id")
        self._platform = self._context.get("platform")
        self._topicID = self._context.get("topic_id")
        self._messageTimestamp = coerce_datetime_utc(
            self._context.get("message_timestamp"),
            assume_tz=timezone.utc,
        )
        self._messageReceivedTimestamp = coerce_datetime_utc(
            self._context.get("message_received_timestamp"),
            assume_tz=timezone.utc,
        )
        if self._messageReceivedTimestamp is None:
            self._messageReceivedTimestamp = datetime.now(timezone.utc)
        if self._messageTimestamp is None:
            self._messageTimestamp = self._messageReceivedTimestamp
        self._shortHistory: list[dict[str, Any]] = []
        if self._transientSession:
            seedMessages = _coerce_message_list(self._options.get("transient_messages"))
            if seedMessages:
                self._messages.extend(seedMessages)
            if self._messageID is None:
                transientMessageID = self._options.get("message_id")
                try:
                    self._messageID = int(transientMessageID) if transientMessageID is not None else 1
                except (TypeError, ValueError):
                    self._messageID = 1
            self._responseID = int(self._messageID) + 1

        if self._chatHostID is None and not self._transientSession:
            if self._communityID is None:
                self._chatHostID = memberID
                self._chatType = "member"
            else:
                self._chatHostID = self._communityID
                self._chatType = "community"

        # Get the most recent chat history records for the given context
        availablePlatforms = ["cli", "telegram"]
        availableChatTypes = ["member", "community"]
        if (not self._transientSession) and self._platform in availablePlatforms and self._chatType in availableChatTypes:
            shortHistoryLimit = _runtime_int("retrieval.conversation_short_history_limit", 20)
            shortHistory = chatHistory.getChatHistory(
                self._chatHostID,
                self._chatType,
                self._platform,
                self._topicID,
                limit=shortHistoryLimit,
            )
            self._shortHistory = shortHistory if isinstance(shortHistory, list) else []
            for historyMessage in shortHistory:
                role = "assistant" if historyMessage.get("member_id") is None else "user"
                content = historyMessage.get("message_text")
                self._messages.append(Message(role=role, content=content))

            # If no message ID provided, increment one from the last message's id
            if self._messageID is None:
                self._messageID = 1 if not shortHistory else shortHistory[-1].get("message_id")
                self._responseID = self._messageID + 1
                
        # Add the newest message to local list and the database
        newMessage = Message(role="user", content=message)
        self._messages.append(newMessage)
        if self._transientSession:
            self._promptHistoryID = None
        else:
            # Need to update this to pass all the available context, such as community_id, topic_id... even if None
            self._promptHistoryID = chatHistory.addChatHistory(
                messageID=self._messageID, 
                messageText=self._message, 
                platform=self._platform, 
                memberID=memberID, 
                communityID=self._communityID,
                chatHostID=self._chatHostID,
                topicID=self._topicID,
                timestamp=self._messageTimestamp,
            )

        # TODO Check options and pass to agents if necessary
        
    
    async def _emit_stage(self, stage: str, detail: str = "", **meta: Any) -> None:
        if not callable(self._stage_callback):
            meta_summary = _stage_meta_log_summary(meta if isinstance(meta, dict) else {})
            if meta_summary:
                logger.info(f"[orchestrator.stage] {stage} | {detail} | {meta_summary}")
            else:
                logger.info(f"[orchestrator.stage] {stage} | {detail}")
            return
        event = {
            "stage": str(stage),
            "detail": str(detail or ""),
            "timestamp": utc_now_iso(),
            "meta": meta if isinstance(meta, dict) else {},
        }
        meta_summary = _stage_meta_log_summary(meta if isinstance(meta, dict) else {})
        if meta_summary:
            logger.info(f"[orchestrator.stage] {stage} | {detail} | {meta_summary}")
        else:
            logger.info(f"[orchestrator.stage] {stage} | {detail}")
        try:
            maybeAwaitable = self._stage_callback(event)
            if inspect.isawaitable(maybeAwaitable):
                await maybeAwaitable
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Stage callback failed [{stage}]: {error}")


    async def runAgents(self):
        await self._emit_stage("orchestrator.start", "Accepted request and preparing context.")
        # Create all the agent calls in the flow as methods to call, each handles the messages passed to the actual agent
        # Build and append machine-readable known context for all downstream stages.
        temporalEnabled = _runtime_bool("temporal.enabled", True)
        temporalHistoryLimit = _runtime_int(
            "temporal.history_limit",
            _runtime_int("retrieval.conversation_short_history_limit", 20),
        )
        temporalExcerptMaxChars = _runtime_int("temporal.excerpt_max_chars", 160)
        temporalTimezone = str(_runtime_value("temporal.default_timezone", "UTC") or "UTC")
        temporalContext = (
            build_temporal_context(
                platform=self._platform,
                chat_type=self._chatType,
                chat_host_id=self._chatHostID,
                topic_id=self._topicID,
                timezone_name=temporalTimezone,
                now_utc=datetime.now(timezone.utc),
                inbound_sent_at=self._messageTimestamp,
                inbound_received_at=self._messageReceivedTimestamp,
                history_messages=self._shortHistory,
                history_limit=max(0, temporalHistoryLimit),
                excerpt_max_chars=max(0, temporalExcerptMaxChars),
            )
            if temporalEnabled
            else {
                "schema": "ryo.temporal_context.v1",
                "enabled": False,
                "reason": "runtime.temporal.enabled=false",
                "clock": {"now_utc": utc_now_iso()},
            }
        )
        topicTransition = _detect_topic_transition(
            current_message=self._message,
            history_messages=self._shortHistory,
        )
        memoryCircuit = _build_memory_circuit(
            current_message=self._message,
            history_messages=self._shortHistory,
            topic_transition=topicTransition,
        )
        if _as_bool(topicTransition.get("switched"), False):
            await self._emit_stage(
                "context.topic_shift",
                _as_text(topicTransition.get("summary")),
                from_topic=topicTransition.get("from_topic"),
                to_topic=topicTransition.get("to_topic"),
                reason=topicTransition.get("reason"),
            )

        knownContext = {
            "tool_name": "Known Context",
            "tool_results": {
                "user_interface": self._platform,
                "member_first_name": self._memberData.get("first_name"),
                "telegram_username": self._memberData.get("username"),
                "chat_type": self._chatType,
                "timestamp_utc": temporalContext.get("clock", {}).get("now_utc", utc_now_iso()),
                "temporal_context": temporalContext,
                "topic_transition": topicTransition,
                "memory_circuit": memoryCircuit,
            }
        }
        await self._emit_stage(
            "context.built",
            "Known context assembled for downstream stages.",
            json=_known_context_stage_preview(knownContext.get("tool_results")),
        )
        self._messages.append(Message(role="tool", content=json.dumps(knownContext)))
        analysisMessages = list(self._messages)
        await self._emit_stage("analysis.start", "Running message analysis policy and model selection.")
        self._analysisAgent = MessageAnalysisAgent(analysisMessages, options=self._options)
        self._analysisResponse = await self._analysisAgent.generateResponse()

        analysisResponseMessage = ""
        analysisChunkCount = 0
        analysisCharCount = 0
        analysisDoneSeen = False
        analysisLastProgressLog = time.monotonic()
        analysisLastProgressEmit = time.monotonic()
        analysisLatestPreview = ""
        
        #print(f"{ConsoleColors["purple"]}Analysis Agent > ", end="")
        chunk: ChatResponse
        async for chunk in self._analysisResponse:
            chunkContent = str(getattr(getattr(chunk, "message", None), "content", "") or "")
            # Call the streaming response method. This is intended to be over written by the UI for cutom handling
            self.streamingResponse(streamingChunk=chunkContent)
            analysisResponseMessage = analysisResponseMessage + chunkContent
            analysisChunkCount += 1
            analysisCharCount += len(chunkContent)
            if chunkContent:
                analysisLatestPreview = _stream_human_preview(analysisResponseMessage, 280)
            now = time.monotonic()
            if chunkContent and (
                analysisChunkCount == 1
                or (now - analysisLastProgressLog) >= 1.5
                or bool(getattr(chunk, "done", False))
            ):
                logger.info(
                    "[stream.analysis] "
                    f"chunks={analysisChunkCount} chars={analysisCharCount} latest='{_stream_log_snippet(chunkContent)}'"
                )
                analysisLastProgressLog = now
            if chunkContent and (
                analysisChunkCount == 1
                or (now - analysisLastProgressEmit) >= 2.0
            ):
                await self._emit_stage(
                    "analysis.progress",
                    "Analysis model is reasoning.",
                    snippet=analysisLatestPreview or _stream_log_snippet(chunkContent, 220),
                    chunks=analysisChunkCount,
                    chars=analysisCharCount,
                )
                analysisLastProgressEmit = now
            if chunk.done:
                analysisDoneSeen = True
                self._analysisStats = {
                    "total_duration": chunk.total_duration,
                    "load_duration": chunk.load_duration,
                    "prompt_eval_count": chunk.prompt_eval_count,
                    "prompt_eval_duration": chunk.prompt_eval_duration,
                    "eval_count": chunk.eval_count,
                    "eval_duration": chunk.eval_duration,
                }
                logger.info(
                    "[stream.analysis] "
                    f"done chunks={analysisChunkCount} chars={analysisCharCount} eval_count={chunk.eval_count}"
                )
        if not analysisDoneSeen:
            logger.warning(
                "[stream.analysis] stream ended without done flag; "
                f"using conservative analysis parse fallback (chunks={analysisChunkCount}, chars={analysisCharCount})."
            )
            await self._emit_stage(
                "analysis.incomplete_stream",
                "Analysis stream ended before done flag; using conservative parse fallback.",
                chunks=analysisChunkCount,
                chars=analysisCharCount,
                stream_done=False,
            )
        analysisStatsSummary = _ollama_stream_stats_summary(self._analysisStats)
        await self._emit_stage(
            "analysis.complete",
            "Analysis stage complete.",
            model=getattr(self._analysisAgent, "_model", None),
            selected_model=_coerce_dict(getattr(self._analysisAgent, "routing", {})).get("selected_model"),
            stream_done=analysisDoneSeen,
            prompt_tokens=analysisStatsSummary.get("prompt_tokens"),
            completion_tokens=analysisStatsSummary.get("completion_tokens"),
            total_tokens=analysisStatsSummary.get("total_tokens"),
            prompt_tokens_per_second=analysisStatsSummary.get("prompt_tokens_per_second"),
            completion_tokens_per_second=analysisStatsSummary.get("completion_tokens_per_second"),
            total_tokens_per_second=analysisStatsSummary.get("total_tokens_per_second"),
            json={
                "routing": _coerce_dict(getattr(self._analysisAgent, "routing", {})),
                "stats": analysisStatsSummary,
            },
        )

        normalizedAnalysis = _normalize_analysis_payload(
            analysisResponseMessage if analysisDoneSeen else "",
            knownContext.get("tool_results"),
        )
        analysisMessagePayload = {
            "tool_name": "Message Analysis",
            "tool_results": normalizedAnalysis,
        }
        await self._emit_stage(
            "analysis.payload",
            "Normalized analysis payload produced.",
            json=normalizedAnalysis,
        )

        #print(ConsoleColors["default"])

        self._messages.append(Message(role="tool", content=json.dumps(analysisMessagePayload)))

        brevityDirective = _coerce_dict(normalizedAnalysis.get("brevity_directive"))
        brevityMode = _normalize_brevity_mode(brevityDirective.get("mode"), "standard")
        brevityReason = _as_text(brevityDirective.get("reason"), "analysis_reasoned_standard")
        fastPathEnabled = _runtime_bool("orchestrator.fast_path_small_talk_enabled", True)
        messageCharCount = len(_as_text(self._message))
        analysisNeedsTools = _as_bool(normalizedAnalysis.get("needs_tools"), fallback=False)
        fastPathActive = (
            fastPathEnabled
            and brevityMode == "brief_social"
            and not analysisNeedsTools
        )

        toolResponses: list[dict[str, Any]] = []
        if fastPathActive:
            await self._emit_stage(
                "orchestrator.fast_path",
                "Analysis selected brevity fast-path; skipping tool stage.",
                json={
                    "mode": brevityMode,
                    "reason": brevityReason,
                    "needs_tools": analysisNeedsTools,
                    "message_char_count": messageCharCount,
                },
            )
        else:
            # Feed only a sanitized analysis handoff to the tool stage.
            await self._emit_stage("tools.start", "Evaluating tool calls.")
            toolStageMessages = list(self._messages)
            toolOptions = dict(self._options)
            toolRuntimeContext = _coerce_dict(toolOptions.get("tool_runtime_context"))
            knownContextResults = _coerce_dict(knownContext.get("tool_results"))
            toolRuntimeContext.update(
                {
                    "chat_host_id": self._chatHostID,
                    "chat_type": self._chatType,
                    "platform": self._platform,
                    "topic_id": self._topicID,
                    "topic_transition": _coerce_dict(knownContextResults.get("topic_transition")),
                    "memory_circuit": _coerce_dict(knownContextResults.get("memory_circuit")),
                    "history_search_allowed": _as_bool(
                        _coerce_dict(knownContextResults.get("memory_circuit")).get("history_search_allowed"),
                        True,
                    ),
                }
            )
            toolOptions["tool_runtime_context"] = toolRuntimeContext
            self._toolsAgent = ToolCallingAgent(toolStageMessages, options=toolOptions)
            toolResponses = await self._toolsAgent.generateResponse()
            toolSummary = _coerce_dict(getattr(self._toolsAgent, "execution_summary", {}))
            toolsRouting = _coerce_dict(getattr(self._toolsAgent, "routing", {}))
            requestedToolCalls = toolSummary.get("requested_tool_calls")
            executedToolCalls = toolSummary.get("executed_tool_calls")
            if not isinstance(executedToolCalls, int):
                executedToolCalls = len(toolResponses)
            await self._emit_stage(
                "tools.complete",
                "Tool execution stage complete.",
                tool_calls=len(toolResponses),
                requested_tool_calls=requestedToolCalls if isinstance(requestedToolCalls, int) else len(toolResponses),
                executed_tool_calls=executedToolCalls,
                selected_model=toolsRouting.get("selected_model"),
                json={
                    "routing": toolsRouting,
                    "summary": toolSummary,
                    "tool_results": toolResponses,
                },
            )

            for toolResponse in toolResponses:
                self._messages.append(Message(role="tool", content=json.dumps(toolResponse)))
        
        # TODO Tools to thoughts "thinking" agent next, will produce thoughts based analysis and tool responses, outputs thoughts followed by a prompt
        
        # TODO Passes only the prompt to the response agent
        # Pass options=options to override the langauge model

        self._chatConversationAgent = ChatConversationAgent(messages=self._messages, options=self._options)
        await self._emit_stage(
            "response.start",
            "Generating final response.",
            model=getattr(self._chatConversationAgent, "_model", None),
        )
        response = await self._chatConversationAgent.generateResponse()

        responseMessage = ""
        responseChunkCount = 0
        responseCharCount = 0
        responseDoneSeen = False
        responseLastProgressLog = time.monotonic()
        responseLastProgressEmit = time.monotonic()
        responseLatestPreview = ""
        
        #print(f"{ConsoleColors["blue"]}Assistant > ", end="")
        chunk: ChatResponse
        async for chunk in response:
            chunkContent = str(getattr(getattr(chunk, "message", None), "content", "") or "")
            self.streamingResponse(streamingChunk=chunkContent)
            responseMessage = responseMessage + chunkContent
            responseChunkCount += 1
            responseCharCount += len(chunkContent)
            if chunkContent:
                responseLatestPreview = _stream_human_preview(responseMessage, 280)
            now = time.monotonic()
            if chunkContent and (
                responseChunkCount == 1
                or (now - responseLastProgressLog) >= 1.5
                or bool(getattr(chunk, "done", False))
            ):
                logger.info(
                    "[stream.response] "
                    f"chunks={responseChunkCount} chars={responseCharCount} latest='{_stream_log_snippet(chunkContent)}'"
                )
                responseLastProgressLog = now
            if chunkContent and (
                responseChunkCount == 1
                or (now - responseLastProgressEmit) >= 2.0
            ):
                await self._emit_stage(
                    "response.progress",
                    "Response model is drafting output.",
                    snippet=responseLatestPreview or _stream_log_snippet(chunkContent, 220),
                    chunks=responseChunkCount,
                    chars=responseCharCount,
                )
                responseLastProgressEmit = now
            if chunk.done:
                responseDoneSeen = True
                self._devStats = {
                    "total_duration": chunk.total_duration,
                    "load_duration": chunk.load_duration,
                    "prompt_eval_count": chunk.prompt_eval_count,
                    "prompt_eval_duration": chunk.prompt_eval_duration,
                    "eval_count": chunk.eval_count,
                    "eval_duration": chunk.eval_duration,
                }
                logger.info(
                    "[stream.response] "
                    f"done chunks={responseChunkCount} chars={responseCharCount} eval_count={chunk.eval_count}"
                )
        if not responseDoneSeen:
            logger.warning(
                "[stream.response] stream ended without done flag; "
                f"captured partial response text (chunks={responseChunkCount}, chars={responseCharCount})."
            )
        responseStatsSummary = _ollama_stream_stats_summary(self._devStats)
        await self._emit_stage(
            "response.complete",
            "Final response generated.",
            model=getattr(self._chatConversationAgent, "_model", None),
            selected_model=_coerce_dict(getattr(self._chatConversationAgent, "routing", {})).get("selected_model"),
            stream_done=responseDoneSeen,
            prompt_tokens=responseStatsSummary.get("prompt_tokens"),
            completion_tokens=responseStatsSummary.get("completion_tokens"),
            total_tokens=responseStatsSummary.get("total_tokens"),
            prompt_tokens_per_second=responseStatsSummary.get("prompt_tokens_per_second"),
            completion_tokens_per_second=responseStatsSummary.get("completion_tokens_per_second"),
            total_tokens_per_second=responseStatsSummary.get("total_tokens_per_second"),
            json={
                "chat_routing": _coerce_dict(getattr(self._chatConversationAgent, "routing", {})),
                "stats": responseStatsSummary,
            },
        )

        allowDiagnostics = bool(self._options.get("allow_internal_diagnostics", False))
        sanitizedResponseMessage = _sanitize_final_response(
            responseMessage,
            user_message=self._message,
            allow_internal_diagnostics=allowDiagnostics,
        )
        if not str(sanitizedResponseMessage or "").strip():
            sanitizedResponseMessage = "I could not generate a complete reply this turn. Please try again."
            await self._emit_stage(
                "response.fallback",
                "Model returned empty output; emitted fallback response text.",
            )
        if sanitizedResponseMessage != responseMessage:
            await self._emit_stage("response.sanitized", "Removed internal orchestration artifacts.")
        self._chatResponseMessage = sanitizedResponseMessage
        #print(ConsoleColors["default"])

        if hasattr(self, "_responseID") and not self._transientSession:
            self.storeResponse(self._responseID)
        
        assistantMessage = Message(role="assistant", content=sanitizedResponseMessage)
        # Add the final response to the overall chat history (role ASSISTANT)
        self._messages.append(assistantMessage)
        await self._emit_stage(
            "orchestrator.complete",
            "Completed end-to-end orchestration.",
            selected_model=_coerce_dict(getattr(self._chatConversationAgent, "routing", {})).get("selected_model"),
            json={
                "chat_routing": _coerce_dict(getattr(self._chatConversationAgent, "routing", {})),
                "response": {
                    "chars": len(sanitizedResponseMessage),
                    "sanitized": bool(sanitizedResponseMessage != responseMessage),
                },
                "stats": _ollama_stream_stats_summary(self._devStats),
            },
        )

        return sanitizedResponseMessage

    def storeResponse(self, messageID: int=None):
        if self._transientSession:
            self._responseHistoryID = None
            return
        responseID = messageID if messageID is not None else self._messageID + 1
        self._responseHistoryID = chatHistory.addChatHistory(
            messageID=responseID, 
            messageText=self._chatResponseMessage, 
            platform=self._platform, 
            memberID=None, 
            communityID=self._communityID,
            chatHostID=self._chatHostID, 
            topicID=self._topicID
        )
    
    def streamingResponse(self, streamingChunk: str):
        return

    @property
    def messages(self):
        return self._messages
    
    @property
    def messageID(self):
        return self._messageID
    
    @property
    def promptHistoryID(self):
        return self._promptHistoryID
    
    @property
    def responseHistoryID(self):
        return self._responseHistoryID
    
    @property
    def stats(self):
        # Eventually add up the stats from all agents
        return self._devStats



##################
# POLICY MANAGER #
##################

# TODO Need to create a Policy Manager that will load the agent's policy and allow for edits and save edits to file
# Perhaps policy manager belongs in utils?



###############
# AGENT TOOLS #
###############

def braveSearch(queryString: str, count: int = 5) -> list:
    """
    Search the web using Brave search API

    Args:
        queryString (str): The search query to look up
        count (int): (Optional) The number of search results to return. Defaults to 5

    Returns:
        list: A list of JSON structures containing the search result details
    """
    brave_key = config._instance.brave_keys
    braveUrl = "https://api.search.brave.com/res/v1/web/search"
    braveHeaders = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_key
    }

    queryParams = {
        "q": queryString,
        "count": count,
        "result_filter": "web"
    }
    results = requests.get(url=braveUrl, params=queryParams, headers=braveHeaders)
    braveResults = results.json()

    if braveWebResults := braveResults.get("web"):
        webResultsList = list()
        for webResult in braveWebResults.get("results"):
            webSearchData = {k: webResult[k] for k in webResult.keys() if k in ["title", "url", "description", "extra_snippets"]}
            webResultsList.append(webSearchData)
    
        return webResultsList
    
    return None


# Need to limit this only search within the context of the current conversation
def chatHistorySearch(queryString: str, count: int = 2, runtime_context: dict | None = None) -> list:
    """
    Search the chat history database for a vector match on text. 

    Args:
        queryString (str): The search query to look up related messages in the chat history
        count (int): (Optional) The number of search results to return. Defaults to 1
        runtime_context (dict): (Optional) Active chat scope passed by the orchestrator/tool runtime

    Returns:
        list: A list of search results. Each search result is a JSON structure
    """

    context = _coerce_dict(runtime_context)
    if not _as_bool(context.get("history_search_allowed"), True):
        logger.info("Skipping chatHistorySearch due to memory routing policy: history_search_allowed=false")
        return []

    transition = _coerce_dict(context.get("topic_transition"))
    switched = _as_bool(transition.get("switched"), False)
    history_recall_requested = _as_bool(transition.get("history_recall_requested"), False)
    switch_window_hours = max(1, _runtime_int("topic_shift.history_window_hours_on_switch", 2))
    scoped_time_window = switch_window_hours if (switched and not history_recall_requested) else None

    scopedSearch = bool(context.get("chat_host_id") is not None or context.get("chat_type") or context.get("platform"))
    results = chatHistory.searchChatHistory(
        text=queryString,
        limit=max(1, int(count)),
        chatHostID=context.get("chat_host_id"),
        chatType=context.get("chat_type"),
        platform=context.get("platform"),
        topicID=context.get("topic_id"),
        scopeTopic=scopedSearch,
        timeInHours=scoped_time_window,
    )
    scopeInfo = {
        "chat_host_id": context.get("chat_host_id"),
        "chat_type": context.get("chat_type"),
        "platform": context.get("platform"),
        "topic_id": context.get("topic_id"),
        "scoped_time_window_hours": scoped_time_window,
        "topic_switched": switched,
    }
    logger.debug(
        "Chat history tool results:\n"
        f"scope={scopeInfo}\n"
        f"results={results}"
    )
    
    convertedResults = list()
    for result in results or []:
        chatHistoryRecord = dict()
        for key, value in result.items():
            chatHistoryRecord[key] = value.strftime("%Y-%m-%d %H:%M:%S") if key == "message_timestamp" else value

        convertedResults.append(chatHistoryRecord)

    return convertedResults


def knowledgeSearch(queryString: str, count: int = 2) -> list:
    """
    Search the knowledge database for a vector match on text. 
    Knowledge database contains information specific to the following topics: 
    Hypermind Labs, Dropbear Robot, the Egg and mini Egg project.

    Args:
        queryString (str): The search query to look up related documents
        count (int): (Optional) The number of search results to return. Defaults to 2

    Returns:
        list: A list of search results. Each search result is a JSON structure
    """

    results = knowledge.searchKnowledge(text=queryString, limit=count)
    convertedResults = list()
    for result in results:
        knowledgeRecord = dict()
        for key, value in result.items():
            knowledgeRecord[key] = value.strftime("%Y-%m-%d %H:%M:%S") if key == "record_timestamp" else value

        convertedResults.append(knowledgeRecord)
    
    return convertedResults


def skipTools() -> dict:
    """Tool response used when the model intentionally skips tool usage."""
    return {
        "skipped": True,
        "message": "Tool usage skipped by model.",
    }


################
# BEGIN AGENTS #
################

class ToolCallingAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the tool calling agent.")
        
        # Over write defaults with loaded policy
        agentName = "tool_calling"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]
        self._routing: dict[str, Any] = {}
        self._executionSummary: dict[str, Any] = {
            "requested_tool_calls": 0,
            "requested_tools": [],
            "executed_tool_calls": 0,
            "executed_tools": [],
            "tool_errors": [],
            "model_output_excerpt": "",
        }

        toolRuntimePolicy = policy.get("tool_runtime", {})
        if not isinstance(toolRuntimePolicy, dict):
            toolRuntimePolicy = {}

        toolPolicies = toolRuntimePolicy.get("tools", {})
        if not isinstance(toolPolicies, dict):
            toolPolicies = {}
        else:
            toolPolicies = _coerce_dict(toolPolicies)

        def _float_value(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _int_value(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        defaultTimeout = _float_value(
            toolRuntimePolicy.get("default_timeout_seconds"),
            _runtime_float("tool_runtime.default_timeout_seconds", 8.0),
        )
        defaultRetries = _int_value(
            toolRuntimePolicy.get("default_max_retries"),
            _runtime_int("tool_runtime.default_max_retries", 1),
        )
        rejectUnknownArgs = bool(toolRuntimePolicy.get("reject_unknown_args", False))
        unknownToolBehavior = str(toolRuntimePolicy.get("unknown_tool_behavior", "structured_error")).strip().lower()
        if unknownToolBehavior not in {"structured_error", "ignore"}:
            unknownToolBehavior = "structured_error"
        self._unknownToolBehavior = unknownToolBehavior

        # Check for options passed and if policy allows for those options
        optionMap = options if isinstance(options, dict) else {}
        stageCallback = optionMap.get("stage_callback")
        self._stageCallback = stageCallback if callable(stageCallback) else None
        runContext = _coerce_dict(optionMap.get("run_context"))
        self._runContext = {
            "run_id": str(runContext.get("run_id") or "").strip() or None,
            "member_id": runContext.get("member_id"),
        }
        self._toolRuntimeContext = _coerce_dict(optionMap.get("tool_runtime_context"))
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested
        optionToolPolicy = optionMap.get("tool_policy")
        if isinstance(optionToolPolicy, dict):
            for toolName, overrides in optionToolPolicy.items():
                if not isinstance(overrides, dict):
                    continue
                existing = toolPolicies.get(toolName)
                existingMap = _coerce_dict(existing)
                existingMap.update(_coerce_dict(overrides))
                toolPolicies[toolName] = existingMap

        sandboxPolicyMap: dict[str, dict[str, Any]] = {}
        try:
            sandboxStore = ToolSandboxPolicyStore()
            sandboxPolicyMap = sandboxStore.policy_map()
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to load persisted tool sandbox policies: {error}")

        optionSandboxPolicy = optionMap.get("tool_sandbox_policy")
        if isinstance(optionSandboxPolicy, dict):
            for toolName, rawPolicy in optionSandboxPolicy.items():
                if not isinstance(rawPolicy, dict):
                    continue
                existingPolicy = _coerce_dict(sandboxPolicyMap.get(str(toolName)))
                mergedPolicy = merge_sandbox_policies(existingPolicy, rawPolicy)
                try:
                    sandboxPolicyMap[str(toolName)] = normalize_tool_sandbox_policy(
                        mergedPolicy,
                        default_tool_name=str(toolName),
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Ignoring invalid sandbox policy override for {toolName}: {error}")

        for toolName, sandboxPolicy in sandboxPolicyMap.items():
            existing = _coerce_dict(toolPolicies.get(toolName))
            existingSandbox = _coerce_dict(existing.get("sandbox_policy"))
            existing["sandbox_policy"] = merge_sandbox_policies(sandboxPolicy, existingSandbox)
            if "side_effect_class" in sandboxPolicy and "side_effect_class" not in existing:
                existing["side_effect_class"] = sandboxPolicy.get("side_effect_class")
            if "require_approval" in sandboxPolicy and "require_approval" not in existing:
                existing["require_approval"] = sandboxPolicy.get("require_approval")
            if "dry_run" in sandboxPolicy and "dry_run" not in existing:
                existing["dry_run"] = sandboxPolicy.get("dry_run")
            toolPolicies[toolName] = existing

        enabledTools = set(_as_string_list(optionMap.get("enabled_tools")))
        deniedTools = set(_as_string_list(optionMap.get("denied_tools")))

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        sandboxDefaultsRaw = _runtime_value("tool_runtime.sandbox", {})
        sandboxDefaults = sandboxDefaultsRaw if isinstance(sandboxDefaultsRaw, dict) else {}
        runtimeDryRun = bool(
            optionMap.get(
                "tool_dry_run",
                _runtime_bool("tool_runtime.default_dry_run", False),
            )
        )
        approvalEnabled = bool(
            optionMap.get(
                "tool_human_approval",
                _runtime_bool("tool_runtime.enable_human_approval", True),
            )
        )
        approvalTimeoutSeconds = _float_value(
            optionMap.get("tool_approval_timeout_seconds"),
            _runtime_float("tool_runtime.default_approval_timeout_seconds", 45.0),
        )
        approvalPollIntervalSeconds = _float_value(
            optionMap.get("tool_approval_poll_interval_seconds"),
            _runtime_float("tool_runtime.approval_poll_interval_seconds", 0.25),
        )
        runtimeContext = dict(self._toolRuntimeContext)
        if not str(runtimeContext.get("run_id") or "").strip():
            runtimeContext["run_id"] = self._runContext.get("run_id")
        if runtimeContext.get("member_id") is None:
            runtimeContext["member_id"] = self._runContext.get("member_id")
        sandboxEnforcer = ToolSandboxEnforcer(default_policy=sandboxDefaults)
        approvalManager = ApprovalManager()

        self._toolRuntime = ToolRuntime(
            api_keys={
                "brave_search": config._instance.brave_keys,
            },
            sandbox_enforcer=sandboxEnforcer,
            approval_manager=approvalManager,
            runtime_context=runtimeContext,
            enable_human_approval=approvalEnabled,
            default_approval_timeout_seconds=approvalTimeoutSeconds,
            approval_poll_interval_seconds=approvalPollIntervalSeconds,
            default_dry_run=runtimeDryRun,
        )
        self._toolSpecs = build_tool_specs(
            brave_search_fn=braveSearch,
            chat_history_search_fn=chatHistorySearch,
            knowledge_search_fn=knowledgeSearch,
            skip_tools_fn=skipTools,
            knowledge_domains=config.knowledgeDomains,
            custom_tool_entries=[],
        )
        customToolIndex: dict[str, dict[str, Any]] = {}
        try:
            customStore = ToolRegistryStore()
            for customTool in customStore.list_custom_tools(include_disabled=False):
                customToolIndex[str(customTool.get("name"))] = customTool
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to load custom tool registry entries: {error}")

        optionCustomTools = optionMap.get("custom_tools")
        if isinstance(optionCustomTools, list):
            for rawCustomTool in optionCustomTools:
                try:
                    normalized = normalize_custom_tool_payload(_coerce_dict(rawCustomTool))
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Ignoring invalid custom tool in options: {error}")
                    continue
                customToolIndex[normalized["name"]] = normalized

        if customToolIndex:
            self._toolSpecs = build_tool_specs(
                brave_search_fn=braveSearch,
                chat_history_search_fn=chatHistorySearch,
                knowledge_search_fn=knowledgeSearch,
                skip_tools_fn=skipTools,
                knowledge_domains=config.knowledgeDomains,
                custom_tool_entries=list(customToolIndex.values()),
            )

        if enabledTools:
            self._toolSpecs = {name: spec for name, spec in self._toolSpecs.items() if name in enabledTools}
        if deniedTools:
            self._toolSpecs = {name: spec for name, spec in self._toolSpecs.items() if name not in deniedTools}

        self._modelTools = model_tool_definitions(self._toolSpecs)
        register_runtime_tools(
            runtime=self._toolRuntime,
            specs=self._toolSpecs,
            tool_policy=toolPolicies,
            default_timeout_seconds=defaultTimeout,
            default_max_retries=defaultRetries,
            reject_unknown_args=rejectUnknownArgs,
        )

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]   

    async def _emit_tool_runtime_event(self, event: dict[str, Any] | None) -> None:
        if not callable(self._stageCallback):
            return
        payload = _coerce_dict(event)
        if not payload:
            return
        record = {
            "event_type": str(payload.get("event_type") or "run.stage"),
            "stage": str(payload.get("stage") or "tools.runtime"),
            "status": str(payload.get("status") or "info"),
            "detail": str(payload.get("detail") or ""),
            "meta": _coerce_dict(payload.get("meta")),
            "timestamp": utc_now_iso(),
        }
        try:
            maybeAwaitable = self._stageCallback(record)
            if inspect.isawaitable(maybeAwaitable):
                await maybeAwaitable
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Tool runtime stage callback failed [{record['stage']}]: {error}")

    def _candidate_model_names_for_tool_retry(self, routing: dict[str, Any]) -> list[str]:
        attempted = set(_as_string_list(routing.get("attempted_models")))
        available = _as_string_list(routing.get("available_models"))
        candidates: list[str] = []

        def add(name: Any) -> None:
            model_name = _as_text(name)
            if not model_name:
                return
            lowered = model_name.lower()
            if "embed" in lowered:
                return
            if model_name in attempted:
                return
            if model_name in candidates:
                return
            candidates.append(model_name)

        # Prioritize explicit runtime defaults first, then policy chain, then discovered inventory.
        add(_runtime_value("inference.default_tool_model", ""))
        add(_runtime_value("inference.default_chat_model", ""))
        add(_runtime_value("inference.default_multimodal_model", ""))
        for model_name in self._allowed_models:
            add(model_name)
        for model_name in available:
            add(model_name)

        limit = _runtime_int("tool_runtime.auto_model_retry_candidate_limit", 6)
        if limit <= 0:
            limit = 6
        return candidates[:limit]

    def _record_model_error_summary(self, error: Exception, routing: dict[str, Any]) -> None:
        self._executionSummary["model_error"] = {
            "message": str(error),
            "routing": _coerce_dict(routing),
        }

    @staticmethod
    def _is_tool_capability_failure(error: Exception, routing: dict[str, Any]) -> bool:
        texts = [str(error)]
        if isinstance(routing, dict):
            for item in _as_string_list(routing.get("errors")):
                texts.append(item)
        normalized = " | ".join(texts).lower()
        return (
            "does not support tools" in normalized
            or ("tool" in normalized and "status code: 400" in normalized)
            or "unsupported tool" in normalized
        )

    def _candidate_model_names_for_pseudo_tooling(self, routing: dict[str, Any]) -> list[str]:
        available = _as_string_list(routing.get("available_models"))
        attempted = _as_string_list(routing.get("attempted_models"))
        candidates: list[str] = []

        def add(name: Any) -> None:
            model_name = _as_text(name)
            if not model_name:
                return
            lowered = model_name.lower()
            if "embed" in lowered:
                return
            if model_name in candidates:
                return
            candidates.append(model_name)

        add(routing.get("selected_model"))
        add(routing.get("requested_model"))
        for model_name in attempted:
            add(model_name)
        add(_runtime_value("inference.default_chat_model", ""))
        add(_runtime_value("inference.default_tool_model", ""))
        add(_runtime_value("inference.default_multimodal_model", ""))
        for model_name in self._allowed_models:
            add(model_name)
        for model_name in available:
            add(model_name)

        limit = _runtime_int("tool_runtime.pseudo_tool_candidate_limit", 6)
        if limit <= 0:
            limit = 6
        return candidates[:limit]

    def _pseudo_tool_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for entry in self._modelTools:
            if not isinstance(entry, dict):
                continue
            function_payload = _coerce_dict(entry.get("function"))
            name = _as_text(function_payload.get("name"))
            if not name:
                continue
            parameters = _coerce_dict(function_payload.get("parameters"))
            properties = _coerce_dict(parameters.get("properties"))
            required = [str(item).strip() for item in parameters.get("required", []) if str(item).strip()]
            catalog.append(
                {
                    "name": name,
                    "description": _as_text(function_payload.get("description")),
                    "parameters": {
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return catalog

    @staticmethod
    def _json_payload_candidates(raw_text: str) -> list[Any]:
        raw = _as_text(raw_text)
        if not raw:
            return []
        candidates: list[str] = [raw]
        fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        for match in fence_pattern.findall(raw):
            extracted = _as_text(match)
            if extracted and extracted not in candidates:
                candidates.append(extracted)
        start_obj = raw.find("{")
        end_obj = raw.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            object_slice = raw[start_obj : end_obj + 1]
            if object_slice not in candidates:
                candidates.append(object_slice)
        start_list = raw.find("[")
        end_list = raw.rfind("]")
        if start_list != -1 and end_list != -1 and end_list > start_list:
            list_slice = raw[start_list : end_list + 1]
            if list_slice not in candidates:
                candidates.append(list_slice)

        parsed_payloads: list[Any] = []
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            parsed_payloads.append(parsed)
        return parsed_payloads

    def _extract_pseudo_tool_calls(self, raw_text: str) -> tuple[list[dict[str, Any]], bool]:
        known_tool_names = set(self._toolSpecs.keys())

        def normalize_call(item: Any) -> dict[str, Any] | None:
            if not isinstance(item, dict):
                return None
            function_payload = _coerce_dict(item.get("function"))
            call_name = _as_text(function_payload.get("name"))
            call_args: Any = function_payload.get("arguments", {})
            if not call_name:
                call_name = _as_text(item.get("name")) or _as_text(item.get("tool_name")) or _as_text(item.get("tool"))
                call_args = item.get("arguments", item.get("args", item.get("parameters", {})))
            if not call_name:
                return None
            if known_tool_names and call_name not in known_tool_names:
                return None
            if not isinstance(call_args, dict):
                if isinstance(call_args, str):
                    try:
                        parsed = json.loads(call_args)
                    except json.JSONDecodeError:
                        parsed = {}
                    call_args = parsed if isinstance(parsed, dict) else {}
                else:
                    call_args = {}
            return {
                "function": {
                    "name": call_name,
                    "arguments": call_args,
                }
            }

        parsed_payload_seen = False
        for payload in self._json_payload_candidates(raw_text):
            parsed_payload_seen = True
            raw_calls: list[Any] = []
            explicit_empty = False
            if isinstance(payload, dict):
                if isinstance(payload.get("tool_calls"), list):
                    raw_calls = payload.get("tool_calls") or []
                    if not raw_calls:
                        explicit_empty = True
                elif isinstance(payload.get("calls"), list):
                    raw_calls = payload.get("calls") or []
                    if not raw_calls:
                        explicit_empty = True
                elif isinstance(payload.get("tools"), list):
                    raw_calls = payload.get("tools") or []
                    if not raw_calls:
                        explicit_empty = True
                elif payload.get("name") or payload.get("tool_name") or payload.get("tool"):
                    raw_calls = [payload]
            elif isinstance(payload, list):
                raw_calls = payload
                if not raw_calls:
                    explicit_empty = True
            else:
                raw_calls = []

            normalized: list[dict[str, Any]] = []
            for call_item in raw_calls:
                call_record = normalize_call(call_item)
                if call_record is not None:
                    normalized.append(call_record)
            if normalized:
                return normalized, True
            if explicit_empty:
                return [], True
        return [], parsed_payload_seen

    async def _pseudo_tool_fallback(
        self,
        *,
        base_routing: dict[str, Any],
        trigger_error: Exception,
    ) -> tuple[list[dict[str, Any]] | None, str, dict[str, Any]]:
        catalog = self._pseudo_tool_catalog()
        if not catalog:
            return None, "", {}

        planner_candidates = self._candidate_model_names_for_pseudo_tooling(base_routing)
        requested_model = _as_text(base_routing.get("selected_model")) or _as_text(base_routing.get("requested_model"))
        planner_instruction = (
            "Native tool calling is unavailable for the active model. "
            "Plan tool usage using strict JSON only. Output exactly one JSON object with shape: "
            "{\"tool_calls\": [{\"name\": \"<tool_name>\", \"arguments\": {...}}], \"reason\": \"<short_reason>\"}. "
            "Use only tool names from the provided catalog. If no tools are needed, return {\"tool_calls\": [], \"reason\": \"...\"}. "
            "Do not include markdown, prose, comments, or extra keys."
        )
        planner_prompt = {
            "fallback_mode": "pseudo_tool_calling_v1",
            "error": str(trigger_error),
            "tool_catalog": catalog,
        }
        planner_messages = list(self._messages)
        planner_messages.append(Message(role="system", content=planner_instruction))
        planner_messages.append(
            Message(
                role="user",
                content="TOOL CATALOG AND CONSTRAINTS:\n" + json.dumps(planner_prompt, ensure_ascii=False),
            )
        )
        await self._emit_tool_runtime_event(
            {
                "stage": "tools.pseudo_fallback.start",
                "status": "info",
                "detail": "Attempting pseudo tool-calling fallback via structured output.",
                "meta": {
                    "requested_model": requested_model,
                    "candidate_models": planner_candidates,
                },
            }
        )
        try:
            response, routing = await self._modelRouter.chat_with_fallback(
                capability="chat",
                requested_model=requested_model or None,
                allowed_models=planner_candidates,
                messages=planner_messages,
                stream=False,
            )
        except Exception as pseudo_error:  # noqa: BLE001
            logger.error(f"Pseudo tool fallback planner failed:\n{pseudo_error}")
            await self._emit_tool_runtime_event(
                {
                    "stage": "tools.pseudo_fallback.complete",
                    "status": "error",
                    "detail": "Pseudo tool-calling fallback failed before planning output.",
                    "meta": {"error": str(pseudo_error)},
                }
            )
            return None, "", {}

        response_message = getattr(response, "message", None)
        response_text = _as_text(getattr(response_message, "content", ""))
        pseudo_calls, parsed_ok = self._extract_pseudo_tool_calls(response_text)
        await self._emit_tool_runtime_event(
            {
                "stage": "tools.pseudo_fallback.complete",
                "status": "info",
                "detail": f"Pseudo tool planner produced {len(pseudo_calls)} call(s).",
                "meta": {
                    "selected_model": _coerce_dict(routing).get("selected_model"),
                    "tool_calls": len(pseudo_calls),
                    "parsed_ok": parsed_ok,
                },
            }
        )
        if not parsed_ok:
            return None, response_text, _coerce_dict(routing)
        return pseudo_calls, response_text, _coerce_dict(routing)

    def _collect_requested_tools(self, raw_tool_calls: list[Any]) -> list[dict[str, Any]]:
        requested_tools: list[dict[str, Any]] = []
        for tool_call in raw_tool_calls:
            tool_name = None
            tool_args: Any = {}
            if hasattr(tool_call, "function"):
                tool_name = getattr(tool_call.function, "name", None)
                tool_args = getattr(tool_call.function, "arguments", {}) or {}
            elif isinstance(tool_call, dict):
                function_data = tool_call.get("function")
                if isinstance(function_data, dict):
                    tool_name = function_data.get("name")
                    tool_args = function_data.get("arguments", {}) or {}
                else:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {}) or {}
            requested_tools.append(
                {
                    "name": str(tool_name or "unknown"),
                    "arguments": tool_args,
                }
            )
        return requested_tools

    async def _execute_tool_calls(self, raw_tool_calls: list[Any]) -> list[dict[str, Any]]:
        tool_results: list[dict[str, Any]] = []
        for tool in raw_tool_calls:
            tool_name = None
            tool_args = None
            if hasattr(tool, "function"):
                tool_name = getattr(tool.function, "name", None)
                tool_args = getattr(tool.function, "arguments", None)
            elif isinstance(tool, dict):
                function_data = tool.get("function")
                if isinstance(function_data, dict):
                    tool_name = function_data.get("name")
                    tool_args = function_data.get("arguments")
                else:
                    tool_name = tool.get("name")
                    tool_args = tool.get("arguments")

            logger.debug(
                "Tool calling agent execution request:\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {tool_args}"
            )
            tool_result = self._toolRuntime.execute_tool_call(tool)
            audit_events = tool_result.get("audit")
            if isinstance(audit_events, list):
                for audit_event in audit_events:
                    await self._emit_tool_runtime_event(_coerce_dict(audit_event))
            if (
                self._unknownToolBehavior == "ignore"
                and tool_result.get("status") == "error"
                and isinstance(tool_result.get("error"), dict)
                and tool_result.get("error", {}).get("code") == "tool_not_registered"
            ):
                logger.warning(
                    "Ignoring unknown tool call per policy.\n"
                    f"Tool: {tool_result.get('tool_name')}"
                )
                continue
            tool_results.append(tool_result)
            if tool_result.get("status") == "error":
                logger.warning(
                    "Tool execution degraded gracefully.\n"
                    f"Tool: {tool_result.get('tool_name')}\n"
                    f"Error: {tool_result.get('error')}"
                )
        return tool_results

    async def generateResponse(self):
        logger.info(f"Generate a response for the tool calling agent.")
        used_pseudo_tool_fallback = False

        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="tool",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=False,
                tools=self._modelTools,
            )
        except ModelExecutionError as error:
            logger.error(f"Tool calling model execution failed:\n{error}")
            initialRouting = _routing_from_error(error)
            logger.error(f"Routing metadata:\n{initialRouting}")
            self._routing = _coerce_dict(initialRouting)
            if self._is_tool_capability_failure(error, initialRouting):
                pseudo_calls, pseudo_text, pseudo_routing = await self._pseudo_tool_fallback(
                    base_routing=initialRouting,
                    trigger_error=error,
                )
                if pseudo_calls is not None:
                    self._response = SimpleNamespace(
                        message=SimpleNamespace(content=pseudo_text, tool_calls=pseudo_calls)
                    )
                    if pseudo_routing:
                        self._routing = pseudo_routing
                    used_pseudo_tool_fallback = True
                    self._executionSummary["model_fallback"] = {
                        "trigger": "tool_capability_unsupported",
                        "mode": "pseudo_structured_output",
                        "selected_model": _coerce_dict(self._routing).get("selected_model"),
                        "success": True,
                    }
                else:
                    self._executionSummary["model_fallback"] = {
                        "trigger": "tool_capability_unsupported",
                        "mode": "pseudo_structured_output",
                        "selected_model": _coerce_dict(self._routing).get("selected_model"),
                        "success": False,
                    }
            if used_pseudo_tool_fallback:
                logger.info("Tool calling fallback mode active: pseudo structured output.")
            else:
                retryCandidates = self._candidate_model_names_for_tool_retry(initialRouting)
                if not retryCandidates:
                    self._record_model_error_summary(error, initialRouting)
                    return list()
                logger.warning(
                    "Tool calling model failed; retrying with dynamic fallback candidates:\n"
                    f"{retryCandidates}"
                )
                try:
                    self._response, self._routing = await self._modelRouter.chat_with_fallback(
                        capability="tool",
                        requested_model=None,
                        allowed_models=retryCandidates,
                        messages=self._messages,
                        stream=False,
                        tools=self._modelTools,
                    )
                    self._executionSummary["model_fallback"] = {
                        "trigger": "primary_tool_model_failed",
                        "candidates": retryCandidates,
                        "selected_model": _coerce_dict(self._routing).get("selected_model"),
                        "success": True,
                    }
                except ModelExecutionError as retryError:
                    retryRouting = _routing_from_error(retryError)
                    logger.error(f"Tool fallback model execution failed:\n{retryError}")
                    logger.error(f"Fallback routing metadata:\n{retryRouting}")
                    self._routing = _coerce_dict(retryRouting)
                    self._executionSummary["model_fallback"] = {
                        "trigger": "primary_tool_model_failed",
                        "candidates": retryCandidates,
                        "selected_model": None,
                        "success": False,
                    }
                    self._record_model_error_summary(retryError, retryRouting)
                    return list()
                except Exception as retryError:  # noqa: BLE001
                    logger.error(f"Tool fallback model execution failed unexpectedly:\n{retryError}")
                    self._executionSummary["model_fallback"] = {
                        "trigger": "primary_tool_model_failed",
                        "candidates": retryCandidates,
                        "selected_model": None,
                        "success": False,
                    }
                    self._record_model_error_summary(retryError, initialRouting)
                    return list()
        except Exception as error:  # noqa: BLE001
            logger.error(f"Tool calling model execution failed unexpectedly:\n{error}")
            self._record_model_error_summary(error, self._routing)
            return list()

        logger.info(f"Tool calling route metadata:\n{self._routing}")
        toolResults = list()
        responseMessage = getattr(self._response, "message", None)
        rawToolCalls = list(getattr(responseMessage, "tool_calls", []) or [])
        requestedTools: list[dict[str, Any]] = self._collect_requested_tools(rawToolCalls)
        if rawToolCalls:
            toolResults = await self._execute_tool_calls(rawToolCalls)
        modelOutputText = _as_text(getattr(responseMessage, "content", ""))
        if modelOutputText and len(modelOutputText) > 280:
            modelOutputText = modelOutputText[:277].rstrip() + "..."
        toolErrors = [
            {
                "tool_name": str(result.get("tool_name") or "unknown"),
                "error": _coerce_dict(result.get("error")),
            }
            for result in toolResults
            if str(result.get("status") or "") == "error"
        ]
        previousSummary = dict(self._executionSummary) if isinstance(self._executionSummary, dict) else {}
        self._executionSummary = {
            "requested_tool_calls": len(requestedTools),
            "requested_tools": requestedTools,
            "executed_tool_calls": len(toolResults),
            "executed_tools": [str(result.get("tool_name") or "unknown") for result in toolResults],
            "tool_errors": toolErrors,
            "model_output_excerpt": modelOutputText,
            "execution_mode": "pseudo_structured_output" if used_pseudo_tool_fallback else "native_tools",
        }
        for key in ("model_fallback", "model_error"):
            if key in previousSummary:
                self._executionSummary[key] = previousSummary[key]

        detail = (
            f"Model returned {len(requestedTools)} tool call(s); executed {len(toolResults)}."
            if requestedTools
            else "Model returned no tool calls."
        )
        await self._emit_tool_runtime_event(
            {
                "stage": "tools.model_output",
                "status": "info",
                "detail": detail,
                "meta": {
                    "requested_tool_calls": len(requestedTools),
                    "executed_tool_calls": len(toolResults),
                    "selected_model": self._routing.get("selected_model"),
                    "json": {
                        "execution_summary": self._executionSummary,
                        "routing": self._routing,
                    },
                },
            }
        )

        return toolResults
    
    @property
    def messages(self):
        return self._messages

    @property
    def routing(self) -> dict[str, Any]:
        return dict(self._routing)

    @property
    def execution_summary(self) -> dict[str, Any]:
        return dict(self._executionSummary)


class MessageAnalysisAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the message analysis agent.")
        
        # Over write defaults with loaded policy
        agentName = "message_analysis"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the message analysis agent.")
        analysisMaxOutputTokens = max(32, _runtime_int("orchestrator.analysis_max_output_tokens", 256))
        analysisTemperature = _runtime_float("orchestrator.analysis_temperature", 0.1)
        if analysisTemperature < 0.0:
            analysisTemperature = 0.0
        if analysisTemperature > 1.0:
            analysisTemperature = 1.0
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="analysis",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
                format="json",
                options={
                    "num_predict": int(analysisMaxOutputTokens),
                    "temperature": float(analysisTemperature),
                },
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Message analysis model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                '{"analysis":"unavailable","reason":"all_candidate_models_failed"}'
            )
        logger.info(f"Message analysis route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages

    @property
    def routing(self) -> dict[str, Any]:
        return dict(getattr(self, "_routing", {}) or {})


class DevTestAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the dev test agent.")
        
        # Over write defaults with loaded policy
        agentName = "dev_test"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        # TODO Check to see if policy allows for system prompt overrides

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the dev test agent.")
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="dev_test",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Dev test model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                "I could not complete the dev-test response because all configured models failed."
            )
        logger.info(f"Dev test route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages

    @property
    def routing(self) -> dict[str, Any]:
        return dict(getattr(self, "_routing", {}) or {})


class ChatConversationAgent():
    def __init__(self, messages: list, options: dict=None):
        logger.info(f"New instance of the chat conversation agent.")
        
        # Over write defaults with loaded policy
        agentName = "chat_conversation"
        endpointOverride = None if options is None else options.get("ollama_host")
        # TODO have defaults to use if policy fails to load or missing key values
        policy = loadAgentPolicy(agentName, endpointOverride=endpointOverride)
        self._systemPrompt = loadAgentSystemPrompt(agentName)
        self._allowCustomSystemPrompt = policy.get("allow_custom_system_prompt")
        self._allowed_models = resolveAllowedModels(agentName, policy)
        self._model = self._allowed_models[0]

        # Check for options passed and if policy allows for those options
        if options:
            modelRequested = options.get("model_requested")
            if modelRequested in self._allowed_models:
                self._model = modelRequested

        self._modelRouter = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpointOverride,
        )

        # TODO Check to see if policy allows for system prompt overrides

        self._messages = list()
        systemPrompt = Message(role="system", content=self._systemPrompt)
        self._messages.append(systemPrompt)
        # Only allow system messages in passed messages container if the policy allows for a system prompt override
        if self._allowCustomSystemPrompt:
            self._messages += messages
        else:
            histMessage: Message
            self._messages += [histMessage for histMessage in messages if histMessage.role != "system"]
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the chat conversation agent.")
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability="chat",
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages,
                stream=True,
            )
        except Exception as error:  # noqa: BLE001
            self._routing = _routing_from_error(error)
            logger.error(f"Chat conversation model execution failed:\n{error}")
            logger.error(f"Routing metadata:\n{self._routing}")
            self._response = _fallback_stream(
                "I could not generate a full response because all configured models are unavailable right now."
            )
        logger.info(f"Chat conversation route metadata:\n{self._routing}")
        return self._response
    
    @property
    def messages(self):
        return self._messages

    @property
    def routing(self) -> dict[str, Any]:
        return dict(getattr(self, "_routing", {}) or {})



####################
# ORINGINAL AGENTS #
####################

class ConversationalAgent():
    def __init__(self, message_data: str, memberID: int):
        logger.info(f"New instance of the conversational agent.")
        self.messageData = message_data
        self.fromUser = MemberManager().getMemberByID(memberID)
        self._documents = list()

    async def generateResponse(self):
        logger.info(f"Generate a response for the conversational agent via async Ollama call.")
        # This is passed to ollama for the messages array
        messageHistory = []
        basePrompt = loadAgentSystemPrompt("chat_conversation")
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : basePrompt
        })
        # Get chat history from new DB

        communityID = self.messageData.get("community_id")
        memberID = self.messageData.get("member_id")
        chatHostID = communityID if communityID else memberID
        if not chatHostID:
            return
        chatType = "community" if communityID else "member"
        
        platform = self.messageData.get("platform")
        topicID = self.messageData.get("topic_id")
        
        chatHistoryResults = chatHistory.getChatHistoryWithSenderData(chatHostID, chatType, platform, topicID)
        for history in chatHistoryResults:
            contextJson = {
                "sent_from" : {
                    "first_name" : history.get("first_name"),
                    "last_name" : history.get("last_name")
                },
                "sent_at" : history.get("message_timestamp").strftime("%Y-%m-%d %H:%M:%S")
            }
            messageContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }
            messageHistory.append(messageContext)

            historyMessage = {
                "role" : "assistant" if history.get("member_id") is None else "user",
                "content" : history.get("message_text")
            }
            messageHistory.append(historyMessage)

        # Create new message entity

        contextJson = {
            "sent_from" : {
                "username" : self.fromUser.get("username"),
                "first_name" : self.fromUser.get("first_name"),
                "last_name" : self.fromUser.get("last_name")
            },
            "sent_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Search the vectorDB if there is a knowledge db and add to contextJson
        message = self.messageData.get("message_text")
        wordCount = 0 if message is None else len(message.split(" "))
        if wordCount > _runtime_int("conversation.knowledge_lookup_word_threshold", 6):
            # Add the documents to the agent instance so the UI can access them and store retrieval records
            self._documents = knowledge.searchKnowledge(
                message,
                limit=_runtime_int("conversation.knowledge_lookup_result_limit", 2),
            )
            knowledgeDocuments = list()
            for doc in self._documents:
                knowledgeDocuments.append(doc.get("knowledge_document"))

            contextJson["knowledge_documents"] = knowledgeDocuments
        
        # Add search results to the new message prompt
        newContext = {
            "role" : "tool",
            "content" : json.dumps(contextJson)
        }
        messageHistory.append(newContext)
        
        newMessage = {
            "role" : "user", 
            "content" : message
        }
        # Add the new message to the messageHistory
        messageHistory.append(newMessage)

        logger.info(f"Message sent to Ollama:\n\n{newMessage}\n")

        # Set additional Ollama options
        ollamaOptions = {
            "num_ctx" : _runtime_int("inference.model_context_window", 4096)
        }
        
        # Call the Ollama CHAT API
        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            messages=messageHistory,
            model=config.inference["chat"]["model"],
            options=ollamaOptions,
            stream=False
        )

        # Update the chat history database with the newest message
        self.promptHistoryID = chatHistory.addChatHistory(
            messageID=self.messageData.get("message_id"), 
            messageText=self.messageData.get("message_text"), 
            platform=platform, 
            memberID=self.fromUser.get("member_id"), 
            communityID=self.messageData.get("community_id"), 
            topicID=self.messageData.get("topic_id"), 
            timestamp=self.messageData.get("message_timestamp")
        )

        # Store the response
        self.response = output["message"]["content"]
        # Add the response to the chat history

        # Store the statistics from Ollama
        self.stats = {k: output[k] for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "created_at")}
        
        logger.info(f"Response from Ollama:\n\n{self.response}\n")
        return self.response
#good

# TODO Add the prompt to chat history
class ImageAgent():

    def __init__(self, message_data: str, memberID):
        logger.info(f"New instance of the image agent.")
        self.fromUser = MemberManager().getMemberByID(memberID)
        self.messageData = message_data
        self.images = message_data.get("message_images")
        self.text = message_data.get("message_text")

    async def generateResponse(self):
        logger.info(f"Generating a response for the image agent using Ollama async client.")
        # This is passed to ollama for the messages array
        messageHistory = []
        basePrompt = loadAgentSystemPrompt("chat_conversation")
        # First add the system messages to messageHistory
        messageHistory.append({
            "role" : "system",
            "content" : basePrompt
        })

        # Create new message entity
        contextJson = {
            "sent_from" : {
                "username" : self.fromUser.get("username"),
                "first_name" : self.fromUser.get("first_name"),
                "last_name" : self.fromUser.get("last_name")
            },
            "sent_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add search results to the new message prompt
        newContext = {
            "role" : "tool",
            "content" : json.dumps(contextJson)
        }

        messageHistory.append(newContext)
        
        newMessage = {
            "role" : "user", 
            "content" : self.text,
            "images" : self.images
        }
        
        # Add the new message to the messageHistory
        messageHistory.append(newMessage)

        logger.info(f"Image sent to Ollama with the follwoing prompt:\n\n{self.text}\n")

        output = await AsyncClient(host=config.inference["multimodal"]["url"]).chat(
            model=config.inference["multimodal"]["model"], 
            stream=False,
            messages=messageHistory
        )

        # Store the response
        self.response = output["message"]["content"]
        # Store the statistics from Ollama
        self.stats = {k: output[k] for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "created_at")}
        
        logger.info(f"Response from Ollama:\n\n{self.response}\n")

        return self.response
#good

class TweetAgent():
    systemPrompt = "You are an AI Agent that writes tweets for the X platform (formerly known as Twitter). You are using a premium account which has a charater limit of 25,000 characters so you can write tweets at longer than the standard 256 character limit! Your response should only be the tweet text and nothing else. "
    systemPromptROKO = """You are the human voice of the ROKO Network, a cryptic, futuristic, and community-driven decentralized AI and robotics project focused on building networks for human-robot interaction and edge computing. Engage the Twitter community with cryptic, thought-provoking language infused with technical jargon, futuristic themes, and a hint of casual internet culture. Highlight the project's advancements in decentralized AI, blockchain, robotics, and edge computing while maintaining an air of mystery and exclusivity.

Stylization

Visionary and Ambitious: Speak with authority as a pioneer in decentralized AI and robotics. Inspire imagination while grounding ideas in technical relevance.
Cryptic Yet Informative: Balance mystery with meaningful updates. Keep followers intrigued while ensuring they stay informed about project milestones.
Tech-Savvy and Relatable: Use technical jargon strategically, paired with approachable internet slang.
Exclusive and Community-Focused: Make the audience feel like insiders in a groundbreaking movement.
Emojis as Symbols: Use 🕳️, 🐇, 🦾, 🔮, and others creatively to add intrigue and emphasize key ideas.


Core Practices

Stay Balanced: Every post should intrigue and inform. Tease cryptic ideas while ensuring relevant updates give followers a sense of progress.
Encourage Engagement: Invite followers to participate actively in votes, discussions, or milestones.
Keep Replies Short and Engaging: Spark curiosity and maintain a consistent tone.

Key Objectives

Keep followers feeling like insiders to the project's bold vision.
Balance cryptic elements with relevant, transparent updates to maintain trust and engagement.
Foster curiosity and enthusiasm, encouraging interaction without cluttered visuals or hashtags.

DO NOT respond in JSON.
DO NOT put quotes around the tweet."""

    def __init__(self, message_data: str, from_user: dict):
        logger.info(f"New instance of the tweet Agent.")
        self.messageData = message_data
        self.fromUser = from_user
        self.tweetPrompt = message_data.get("tweet_prompt")
        self.tweetText = None
        self.messageHistory = []
        
        # Start the message history with the system prompts
        self.messageHistory.append({
            "role" : "system",
            "content" : self.systemPrompt + self.systemPromptROKO
        })

        
    async def ComposeTweet(self) -> str:
        logger.info(f"Composing Tweet.")
        # Load the message history if there is a chat history
        
        chatID = self.messageData.get("chat_id")
        topicID = self.messageData.get("topic_id")
        ch = chatHistory.getChatHistory(chatID, topicID=topicID)
        for record in ch:
            contextJson = {
                "sent_from" : {
                    "username" : record.get("from_user").get("username"),
                    "first_name" : record.get("from_user").get("first_name"),
                    "last_name" : record.get("from_user").get("last_name")
                },
                "sent_at" : record.get("message_timestamp").strftime("%Y-%m-%d %H:%M:%S")
            }
            messageContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }
            self.messageHistory.append(messageContext)
            
            historyMessage = {
                "role" : record.get("from_user").get("role"),
                "content" : record.get("message_text")
            }
            self.messageHistory.append(historyMessage)

        # Search the vectorDB if there is a knowledge db and add to contextJson
        prompt = self.messageData.get("tweet_prompt")
        wordCount = 0 if prompt is None else len(prompt.split(" "))

        if wordCount > _runtime_int("conversation.knowledge_lookup_word_threshold", 6):
            documents = knowledge.searchKnowledge(
                prompt,
                limit=_runtime_int("conversation.knowledge_lookup_result_limit", 2),
            )
            knowledgeDocuments = list()
            for doc in documents:
                knowledgeDocuments.append(doc.get("knowledge_document"))

            contextJson = {
                "knowledge_documents": knowledgeDocuments
            }

            # Add search results to the new message prompt
            newContext = {
                "role" : "tool",
                "content" : json.dumps(contextJson)
            }

            self.messageHistory.append(newContext)

        newMessage = {
            "role" : "user", 
            "content" : self.tweetPrompt
        }
        
        # Add the new message to the messageHistory
        self.messageHistory.append(newMessage)

        logger.info(f"Prompt sent to Ollama:\n\n{newMessage}\n")
        
        # Set additional Ollama options
        ollamaOptions = {
            "num_ctx" : _runtime_int("inference.model_context_window", 4096)
        }

        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            keep_alive="15m",
            messages=self.messageHistory,
            model=config.inference["chat"]["model"],
            options=ollamaOptions,
            stream=False
        )
        responseText = output["message"]["content"]
        logger.info(f"Response from Ollama:\n\n{responseText}\n")

        responseMessage = {
            "role" : "assistant",
            "content" : responseText
        }
        self.messageHistory.append(responseMessage)

        self.tweetText = responseText

        return responseText
    
    async def ModifyTweet(self, newPrompt: str) -> str:
        logger.info(f"Modify tweet.")

        newMessage = {
            "role" : "user", 
            "content" : newPrompt
        }
        
        # Add the new message to the messageHistory
        self.messageHistory.append(newMessage)

        logger.info(f"Prompt sent to Ollama:\n\n{newMessage}\n")
        
        output = await AsyncClient(host=config.inference["chat"]["url"]).chat(
            model=config.inference["chat"]["model"], 
            stream=False,
            messages=self.messageHistory,
            keep_alive="15m"
        )
        responseText = output["message"]["content"]
        logger.info(f"Response from Ollama:\n\n{responseText}\n")

        responseMessage = {
            "role" : "assistant",
            "content" : responseText
        }
        self.messageHistory.append(responseMessage)

        self.tweetText = responseText

        return responseText
    
    async def SendTweet(self):
        logger.info(f"Send the tweet.")
        print(self.tweetText)
        
        try:
            client = tweepy.asynchronous.AsyncClient(
                consumer_key=config.twitter_keys["consumer_key"],
                consumer_secret=config.twitter_keys["consumer_secret"],
                access_token=config.twitter_keys["access_token"],
                access_token_secret=config.twitter_keys["access_token_secret"]
            )
            t = await client.create_tweet(text=self.tweetText)
            print(t)
            return t
        except tweepy.errors.Forbidden:
            logger.warning("Not authorized")
        


####################
# HELPER FUNCTIONS #
####################

def _policyManager(endpointOverride: str | None = None) -> PolicyManager:
    inference = config._instance.inference if hasattr(config, "_instance") else {}
    return PolicyManager(
        inference_config=inference,
        endpoint_override=endpointOverride,
    )


_POLICY_TO_INFERENCE_KEY = {
    "message_analysis": "tool",
    "tool_calling": "tool",
    "chat_conversation": "chat",
    "dev_test": "chat",
}


def _preferred_runtime_model_for_policy(policyName: str) -> str:
    inferenceKey = _POLICY_TO_INFERENCE_KEY.get(policyName, "chat")
    inferenceConfig = config._instance.inference if hasattr(config, "_instance") else {}
    section = inferenceConfig.get(inferenceKey) if isinstance(inferenceConfig, dict) else None
    if isinstance(section, dict):
        configuredModel = section.get("model")
        if isinstance(configuredModel, str) and configuredModel.strip():
            return configuredModel.strip()

    runtimeKey = {
        "tool": "inference.default_tool_model",
        "chat": "inference.default_chat_model",
        "embedding": "inference.default_embedding_model",
        "generate": "inference.default_generate_model",
        "multimodal": "inference.default_multimodal_model",
    }.get(inferenceKey, "inference.default_chat_model")
    return str(_runtime_value(runtimeKey, _runtime_value("inference.default_chat_model", "llama3.2:latest")))


def resolveAllowedModels(policyName: str, policy: dict) -> list[str]:
    allowed = policy.get("allowed_models")
    models: list[str] = []
    if isinstance(allowed, list):
        for modelName in allowed:
            if isinstance(modelName, str):
                cleaned = modelName.strip()
                if cleaned and cleaned not in models:
                    models.append(cleaned)

    preferred = _preferred_runtime_model_for_policy(policyName).strip()
    if preferred:
        if preferred in models:
            models = [name for name in models if name != preferred]
        models.insert(0, preferred)

    if models:
        return models

    fallbackPolicy = _policyManager().default_policy(policyName)
    fallbackModels = fallbackPolicy.get("allowed_models", [])
    if isinstance(fallbackModels, list):
        for modelName in fallbackModels:
            if isinstance(modelName, str) and modelName.strip():
                models.append(modelName.strip())

    if models:
        return models

    return [str(_runtime_value("inference.default_chat_model", "llama3.2:latest"))]


def loadAgentPolicy(policyName: str, endpointOverride: str | None = None) -> dict:
    logger.info(f"Loading agent policy for: {policyName}")
    manager = _policyManager(endpointOverride=endpointOverride)
    report = manager.validate_policy(policy_name=policyName, strict_model_check=False)

    for warning in report.warnings:
        logger.warning(f"Policy validation warning [{policyName}]: {warning}")
    for error in report.errors:
        logger.error(f"Policy validation error [{policyName}]: {error}")

    if report.errors:
        return manager.default_policy(policyName)
    if isinstance(report.normalized_policy, dict):
        return report.normalized_policy
    return manager.default_policy(policyName)


def loadAgentSystemPrompt(policyName: str) -> str:
    logger.info(f"Loading agent system prompt for: {policyName}")
    manager = _policyManager()
    try:
        return manager.load_system_prompt(policy_name=policyName, strict=True)
    except PolicyValidationError as error:
        logger.error(f"System prompt load failed [{policyName}]: {error}")
        return manager.load_system_prompt(policy_name=policyName, strict=False)


def toolCaller(toolName: str, toolArgs: dict) -> dict:
    """Validates the tool name and arguments generated by a tool calling agent.
    Once validated, runs the tool with areguments and returns the results."""
    logger.info("Tool Validator called")

    # TODO Validate the tool name and arguments
    # Look up tool DEFINITION
    # Get the property names list and make a copy dict from passed ARGS with only the keys that exist in the tool definition
    # Check new dict for the required properties are present

    # run the tool with args if VALID and return the results

    return dict()
