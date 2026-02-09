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

import asyncio
import copy
from datetime import datetime, timedelta, timezone
import ipaddress
import inspect
import json
import logging
import re
import requests
import shlex
import subprocess
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
from urllib.parse import urlparse
from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.history_recall import (
    ProgressiveHistoryExplorer,
    ProgressiveHistoryExplorerConfig,
)
from hypermindlabs.model_router import ModelExecutionError, ModelRouter
from hypermindlabs.personality_engine import PersonalityEngine, PersonalityRuntimeConfig
from hypermindlabs.personality_injector import PersonalityInjector
from hypermindlabs.personality_rollup import NarrativeRollupEngine
from hypermindlabs.personality_store import PersonalityStoreManager
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
    CollaborationWorkspaceManager,
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
collaboration = CollaborationWorkspaceManager()
personalityStore = PersonalityStoreManager()
personalityEngine = PersonalityEngine()
personalityInjector = PersonalityInjector()
narrativeRollup = NarrativeRollupEngine()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")


_CONTROL_PLANE_MANAGER_CACHE: dict[str, dict[str, Any]] = {}
_CONTROL_PLANE_POLICY_CACHE: dict[str, dict[str, Any]] = {}
_CONTROL_PLANE_PROMPT_CACHE: dict[str, dict[str, Any]] = {}
_TOOL_CAPABILITY_CACHE: dict[str, dict[str, Any]] = {}


def _runtime_int(path: str, default: int) -> int:
    return config.runtimeInt(path, default)


def _runtime_float(path: str, default: float) -> float:
    return config.runtimeFloat(path, default)


def _runtime_value(path: str, default: Any = None) -> Any:
    return config.runtimeValue(path, default)


def _runtime_bool(path: str, default: bool) -> bool:
    return config.runtimeBool(path, default)


def _runtime_int_list(path: str, default: list[int]) -> list[int]:
    raw_value = _runtime_value(path, default)
    parsed: list[int] = []
    source: list[Any]
    if isinstance(raw_value, list):
        source = raw_value
    elif isinstance(raw_value, str):
        source = [item.strip() for item in re.split(r"[,;\n]+", raw_value)]
    else:
        source = list(default)

    for item in source:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        if value in parsed:
            continue
        parsed.append(value)
    if not parsed:
        return list(default)
    return parsed


def _progressive_history_runtime_payload(*, max_selected_override: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": _runtime_bool("retrieval.progressive_history_enabled", True),
        "max_rounds": max(1, _runtime_int("retrieval.progressive_history_max_rounds", 5)),
        "round_windows_hours": _runtime_int_list(
            "retrieval.progressive_history_round_windows_hours",
            [12, 48, 168, 720],
        ),
        "semantic_limit_start": max(1, _runtime_int("retrieval.progressive_history_semantic_limit_start", 3)),
        "semantic_limit_step": max(0, _runtime_int("retrieval.progressive_history_semantic_limit_step", 2)),
        "timeline_limit_start": max(2, _runtime_int("retrieval.progressive_history_timeline_limit_start", 24)),
        "timeline_limit_step": max(0, _runtime_int("retrieval.progressive_history_timeline_limit_step", 24)),
        "context_radius": max(1, _runtime_int("retrieval.progressive_history_context_radius", 2)),
        "match_threshold": max(0.05, min(0.98, _runtime_float("retrieval.progressive_history_match_threshold", 0.42))),
        "max_selected": max(1, _runtime_int("retrieval.progressive_history_max_selected", 8)),
        "max_message_chars": max(40, _runtime_int("retrieval.progressive_history_max_message_chars", 220)),
    }
    if isinstance(max_selected_override, int) and max_selected_override > 0:
        payload["max_selected"] = max(1, min(int(max_selected_override), int(payload["max_selected"])))
    return payload


def _control_plane_cache_enabled() -> bool:
    return _runtime_bool("orchestrator.control_plane_cache_enabled", True)


def _control_plane_cache_ttl_seconds() -> float:
    ttl = _runtime_float("orchestrator.control_plane_cache_ttl_seconds", 60.0)
    if ttl <= 0:
        return 0.0
    return max(0.25, float(ttl))


def _normalize_model_name(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_embedding_model(model_name: str) -> bool:
    lowered = _normalize_model_name(model_name).lower()
    return "embed" in lowered if lowered else False


def _dedupe_models(candidates: list[Any]) -> list[str]:
    ordered: list[str] = []
    for raw_name in candidates:
        model_name = _normalize_model_name(raw_name)
        if not model_name:
            continue
        if model_name in ordered:
            continue
        ordered.append(model_name)
    return ordered


def _runtime_model_list(path: str, default: list[str] | None = None) -> list[str]:
    raw_value = _runtime_value(path, default if default is not None else [])
    if isinstance(raw_value, list):
        return _dedupe_models(raw_value)
    if isinstance(raw_value, str):
        separators = [",", "\n", ";"]
        parts = [raw_value]
        for separator in separators:
            expanded: list[str] = []
            for part in parts:
                expanded.extend(part.split(separator))
            parts = expanded
        return _dedupe_models(parts)
    return _dedupe_models(default or [])


def _cache_valid(entry: dict[str, Any], ttl_seconds: float) -> bool:
    if not isinstance(entry, dict):
        return False
    cached_at = entry.get("cached_at")
    if not isinstance(cached_at, (int, float)):
        return False
    return (time.monotonic() - float(cached_at)) <= ttl_seconds


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


def _timeout_seconds(path: str, default: float) -> float | None:
    timeout = _runtime_float(path, default)
    if timeout <= 0:
        return None
    return float(timeout)


async def _next_stream_chunk(
    stream_iterator: AsyncIterator[Any],
    *,
    idle_timeout_seconds: float | None = None,
    deadline_monotonic: float | None = None,
) -> Any:
    timeout_seconds = idle_timeout_seconds if isinstance(idle_timeout_seconds, (int, float)) else None
    if isinstance(timeout_seconds, (int, float)) and timeout_seconds <= 0:
        timeout_seconds = None

    if isinstance(deadline_monotonic, (int, float)):
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Stream total timeout exceeded.")
        timeout_seconds = min(timeout_seconds, remaining) if timeout_seconds else remaining

    if timeout_seconds:
        return await asyncio.wait_for(anext(stream_iterator), timeout=timeout_seconds)
    return await anext(stream_iterator)


def _routing_from_error(error: Exception) -> dict[str, Any]:
    if isinstance(error, ModelExecutionError):
        metadata = getattr(error, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
    return {"errors": [str(error)], "status": "failed_all_candidates"}


_ALLOWED_TOOL_HINTS = {
    "braveSearch",
    "curlRequest",
    "chatHistorySearch",
    "knowledgeSearch",
    "skipTools",
    "knownUsersList",
    "messageKnownUser",
    "upsertProcessWorkspace",
    "listProcessWorkspace",
    "updateProcessWorkspaceStep",
    "listOutboxMessages",
}
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


def _chunk_field(chunk: Any, field: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(field, default)
    return getattr(chunk, field, default)


def _chunk_message_content(chunk: Any) -> str:
    message_payload = _chunk_field(chunk, "message")
    if isinstance(message_payload, dict):
        return str(message_payload.get("content") or "")
    return str(getattr(message_payload, "content", "") or "")


def _chunk_done(chunk: Any) -> bool:
    return _as_bool(_chunk_field(chunk, "done", False), False)


def _chunk_stream_stats(chunk: Any) -> dict[str, Any]:
    return {
        "total_duration": _chunk_field(chunk, "total_duration"),
        "load_duration": _chunk_field(chunk, "load_duration"),
        "prompt_eval_count": _chunk_field(chunk, "prompt_eval_count"),
        "prompt_eval_duration": _chunk_field(chunk, "prompt_eval_duration"),
        "eval_count": _chunk_field(chunk, "eval_count"),
        "eval_duration": _chunk_field(chunk, "eval_duration"),
    }


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


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


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
    for key in (
        "model",
        "selected_model",
        "tool_calls",
        "requested_tool_calls",
        "executed_tool_calls",
        "action",
        "active_process_count",
        "pending_outbox_count",
    ):
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
        if "suggestion_count" in json_payload:
            summary["suggestion_count"] = json_payload.get("suggestion_count")
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
        "image_context": {
            "present": _as_bool(_coerce_dict(payload.get("image_context")).get("present"), False),
            "image_count": _safe_int(_coerce_dict(payload.get("image_context")).get("image_count"), 0),
            "caption_excerpt": _truncate_for_prompt(_coerce_dict(payload.get("image_context")).get("caption"), 120),
            "width": _safe_int(_coerce_dict(payload.get("image_context")).get("width"), 0),
            "height": _safe_int(_coerce_dict(payload.get("image_context")).get("height"), 0),
        },
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
        "workspace_context": _workspace_context_stage_preview(payload.get("workspace_context")),
        "personality_context": _personality_context_stage_preview(payload.get("personality_context")),
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
    progressive_payload = _coerce_dict(selection.get("progressive_recall"))
    rounds_raw = progressive_payload.get("rounds")
    rounds = rounds_raw if isinstance(rounds_raw, list) else []
    progressive_rounds_preview: list[dict[str, Any]] = []
    for round_item in rounds[:6]:
        round_data = _coerce_dict(round_item)
        progressive_rounds_preview.append(
            {
                "round": round_data.get("round"),
                "time_window_hours": round_data.get("time_window_hours"),
                "scope_topic": round_data.get("scope_topic"),
                "semantic_hits": round_data.get("semantic_hits"),
                "timeline_hits": round_data.get("timeline_hits"),
                "candidate_count": round_data.get("candidate_count"),
                "best_history_id": round_data.get("best_history_id"),
                "best_score": round_data.get("best_score"),
                "found": round_data.get("found"),
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
        "progressive_recall": {
            "enabled": progressive_payload.get("enabled"),
            "used": _as_bool(selection.get("progressive_used"), False),
            "found": progressive_payload.get("found"),
            "found_round": progressive_payload.get("found_round"),
            "decision_reason": progressive_payload.get("decision_reason"),
            "target_history_id": progressive_payload.get("target_history_id"),
            "target_context_index": progressive_payload.get("target_context_index"),
            "target_score": progressive_payload.get("target_score"),
            "round_count": len(rounds),
            "rounds": progressive_rounds_preview,
        },
    }


def _workspace_context_stage_preview(workspace_context: dict[str, Any] | None) -> dict[str, Any]:
    payload = _coerce_dict(workspace_context)
    active_raw = payload.get("active_processes")
    active = active_raw if isinstance(active_raw, list) else []
    outbox_raw = payload.get("recent_outbox")
    outbox = outbox_raw if isinstance(outbox_raw, list) else []

    active_preview: list[dict[str, Any]] = []
    for process in active[:6]:
        record = _coerce_dict(process)
        active_preview.append(
            {
                "process_id": record.get("process_id"),
                "process_label": _truncate_for_prompt(record.get("process_label"), 80),
                "process_status": _as_text(record.get("process_status"), "active"),
                "completion_percent": record.get("completion_percent"),
                "steps_completed": record.get("steps_completed"),
                "steps_total": record.get("steps_total"),
                "next_step_label": _truncate_for_prompt(record.get("next_step_label"), 100),
                "next_step_status": _as_text(record.get("next_step_status")),
                "updated_at": record.get("updated_at"),
            }
        )

    outbox_preview: list[dict[str, Any]] = []
    for message in outbox[:6]:
        record = _coerce_dict(message)
        outbox_preview.append(
            {
                "outbox_id": record.get("outbox_id"),
                "target_username": _as_text(record.get("target_username")),
                "delivery_status": _as_text(record.get("delivery_status"), "queued"),
                "process_id": record.get("process_id"),
                "message_excerpt": _truncate_for_prompt(record.get("message_excerpt"), 100),
                "created_at": record.get("created_at"),
            }
        )

    return {
        "schema": _as_text(payload.get("schema"), "ryo.workspace_context.v1"),
        "enabled": _as_bool(payload.get("enabled"), True),
        "active_process_count": _safe_int(payload.get("active_process_count"), len(active_preview)),
        "pending_outbox_count": _safe_int(payload.get("pending_outbox_count"), 0),
        "resume_recommended": _as_bool(payload.get("resume_recommended"), bool(active_preview)),
        "active_processes": active_preview,
        "recent_outbox": outbox_preview,
    }


def _personality_runtime_config() -> PersonalityRuntimeConfig:
    payload = _coerce_dict(_runtime_value("personality", {}))
    return PersonalityRuntimeConfig.from_runtime(payload)


def _personality_context_stage_preview(payload: dict[str, Any] | None) -> dict[str, Any]:
    injection = _coerce_dict(payload)
    style = _coerce_dict(injection.get("effective_style"))
    return {
        "schema": _as_text(injection.get("schema"), "ryo.personality_injection.v1"),
        "member_id": injection.get("member_id"),
        "effective_style": {
            "tone": style.get("tone"),
            "verbosity": style.get("verbosity"),
            "reading_level": style.get("reading_level"),
            "format": style.get("format"),
        },
        "adaptive": _coerce_dict(injection.get("adaptive")),
        "narrative_summary_excerpt": _truncate_for_prompt(injection.get("narrative_summary"), 180),
        "directive_rule_count": len(_as_string_list(injection.get("directive_rules"))),
        "behavior_rule_count": len(_as_string_list(injection.get("behavior_rules"))),
        "hard_constraint_count": len(_as_string_list(injection.get("hard_constraints"))),
    }


def _workspace_context_for_member(member_id: Any) -> dict[str, Any]:
    if not _runtime_bool("orchestrator.workspace_context_enabled", True):
        return {
            "schema": "ryo.workspace_context.v1",
            "enabled": False,
            "reason": "runtime.workspace_context_disabled",
            "active_process_count": 0,
            "pending_outbox_count": 0,
            "resume_recommended": False,
            "active_processes": [],
            "recent_outbox": [],
        }

    member_id_int = _safe_int(member_id, 0)
    if member_id_int <= 0:
        return {
            "schema": "ryo.workspace_context.v1",
            "enabled": False,
            "reason": "missing_member_id",
            "active_process_count": 0,
            "pending_outbox_count": 0,
            "resume_recommended": False,
            "active_processes": [],
            "recent_outbox": [],
        }

    process_limit = max(1, min(12, _runtime_int("orchestrator.workspace_process_limit", 6)))
    outbox_limit = max(1, min(20, _runtime_int("orchestrator.workspace_outbox_limit", 8)))
    active_processes: list[dict[str, Any]] = []
    recent_outbox: list[dict[str, Any]] = []
    pending_outbox_count = 0

    try:
        process_rows = collaboration.listProcesses(
            ownerMemberID=member_id_int,
            processStatus="all",
            count=process_limit,
            include_steps=True,
        )
        for process_row in process_rows:
            process = _coerce_dict(process_row)
            process_status = _as_text(process.get("process_status"), "active")
            if process_status in {"completed", "cancelled"}:
                continue
            next_step_label = ""
            next_step_status = ""
            steps = process.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    step_data = _coerce_dict(step)
                    status = _as_text(step_data.get("step_status"), "pending")
                    if status in {"completed", "skipped", "cancelled"}:
                        continue
                    next_step_label = _as_text(step_data.get("step_label"))
                    next_step_status = status
                    break
            active_processes.append(
                {
                    "process_id": process.get("process_id"),
                    "process_label": _as_text(process.get("process_label")),
                    "process_status": process_status,
                    "completion_percent": process.get("completion_percent"),
                    "steps_total": process.get("steps_total"),
                    "steps_completed": process.get("steps_completed"),
                    "next_step_label": next_step_label,
                    "next_step_status": next_step_status,
                    "updated_at": process.get("updated_at"),
                }
            )

        outbox_rows = collaboration.listOutboxForMember(memberID=member_id_int, count=outbox_limit)
        for outbox_row in outbox_rows:
            outbox = _coerce_dict(outbox_row)
            delivery_status = _as_text(outbox.get("delivery_status"), "queued")
            if delivery_status in {"queued", "failed"}:
                pending_outbox_count += 1
            recent_outbox.append(
                {
                    "outbox_id": outbox.get("outbox_id"),
                    "target_username": _as_text(outbox.get("target_username")),
                    "target_member_id": outbox.get("target_member_id"),
                    "delivery_status": delivery_status,
                    "process_id": outbox.get("process_id"),
                    "message_excerpt": _truncate_for_prompt(outbox.get("message_text"), 120),
                    "created_at": outbox.get("created_at"),
                }
            )
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Unable to build workspace context for member {member_id_int}: {error}")
        return {
            "schema": "ryo.workspace_context.v1",
            "enabled": False,
            "reason": "workspace_context_error",
            "error": str(error),
            "active_process_count": 0,
            "pending_outbox_count": 0,
            "resume_recommended": False,
            "active_processes": [],
            "recent_outbox": [],
        }

    return {
        "schema": "ryo.workspace_context.v1",
        "enabled": True,
        "active_process_count": len(active_processes),
        "pending_outbox_count": pending_outbox_count,
        "resume_recommended": bool(active_processes),
        "active_processes": active_processes,
        "recent_outbox": recent_outbox,
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


def _preflight_tool_intent_signal(message_text: str) -> dict[str, Any]:
    raw_text = _as_text(message_text)
    lowered = raw_text.lower()
    if not lowered:
        return {
            "explicit_request": False,
            "tool_hints": [],
            "reasons": [],
            "first_url": "",
        }

    hints: list[str] = []
    reasons: list[str] = []
    explicit_request = False

    def add_hint(hint_name: str) -> None:
        if hint_name in _ALLOWED_TOOL_HINTS and hint_name not in hints:
            hints.append(hint_name)

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    first_url = _extract_first_url(raw_text)
    if first_url:
        add_hint("curlRequest")
        add_reason("direct_url_present")
        explicit_request = True

    explicit_tool_phrases = (
        r"\buse (?:a |the )?tool(?:s)?\b",
        r"\bcall (?:a |the )?tool(?:s)?\b",
        r"\binvoke (?:a |the )?tool(?:s)?\b",
        r"\brun (?:a |the )?tool(?:s)?\b",
    )
    if any(re.search(pattern, lowered) for pattern in explicit_tool_phrases):
        add_reason("explicit_tool_phrase")
        explicit_request = True

    if re.search(r"\bbrave(?:\s+search)?\b", lowered):
        add_hint("braveSearch")
        add_reason("brave_search_requested")
        explicit_request = True
    if re.search(r"\bcurl\b|\bhttp\s+request\b|\bfetch\s+(?:the\s+)?(?:url|endpoint)\b|\bcall\s+(?:the\s+)?api\b", lowered):
        add_hint("curlRequest")
        add_reason("http_fetch_requested")
        explicit_request = True
    if re.search(r"\bchat\s+history\b|\bsearch\s+history\b|\bhistory\s+search\b", lowered):
        add_hint("chatHistorySearch")
        add_reason("chat_history_requested")
        explicit_request = True
    if re.search(r"\bknowledge\s+search\b|\bsearch\s+knowledge\b|\bknowledge\s+base\b|\bkb\b", lowered):
        add_hint("knowledgeSearch")
        add_reason("knowledge_search_requested")
        explicit_request = True

    if re.search(r"\b(search|look\s*up|find|browse|investigate|check)\b", lowered):
        add_hint("braveSearch")
        add_reason("research_action_detected")
        explicit_request = True

    if re.search(r"\b(message|dm|send)\b", lowered) and (
        "@" in lowered or re.search(r"\b(user|member|username)\b", lowered)
    ):
        add_hint("knownUsersList")
        add_hint("messageKnownUser")
        add_reason("inter_user_message_requested")
        explicit_request = True

    if re.search(r"\b(process|workflow|step|resume|continue|progress|outbox)\b", lowered):
        add_hint("listProcessWorkspace")
        add_hint("upsertProcessWorkspace")
        add_hint("updateProcessWorkspaceStep")
        add_reason("process_workspace_requested")
        explicit_request = True

    if re.search(r"\blist\s+(users?|members?)\b|\bwho\s+(?:can|is|are)\b", lowered):
        add_hint("knownUsersList")
        add_reason("known_user_listing_requested")
        explicit_request = True

    if explicit_request and not hints:
        # Keep this generic: trigger lightweight retrieval tooling if user explicitly requests tools
        # but doesn't name one.
        add_hint("braveSearch")
        add_reason("default_research_tool_injected")

    return {
        "explicit_request": explicit_request,
        "tool_hints": hints,
        "reasons": reasons,
        "first_url": first_url,
    }


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
    low_token_switch_enabled = _runtime_bool("topic_shift.low_token_switch_enabled", True)

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

    lexical_switch = (
        len(current_tokens) >= max(1, min_token_count - 1)
        and len(previous_tokens) >= min_token_count
        and jaccard_similarity <= jaccard_threshold
        and not small_talk_turn
    )
    low_token_switch = (
        low_token_switch_enabled
        and len(current_tokens) >= 1
        and len(previous_tokens) >= max(1, min_token_count - 1)
        and jaccard_similarity <= max(0.0, jaccard_threshold * 0.6)
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
    elif lexical_switch and to_topic != from_topic:
        switched = True
        reason = "lexical_divergence"
        confidence = "medium"
    elif low_token_switch and to_topic != from_topic:
        switched = True
        reason = "low_token_divergence"
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
    allow_recent_topic_fallback = _runtime_bool("topic_shift.allow_recent_topic_fallback", False)

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
    switched = _as_bool(transition.get("switched"), False)
    history_recall_requested = _as_bool(transition.get("history_recall_requested"), False) or _history_recall_requested(current_message)
    small_talk_turn = _as_bool(transition.get("small_talk_turn"), False) or _is_small_talk_message(current_message)
    if (
        (not active_topic or active_topic == "general")
        and recent_user_topics
        and allow_recent_topic_fallback
        and not switched
        and not small_talk_turn
    ):
        active_topic = recent_user_topics[-1]

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
    progressive_used = False
    progressive_result: dict[str, Any] = {}

    records = history_messages if isinstance(history_messages, list) else []
    candidates_considered = min(len(records), max(1, _runtime_int("retrieval.memory_candidate_limit", 30)))
    records = records[-candidates_considered:]

    history_recall_requested = _as_bool(circuit.get("history_recall_requested"), False)
    small_talk_turn = _as_bool(circuit.get("small_talk_turn"), False)
    switched = _as_bool(transition.get("switched"), False)
    min_overlap_default = max(0.0, min(1.0, _runtime_float("topic_shift.history_min_overlap_default", 0.08)))
    min_overlap_on_switch = max(0.0, min(1.0, _runtime_float("topic_shift.history_min_overlap_on_switch", 0.22)))
    required_overlap = min_overlap_on_switch if (switched and not history_recall_requested) else min_overlap_default

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
            "progressive_used": False,
            "progressive_recall": {},
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
            "progressive_used": False,
            "progressive_recall": {},
        }

    if recall_scope == "none":
        history_search_allowed = False

    query_text = _as_text(current_message)
    progressive_requested = history_recall_requested or _as_bool(
        directive.get("progressive_recall"),
        False,
    )
    progressive_config_payload = _progressive_history_runtime_payload(max_selected_override=max_items)
    progressive_enabled = _as_bool(progressive_config_payload.get("enabled"), True)
    if (
        progressive_enabled
        and progressive_requested
        and history_search_allowed
        and bool(query_text)
        and chat_host_id is not None
        and _as_text(chat_type)
        and _as_text(platform)
    ):
        progressive_used = True
        explorer = ProgressiveHistoryExplorer(
            chat_history_manager=chatHistory,
            config=ProgressiveHistoryExplorerConfig.from_runtime(progressive_config_payload),
        )
        progressive_result = explorer.explore(
            query_text=query_text,
            chat_host_id=chat_host_id,
            chat_type=_as_text(chat_type),
            platform=_as_text(platform),
            topic_id=topic_id,
            history_recall_requested=history_recall_requested,
            switched=switched,
            allow_history_search=history_search_allowed,
        )
        progressive_selected_raw = progressive_result.get("selected")
        progressive_selected = progressive_selected_raw if isinstance(progressive_selected_raw, list) else []
        progressive_found = _as_bool(progressive_result.get("found"), False)
        if progressive_selected and (progressive_found or history_recall_requested):
            selected = progressive_selected[:max_items]
            rounds_raw = progressive_result.get("rounds")
            rounds = rounds_raw if isinstance(rounds_raw, list) else []
            return {
                "schema": "ryo.memory_recall.v1",
                "mode": mode,
                "recall_scope": recall_scope,
                "history_search_allowed": history_search_allowed,
                "selected": selected,
                "selected_count": len(selected),
                "candidates_considered": sum(
                    _safe_int(_coerce_dict(item).get("candidate_count"), 0) for item in rounds
                ),
                "decision_reason": _as_text(
                    progressive_result.get("decision_reason"),
                    "progressive_recall_selected",
                ),
                "selection_threshold": min_score,
                "topic_transition": transition,
                "memory_circuit": {
                    "active_topic": circuit.get("active_topic"),
                    "routing_note": circuit.get("routing_note"),
                    "small_talk_turn": small_talk_turn,
                },
                "progressive_used": True,
                "progressive_recall": progressive_result,
            }

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

        if not history_recall_requested and required_overlap > 0.0:
            if switched and lexical_overlap < required_overlap:
                continue
            if lexical_overlap < required_overlap and semantic_weight < 0.55:
                continue

        score = (lexical_overlap * 2.2) + (semantic_weight * 1.4) + (recency_weight * 0.5)
        if role == "user":
            score += 0.08
        if switched and not history_recall_requested and lexical_overlap < max(required_overlap, 0.2):
            score -= 0.25

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
        "progressive_used": progressive_used,
        "progressive_recall": progressive_result,
    }


def _normalize_knowledge_classification(value: Any, fallback: str = "known") -> str:
    label = _as_text(value, fallback).strip().lower()
    if label in {"unknown", "probably_unknown", "known"}:
        return label
    if label in {"likely_unknown", "uncertain", "unclear", "partially_unknown"}:
        return "probably_unknown"
    return fallback if fallback in {"unknown", "probably_unknown", "known"} else "known"


def _analysis_question_form_score(message_text: str) -> float:
    text = _as_text(message_text).strip()
    if not text:
        return 0.0
    has_question = "?" in text
    token_count = len(re.findall(r"[A-Za-z0-9']+", text))
    if has_question and token_count >= 4:
        return 1.0
    if has_question:
        return 0.7
    if token_count >= 10:
        return 0.35
    return 0.0


def _derive_knowledge_state(
    *,
    payload: dict[str, Any],
    current_message: str,
    process_action: str,
) -> dict[str, Any]:
    raw_state = _coerce_dict(payload.get("knowledge_state"))
    provided_class = _normalize_knowledge_classification(raw_state.get("classification"), "")
    if provided_class not in {"known", "probably_unknown", "unknown"}:
        provided_class = ""
    provided_confidence = _as_text(raw_state.get("confidence"), "").strip().lower()
    if provided_confidence not in {"low", "medium", "high"}:
        provided_confidence = ""

    unknown_threshold = max(
        0.1,
        min(0.95, _runtime_float("orchestrator.discovery_unknown_threshold", 0.67)),
    )
    probably_unknown_threshold = max(
        0.05,
        min(unknown_threshold - 0.05, _runtime_float("orchestrator.discovery_probably_unknown_threshold", 0.4)),
    )

    needs_tools = _as_bool(payload.get("needs_tools"), False)
    tool_hints = [hint for hint in _as_string_list(payload.get("tool_hints")) if hint and hint != "skipTools"]
    risk_flags = [flag.lower() for flag in _as_string_list(payload.get("risk_flags"))]
    question_form_score = _analysis_question_form_score(current_message)

    weights: dict[str, float] = {}

    def add_weight(name: str, score: float) -> None:
        if score <= 0:
            return
        weights[name] = round(score, 4)

    if process_action == "none" and needs_tools:
        add_weight("analysis_needs_tools", 0.46)
    if process_action == "none" and tool_hints:
        add_weight("tool_hints", min(0.28, 0.09 * len(tool_hints)))
    if "uncertain" in risk_flags:
        add_weight("risk_uncertain", 0.32)
    if "none" not in risk_flags and risk_flags:
        add_weight("risk_signal", 0.16)
    if process_action == "none" and question_form_score > 0:
        add_weight("question_form", 0.1 * question_form_score)

    unknown_score = _coerce_float(raw_state.get("unknown_score"), -1.0)
    certainty_score = _coerce_float(raw_state.get("certainty_score"), -1.0)
    heuristic_score = max(0.0, min(1.0, sum(weights.values())))

    if unknown_score < 0.0 or unknown_score > 1.0:
        unknown_score = heuristic_score
    elif heuristic_score > 0:
        # Blend model-provided score with deterministic safety signals.
        unknown_score = max(0.0, min(1.0, (unknown_score * 0.75) + (heuristic_score * 0.25)))

    if provided_class:
        class_baseline = {
            "known": 0.2,
            "probably_unknown": max(probably_unknown_threshold + 0.02, 0.5),
            "unknown": max(unknown_threshold + 0.02, 0.82),
        }.get(provided_class, 0.2)
        unknown_score = max(0.0, min(1.0, (unknown_score * 0.8) + (class_baseline * 0.2)))

    if certainty_score < 0.0 or certainty_score > 1.0:
        certainty_score = max(0.0, min(1.0, 1.0 - unknown_score))
    else:
        certainty_score = max(0.0, min(1.0, (certainty_score * 0.8) + ((1.0 - unknown_score) * 0.2)))

    if unknown_score >= unknown_threshold:
        derived_class = "unknown"
    elif unknown_score >= probably_unknown_threshold:
        derived_class = "probably_unknown"
    else:
        derived_class = "known"

    classification = provided_class or derived_class
    if provided_class == "known" and derived_class in {"probably_unknown", "unknown"}:
        classification = derived_class
    elif provided_class == "probably_unknown" and derived_class == "unknown":
        classification = "unknown"

    discovery_force = _runtime_bool("orchestrator.discovery_force_tools_on_uncertainty", True)
    discovery_required = _as_bool(
        raw_state.get("discovery_required"),
        fallback=(classification != "known"),
    )
    if discovery_force and classification in {"probably_unknown", "unknown"}:
        discovery_required = True

    reason = _as_text(raw_state.get("reason"))
    if not reason:
        if not weights:
            reason = "No strong uncertainty signals; proceed with direct response unless tools are explicitly requested."
        else:
            top_signals = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:2]
            label_map = {
                "analysis_needs_tools": "analysis marked tool usage",
                "tool_hints": "tool hints indicate external lookup",
                "risk_uncertain": "analysis flagged uncertainty",
                "risk_signal": "risk flags require verification",
                "question_form": "question structure suggests factual lookup",
            }
            reason = " + ".join(label_map.get(name, name) for name, _ in top_signals)
            reason = f"Discovery confidence derived from: {reason}."
    reason = _truncate_for_prompt(reason, 180)

    confidence = provided_confidence
    if not confidence:
        if classification == "known":
            confidence = "high" if unknown_score <= (probably_unknown_threshold * 0.55) else "medium"
        elif classification == "unknown":
            confidence = "high" if unknown_score >= min(1.0, unknown_threshold + 0.12) else "medium"
        else:
            mid_point = (unknown_threshold + probably_unknown_threshold) / 2.0
            confidence = "high" if abs(unknown_score - mid_point) >= 0.14 else "medium"
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    return {
        "classification": classification,
        "confidence": confidence,
        "unknown_score": round(max(0.0, min(1.0, unknown_score)), 4),
        "certainty_score": round(max(0.0, min(1.0, certainty_score)), 4),
        "discovery_required": bool(discovery_required),
        "reason": reason,
        "weights": weights,
    }


def _normalize_analysis_payload(
    raw_analysis_text: str,
    known_context: dict[str, Any] | None = None,
    current_message: str | None = None,
) -> dict[str, Any]:
    parsed = _parse_json_like(raw_analysis_text)
    payload = parsed if isinstance(parsed, dict) else {}
    tool_results = payload.get("tool_results")
    if isinstance(tool_results, dict):
        payload = tool_results
    if _as_text(payload.get("analysis")).lower() == "unavailable":
        payload.setdefault("risk_flags", ["uncertain"])
    if not any(
        key in payload
        for key in ("topic", "intent", "needs_tools", "tool_hints", "process_directive", "knowledge_state")
    ):
        payload.setdefault("risk_flags", ["uncertain"])
        payload.setdefault(
            "context_summary",
            "Analysis output unavailable; defaulting to conservative discovery posture.",
        )

    context_data = known_context if isinstance(known_context, dict) else {}
    context_topic_transition = _coerce_dict(context_data.get("topic_transition"))
    context_memory_circuit = _coerce_dict(context_data.get("memory_circuit"))
    context_workspace = _coerce_dict(context_data.get("workspace_context"))
    context_active_processes_raw = context_workspace.get("active_processes")
    context_active_processes = (
        context_active_processes_raw
        if isinstance(context_active_processes_raw, list)
        else []
    )
    payload_topic_transition = _coerce_dict(payload.get("topic_transition"))
    payload_memory_directive = _coerce_dict(payload.get("memory_directive"))
    payload_process_directive = _coerce_dict(payload.get("process_directive"))

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

    action_raw = _as_text(payload_process_directive.get("action"), "none").lower()
    process_action_aliases = {
        "none": "none",
        "list_users": "list_users",
        "users": "list_users",
        "known_users": "list_users",
        "send_message": "send_message",
        "message_user": "send_message",
        "direct_message": "send_message",
        "start_process": "start_process",
        "create_process": "start_process",
        "new_process": "start_process",
        "resume_process": "resume_process",
        "continue_process": "resume_process",
        "list_processes": "resume_process",
        "update_process_step": "update_process_step",
        "complete_step": "update_process_step",
        "list_outbox": "list_outbox",
        "outbox": "list_outbox",
    }
    process_action = process_action_aliases.get(action_raw, "none")
    process_id = _safe_int(payload_process_directive.get("process_id"), 0)
    if process_id <= 0:
        process_id = 0
    if process_id == 0 and process_action in {"resume_process", "update_process_step"} and context_active_processes:
        first_active = _coerce_dict(context_active_processes[0])
        process_id = _safe_int(first_active.get("process_id"), 0)
    process_label = _as_text(payload_process_directive.get("process_label"))
    if not process_label and process_action == "start_process":
        process_label = f"{_as_text(payload.get('topic'), 'process')} workflow"
    process_directive = {
        "action": process_action,
        "process_id": process_id if process_id > 0 else None,
        "process_label": _truncate_for_prompt(process_label, 140),
        "step_label": _truncate_for_prompt(payload_process_directive.get("step_label"), 140),
        "step_status": _as_text(payload_process_directive.get("step_status"), "completed").lower(),
        "step_details": _truncate_for_prompt(payload_process_directive.get("step_details"), 180),
        "target_username": _as_text(payload_process_directive.get("target_username")).lstrip("@"),
        "target_member_id": (
            _safe_int(payload_process_directive.get("target_member_id"), 0)
            if _safe_int(payload_process_directive.get("target_member_id"), 0) > 0
            else None
        ),
        "message_text": _truncate_for_prompt(payload_process_directive.get("message_text"), 300),
        "reason": _truncate_for_prompt(
            payload_process_directive.get("reason"),
            180,
        ),
    }
    if process_directive["step_status"] not in {"pending", "in_progress", "blocked", "completed", "skipped", "cancelled"}:
        process_directive["step_status"] = "completed"
    process_action_hints = {
        "list_users": ["knownUsersList"],
        "send_message": ["knownUsersList", "messageKnownUser"],
        "start_process": ["upsertProcessWorkspace"],
        "resume_process": ["listProcessWorkspace"],
        "update_process_step": ["listProcessWorkspace", "updateProcessWorkspaceStep"],
        "list_outbox": ["listOutboxMessages"],
    }
    for hinted_tool in process_action_hints.get(process_action, []):
        if hinted_tool in _ALLOWED_TOOL_HINTS and hinted_tool not in tool_hints:
            tool_hints.append(hinted_tool)

    risk_flags = _as_string_list(payload.get("risk_flags")) or ["none"]
    needs_tools = _as_bool(payload.get("needs_tools"), fallback=False)
    if process_action != "none":
        needs_tools = True

    normalized_payload_for_knowledge = {
        "needs_tools": needs_tools,
        "tool_hints": list(tool_hints),
        "risk_flags": list(risk_flags),
        "process_directive": {"action": process_action},
        "knowledge_state": _coerce_dict(payload.get("knowledge_state")),
    }
    knowledge_state = _derive_knowledge_state(
        payload=normalized_payload_for_knowledge,
        current_message=_as_text(current_message, ""),
        process_action=process_action,
    )
    discovery_layer_enabled = _runtime_bool("orchestrator.discovery_layer_enabled", True)
    discovery_required = _as_bool(
        knowledge_state.get("discovery_required"),
        fallback=knowledge_state.get("classification") in {"probably_unknown", "unknown"},
    )
    if not discovery_layer_enabled:
        discovery_required = False
        knowledge_state["discovery_required"] = False
    if discovery_required:
        needs_tools = True
        default_discovery_hints = [
            hint
            for hint in _as_string_list(
                _runtime_value(
                    "orchestrator.discovery_default_tool_hints",
                    ["braveSearch", "curlRequest"],
                )
            )
            if hint in _ALLOWED_TOOL_HINTS and hint != "skipTools"
        ]
        for hinted_tool in default_discovery_hints:
            if hinted_tool not in tool_hints:
                tool_hints.append(hinted_tool)

    max_hints = max(1, min(8, _runtime_int("orchestrator.analysis_tool_hints_max", 3)))
    if len(tool_hints) > max_hints:
        tool_hints = tool_hints[:max_hints]

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
        "needs_tools": needs_tools,
        "tool_hints": tool_hints,
        "process_directive": process_directive,
        "risk_flags": risk_flags,
        "knowledge_state": knowledge_state,
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


def _is_low_signal_response(text: str) -> bool:
    cleaned = _as_text(text)
    if not cleaned:
        return True
    alnum_count = len(re.sub(r"[^A-Za-z0-9]+", "", cleaned))
    if alnum_count < 6:
        return True
    if re.fullmatch(r"[\s\{\}\[\]\":,`'.\-_/\\]+", cleaned):
        return True
    return False


def _truncate_for_prompt(text: Any, max_chars: int = 300) -> str:
    cleaned = _as_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


def _tool_results_repair_preview(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for item in tool_results[:4]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "tool_name": _as_text(item.get("tool_name"), "unknown"),
                "status": _as_text(item.get("status"), "unknown"),
                "error": _coerce_dict(item.get("error")),
                "tool_results_excerpt": _truncate_for_prompt(item.get("tool_results"), 260),
            }
        )
    return preview


def _process_tool_stage_summary(tool_results: list[dict[str, Any]] | None) -> dict[str, Any]:
    results = tool_results if isinstance(tool_results, list) else []
    process_tool_names = {
        "knownUsersList",
        "messageKnownUser",
        "upsertProcessWorkspace",
        "listProcessWorkspace",
        "updateProcessWorkspaceStep",
        "listOutboxMessages",
    }

    process_tools_executed: list[str] = []
    process_ids: list[int] = []
    outbox_statuses: list[str] = []
    errors: list[dict[str, Any]] = []
    active_process_count: int | None = None

    for item in results:
        result = _coerce_dict(item)
        tool_name = _as_text(result.get("tool_name"))
        if tool_name not in process_tool_names:
            continue
        process_tools_executed.append(tool_name)
        status = _as_text(result.get("status"), "ok")
        payload = _coerce_dict(result.get("tool_results"))
        if status == "error":
            errors.append(
                {
                    "tool_name": tool_name,
                    "error": _coerce_dict(result.get("error")),
                }
            )
            continue

        if tool_name in {"upsertProcessWorkspace", "updateProcessWorkspaceStep"}:
            process = _coerce_dict(payload.get("process"))
            process_id = _safe_int(process.get("process_id"), 0)
            if process_id > 0 and process_id not in process_ids:
                process_ids.append(process_id)

        if tool_name == "listProcessWorkspace":
            active_process_count = _safe_int(payload.get("count"), 0)
            processes = payload.get("processes")
            if isinstance(processes, list):
                for process in processes:
                    process_id = _safe_int(_coerce_dict(process).get("process_id"), 0)
                    if process_id > 0 and process_id not in process_ids:
                        process_ids.append(process_id)

        if tool_name == "messageKnownUser":
            outbox = _coerce_dict(payload.get("outbox"))
            delivery_status = _as_text(outbox.get("delivery_status"))
            if delivery_status:
                outbox_statuses.append(delivery_status)

        if tool_name == "listOutboxMessages":
            if active_process_count is None:
                active_process_count = _safe_int(payload.get("count"), 0)

    deduped_tools = []
    for name in process_tools_executed:
        if name not in deduped_tools:
            deduped_tools.append(name)
    deduped_outbox_statuses = []
    for status in outbox_statuses:
        if status not in deduped_outbox_statuses:
            deduped_outbox_statuses.append(status)

    return {
        "has_updates": bool(deduped_tools),
        "process_tools_executed": deduped_tools,
        "process_ids": process_ids,
        "active_process_count": active_process_count,
        "outbox_statuses": deduped_outbox_statuses,
        "errors": errors,
    }


def _tool_suggestion_plan(
    *,
    analysis_payload: dict[str, Any] | None,
    current_message: str,
) -> dict[str, Any]:
    payload = _coerce_dict(analysis_payload)
    hints = [hint for hint in _as_string_list(payload.get("tool_hints")) if hint and hint != "skipTools"]
    needs_tools = _as_bool(payload.get("needs_tools"), False)
    process_directive = _coerce_dict(payload.get("process_directive"))
    process_action = _as_text(process_directive.get("action"), "none")
    knowledge_state = _coerce_dict(payload.get("knowledge_state"))
    knowledge_class = _normalize_knowledge_classification(knowledge_state.get("classification"), "known")
    discovery_required = _as_bool(
        knowledge_state.get("discovery_required"),
        fallback=knowledge_class in {"probably_unknown", "unknown"},
    )
    if discovery_required:
        needs_tools = True
    first_url = _extract_first_url(current_message)

    suggestions: list[dict[str, Any]] = []

    def add_suggestion(tool_name: str, reason: str, priority: int) -> None:
        if any(_as_text(item.get("tool_name")) == tool_name for item in suggestions):
            return
        suggestions.append(
            {
                "tool_name": tool_name,
                "reason": _truncate_for_prompt(reason, 160),
                "priority": int(priority),
            }
        )

    process_action_map = {
        "list_users": [("knownUsersList", "Resolve known users before inter-user actions.")],
        "send_message": [
            ("knownUsersList", "Resolve recipient identity before sending."),
            ("messageKnownUser", "Deliver or queue the inter-user message."),
        ],
        "start_process": [("upsertProcessWorkspace", "Create a persisted multi-step process workspace.")],
        "resume_process": [("listProcessWorkspace", "Load active process state for continuity.")],
        "update_process_step": [
            ("listProcessWorkspace", "Load process context before applying step updates."),
            ("updateProcessWorkspaceStep", "Write the new step status and progress."),
        ],
        "list_outbox": [("listOutboxMessages", "Inspect queued/sent inter-user messages.")],
    }
    for idx, (tool_name, reason) in enumerate(process_action_map.get(process_action, []), start=1):
        add_suggestion(tool_name, reason, idx)

    if discovery_required and process_action == "none":
        add_suggestion(
            "braveSearch",
            "Knowledge confidence is uncertain; discover grounded external sources before replying.",
            1,
        )
        if first_url:
            add_suggestion(
                "curlRequest",
                "A direct URL is present; fetch it to verify concrete details.",
                2,
            )
        else:
            add_suggestion(
                "curlRequest",
                "Follow search discoveries with direct URL fetch for factual validation.",
                3,
            )

    for idx, hint in enumerate(hints, start=1):
        if hint == "braveSearch":
            add_suggestion(
                "braveSearch",
                "Gather candidate sources and current web facts quickly before deeper fetches.",
                idx,
            )
            if first_url:
                add_suggestion(
                    "curlRequest",
                    "A direct URL is present; fetch the endpoint content to ground the final answer.",
                    idx + 1,
                )
            else:
                add_suggestion(
                    "curlRequest",
                    "After web search identifies a target URL, fetch that page/API directly for precise details.",
                    idx + 2,
                )
        elif hint == "curlRequest":
            if first_url:
                add_suggestion(
                    "curlRequest",
                    "Direct URL detected in user request; fetch endpoint data directly.",
                    idx,
                )
            else:
                add_suggestion(
                    "braveSearch",
                    "No explicit URL provided; discover candidate URL(s) before HTTP fetch.",
                    idx,
                )
                add_suggestion(
                    "curlRequest",
                    "Fetch best candidate URL discovered by search to validate details.",
                    idx + 1,
                )
        elif hint == "chatHistorySearch":
            add_suggestion("chatHistorySearch", "Recover relevant prior chat context for grounding.", idx)
        elif hint == "knowledgeSearch":
            add_suggestion("knowledgeSearch", "Retrieve project-specific indexed knowledge context.", idx)
        elif hint == "knownUsersList":
            add_suggestion("knownUsersList", "Resolve users for communication workflows.", idx)
        elif hint == "listProcessWorkspace":
            add_suggestion("listProcessWorkspace", "Load current process state before continuing steps.", idx)
        elif hint == "updateProcessWorkspaceStep":
            add_suggestion("updateProcessWorkspaceStep", "Persist step completion/progress updates.", idx)
        elif hint == "listOutboxMessages":
            add_suggestion("listOutboxMessages", "Inspect queued/sent message delivery state.", idx)

    if needs_tools and not suggestions:
        if first_url:
            add_suggestion("curlRequest", "Direct URL available for immediate retrieval.", 1)
        else:
            add_suggestion("braveSearch", "Start with web retrieval to gather grounding context.", 1)

    suggestions.sort(key=lambda item: (_safe_int(item.get("priority"), 1000), _as_text(item.get("tool_name"))))
    return {
        "needs_tools": needs_tools,
        "process_action": process_action,
        "knowledge_state": {
            "classification": knowledge_class,
            "discovery_required": discovery_required,
        },
        "suggested_tools": suggestions[:6],
        "suggestion_count": len(suggestions[:6]),
        "has_direct_url": bool(first_url),
    }



################
# ORCHESTRATOR #
################

class ConversationOrchestrator:

    def __init__(self, message: str, memberID: int, context: dict=None, messageID: int=None, options: dict=None):
        self._messages: list[Message] = []
        self._analysisStats: dict[str, Any] = {}
        self._devStats: dict[str, Any] = {}
        self._chatResponseMessage = ""
        self._runSummary: dict[str, Any] = {}
        self._responseID: int | None = None
        self._responseHistoryID: int | None = None

        self._message = message
        self._memberID = memberID
        self._messageID = messageID
        self._context = context if isinstance(context, dict) else {}
        self._options = options if isinstance(options, dict) else {}
        stageCallback = self._options.get("stage_callback")
        self._stage_callback = stageCallback if callable(stageCallback) else None
        self._ingressImages = [item for item in _as_string_list(self._options.get("ingress_images")) if item]
        contextImage = _coerce_dict(self._context.get("image_context"))
        optionImage = _coerce_dict(self._options.get("image_context"))
        mergedImageContext = dict(contextImage)
        mergedImageContext.update(optionImage)
        if self._ingressImages:
            mergedImageContext.setdefault("present", True)
            mergedImageContext.setdefault("image_count", len(self._ingressImages))
        self._imageContext = mergedImageContext
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
        if self._messageID is None:
            fallbackMessageID = self._options.get("message_id")
            try:
                self._messageID = int(fallbackMessageID) if fallbackMessageID is not None else 1
            except (TypeError, ValueError):
                self._messageID = 1
        if self._messageID is not None:
            try:
                self._messageID = int(self._messageID)
            except (TypeError, ValueError):
                self._messageID = 1
            self._responseID = int(self._messageID) + 1
                
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

    async def _attempt_response_repair(
        self,
        *,
        normalized_analysis: dict[str, Any],
        tool_responses: list[dict[str, Any]],
        tool_execution_mode: str,
    ) -> tuple[str, dict[str, Any]]:
        endpoint_override = self._options.get("ollama_host")
        router = ModelRouter(
            inference_config=config._instance.inference,
            endpoint_override=endpoint_override,
        )
        chat_routing = _coerce_dict(getattr(self._chatConversationAgent, "routing", {}))
        requested_model = (
            _as_text(chat_routing.get("selected_model"))
            or _as_text(chat_routing.get("requested_model"))
            or _as_text(getattr(self._chatConversationAgent, "_model", ""))
        )
        allowed_models = _as_string_list(getattr(self._chatConversationAgent, "_allowed_models", []))
        if requested_model and requested_model not in allowed_models:
            allowed_models.insert(0, requested_model)

        repair_prompt = (
            "You are producing the final user-facing answer for a chat assistant. "
            "Respond directly to the user's request in plain text. "
            "Do not output JSON, code fences, analysis metadata, or internal orchestration details."
        )
        repair_payload = {
            "user_message": _truncate_for_prompt(self._message, 500),
            "analysis": {
                "topic": _as_text(normalized_analysis.get("topic"), "general"),
                "intent": _as_text(normalized_analysis.get("intent"), "answer_user"),
                "context_summary": _truncate_for_prompt(normalized_analysis.get("context_summary"), 240),
                "needs_tools": bool(normalized_analysis.get("needs_tools")),
            },
            "tool_execution_mode": _as_text(tool_execution_mode, "native_tools"),
            "tool_results": _tool_results_repair_preview(tool_responses),
        }
        repair_messages = [
            Message(role="system", content=repair_prompt),
            Message(
                role="user",
                content="Produce a clean final reply from this context:\n"
                + json.dumps(repair_payload, ensure_ascii=False),
            ),
        ]
        response, routing = await router.chat_with_fallback(
            capability="chat",
            requested_model=requested_model or None,
            allowed_models=allowed_models or None,
            messages=repair_messages,
            stream=False,
        )
        response_message = getattr(response, "message", None)
        repaired_text = _as_text(getattr(response_message, "content", ""))
        repaired_text = _sanitize_final_response(
            repaired_text,
            user_message=self._message,
            allow_internal_diagnostics=False,
        )
        return repaired_text, _coerce_dict(routing)


    async def runAgents(self):
        await self._emit_stage("orchestrator.start", "Accepted request and preparing context.")
        emitProgressStages = _runtime_bool("orchestrator.progress_stage_events_enabled", False)
        analysisBypassSmallTalk = _runtime_bool("orchestrator.analysis_bypass_small_talk_enabled", False)
        autoExpandToolStage = _runtime_bool("orchestrator.auto_expand_tool_stage_enabled", True)
        autoExpandMaxRounds = max(0, _runtime_int("orchestrator.auto_expand_max_rounds", 1))
        personalityRuntime = _personality_runtime_config()
        personalityEnabled = (
            personalityRuntime.enabled
            and not self._transientSession
            and isinstance(self._memberID, int)
            and self._memberID > 0
        )
        personalityProfile: dict[str, Any] = {}
        personalityPromptPayload: dict[str, Any] = {}
        personalityPromptMessage: dict[str, Any] = {}
        personalityStagePreview: dict[str, Any] = {}
        personalityAdaptationState: dict[str, Any] = {}
        personalityRollupState: dict[str, Any] = {}

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
        workspaceContext = _workspace_context_for_member(
            self._memberID if not self._transientSession else None
        )
        workspaceContextPreview = _workspace_context_stage_preview(workspaceContext)
        if _as_bool(topicTransition.get("switched"), False):
            await self._emit_stage(
                "context.topic_shift",
                _as_text(topicTransition.get("summary")),
                from_topic=topicTransition.get("from_topic"),
                to_topic=topicTransition.get("to_topic"),
                reason=topicTransition.get("reason"),
            )
        if _as_bool(workspaceContext.get("enabled"), False):
            await self._emit_stage(
                "context.workspace",
                "Loaded workspace state for process/message continuity.",
                active_process_count=workspaceContextPreview.get("active_process_count", 0),
                pending_outbox_count=workspaceContextPreview.get("pending_outbox_count", 0),
                json=workspaceContextPreview,
            )

        if personalityEnabled:
            try:
                latestNarrativeChunk = personalityStore.get_latest_narrative_chunk(self._memberID)
                narrativeSummary = (
                    _as_text(_coerce_dict(latestNarrativeChunk).get("summary_text"))
                    if isinstance(latestNarrativeChunk, dict)
                    else ""
                )
                narrativeChunkCount = _safe_int(
                    _coerce_dict(latestNarrativeChunk).get("chunk_index"),
                    0,
                )
                storedProfile = personalityStore.get_profile(self._memberID)
                personalityProfile = personalityEngine.resolve_profile(
                    member_id=self._memberID,
                    stored_profile=storedProfile,
                    runtime_config=personalityRuntime,
                    latest_narrative_summary=narrativeSummary,
                    narrative_chunk_count=narrativeChunkCount,
                    last_chunk_index=narrativeChunkCount,
                )
                explicitDirective = _coerce_dict(self._context.get("personality_directive"))
                if explicitDirective:
                    personalityProfile = personalityEngine.apply_explicit_directive(
                        profile=personalityProfile,
                        directive=explicitDirective,
                        runtime_config=personalityRuntime,
                    )
                    persistedDirectiveProfile = personalityStore.upsert_profile_payload(
                        self._memberID,
                        personalityProfile,
                    )
                    personalityStore.append_event(
                        member_id=self._memberID,
                        event_type="explicit_update",
                        before_json={"explicit": _coerce_dict(_coerce_dict(storedProfile).get("explicit_directive_json"))},
                        after_json={"explicit": _coerce_dict(_coerce_dict(personalityProfile).get("explicit"))},
                        reason_code="context_personality_directive",
                        reason_detail="Explicit personality directive applied from orchestrator context.",
                    )
                    await self._emit_stage(
                        "persona.directive",
                        "Applied explicit personality directive update for this user.",
                        profile_version=_safe_int(_coerce_dict(persistedDirectiveProfile).get("profile_version"), 0),
                        json={"directive": explicitDirective},
                    )
                personalityPromptPayload = personalityInjector.build_payload(
                    profile=personalityProfile,
                    narrative_summary=narrativeSummary,
                    max_injection_chars=personalityRuntime.max_injection_chars,
                )
                personalityStagePreview = _personality_context_stage_preview(personalityPromptPayload)
                personalityPromptMessage = {
                    "tool_name": "Personality Context",
                    "tool_results": personalityPromptPayload,
                }
                await self._emit_stage(
                    "persona.load",
                    "Loaded user personality profile and narrative continuity.",
                    profile_version=_safe_int(_coerce_dict(storedProfile).get("profile_version"), 0),
                    chunk_count=_safe_int(
                        _coerce_dict(_coerce_dict(personalityProfile).get("narrative")).get("chunk_count"),
                        0,
                    ),
                    json=personalityStagePreview,
                )
            except Exception as error:  # noqa: BLE001
                personalityEnabled = False
                personalityProfile = {}
                personalityPromptPayload = {}
                personalityPromptMessage = {}
                personalityStagePreview = {}
                await self._emit_stage(
                    "persona.error",
                    "Personality profile load failed; falling back to default response style.",
                    error=str(error),
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
                "workspace_context": workspaceContext,
                "image_context": _coerce_dict(self._imageContext),
                "personality_context": personalityStagePreview,
            }
        }
        knownContextPreview = _known_context_stage_preview(knownContext.get("tool_results"))
        knownContextPrompt = {
            "tool_name": "Known Context",
            "tool_results": knownContextPreview,
        }
        await self._emit_stage(
            "context.built",
            "Known context assembled for downstream stages.",
            json=knownContextPreview,
        )
        self._messages.append(Message(role="tool", content=json.dumps(knownContextPrompt, ensure_ascii=False)))
        if personalityEnabled and personalityPromptMessage:
            self._messages.append(
                Message(role="tool", content=json.dumps(personalityPromptMessage, ensure_ascii=False))
            )
            await self._emit_stage(
                "persona.inject",
                "Injected personality and narrative guidance for downstream stages.",
                json=personalityStagePreview,
            )

        def _recent_dialogue_messages(limit: int) -> list[Message]:
            dialogue = [
                message
                for message in self._messages
                if _as_text(getattr(message, "role", "")).lower() in {"user", "assistant"}
            ]
            if limit > 0 and len(dialogue) > limit:
                dialogue = dialogue[-limit:]
            return list(dialogue)

        self._analysisStats = {}
        analysisStatsSummary: dict[str, Any] = {}
        normalizedAnalysis: dict[str, Any]
        smallTalkTurn = _as_bool(memoryCircuit.get("small_talk_turn"), False)
        historyRecallRequested = _as_bool(memoryCircuit.get("history_recall_requested"), False)
        workspaceResumeRecommended = _as_bool(workspaceContext.get("resume_recommended"), False)
        preflightToolIntent = _coerce_dict(_preflight_tool_intent_signal(self._message))
        preflightToolRequested = _as_bool(preflightToolIntent.get("explicit_request"), False)
        preflightToolHints = [
            hint
            for hint in _as_string_list(preflightToolIntent.get("tool_hints"))
            if hint in _ALLOWED_TOOL_HINTS and hint != "skipTools"
        ]
        analysisBypassed = (
            analysisBypassSmallTalk
            and smallTalkTurn
            and not historyRecallRequested
            and not workspaceResumeRecommended
            and not preflightToolRequested
        )

        if analysisBypassed:
            normalizedAnalysis = {
                "topic": _as_text(topicTransition.get("to_topic"), "general"),
                "intent": "social_reply",
                "needs_tools": False,
                "tool_hints": [],
                "process_directive": {
                    "action": "none",
                    "process_id": None,
                    "process_label": "",
                    "step_label": "",
                    "step_status": "completed",
                    "target_username": "",
                    "message_text": "",
                    "reason": "analysis_bypass_small_talk",
                },
                "risk_flags": ["none"],
                "knowledge_state": {
                    "classification": "known",
                    "confidence": "high",
                    "unknown_score": 0.05,
                    "certainty_score": 0.95,
                    "discovery_required": False,
                    "reason": "Social brevity turn; no discovery needed.",
                    "weights": {},
                },
                "topic_transition": _coerce_dict(topicTransition),
                "memory_directive": {
                    "mode": "lightweight_social",
                    "history_search_allowed": False,
                    "reason": "small_talk_turn_without_recall_request",
                },
                "brevity_directive": {
                    "mode": "brief_social",
                    "reason": "analysis_bypass_small_talk",
                },
                "response_style": {
                    "tone": "friendly",
                    "length": "concise",
                },
                "context_summary": "Short social turn detected; bypassed heavy analysis and tool planning.",
            }
            self._analysisAgent = SimpleNamespace(
                routing={
                    "capability": "analysis",
                    "mode": "bypass_small_talk",
                    "selected_model": None,
                }
            )
            await self._emit_stage(
                "analysis.skipped",
                "Bypassed analysis model for a lightweight social turn.",
                json=normalizedAnalysis,
            )
        else:
            analysisHistoryLimit = max(2, _runtime_int("orchestrator.analysis_history_limit", 8))
            if _as_bool(topicTransition.get("switched"), False) and not historyRecallRequested:
                analysisHistoryLimit = min(
                    analysisHistoryLimit,
                    max(2, _runtime_int("orchestrator.analysis_history_limit_on_switch", 4)),
                )
            if smallTalkTurn and not historyRecallRequested:
                analysisHistoryLimit = min(
                    analysisHistoryLimit,
                    max(1, _runtime_int("orchestrator.analysis_history_limit_small_talk", 2)),
                )

            analysisMessages = _recent_dialogue_messages(analysisHistoryLimit)
            analysisMessages.append(Message(role="tool", content=json.dumps(knownContextPrompt, ensure_ascii=False)))
            if personalityEnabled and personalityPromptMessage:
                analysisMessages.append(
                    Message(role="tool", content=json.dumps(personalityPromptMessage, ensure_ascii=False))
                )

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
            analysisChunkTimeout = _timeout_seconds("inference.stream_chunk_timeout_seconds", 45.0)
            analysisTotalTimeout = _timeout_seconds("inference.stream_total_timeout_seconds", 240.0)
            analysisDeadline = (
                time.monotonic() + float(analysisTotalTimeout)
                if isinstance(analysisTotalTimeout, (int, float))
                else None
            )
            analysisIterator = self._analysisResponse.__aiter__()

            chunk: ChatResponse
            while True:
                try:
                    chunk = await _next_stream_chunk(
                        analysisIterator,
                        idle_timeout_seconds=analysisChunkTimeout,
                        deadline_monotonic=analysisDeadline,
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError as error:
                    logger.warning(
                        "[stream.analysis] stream timeout; "
                        f"using conservative parse fallback (chunks={analysisChunkCount}, chars={analysisCharCount}): {error}"
                    )
                    await self._emit_stage(
                        "analysis.timeout",
                        "Analysis stream timed out; using conservative parse fallback.",
                        chunks=analysisChunkCount,
                        chars=analysisCharCount,
                        timeout_seconds=analysisChunkTimeout,
                        total_timeout_seconds=analysisTotalTimeout,
                    )
                    break
                except Exception as error:  # noqa: BLE001
                    logger.exception(
                        "[stream.analysis] stream iteration failed; "
                        f"using conservative parse fallback (chunks={analysisChunkCount}, chars={analysisCharCount})"
                    )
                    await self._emit_stage(
                        "analysis.error",
                        "Analysis stream failed; using conservative parse fallback.",
                        error=str(error),
                        chunks=analysisChunkCount,
                        chars=analysisCharCount,
                    )
                    break
                chunkContent = _chunk_message_content(chunk)
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
                if emitProgressStages and chunkContent and (
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
                if _chunk_done(chunk):
                    analysisDoneSeen = True
                    self._analysisStats = _chunk_stream_stats(chunk)
                    evalCount = _safe_int(_chunk_field(chunk, "eval_count"), 0)
                    logger.info(
                        "[stream.analysis] "
                        f"done chunks={analysisChunkCount} chars={analysisCharCount} eval_count={evalCount}"
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
                current_message=self._message,
            )
            await self._emit_stage(
                "analysis.payload",
                "Normalized analysis payload produced.",
                json=normalizedAnalysis,
            )
            processDirective = _coerce_dict(normalizedAnalysis.get("process_directive"))
            processAction = _as_text(processDirective.get("action"), "none")
            if processAction != "none":
                await self._emit_stage(
                    "process.directive",
                    "Process workflow directive selected from analysis.",
                    action=processAction,
                    process_id=processDirective.get("process_id"),
                    json=processDirective,
                )

        if personalityEnabled and personalityProfile:
            normalizedAnalysis = personalityEngine.apply_analysis_style(
                analysis_payload=normalizedAnalysis,
                profile=personalityProfile,
            )
            effectiveStyle = _coerce_dict(_coerce_dict(personalityProfile).get("effective"))
            await self._emit_stage(
                "persona.style",
                "Applied personality style targets to response planning.",
                tone=effectiveStyle.get("tone"),
                verbosity=effectiveStyle.get("verbosity"),
                reading_level=effectiveStyle.get("reading_level"),
            )

        normalizedToolHints = [
            hint
            for hint in _as_string_list(normalizedAnalysis.get("tool_hints"))
            if hint in _ALLOWED_TOOL_HINTS and hint != "skipTools"
        ]
        for hint in preflightToolHints:
            if hint not in normalizedToolHints:
                normalizedToolHints.append(hint)
        if normalizedToolHints:
            normalizedAnalysis["tool_hints"] = normalizedToolHints

        discoveryState = _coerce_dict(normalizedAnalysis.get("knowledge_state"))
        discoveryClassification = _normalize_knowledge_classification(
            discoveryState.get("classification"),
            "known",
        )
        discoveryRequired = _as_bool(
            discoveryState.get("discovery_required"),
            fallback=discoveryClassification in {"probably_unknown", "unknown"},
        )
        if _runtime_bool("orchestrator.discovery_layer_enabled", True):
            await self._emit_stage(
                "discovery.state",
                "Epistemic confidence state computed for discovery routing.",
                classification=discoveryClassification,
                unknown_score=discoveryState.get("unknown_score"),
                certainty_score=discoveryState.get("certainty_score"),
                discovery_required=discoveryRequired,
                json=discoveryState,
            )

        if discoveryRequired and not _as_bool(normalizedAnalysis.get("needs_tools"), False):
            normalizedAnalysis["needs_tools"] = True
            normalizedAnalysis["brevity_directive"] = {
                "mode": "standard",
                "reason": "knowledge_uncertainty_requires_discovery",
            }
            await self._emit_stage(
                "orchestrator.expand",
                "Escalating to discovery tools due to uncertain/unknown knowledge state.",
                json={
                    "reason": "knowledge_state_requires_discovery",
                    "classification": discoveryClassification,
                    "unknown_score": discoveryState.get("unknown_score"),
                    "hints": normalizedToolHints,
                },
            )

        if preflightToolRequested and not _as_bool(normalizedAnalysis.get("needs_tools"), False):
            normalizedAnalysis["needs_tools"] = True
            normalizedAnalysis["brevity_directive"] = {
                "mode": "standard",
                "reason": "explicit_tool_request_preflight",
            }
            await self._emit_stage(
                "orchestrator.expand",
                "Escalating to tool stage from explicit user tool request.",
                json={
                    "reason": "explicit_tool_request_preflight",
                    "hints": normalizedToolHints,
                    "signals": preflightToolIntent.get("reasons"),
                },
            )

        analysisMessagePayload = {
            "tool_name": "Message Analysis",
            "tool_results": normalizedAnalysis,
        }
        self._messages.append(Message(role="tool", content=json.dumps(analysisMessagePayload, ensure_ascii=False)))
        toolSuggestionPlan = _tool_suggestion_plan(
            analysis_payload=normalizedAnalysis,
            current_message=self._message,
        )
        toolSuggestionPayload = {
            "tool_name": "Tool Suggestions",
            "tool_results": toolSuggestionPlan,
        }
        if _safe_int(toolSuggestionPlan.get("suggestion_count"), 0) > 0:
            await self._emit_stage(
                "tools.suggested",
                "Prepared suggested tool sequence before execution.",
                json=toolSuggestionPlan,
            )
            self._messages.append(
                Message(role="tool", content=json.dumps(toolSuggestionPayload, ensure_ascii=False))
            )

        memorySelection = _select_memory_recall(
            current_message=self._message,
            history_messages=self._shortHistory,
            analysis_payload=normalizedAnalysis,
            topic_transition=topicTransition,
            memory_circuit=memoryCircuit,
            chat_host_id=self._chatHostID,
            chat_type=self._chatType,
            platform=self._platform,
            topic_id=self._topicID,
        )
        memorySelectionPreview = _memory_selection_stage_preview(memorySelection)
        for record in memorySelectionPreview.get("selected", []):
            if isinstance(record, dict):
                record["message_text"] = _truncate_for_prompt(record.get("message_text"), 220)
        memoryMessagePayload = {
            "tool_name": "Memory Recall",
            "tool_results": memorySelectionPreview,
        }
        self._messages.append(Message(role="tool", content=json.dumps(memoryMessagePayload, ensure_ascii=False)))

        brevityDirective = _coerce_dict(normalizedAnalysis.get("brevity_directive"))
        brevityMode = _normalize_brevity_mode(brevityDirective.get("mode"), "standard")
        brevityReason = _as_text(brevityDirective.get("reason"), "analysis_reasoned_standard")
        fastPathEnabled = _runtime_bool("orchestrator.fast_path_small_talk_enabled", True)
        fastPathMaxChars = max(1, _runtime_int("orchestrator.fast_path_small_talk_max_chars", 96))
        messageCharCount = len(_as_text(self._message))
        analysisNeedsTools = _as_bool(normalizedAnalysis.get("needs_tools"), fallback=False)
        analysisToolHints = [hint for hint in _as_string_list(normalizedAnalysis.get("tool_hints")) if hint != "skipTools"]
        if autoExpandToolStage and not analysisNeedsTools and analysisToolHints:
            analysisNeedsTools = True
            normalizedAnalysis["needs_tools"] = True
            await self._emit_stage(
                "orchestrator.expand",
                "Escalating to tool stage based on analysis hints.",
                json={
                    "reason": "analysis_hints_auto_expand",
                    "hints": analysisToolHints,
                },
            )
        suggestionCount = _safe_int(toolSuggestionPlan.get("suggestion_count"), 0)
        processDirectiveForFastPath = _coerce_dict(normalizedAnalysis.get("process_directive"))
        processActionForFastPath = _as_text(processDirectiveForFastPath.get("action"), "none")
        knowledgeStateForFastPath = _coerce_dict(normalizedAnalysis.get("knowledge_state"))
        discoveryRequiredForFastPath = _as_bool(
            knowledgeStateForFastPath.get("discovery_required"),
            fallback=_normalize_knowledge_classification(
                knowledgeStateForFastPath.get("classification"),
                "known",
            )
            in {"probably_unknown", "unknown"},
        )
        fastPathBlockedReasons: list[str] = []
        if messageCharCount > fastPathMaxChars:
            fastPathBlockedReasons.append("message_too_long")
        if suggestionCount > 0:
            fastPathBlockedReasons.append("tool_suggestions_present")
        if processActionForFastPath != "none":
            fastPathBlockedReasons.append("process_action_present")
        if analysisNeedsTools:
            fastPathBlockedReasons.append("analysis_requires_tools")
        if discoveryRequiredForFastPath:
            fastPathBlockedReasons.append("knowledge_uncertainty_requires_discovery")
        if preflightToolRequested:
            fastPathBlockedReasons.append("preflight_tool_intent")
        fastPathActive = (
            fastPathEnabled
            and brevityMode == "brief_social"
            and not analysisNeedsTools
            and not preflightToolRequested
            and messageCharCount <= fastPathMaxChars
            and suggestionCount <= 0
            and processActionForFastPath == "none"
        )
        if fastPathEnabled and not fastPathActive and fastPathBlockedReasons:
            await self._emit_stage(
                "orchestrator.fast_path_blocked",
                "Fast-path disabled for this turn due to tool/process signals.",
                json={
                    "reasons": fastPathBlockedReasons,
                    "message_chars": messageCharCount,
                    "max_chars": fastPathMaxChars,
                    "suggestion_count": suggestionCount,
                    "process_action": processActionForFastPath,
                },
            )

        toolResponses: list[dict[str, Any]] = []
        toolSummary: dict[str, Any] = {}
        processToolState: dict[str, Any] = {"has_updates": False}
        toolExecutionMode = "skipped_fast_path" if fastPathActive else "native_tools"
        knownContextResults = _coerce_dict(knownContext.get("tool_results"))

        def _build_tool_stage_messages() -> list[Message]:
            toolHistoryLimit = max(2, _runtime_int("orchestrator.tool_history_limit", 6))
            stageMessages = _recent_dialogue_messages(toolHistoryLimit)
            stageMessages.append(Message(role="tool", content=json.dumps(knownContextPrompt, ensure_ascii=False)))
            if personalityEnabled and personalityPromptMessage:
                stageMessages.append(
                    Message(role="tool", content=json.dumps(personalityPromptMessage, ensure_ascii=False))
                )
            stageMessages.append(Message(role="tool", content=json.dumps(analysisMessagePayload, ensure_ascii=False)))
            if _safe_int(toolSuggestionPlan.get("suggestion_count"), 0) > 0:
                stageMessages.append(Message(role="tool", content=json.dumps(toolSuggestionPayload, ensure_ascii=False)))
            if memorySelectionPreview.get("selected_count", 0):
                stageMessages.append(Message(role="tool", content=json.dumps(memoryMessagePayload, ensure_ascii=False)))
            return stageMessages

        async def _run_tool_stage(analysisPayload: dict[str, Any], reason: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
            await self._emit_stage(
                "tools.start",
                "Evaluating tool calls.",
                reason=reason,
            )
            toolOptions = dict(self._options)
            toolRuntimeContext = _coerce_dict(toolOptions.get("tool_runtime_context"))
            toolRuntimeContext.update(
                {
                    "chat_host_id": self._chatHostID,
                    "chat_type": self._chatType,
                    "platform": self._platform,
                    "topic_id": self._topicID,
                    "member_id": self._memberID,
                    "topic_transition": _coerce_dict(knownContextResults.get("topic_transition")),
                    "memory_circuit": _coerce_dict(knownContextResults.get("memory_circuit")),
                    "workspace_context": _coerce_dict(knownContextResults.get("workspace_context")),
                    "history_search_allowed": _as_bool(
                        _coerce_dict(knownContextResults.get("memory_circuit")).get("history_search_allowed"),
                        True,
                    ),
                }
            )
            toolOptions["tool_runtime_context"] = toolRuntimeContext
            toolRunContext = _coerce_dict(toolOptions.get("run_context"))
            if self._memberID is not None:
                toolRunContext["member_id"] = self._memberID
            if self._transientSession:
                toolRunContext["guest_mode"] = True
            toolOptions["run_context"] = toolRunContext
            toolOptions["analysis_payload"] = _coerce_dict(analysisPayload)
            toolOptions["latest_user_message"] = self._message
            self._toolsAgent = ToolCallingAgent(_build_tool_stage_messages(), options=toolOptions)
            responses = await self._toolsAgent.generateResponse()
            summary = _coerce_dict(getattr(self._toolsAgent, "execution_summary", {}))
            executionMode = _as_text(summary.get("execution_mode"), "native_tools")
            toolsRouting = _coerce_dict(getattr(self._toolsAgent, "routing", {}))
            requestedToolCalls = summary.get("requested_tool_calls")
            executedToolCalls = summary.get("executed_tool_calls")
            if not isinstance(executedToolCalls, int):
                executedToolCalls = len(responses)
            await self._emit_stage(
                "tools.complete",
                "Tool execution stage complete.",
                tool_calls=len(responses),
                requested_tool_calls=requestedToolCalls if isinstance(requestedToolCalls, int) else len(responses),
                executed_tool_calls=executedToolCalls,
                selected_model=toolsRouting.get("selected_model"),
                json={
                    "routing": toolsRouting,
                    "summary": summary,
                    "tool_results": responses,
                },
            )
            return responses, summary, executionMode

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
            toolResponses, toolSummary, toolExecutionMode = await _run_tool_stage(
                normalizedAnalysis,
                reason="analysis",
            )
            roundsUsed = 0
            while (
                autoExpandToolStage
                and roundsUsed < autoExpandMaxRounds
                and analysisNeedsTools
                and not toolResponses
                and bool(analysisToolHints)
            ):
                roundsUsed += 1
                await self._emit_stage(
                    "orchestrator.expand",
                    "Tool stage returned no calls; expanding with stricter planning constraints.",
                    json={
                        "round": roundsUsed,
                        "max_rounds": autoExpandMaxRounds,
                        "hints": analysisToolHints,
                    },
                )
                forcedAnalysisPayload = copy.deepcopy(normalizedAnalysis)
                forcedAnalysisPayload["needs_tools"] = True
                forcedAnalysisPayload["tool_hints"] = analysisToolHints
                toolResponses, toolSummary, toolExecutionMode = await _run_tool_stage(
                    forcedAnalysisPayload,
                    reason=f"auto_expand_round_{roundsUsed}",
                )
                if toolResponses:
                    break

            for toolResponse in toolResponses:
                self._messages.append(Message(role="tool", content=json.dumps(toolResponse, ensure_ascii=False)))
            processToolState = _process_tool_stage_summary(toolResponses)
            if _as_bool(processToolState.get("has_updates"), False):
                await self._emit_stage(
                    "process.state",
                    "Process/message workspace updated during tool execution.",
                    json=processToolState,
                )
        
        # TODO Tools to thoughts "thinking" agent next, will produce thoughts based analysis and tool responses, outputs thoughts followed by a prompt
        
        # TODO Passes only the prompt to the response agent
        # Pass options=options to override the langauge model

        responseHistoryLimit = max(2, _runtime_int("orchestrator.response_history_limit", 10))
        responseMessages = _recent_dialogue_messages(responseHistoryLimit)
        responseMessages.append(Message(role="tool", content=json.dumps(knownContextPrompt, ensure_ascii=False)))
        if personalityEnabled and personalityPromptMessage:
            responseMessages.append(
                Message(role="tool", content=json.dumps(personalityPromptMessage, ensure_ascii=False))
            )
        responseMessages.append(Message(role="tool", content=json.dumps(analysisMessagePayload, ensure_ascii=False)))
        if memorySelectionPreview.get("selected_count", 0):
            responseMessages.append(Message(role="tool", content=json.dumps(memoryMessagePayload, ensure_ascii=False)))
        for toolResponse in toolResponses:
            responseMessages.append(Message(role="tool", content=json.dumps(toolResponse, ensure_ascii=False)))

        self._chatConversationAgent = ChatConversationAgent(messages=responseMessages, options=self._options)
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
        responseTimedOut = False
        responseLastProgressLog = time.monotonic()
        responseLastProgressEmit = time.monotonic()
        responseLatestPreview = ""
        responseChunkTimeout = _timeout_seconds("inference.stream_chunk_timeout_seconds", 45.0)
        responseTotalTimeout = _timeout_seconds("inference.stream_total_timeout_seconds", 240.0)
        responseDeadline = (
            time.monotonic() + float(responseTotalTimeout)
            if isinstance(responseTotalTimeout, (int, float))
            else None
        )
        responseIterator = response.__aiter__()
        
        #print(f"{ConsoleColors["blue"]}Assistant > ", end="")
        chunk: ChatResponse
        while True:
            try:
                chunk = await _next_stream_chunk(
                    responseIterator,
                    idle_timeout_seconds=responseChunkTimeout,
                    deadline_monotonic=responseDeadline,
                )
            except StopAsyncIteration:
                break
            except TimeoutError as error:
                responseTimedOut = True
                logger.warning(
                    "[stream.response] stream timeout; "
                    f"finalizing with available content (chunks={responseChunkCount}, chars={responseCharCount}): {error}"
                )
                await self._emit_stage(
                    "response.timeout",
                    "Response stream timed out; finalizing with available content.",
                    chunks=responseChunkCount,
                    chars=responseCharCount,
                    timeout_seconds=responseChunkTimeout,
                    total_timeout_seconds=responseTotalTimeout,
                )
                break
            except Exception as error:  # noqa: BLE001
                logger.exception(
                    "[stream.response] stream iteration failed; "
                    f"finalizing with available content (chunks={responseChunkCount}, chars={responseCharCount})"
                )
                await self._emit_stage(
                    "response.error",
                    "Response stream failed; finalizing with available content.",
                    error=str(error),
                    chunks=responseChunkCount,
                    chars=responseCharCount,
                )
                break
            chunkContent = _chunk_message_content(chunk)
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
            if emitProgressStages and chunkContent and (
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
            if _chunk_done(chunk):
                responseDoneSeen = True
                self._devStats = _chunk_stream_stats(chunk)
                evalCount = _safe_int(_chunk_field(chunk, "eval_count"), 0)
                logger.info(
                    "[stream.response] "
                    f"done chunks={responseChunkCount} chars={responseCharCount} eval_count={evalCount}"
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
        if _is_low_signal_response(sanitizedResponseMessage) and not responseTimedOut:
            await self._emit_stage(
                "response.repair",
                "Primary response was low-signal; running recovery generation.",
                json={
                    "execution_mode": toolExecutionMode,
                    "response_chars": len(_as_text(sanitizedResponseMessage)),
                },
            )
            try:
                repairedText, repairRouting = await self._attempt_response_repair(
                    normalized_analysis=normalizedAnalysis,
                    tool_responses=toolResponses,
                    tool_execution_mode=toolExecutionMode,
                )
                if not _is_low_signal_response(repairedText):
                    sanitizedResponseMessage = repairedText
                    await self._emit_stage(
                        "response.repair.complete",
                        "Recovery generation produced a user-facing reply.",
                        selected_model=repairRouting.get("selected_model"),
                        json={"routing": repairRouting},
                    )
                else:
                    await self._emit_stage(
                        "response.repair.complete",
                        "Recovery generation returned low-signal output.",
                        selected_model=repairRouting.get("selected_model"),
                        json={"routing": repairRouting},
                    )
            except Exception as repairError:  # noqa: BLE001
                logger.warning(f"Response repair generation failed: {repairError}")
                await self._emit_stage(
                    "response.repair.complete",
                    "Recovery generation failed; using fallback response text.",
                    error=str(repairError),
                )
        if not str(sanitizedResponseMessage or "").strip():
            if toolExecutionMode == "pseudo_structured_output":
                sanitizedResponseMessage = (
                    "I completed tool planning in compatibility mode, but the final response model "
                    "did not return usable text. Try a different chat model for this route."
                )
            else:
                sanitizedResponseMessage = "I could not generate a complete reply this turn. Please try again."
            await self._emit_stage(
                "response.fallback",
                "Model returned empty output; emitted fallback response text.",
            )
        if sanitizedResponseMessage != responseMessage:
            await self._emit_stage("response.sanitized", "Removed internal orchestration artifacts.")
        self._chatResponseMessage = sanitizedResponseMessage
        #print(ConsoleColors["default"])

        if self._responseID is not None and not self._transientSession:
            self.storeResponse(self._responseID)
        
        assistantMessage = Message(role="assistant", content=sanitizedResponseMessage)
        # Add the final response to the overall chat history (role ASSISTANT)
        self._messages.append(assistantMessage)
        if personalityEnabled and personalityProfile:
            try:
                adaptationResult = personalityEngine.adapt_after_turn(
                    profile=personalityProfile,
                    user_message=self._message,
                    assistant_message=sanitizedResponseMessage,
                    analysis_payload=normalizedAnalysis,
                    runtime_config=personalityRuntime,
                )
                personalityProfile = _coerce_dict(adaptationResult.get("profile"))
                persistedProfile = personalityStore.upsert_profile_payload(
                    self._memberID,
                    personalityProfile,
                )
                personalityAdaptationState = {
                    "changed_fields": _coerce_list(adaptationResult.get("changed_fields")),
                    "signals": _coerce_dict(adaptationResult.get("signals")),
                    "reason_code": _as_text(adaptationResult.get("reason_code")),
                    "reason_detail": _as_text(adaptationResult.get("reason_detail")),
                    "profile_version": _safe_int(_coerce_dict(persistedProfile).get("profile_version"), 0),
                }
                personalityStore.append_event(
                    member_id=self._memberID,
                    event_type="adaptive_update",
                    before_json=_coerce_dict(adaptationResult.get("before")),
                    after_json=_coerce_dict(adaptationResult.get("after")),
                    reason_code=_as_text(adaptationResult.get("reason_code")),
                    reason_detail=_as_text(adaptationResult.get("reason_detail")),
                )
                await self._emit_stage(
                    "persona.adapt",
                    "Updated adaptive personality state from turn-level behavior signals.",
                    changed_fields=personalityAdaptationState.get("changed_fields"),
                    profile_version=personalityAdaptationState.get("profile_version"),
                    json=personalityAdaptationState,
                )

                if personalityRuntime.narrative_enabled and narrativeRollup.should_rollup(
                    profile=personalityProfile,
                    turn_threshold=personalityRuntime.rollup_turn_threshold,
                    char_threshold=personalityRuntime.rollup_char_threshold,
                ):
                    rollupResult = narrativeRollup.build_rollup(
                        profile=personalityProfile,
                        history_messages=self._shortHistory,
                        user_message=self._message,
                        assistant_message=sanitizedResponseMessage,
                        analysis_payload=normalizedAnalysis,
                        max_source_messages=personalityRuntime.narrative_source_history_limit,
                        max_summary_chars=personalityRuntime.narrative_summary_max_chars,
                    )
                    chunkRecord = personalityStore.insert_narrative_chunk(
                        member_id=self._memberID,
                        chunk_index=_safe_int(rollupResult.get("chunk_index"), 1),
                        summary_text=_as_text(rollupResult.get("summary_text")),
                        summary_json=_coerce_dict(rollupResult.get("summary_json")),
                        compression_ratio=_coerce_float(rollupResult.get("compression_ratio"), 1.0),
                        source_turn_start_id=self.promptHistoryID,
                        source_turn_end_id=self.responseHistoryID,
                    )
                    personalityProfile = narrativeRollup.apply_rollup_to_profile(
                        profile=personalityProfile,
                        rollup_result=rollupResult,
                    )
                    personalityStore.upsert_profile_payload(self._memberID, personalityProfile)
                    personalityStore.append_event(
                        member_id=self._memberID,
                        event_type="rollup",
                        before_json={"narrative": _coerce_dict(_coerce_dict(adaptationResult.get("before")).get("narrative"))},
                        after_json={"narrative": _coerce_dict(_coerce_dict(personalityProfile).get("narrative"))},
                        reason_code="narrative_rollup_threshold",
                        reason_detail="Narrative rollup triggered from turn/char threshold.",
                    )
                    personalityRollupState = {
                        "chunk_index": _safe_int(rollupResult.get("chunk_index"), 0),
                        "compression_ratio": _coerce_float(rollupResult.get("compression_ratio"), 1.0),
                        "summary_chars": len(_as_text(rollupResult.get("summary_text"))),
                        "recorded_chunk_id": _safe_int(_coerce_dict(chunkRecord).get("chunk_id"), 0),
                    }
                    await self._emit_stage(
                        "persona.rollup",
                        "Rolled up narrative continuity into a compact profile chunk.",
                        chunk_index=personalityRollupState.get("chunk_index"),
                        compression_ratio=personalityRollupState.get("compression_ratio"),
                        json=personalityRollupState,
                    )
            except Exception as error:  # noqa: BLE001
                logger.exception("Personality adaptation pipeline failed.")
                await self._emit_stage(
                    "persona.error",
                    "Personality adaptation failed; continuing without profile mutation.",
                    error=str(error),
                )
        personalityRunPreview: dict[str, Any] = {}
        if personalityEnabled and personalityProfile:
            try:
                latestNarrativeSummary = _as_text(
                    _coerce_dict(_coerce_dict(personalityProfile).get("narrative")).get("active_summary")
                )
                latestInjection = personalityInjector.build_payload(
                    profile=personalityProfile,
                    narrative_summary=latestNarrativeSummary,
                    max_injection_chars=personalityRuntime.max_injection_chars,
                )
                personalityRunPreview = _personality_context_stage_preview(latestInjection)
            except Exception:  # noqa: BLE001
                personalityRunPreview = _personality_context_stage_preview(personalityPromptPayload)
        self._runSummary = {
            "known_context": _coerce_dict(knownContext.get("tool_results")),
            "image_context": _coerce_dict(self._imageContext),
            "topic_transition": _coerce_dict(topicTransition),
            "memory_circuit": _coerce_dict(memoryCircuit),
            "memory_recall": memorySelectionPreview,
            "analysis_payload": _coerce_dict(normalizedAnalysis),
            "tool_suggestions": _coerce_dict(toolSuggestionPlan),
            "analysis_routing": _coerce_dict(getattr(self._analysisAgent, "routing", {})),
            "analysis_stats": analysisStatsSummary,
            "tool_execution_mode": toolExecutionMode,
            "tool_summary": _coerce_dict(toolSummary),
            "process_tool_state": _coerce_dict(processToolState),
            "tool_results": toolResponses if isinstance(toolResponses, list) else [],
            "response": {
                "text": sanitizedResponseMessage,
                "sanitized": bool(sanitizedResponseMessage != responseMessage),
            },
            "personality": {
                "enabled": personalityEnabled,
                "profile": personalityRunPreview,
                "adaptation": _coerce_dict(personalityAdaptationState),
                "rollup": _coerce_dict(personalityRollupState),
            },
            "chat_routing": _coerce_dict(getattr(self._chatConversationAgent, "routing", {})),
            "response_stats": responseStatsSummary,
        }
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

    @property
    def run_summary(self) -> dict[str, Any]:
        return copy.deepcopy(_coerce_dict(self._runSummary))



##################
# POLICY MANAGER #
##################

# TODO Need to create a Policy Manager that will load the agent's policy and allow for edits and save edits to file
# Perhaps policy manager belongs in utils?



###############
# AGENT TOOLS #
###############

_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
}
_READ_ONLY_HTTP_METHODS = {"GET", "HEAD", "OPTIONS"}


def _extract_first_url(text: Any) -> str:
    raw = _as_text(text)
    if not raw:
        return ""
    match = re.search(r"https?://[^\s<>\"]+", raw, flags=re.IGNORECASE)
    if match is None:
        return ""
    return _as_text(match.group(0))


def _sanitize_header_map(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    output: dict[str, str] = {}
    for raw_key, raw_value in headers.items():
        key = _as_text(raw_key)
        value = _as_text(raw_value)
        if not key:
            continue
        output[key] = value
    return output


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in headers.items():
        if _as_text(key).lower() in _SENSITIVE_HEADER_NAMES and value:
            output[key] = "***REDACTED***"
        else:
            output[key] = value
    return output


def _hostname_allowed(hostname: str, allowlist: list[str]) -> bool:
    host = _as_text(hostname).lower().strip(".")
    if not host:
        return False
    if not allowlist:
        return True
    for allowed_raw in allowlist:
        allowed = _as_text(allowed_raw).lower().strip(".")
        if not allowed:
            continue
        if allowed.startswith("*."):
            allowed = allowed[2:]
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _is_private_network_hostname(hostname: str) -> bool:
    host = _as_text(hostname).strip("[]")
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.lower() in {"localhost"}
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def curlRequest(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    bodyText: str = "",
    timeoutSeconds: int | None = None,
    followRedirects: bool = True,
    maxResponseChars: int | None = None,
) -> dict:
    """Execute a curl-style HTTP request and return structured response details."""
    request_url = _as_text(url)
    parsed = urlparse(request_url)
    scheme = _as_text(parsed.scheme).lower()
    hostname = _as_text(parsed.hostname).lower()
    if scheme not in {"http", "https"}:
        return {"status": "error", "error": "unsupported_url_scheme", "detail": "Only http/https URLs are supported."}
    if not hostname:
        return {"status": "error", "error": "missing_hostname", "detail": "URL hostname is required."}

    allowlist = _runtime_model_list("tool_runtime.curl_allowlist_domains", [])
    if not _hostname_allowed(hostname, allowlist):
        return {
            "status": "error",
            "error": "domain_not_allowlisted",
            "detail": f"Hostname '{hostname}' is not in curl allowlist.",
            "allowlist": allowlist,
        }

    allow_private_network = _runtime_bool("tool_runtime.curl_allow_private_network", False)
    if _is_private_network_hostname(hostname) and not allow_private_network:
        return {
            "status": "error",
            "error": "private_network_blocked",
            "detail": f"Private/local hostname '{hostname}' is blocked by runtime policy.",
        }

    normalized_method = _as_text(method, "GET").upper()
    if normalized_method not in {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}:
        return {
            "status": "error",
            "error": "unsupported_method",
            "detail": f"HTTP method '{normalized_method}' is unsupported.",
        }

    allow_mutating_methods = _runtime_bool("tool_runtime.curl_allow_mutating_methods", False)
    if normalized_method not in _READ_ONLY_HTTP_METHODS and not allow_mutating_methods:
        return {
            "status": "error",
            "error": "mutating_method_blocked",
            "detail": (
                f"Method '{normalized_method}' is disabled by runtime policy. "
                "Enable tool_runtime.curl_allow_mutating_methods to permit it."
            ),
        }

    timeout_seconds = _safe_int(timeoutSeconds, int(_runtime_float("tool_runtime.curl_timeout_seconds", 12.0)))
    if timeout_seconds <= 0:
        timeout_seconds = 12
    timeout_seconds = min(120, timeout_seconds)
    max_response_chars = _safe_int(
        maxResponseChars,
        max(256, _runtime_int("tool_runtime.curl_max_response_chars", 3500)),
    )
    if max_response_chars <= 0:
        max_response_chars = 3500

    request_headers = _sanitize_header_map(headers)
    body_text = _as_text(bodyText)

    curl_cmd: list[str] = [
        "curl",
        "--silent",
        "--show-error",
        "--request",
        normalized_method,
        "--max-time",
        str(timeout_seconds),
        "--connect-timeout",
        str(min(10, timeout_seconds)),
    ]
    if followRedirects:
        curl_cmd.append("--location")
    else:
        curl_cmd.extend(["--max-redirs", "0"])
    for key, value in request_headers.items():
        curl_cmd.extend(["--header", f"{key}: {value}"])
    if body_text and normalized_method not in {"GET", "HEAD", "OPTIONS"}:
        curl_cmd.extend(["--data-raw", body_text])
    curl_cmd.extend(
        [
            request_url,
            "--write-out",
            "\n__RYO_CURL_META__:%{http_code}|%{content_type}|%{size_download}|%{time_total}|%{url_effective}",
        ]
    )

    display_cmd = list(curl_cmd)
    for idx, token in enumerate(display_cmd):
        if idx > 0 and display_cmd[idx - 1] == "--header":
            parts = token.split(":", 1)
            header_name = _as_text(parts[0])
            if header_name.lower() in _SENSITIVE_HEADER_NAMES:
                display_cmd[idx] = f"{header_name}: ***REDACTED***"
        if idx > 0 and display_cmd[idx - 1] == "--data-raw" and len(token) > 120:
            display_cmd[idx] = token[:117] + "..."

    transport = "curl"
    status_code = 0
    content_type = ""
    size_downloaded = 0
    elapsed_seconds = 0.0
    effective_url = request_url
    response_text = ""

    try:
        completed = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 3,
            check=False,
        )
        stdout = _as_text(completed.stdout)
        stderr = _as_text(completed.stderr)
        marker = "__RYO_CURL_META__:"
        if marker in stdout:
            body, _, meta_text = stdout.rpartition(marker)
            response_text = body.rstrip("\n")
            meta_parts = meta_text.split("|", 4)
            if meta_parts:
                status_code = _safe_int(meta_parts[0], 0)
            if len(meta_parts) > 1:
                content_type = _as_text(meta_parts[1])
            if len(meta_parts) > 2:
                size_downloaded = _safe_int(meta_parts[2], 0)
            if len(meta_parts) > 3:
                try:
                    elapsed_seconds = float(meta_parts[3])
                except (TypeError, ValueError):
                    elapsed_seconds = 0.0
            if len(meta_parts) > 4:
                effective_url = _as_text(meta_parts[4], request_url)
        else:
            response_text = stdout

        if completed.returncode != 0 and status_code == 0:
            return {
                "status": "error",
                "error": "curl_command_failed",
                "detail": stderr or f"curl exited with return code {completed.returncode}",
                "curl_command": shlex.join(display_cmd),
                "request": {
                    "url": request_url,
                    "method": normalized_method,
                    "headers": _redact_headers(request_headers),
                },
            }
    except FileNotFoundError:
        transport = "requests_fallback"
        try:
            req_response = requests.request(
                method=normalized_method,
                url=request_url,
                headers=request_headers or None,
                data=body_text or None,
                timeout=float(timeout_seconds),
                allow_redirects=bool(followRedirects),
            )
        except Exception as error:  # noqa: BLE001
            return {
                "status": "error",
                "error": "http_request_failed",
                "detail": str(error),
                "transport": transport,
                "request": {
                    "url": request_url,
                    "method": normalized_method,
                    "headers": _redact_headers(request_headers),
                },
            }
        status_code = _safe_int(req_response.status_code, 0)
        content_type = _as_text(req_response.headers.get("Content-Type"))
        size_downloaded = len(req_response.content or b"")
        if getattr(req_response, "elapsed", None):
            elapsed_seconds = float(req_response.elapsed.total_seconds())
        effective_url = _as_text(getattr(req_response, "url", request_url), request_url)
        response_text = _as_text(req_response.text)
    except subprocess.TimeoutExpired as error:
        return {
            "status": "error",
            "error": "curl_timeout",
            "detail": f"curl timed out after {timeout_seconds}s",
            "transport": transport,
            "request": {
                "url": request_url,
                "method": normalized_method,
                "headers": _redact_headers(request_headers),
            },
            "stderr": _as_text(getattr(error, "stderr", "")),
        }

    response_excerpt = response_text
    truncated = False
    if len(response_excerpt) > max_response_chars:
        response_excerpt = response_excerpt[: max_response_chars - 3].rstrip() + "..."
        truncated = True

    response_json = None
    if response_text and ("json" in content_type.lower() or response_text.lstrip().startswith(("{", "["))):
        try:
            response_json = json.loads(response_text)
        except Exception:  # noqa: BLE001
            response_json = None

    return {
        "status": "ok",
        "transport": transport,
        "request": {
            "url": request_url,
            "method": normalized_method,
            "headers": _redact_headers(request_headers),
            "body_chars": len(body_text),
            "follow_redirects": bool(followRedirects),
            "timeout_seconds": timeout_seconds,
        },
        "response": {
            "status_code": status_code,
            "ok": status_code < 400 if status_code else False,
            "content_type": content_type,
            "size_downloaded": size_downloaded,
            "elapsed_seconds": round(float(elapsed_seconds), 4),
            "effective_url": effective_url,
            "body_excerpt": response_excerpt,
            "body_excerpt_truncated": truncated,
            "json": response_json,
        },
        "curl_command": shlex.join(display_cmd),
    }

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
    progressive_tool_enabled = _runtime_bool("retrieval.progressive_history_tool_enabled", True)
    if (
        progressive_tool_enabled
        and scopedSearch
        and _as_text(queryString)
        and context.get("chat_host_id") is not None
        and _as_text(context.get("chat_type"))
        and _as_text(context.get("platform"))
    ):
        progressive_config_payload = _progressive_history_runtime_payload(max_selected_override=max(1, int(count)))
        explorer = ProgressiveHistoryExplorer(
            chat_history_manager=chatHistory,
            config=ProgressiveHistoryExplorerConfig.from_runtime(progressive_config_payload),
        )
        progressive_result = explorer.explore(
            query_text=queryString,
            chat_host_id=context.get("chat_host_id"),
            chat_type=_as_text(context.get("chat_type")),
            platform=_as_text(context.get("platform")),
            topic_id=context.get("topic_id"),
            history_recall_requested=history_recall_requested,
            switched=switched,
            allow_history_search=_as_bool(context.get("history_search_allowed"), True),
        )
        progressive_selected_raw = progressive_result.get("selected")
        progressive_selected = progressive_selected_raw if isinstance(progressive_selected_raw, list) else []
        if progressive_selected and (
            _as_bool(progressive_result.get("found"), False) or history_recall_requested
        ):
            converted_results: list[dict[str, Any]] = []
            for result in progressive_selected[: max(1, int(count))]:
                row = _coerce_dict(result)
                converted_results.append(
                    {
                        "history_id": row.get("history_id"),
                        "message_id": row.get("message_id"),
                        "message_text": row.get("message_text"),
                        "message_timestamp": row.get("timestamp_utc"),
                        "score": row.get("score"),
                        "signals": _coerce_dict(row.get("signals")),
                        "role": _as_text(row.get("role")),
                    }
                )
            logger.debug(
                "Chat history tool used progressive recall:\n"
                f"query={queryString}\n"
                f"scope_chat_host={context.get('chat_host_id')}\n"
                f"found={progressive_result.get('found')}\n"
                f"selected={len(converted_results)}"
            )
            return converted_results

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


def _runtime_member_id(runtime_context: dict | None) -> int | None:
    context = _coerce_dict(runtime_context)
    member_id = context.get("member_id")
    try:
        value = int(member_id)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def knownUsersList(queryString: str = "", count: int = 20, runtime_context: dict | None = None) -> dict:
    """List members known in the database for inter-user communication workflows."""
    known_users = collaboration.listKnownUsers(queryString=queryString, count=count)
    return {
        "query": _as_text(queryString),
        "count": len(known_users),
        "users": known_users,
    }


def messageKnownUser(
    targetUsername: str = "",
    targetMemberID: int = None,
    messageText: str = "",
    processID: int = None,
    sendNow: bool = True,
    runtime_context: dict | None = None,
) -> dict:
    """Queue or send a direct message to a known user in the workspace."""
    sender_member_id = _runtime_member_id(runtime_context)
    return collaboration.queueOrSendUserMessage(
        senderMemberID=sender_member_id,
        targetUsername=targetUsername,
        targetMemberID=targetMemberID,
        messageText=messageText,
        deliveryChannel="telegram",
        processID=processID,
        sendNow=sendNow,
    )


def upsertProcessWorkspace(
    processLabel: str,
    processDescription: str = "",
    processSpec: Any = None,
    processID: int = None,
    processStatus: str = "active",
    replaceSteps: bool = True,
    runtime_context: dict | None = None,
) -> dict:
    """Create or update a multi-step process workspace persisted across turns."""
    owner_member_id = _runtime_member_id(runtime_context)
    if owner_member_id is None:
        return {"status": "error", "error": "owner_member_id_required"}
    return collaboration.createOrUpdateProcess(
        ownerMemberID=owner_member_id,
        processLabel=processLabel,
        processDescription=processDescription,
        processSpec=processSpec,
        processID=processID,
        processStatus=processStatus,
        replaceSteps=replaceSteps,
    )


def listProcessWorkspace(
    processStatus: str = "active",
    count: int = 12,
    includeSteps: bool = False,
    processID: int = None,
    runtime_context: dict | None = None,
) -> dict:
    """List active/incomplete persisted processes for the calling member."""
    owner_member_id = _runtime_member_id(runtime_context)
    if owner_member_id is None:
        return {"status": "error", "error": "owner_member_id_required"}

    if processID is not None:
        process = collaboration.getProcessByID(owner_member_id, processID, include_steps=includeSteps)
        if process is None:
            return {"status": "error", "error": "process_not_found", "process_id": processID}
        return {"status": "ok", "count": 1, "processes": [process]}

    processes = collaboration.listProcesses(
        ownerMemberID=owner_member_id,
        processStatus=processStatus,
        count=count,
        include_steps=includeSteps,
    )
    return {"status": "ok", "count": len(processes), "processes": processes}


def updateProcessWorkspaceStep(
    processID: int,
    stepID: int = None,
    stepOrder: int = None,
    stepLabel: str = "",
    stepStatus: str = "completed",
    stepDetails: str = None,
    stepPayload: dict | None = None,
    runtime_context: dict | None = None,
) -> dict:
    """Update a process step state and refresh completion/progress counters."""
    owner_member_id = _runtime_member_id(runtime_context)
    if owner_member_id is None:
        return {"status": "error", "error": "owner_member_id_required"}
    return collaboration.updateProcessStep(
        ownerMemberID=owner_member_id,
        processID=processID,
        stepID=stepID,
        stepOrder=stepOrder,
        stepLabel=stepLabel,
        stepStatus=stepStatus,
        stepDetails=stepDetails,
        stepPayload=_coerce_dict(stepPayload),
    )


def listOutboxMessages(
    count: int = 20,
    deliveryStatus: str = "",
    runtime_context: dict | None = None,
) -> dict:
    """List queued/sent member outbox messages for the calling member."""
    owner_member_id = _runtime_member_id(runtime_context)
    if owner_member_id is None:
        return {"status": "error", "error": "owner_member_id_required"}
    messages = collaboration.listOutboxForMember(
        memberID=owner_member_id,
        count=count,
        deliveryStatus=deliveryStatus,
    )
    return {"status": "ok", "count": len(messages), "messages": messages}


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
        self._model = _select_initial_model_from_allowed(self._allowed_models, "tool")
        if self._model and self._model not in self._allowed_models:
            self._allowed_models = [self._model, *self._allowed_models]
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
        self._analysisPayload = _coerce_dict(optionMap.get("analysis_payload"))
        self._latestUserMessage = _as_text(optionMap.get("latest_user_message"))
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
            curl_request_fn=curlRequest,
            chat_history_search_fn=chatHistorySearch,
            knowledge_search_fn=knowledgeSearch,
            skip_tools_fn=skipTools,
            known_users_list_fn=knownUsersList,
            message_known_user_fn=messageKnownUser,
            process_workspace_upsert_fn=upsertProcessWorkspace,
            process_workspace_list_fn=listProcessWorkspace,
            process_workspace_step_update_fn=updateProcessWorkspaceStep,
            outbox_list_fn=listOutboxMessages,
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
                curl_request_fn=curlRequest,
                chat_history_search_fn=chatHistorySearch,
                knowledge_search_fn=knowledgeSearch,
                skip_tools_fn=skipTools,
                known_users_list_fn=knownUsersList,
                message_known_user_fn=messageKnownUser,
                process_workspace_upsert_fn=upsertProcessWorkspace,
                process_workspace_list_fn=listProcessWorkspace,
                process_workspace_step_update_fn=updateProcessWorkspaceStep,
                outbox_list_fn=listOutboxMessages,
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
        available_set = set(available)
        candidates: list[str] = []
        enforce_capability = _runtime_bool("tool_runtime.enforce_native_tool_capability", True)
        prefer_available = _runtime_bool("tool_runtime.prefer_available_model_inventory", True)

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
            if enforce_capability and self._cached_tool_capability(model_name) is False:
                return
            if prefer_available and available_set and model_name not in available_set:
                return
            candidates.append(model_name)

        # Prefer discovered inventory (currently installed), then runtime defaults and policy chain.
        for model_name in available:
            add(model_name)
        add(_runtime_value("inference.default_tool_model", ""))
        add(_runtime_value("inference.default_chat_model", ""))
        add(_runtime_value("inference.default_multimodal_model", ""))
        for model_name in self._allowed_models:
            add(model_name)

        limit = _runtime_int("tool_runtime.auto_model_retry_candidate_limit", 6)
        if limit <= 0:
            limit = 6
        return candidates[:limit]

    def _tool_capability_cache_key(self, model_name: str) -> str:
        host = self._modelRouter.resolve_host("tool")
        return f"{host}|{model_name}"

    def _cached_tool_capability(self, model_name: str) -> bool | None:
        cache_key = self._tool_capability_cache_key(model_name)
        entry = _TOOL_CAPABILITY_CACHE.get(cache_key)
        if not isinstance(entry, dict):
            return None
        ttl_seconds = max(5.0, _runtime_float("tool_runtime.tool_capability_probe_cache_ttl_seconds", 21600.0))
        cached_at = float(entry.get("cached_at") or 0.0)
        if (time.monotonic() - cached_at) > ttl_seconds:
            _TOOL_CAPABILITY_CACHE.pop(cache_key, None)
            return None
        capable = entry.get("capable")
        if isinstance(capable, bool):
            return capable
        return None

    def _remember_tool_capability(self, model_name: str, *, capable: bool, reason: str = "") -> None:
        cleaned_model = _normalize_model_name(model_name)
        if not cleaned_model:
            return
        cache_key = self._tool_capability_cache_key(cleaned_model)
        _TOOL_CAPABILITY_CACHE[cache_key] = {
            "cached_at": time.monotonic(),
            "capable": bool(capable),
            "reason": _as_text(reason),
        }

    async def _probe_native_tool_capability(self, model_name: str) -> bool:
        cached = self._cached_tool_capability(model_name)
        if isinstance(cached, bool):
            return cached

        model_name = _normalize_model_name(model_name)
        if not model_name:
            return False
        if not self._modelTools:
            self._remember_tool_capability(model_name, capable=True, reason="no_tools_registered")
            return True

        host = self._modelRouter.resolve_host("tool")
        client = AsyncClient(host=host)
        probe_tools = [self._modelTools[0]]
        probe_messages = [
            Message(
                role="system",
                content=(
                    "Native tool capability probe. "
                    "Call exactly one available tool with minimal valid arguments and no prose."
                ),
            ),
            Message(role="user", content="Probe native tool calling support."),
        ]
        timeout_seconds = max(0.5, _runtime_float("tool_runtime.tool_capability_probe_timeout_seconds", 3.0))
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model_name,
                    messages=probe_messages,
                    stream=False,
                    tools=probe_tools,
                    options={"temperature": 0, "num_predict": 32},
                ),
                timeout=timeout_seconds,
            )
            response_message = getattr(response, "message", None)
            tool_calls = list(getattr(response_message, "tool_calls", []) or [])
            capable = bool(tool_calls)
            reason = "probe_tool_calls_present" if capable else "probe_missing_tool_calls"
            self._remember_tool_capability(model_name, capable=capable, reason=reason)
            return capable
        except asyncio.TimeoutError:
            self._remember_tool_capability(model_name, capable=False, reason="probe_timeout")
            return False
        except Exception as error:  # noqa: BLE001
            capable = False
            routing = {"errors": [str(error)], "selected_model": model_name}
            if not self._is_tool_capability_failure(error, routing):
                logger.warning(f"Tool capability probe failed for model '{model_name}': {error}")
            self._remember_tool_capability(model_name, capable=capable, reason=f"probe_error:{error}")
            return capable

    async def _resolve_native_tool_candidates(self, candidates: list[str]) -> list[str]:
        ordered = [name for name in _dedupe_models(candidates) if not _looks_like_embedding_model(name)]
        if not ordered:
            return []

        if not _runtime_bool("tool_runtime.enforce_native_tool_capability", True):
            return ordered
        if not _runtime_bool("tool_runtime.tool_capability_probe_enabled", True):
            return ordered

        known_capable: list[str] = []
        unknown: list[str] = []
        for model_name in ordered:
            cached = self._cached_tool_capability(model_name)
            if cached is True:
                known_capable.append(model_name)
            elif cached is None:
                unknown.append(model_name)

        if known_capable:
            return known_capable

        probe_limit = _runtime_int("tool_runtime.tool_capability_probe_max_models", 3)
        if probe_limit <= 0:
            probe_limit = 3
        probes = 0
        for model_name in unknown:
            if probes >= probe_limit:
                break
            probes += 1
            if await self._probe_native_tool_capability(model_name):
                known_capable.append(model_name)

        if known_capable:
            return known_capable
        return []

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
            or "status code: 404" in normalized
            or "model '" in normalized and "not found" in normalized
        )

    def _candidate_model_names_for_pseudo_tooling(self, routing: dict[str, Any]) -> list[str]:
        available = _as_string_list(routing.get("available_models"))
        available_set = set(available)
        attempted = _as_string_list(routing.get("attempted_models"))
        candidates: list[str] = []
        prefer_available = _runtime_bool("tool_runtime.prefer_available_model_inventory", True)

        def add(name: Any) -> None:
            model_name = _as_text(name)
            if not model_name:
                return
            lowered = model_name.lower()
            if "embed" in lowered:
                return
            if model_name in candidates:
                return
            if prefer_available and available_set and model_name not in available_set:
                return
            candidates.append(model_name)

        add(routing.get("selected_model"))
        add(routing.get("requested_model"))
        for model_name in attempted:
            add(model_name)
        for model_name in available:
            add(model_name)
        add(_runtime_value("inference.default_chat_model", ""))
        add(_runtime_value("inference.default_tool_model", ""))
        add(_runtime_value("inference.default_multimodal_model", ""))
        for model_name in self._allowed_models:
            add(model_name)

        limit = _runtime_int("tool_runtime.pseudo_tool_candidate_limit", 6)
        if limit <= 0:
            limit = 6
        return candidates[:limit]

    def _analysis_tool_hints(self) -> list[str]:
        hints: list[str] = []
        for hint in _as_string_list(self._analysisPayload.get("tool_hints")):
            if hint in self._toolSpecs and hint not in hints:
                hints.append(hint)

        knowledge_state = _coerce_dict(self._analysisPayload.get("knowledge_state"))
        knowledge_class = _normalize_knowledge_classification(
            knowledge_state.get("classification"),
            "known",
        )
        discovery_required = _as_bool(
            knowledge_state.get("discovery_required"),
            fallback=knowledge_class in {"probably_unknown", "unknown"},
        )
        if discovery_required:
            default_hints = _as_string_list(
                _runtime_value(
                    "orchestrator.discovery_default_tool_hints",
                    ["braveSearch", "curlRequest"],
                )
            )
            for hint in default_hints:
                if hint in self._toolSpecs and hint not in hints and hint != "skipTools":
                    hints.append(hint)

        process_directive = self._analysis_process_directive()
        for hint in self._process_action_hints(_as_text(process_directive.get("action"), "none")):
            if hint in self._toolSpecs and hint not in hints:
                hints.append(hint)
        return hints

    @staticmethod
    def _process_action_hints(action: str) -> list[str]:
        mapping = {
            "list_users": ["knownUsersList"],
            "send_message": ["knownUsersList", "messageKnownUser"],
            "start_process": ["upsertProcessWorkspace"],
            "resume_process": ["listProcessWorkspace"],
            "update_process_step": ["listProcessWorkspace", "updateProcessWorkspaceStep"],
            "list_outbox": ["listOutboxMessages"],
        }
        return list(mapping.get(_as_text(action).lower(), []))

    def _analysis_process_directive(self) -> dict[str, Any]:
        directive = _coerce_dict(self._analysisPayload.get("process_directive"))
        action_raw = _as_text(directive.get("action"), "none").lower()
        allowed_actions = {
            "none",
            "list_users",
            "send_message",
            "start_process",
            "resume_process",
            "update_process_step",
            "list_outbox",
        }
        action = action_raw if action_raw in allowed_actions else "none"
        process_id = _safe_int(directive.get("process_id"), 0)
        if process_id <= 0:
            process_id = 0
        target_member_id = _safe_int(directive.get("target_member_id"), 0)
        if target_member_id <= 0:
            target_member_id = 0
        workspace_context = _coerce_dict(self._toolRuntimeContext.get("workspace_context"))
        active_processes_raw = workspace_context.get("active_processes")
        active_processes = active_processes_raw if isinstance(active_processes_raw, list) else []
        if process_id <= 0 and action in {"resume_process", "update_process_step"} and active_processes:
            process_id = _safe_int(_coerce_dict(active_processes[0]).get("process_id"), 0)
        step_status = _as_text(directive.get("step_status"), "completed").lower()
        if step_status not in {"pending", "in_progress", "blocked", "completed", "skipped", "cancelled"}:
            step_status = "completed"
        return {
            "action": action,
            "process_id": process_id if process_id > 0 else None,
            "process_label": _as_text(directive.get("process_label")),
            "process_description": _as_text(directive.get("process_description")),
            "step_label": _as_text(directive.get("step_label")),
            "step_status": step_status,
            "step_details": _as_text(directive.get("step_details")),
            "target_username": _as_text(directive.get("target_username")).lstrip("@"),
            "target_member_id": target_member_id if target_member_id > 0 else None,
            "message_text": _as_text(directive.get("message_text")),
        }

    def _analysis_requires_tools(self) -> bool:
        process_directive = self._analysis_process_directive()
        if _as_text(process_directive.get("action"), "none") != "none":
            return True
        knowledge_state = _coerce_dict(self._analysisPayload.get("knowledge_state"))
        knowledge_class = _normalize_knowledge_classification(
            knowledge_state.get("classification"),
            "known",
        )
        if _as_bool(
            knowledge_state.get("discovery_required"),
            fallback=knowledge_class in {"probably_unknown", "unknown"},
        ):
            return True
        needs_tools = _as_bool(self._analysisPayload.get("needs_tools"), False)
        if not needs_tools:
            return False
        hints = self._analysis_tool_hints()
        if not hints:
            return True
        return any(hint != "skipTools" for hint in hints)

    def _hint_driven_tool_calls(self) -> list[dict[str, Any]]:
        hints = self._analysis_tool_hints()
        query_text = self._latestUserMessage or _as_text(self._toolRuntimeContext.get("latest_user_message"))
        query_text = query_text.strip()
        first_url = _extract_first_url(query_text)
        process_directive = self._analysis_process_directive()
        process_action = _as_text(process_directive.get("action"), "none")
        calls: list[dict[str, Any]] = []

        def add_call(tool_name: str, arguments: dict[str, Any] | None = None) -> None:
            if tool_name not in self._toolSpecs:
                return
            args = arguments if isinstance(arguments, dict) else {}
            calls.append({"function": {"name": tool_name, "arguments": args}})

        if process_action != "none":
            process_id = process_directive.get("process_id")
            process_label = _as_text(process_directive.get("process_label"))
            process_description = _as_text(process_directive.get("process_description"))
            step_label = _as_text(process_directive.get("step_label"))
            step_status = _as_text(process_directive.get("step_status"), "completed")
            step_details = _as_text(process_directive.get("step_details"))
            target_username = _as_text(process_directive.get("target_username"))
            target_member_id = process_directive.get("target_member_id")
            message_text = _as_text(process_directive.get("message_text"), query_text)

            if process_action == "list_users":
                args: dict[str, Any] = {}
                if target_username:
                    args["queryString"] = target_username
                elif query_text:
                    args["queryString"] = query_text
                args["count"] = max(1, _runtime_int("orchestrator.process_user_lookup_count", 8))
                add_call("knownUsersList", args)

            elif process_action == "send_message":
                if target_username:
                    add_call(
                        "knownUsersList",
                        {
                            "queryString": target_username,
                            "count": max(1, _runtime_int("orchestrator.process_user_lookup_count", 8)),
                        },
                    )
                elif target_member_id:
                    add_call(
                        "knownUsersList",
                        {
                            "queryString": str(target_member_id),
                            "count": max(1, _runtime_int("orchestrator.process_user_lookup_count", 8)),
                        },
                    )
                if message_text and (target_username or target_member_id):
                    args: dict[str, Any] = {
                        "messageText": message_text,
                        "sendNow": True,
                    }
                    if target_username:
                        args["targetUsername"] = target_username
                    if target_member_id:
                        args["targetMemberID"] = target_member_id
                    if process_id:
                        args["processID"] = process_id
                    add_call("messageKnownUser", args)

            elif process_action == "start_process":
                if not process_label:
                    topic = _as_text(self._analysisPayload.get("topic"), "workflow")
                    process_label = f"{topic} workflow"
                add_call(
                    "upsertProcessWorkspace",
                    {
                        "processLabel": process_label,
                        "processDescription": process_description or _as_text(self._analysisPayload.get("intent")),
                        "processStatus": "active",
                        "replaceSteps": True,
                    },
                )

            elif process_action == "resume_process":
                args: dict[str, Any] = {
                    "includeSteps": True,
                    "count": max(1, _runtime_int("orchestrator.process_resume_count", 5)),
                    "processStatus": "active",
                }
                if process_id:
                    args["processID"] = process_id
                add_call("listProcessWorkspace", args)

            elif process_action == "update_process_step":
                list_args: dict[str, Any] = {
                    "includeSteps": True,
                    "count": max(1, _runtime_int("orchestrator.process_resume_count", 5)),
                    "processStatus": "active",
                }
                if process_id:
                    list_args["processID"] = process_id
                add_call("listProcessWorkspace", list_args)
                if process_id:
                    update_args: dict[str, Any] = {
                        "processID": process_id,
                        "stepStatus": step_status if step_status else "completed",
                    }
                    if step_label:
                        update_args["stepLabel"] = step_label
                    if step_details:
                        update_args["stepDetails"] = step_details
                    add_call("updateProcessWorkspaceStep", update_args)

            elif process_action == "list_outbox":
                add_call(
                    "listOutboxMessages",
                    {"count": max(1, _runtime_int("orchestrator.process_outbox_count", 10))},
                )

            if calls:
                max_calls = max(1, _runtime_int("tool_runtime.process_hint_max_calls", 3))
                return calls[:max_calls]

        if not hints and query_text:
            if first_url and "curlRequest" in self._toolSpecs:
                return [{"function": {"name": "curlRequest", "arguments": {"url": first_url}}}]
            if "braveSearch" in self._toolSpecs:
                return [{"function": {"name": "braveSearch", "arguments": {"queryString": query_text}}}]
        for hint in hints:
            if hint == "skipTools":
                continue
            if hint not in self._toolSpecs:
                continue
            args: dict[str, Any] = {}
            if hint in {"braveSearch", "chatHistorySearch", "knowledgeSearch"} and query_text:
                args["queryString"] = query_text
            elif hint == "curlRequest":
                if first_url:
                    args["url"] = first_url
                elif "braveSearch" in self._toolSpecs and query_text:
                    fallback_call = {"function": {"name": "braveSearch", "arguments": {"queryString": query_text}}}
                    if fallback_call not in calls:
                        calls.append(fallback_call)
                    continue
                else:
                    continue
            elif hint == "knownUsersList":
                if query_text:
                    args["queryString"] = query_text
                args["count"] = max(1, _runtime_int("orchestrator.process_user_lookup_count", 8))
            elif hint == "listProcessWorkspace":
                args["includeSteps"] = True
                args["count"] = max(1, _runtime_int("orchestrator.process_resume_count", 5))
                args["processStatus"] = "active"
            elif hint == "listOutboxMessages":
                args["count"] = max(1, _runtime_int("orchestrator.process_outbox_count", 10))
            elif hint in {"messageKnownUser", "upsertProcessWorkspace", "updateProcessWorkspaceStep"}:
                # These hints require additional structured arguments from process_directive.
                # Avoid executing malformed calls in generic hint fallback mode.
                continue
            candidate_call = {"function": {"name": hint, "arguments": args}}
            if candidate_call not in calls:
                calls.append(candidate_call)
            if len(calls) >= 3:
                break
        if not calls and "skipTools" in hints and "skipTools" in self._toolSpecs:
            calls.append({"function": {"name": "skipTools", "arguments": {}}})
        return calls

    @staticmethod
    def _first_url_from_brave_results(tool_results: list[dict[str, Any]] | None) -> str:
        results = tool_results if isinstance(tool_results, list) else []
        for item in results:
            result = _coerce_dict(item)
            if _as_text(result.get("tool_name")) != "braveSearch":
                continue
            if _as_text(result.get("status"), "error") != "success":
                continue
            payload = result.get("tool_results")
            rows = payload if isinstance(payload, list) else []
            for row in rows:
                url_value = _as_text(_coerce_dict(row).get("url"))
                if url_value:
                    return url_value
        return ""

    async def _auto_expand_search_then_fetch(
        self,
        *,
        tool_results: list[dict[str, Any]],
        requested_tools: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if "curlRequest" not in self._toolSpecs:
            return [], []
        hints = self._analysis_tool_hints()
        if "curlRequest" not in hints:
            return [], []

        requested_names = [_as_text(_coerce_dict(item).get("name")) for item in requested_tools]
        executed_names = [_as_text(_coerce_dict(item).get("tool_name")) for item in (tool_results or [])]
        if "curlRequest" in requested_names or "curlRequest" in executed_names:
            return [], []
        if "braveSearch" not in requested_names and "braveSearch" not in executed_names:
            return [], []

        discovered_url = self._first_url_from_brave_results(tool_results)
        if not discovered_url:
            return [], []

        await self._emit_tool_runtime_event(
            {
                "stage": "tools.expand.fetch",
                "status": "info",
                "detail": "Expanding from search to URL fetch using curlRequest.",
                "meta": {"url": discovered_url},
            }
        )
        expansion_calls = [{"function": {"name": "curlRequest", "arguments": {"url": discovered_url}}}]
        expansion_results = await self._execute_tool_calls(expansion_calls)
        return expansion_calls, expansion_results

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
        enforce_native_capability = _runtime_bool("tool_runtime.enforce_native_tool_capability", True)
        native_allowed_models = list(self._allowed_models)
        if enforce_native_capability:
            native_allowed_models = await self._resolve_native_tool_candidates(native_allowed_models)
        requested_tool_model = self._model if self._model in native_allowed_models else (native_allowed_models[0] if native_allowed_models else None)

        if enforce_native_capability and not native_allowed_models:
            self._routing = {
                "capability": "tool",
                "requested_model": self._model,
                "selected_model": None,
                "allowed_models": list(self._allowed_models),
                "candidate_models": [],
                "errors": ["No native tool-capable model candidates available."],
            }
            pseudo_calls, pseudo_text, pseudo_routing = await self._pseudo_tool_fallback(
                base_routing=self._routing,
                trigger_error=RuntimeError("no_tool_capable_models_available"),
            )
            if pseudo_calls is not None:
                self._response = SimpleNamespace(
                    message=SimpleNamespace(content=pseudo_text, tool_calls=pseudo_calls)
                )
                if pseudo_routing:
                    self._routing = pseudo_routing
                used_pseudo_tool_fallback = True
                self._executionSummary["model_fallback"] = {
                    "trigger": "no_tool_capable_models",
                    "mode": "pseudo_structured_output",
                    "selected_model": _coerce_dict(self._routing).get("selected_model"),
                    "success": True,
                }
            else:
                self._executionSummary["model_fallback"] = {
                    "trigger": "no_tool_capable_models",
                    "mode": "pseudo_structured_output",
                    "selected_model": None,
                    "success": False,
                }
                return list()

        try:
            if not used_pseudo_tool_fallback:
                self._response, self._routing = await self._modelRouter.chat_with_fallback(
                    capability="tool",
                    requested_model=requested_tool_model,
                    allowed_models=native_allowed_models or self._allowed_models,
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
                attempted_for_capability = _as_string_list(initialRouting.get("attempted_models"))
                attempted_for_capability.extend(
                    [
                        _as_text(initialRouting.get("selected_model")),
                        _as_text(initialRouting.get("requested_model")),
                    ]
                )
                for attempted_model in _dedupe_models(attempted_for_capability):
                    self._remember_tool_capability(
                        attempted_model,
                        capable=False,
                        reason="runtime_tool_capability_error",
                    )
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
        modelOutputText = _as_text(getattr(responseMessage, "content", ""))
        selected_model_name = _as_text(_coerce_dict(self._routing).get("selected_model"))

        if selected_model_name and rawToolCalls:
            self._remember_tool_capability(
                selected_model_name,
                capable=True,
                reason="native_tool_calls_present",
            )
        elif (
            selected_model_name
            and not used_pseudo_tool_fallback
            and self._analysis_requires_tools()
            and not rawToolCalls
        ):
            self._remember_tool_capability(
                selected_model_name,
                capable=False,
                reason="missing_tool_calls_when_tools_required",
            )

        if not rawToolCalls and not used_pseudo_tool_fallback and self._analysis_requires_tools():
            logger.warning(
                "Tool stage returned no tool_calls despite analysis.needs_tools=true; "
                "attempting pseudo fallback."
            )
            pseudo_calls, pseudo_text, pseudo_routing = await self._pseudo_tool_fallback(
                base_routing=self._routing,
                trigger_error=RuntimeError("missing_tool_calls_with_analysis_needs_tools"),
            )
            if pseudo_calls is not None:
                rawToolCalls = list(pseudo_calls)
                if pseudo_text:
                    modelOutputText = _as_text(pseudo_text)
                if pseudo_routing:
                    self._routing = pseudo_routing
                used_pseudo_tool_fallback = True
                self._executionSummary["model_fallback"] = {
                    "trigger": "missing_tool_calls",
                    "mode": "pseudo_structured_output",
                    "selected_model": _coerce_dict(self._routing).get("selected_model"),
                    "success": bool(rawToolCalls),
                }

        if not rawToolCalls and self._analysis_requires_tools():
            hint_calls = self._hint_driven_tool_calls()
            if hint_calls:
                rawToolCalls = hint_calls
                used_pseudo_tool_fallback = True
                self._executionSummary["model_fallback"] = {
                    "trigger": "missing_tool_calls",
                    "mode": "analysis_hint_fallback",
                    "selected_model": _coerce_dict(self._routing).get("selected_model"),
                    "success": True,
                }
                await self._emit_tool_runtime_event(
                    {
                        "stage": "tools.fallback.hints",
                        "status": "warning",
                        "detail": "Synthesizing tool calls from analysis hints after empty native tool output.",
                        "meta": {
                            "hints": self._analysis_tool_hints(),
                            "tool_calls": len(rawToolCalls),
                        },
                    }
                )

        requestedTools: list[dict[str, Any]] = self._collect_requested_tools(rawToolCalls)
        if rawToolCalls:
            toolResults = await self._execute_tool_calls(rawToolCalls)

        expansionCalls, expansionResults = await self._auto_expand_search_then_fetch(
            tool_results=toolResults,
            requested_tools=requestedTools,
        )
        if expansionCalls:
            requestedTools.extend(self._collect_requested_tools(expansionCalls))
        if expansionResults:
            toolResults.extend(expansionResults)
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
            "auto_expanded_search_fetch": bool(expansionCalls),
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
        self._model = _select_initial_model_from_allowed(self._allowed_models, "analysis")
        if self._model and self._model not in self._allowed_models:
            self._allowed_models = [self._model, *self._allowed_models]

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
        self._model = _select_initial_model_from_allowed(self._allowed_models, "dev_test")
        if self._model and self._model not in self._allowed_models:
            self._allowed_models = [self._model, *self._allowed_models]

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
        self._model = _select_initial_model_from_allowed(self._allowed_models, "chat")
        if self._model and self._model not in self._allowed_models:
            self._allowed_models = [self._model, *self._allowed_models]
        optionMap = options if isinstance(options, dict) else {}
        self._ingressImages = [item for item in _as_string_list(optionMap.get("ingress_images")) if item]
        self._imageContext = _coerce_dict(optionMap.get("image_context"))
        self._useMultimodal = bool(self._ingressImages)
        if self._useMultimodal:
            multimodalCandidates: list[str] = []
            configuredMultimodal = _coerce_dict(getattr(config._instance, "inference", {}).get("multimodal")).get("model")

            def _add_candidate(model_name: Any) -> None:
                cleaned = _normalize_model_name(model_name)
                if not cleaned:
                    return
                if cleaned in multimodalCandidates:
                    return
                multimodalCandidates.append(cleaned)

            _add_candidate(_runtime_value("inference.default_multimodal_model", ""))
            _add_candidate(configuredMultimodal)
            for model_name in self._allowed_models:
                _add_candidate(model_name)
            if multimodalCandidates:
                self._allowed_models = multimodalCandidates
                self._model = multimodalCandidates[0]

        # Check for options passed and if policy allows for those options
        if optionMap:
            modelRequested = optionMap.get("model_requested")
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
            filtered_messages: list[Any] = []
            for histMessage in messages:
                if isinstance(histMessage, dict):
                    roleName = _as_text(histMessage.get("role")).lower()
                else:
                    roleName = _as_text(getattr(histMessage, "role", "")).lower()
                if roleName == "system":
                    continue
                filtered_messages.append(histMessage)
            self._messages += filtered_messages

    def _messages_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in self._messages:
            if isinstance(message, dict):
                role = _as_text(message.get("role"))
                content = _as_text(message.get("content"))
                if not role:
                    continue
                entry: dict[str, Any] = {"role": role, "content": content}
                if "images" in message and isinstance(message.get("images"), list):
                    entry["images"] = [item for item in message.get("images", []) if _as_text(item)]
                payload.append(entry)
                continue
            role = _as_text(getattr(message, "role", ""))
            if not role:
                continue
            payload.append(
                {
                    "role": role,
                    "content": _as_text(getattr(message, "content", "")),
                }
            )

        if self._useMultimodal:
            attached = False
            for index in range(len(payload) - 1, -1, -1):
                if _as_text(payload[index].get("role")).lower() != "user":
                    continue
                enriched = dict(payload[index])
                enriched["images"] = list(self._ingressImages)
                payload[index] = enriched
                attached = True
                break
            if not attached:
                fallbackContent = _as_text(self._imageContext.get("caption"), "Please describe this image.")
                payload.append(
                    {
                        "role": "user",
                        "content": fallbackContent,
                        "images": list(self._ingressImages),
                    }
                )
        return payload
        
    async def generateResponse(self):
        logger.info(f"Generate a response for the chat conversation agent.")
        capability = "multimodal" if self._useMultimodal else "chat"
        try:
            self._response, self._routing = await self._modelRouter.chat_with_fallback(
                capability=capability,
                requested_model=self._model,
                allowed_models=self._allowed_models,
                messages=self._messages_payload(),
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
    normalizedOverride = str(endpointOverride or "").strip().rstrip("/")
    cacheKey = normalizedOverride or "__default__"
    cacheEnabled = _control_plane_cache_enabled()
    cacheTtlSeconds = _control_plane_cache_ttl_seconds()
    if cacheEnabled and cacheTtlSeconds > 0:
        cached = _CONTROL_PLANE_MANAGER_CACHE.get(cacheKey)
        if _cache_valid(cached, cacheTtlSeconds):
            manager = cached.get("manager")
            if isinstance(manager, PolicyManager):
                return manager

    inference = config._instance.inference if hasattr(config, "_instance") else {}
    manager = PolicyManager(
        inference_config=inference,
        endpoint_override=normalizedOverride or None,
    )
    if cacheEnabled and cacheTtlSeconds > 0:
        _CONTROL_PLANE_MANAGER_CACHE[cacheKey] = {
            "cached_at": time.monotonic(),
            "manager": manager,
        }
    return manager


_POLICY_TO_INFERENCE_KEY = {
    "message_analysis": "chat",
    "tool_calling": "tool",
    "chat_conversation": "chat",
    "dev_test": "chat",
}


def _runtime_model_for_capability(capability: str) -> str:
    inferenceConfig = config._instance.inference if hasattr(config, "_instance") else {}
    configKey = {
        "tool": "default_tool_model",
        "chat": "default_chat_model",
        "analysis": "default_chat_model",
        "dev_test": "default_chat_model",
        "embedding": "default_embedding_model",
        "generate": "default_generate_model",
        "multimodal": "default_multimodal_model",
    }.get(capability, "default_chat_model")
    if isinstance(inferenceConfig, dict):
        configuredValue = _normalize_model_name(inferenceConfig.get(configKey))
        if configuredValue:
            return configuredValue

    runtimeKey = {
        "tool": "inference.default_tool_model",
        "chat": "inference.default_chat_model",
        "analysis": "inference.default_chat_model",
        "dev_test": "inference.default_chat_model",
        "embedding": "inference.default_embedding_model",
        "generate": "inference.default_generate_model",
        "multimodal": "inference.default_multimodal_model",
    }.get(capability, "inference.default_chat_model")
    return _normalize_model_name(_runtime_value(runtimeKey, ""))


def _select_initial_model_from_allowed(allowed_models: list[str], capability: str) -> str:
    cleaned = _dedupe_models(list(allowed_models or []))
    preferred = _runtime_model_for_capability(capability)
    if preferred and preferred in cleaned:
        return preferred
    if cleaned:
        return cleaned[0]
    return preferred


def _preferred_runtime_model_for_policy(policyName: str) -> str:
    inferenceKey = _POLICY_TO_INFERENCE_KEY.get(policyName, "chat")
    inferenceConfig = config._instance.inference if hasattr(config, "_instance") else {}
    section = inferenceConfig.get(inferenceKey) if isinstance(inferenceConfig, dict) else None
    if isinstance(section, dict):
        configuredModel = section.get("model")
        if isinstance(configuredModel, str) and configuredModel.strip():
            return configuredModel.strip()

    return _runtime_model_for_capability(inferenceKey)


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

    if policyName == "tool_calling":
        runtimeToolDefault = _normalize_model_name(_runtime_value("inference.default_tool_model", ""))
        runtimeToolCapable = _runtime_model_list("inference.tool_capable_models", [])
        runtimeChatDefault = _normalize_model_name(_runtime_value("inference.default_chat_model", ""))
        toolOrdered = _dedupe_models(
            [runtimeToolDefault, *runtimeToolCapable, *models, runtimeChatDefault]
        )
        models = [name for name in toolOrdered if not _looks_like_embedding_model(name)]

    if models:
        return models

    fallbackPolicy = _policyManager().default_policy(policyName)
    fallbackModels = fallbackPolicy.get("allowed_models", [])
    if isinstance(fallbackModels, list):
        for modelName in fallbackModels:
            if isinstance(modelName, str) and modelName.strip():
                models.append(modelName.strip())

    if policyName == "tool_calling":
        runtimeToolDefault = _normalize_model_name(_runtime_value("inference.default_tool_model", ""))
        runtimeToolCapable = _runtime_model_list("inference.tool_capable_models", [])
        runtimeChatDefault = _normalize_model_name(_runtime_value("inference.default_chat_model", ""))
        toolOrdered = _dedupe_models(
            [runtimeToolDefault, *runtimeToolCapable, *models, runtimeChatDefault]
        )
        models = [name for name in toolOrdered if not _looks_like_embedding_model(name)]

    if models:
        return models

    if policyName == "tool_calling":
        toolFallback = _normalize_model_name(_runtime_value("inference.default_tool_model", ""))
        if toolFallback:
            return [toolFallback]
    chatFallback = _normalize_model_name(_runtime_value("inference.default_chat_model", ""))
    if chatFallback:
        return [chatFallback]
    preferredFallback = _preferred_runtime_model_for_policy(policyName)
    if preferredFallback:
        return [preferredFallback]

    manager = _policyManager()
    discovered_models, _ = manager.discover_models(manager.resolve_host(policyName))
    discovered = _dedupe_models(discovered_models)
    if policyName == "tool_calling":
        discovered = [name for name in discovered if not _looks_like_embedding_model(name)]
    if discovered:
        return [discovered[0]]
    return []


def _prune_unavailable_policy_models(
    *,
    policy_name: str,
    policy: dict[str, Any],
    available_models: list[str] | None,
) -> dict[str, Any]:
    if not _runtime_bool("orchestrator.prune_unavailable_policy_models", True):
        return policy
    if not isinstance(policy, dict):
        return {}
    available = _dedupe_models(available_models or [])
    if not available:
        return policy
    allowed = _as_string_list(policy.get("allowed_models"))
    if not allowed:
        return policy

    filtered = [model_name for model_name in allowed if model_name in available]
    if not filtered:
        return policy
    if filtered == allowed:
        return policy

    patched = copy.deepcopy(policy)
    patched["allowed_models"] = filtered
    logger.info(
        "Pruned unavailable policy models [%s]: removed=%s",
        policy_name,
        [name for name in allowed if name not in filtered],
    )
    return patched


def loadAgentPolicy(policyName: str, endpointOverride: str | None = None) -> dict:
    normalizedOverride = str(endpointOverride or "").strip().rstrip("/")
    cacheKey = f"{policyName}|{normalizedOverride or '__default__'}"
    cacheEnabled = _control_plane_cache_enabled()
    cacheTtlSeconds = _control_plane_cache_ttl_seconds()
    if cacheEnabled and cacheTtlSeconds > 0:
        cached = _CONTROL_PLANE_POLICY_CACHE.get(cacheKey)
        if _cache_valid(cached, cacheTtlSeconds):
            payload = cached.get("policy")
            if isinstance(payload, dict):
                return copy.deepcopy(payload)

    logger.info(f"Loading agent policy for: {policyName}")
    manager = _policyManager(endpointOverride=normalizedOverride or None)
    report = manager.validate_policy(policy_name=policyName, strict_model_check=False)

    for warning in report.warnings:
        logger.warning(f"Policy validation warning [{policyName}]: {warning}")
    for error in report.errors:
        logger.error(f"Policy validation error [{policyName}]: {error}")

    if report.errors:
        resolved = manager.default_policy(policyName)
    elif isinstance(report.normalized_policy, dict):
        resolved = report.normalized_policy
    else:
        resolved = manager.default_policy(policyName)

    resolved = _prune_unavailable_policy_models(
        policy_name=policyName,
        policy=resolved,
        available_models=report.available_models,
    )

    if cacheEnabled and cacheTtlSeconds > 0:
        _CONTROL_PLANE_POLICY_CACHE[cacheKey] = {
            "cached_at": time.monotonic(),
            "policy": copy.deepcopy(resolved),
        }
    return copy.deepcopy(resolved)


def loadAgentSystemPrompt(policyName: str) -> str:
    cacheEnabled = _control_plane_cache_enabled()
    cacheTtlSeconds = _control_plane_cache_ttl_seconds()
    if cacheEnabled and cacheTtlSeconds > 0:
        cached = _CONTROL_PLANE_PROMPT_CACHE.get(policyName)
        if _cache_valid(cached, cacheTtlSeconds):
            payload = cached.get("prompt")
            if isinstance(payload, str):
                return payload

    logger.info(f"Loading agent system prompt for: {policyName}")
    manager = _policyManager()
    try:
        prompt = manager.load_system_prompt(policy_name=policyName, strict=True)
    except PolicyValidationError as error:
        logger.error(f"System prompt load failed [{policyName}]: {error}")
        prompt = manager.load_system_prompt(policy_name=policyName, strict=False)

    if cacheEnabled and cacheTtlSeconds > 0:
        _CONTROL_PLANE_PROMPT_CACHE[policyName] = {
            "cached_at": time.monotonic(),
            "prompt": str(prompt),
        }
    return str(prompt)


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
