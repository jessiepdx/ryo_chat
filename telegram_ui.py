##########################################################################
#                                                                        #
#  This file (telegram_ui.py) contains the telegram user interface for   #
#  project ryo (run your own) chat                                       #
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

import base64
import asyncio
from datetime import datetime, timedelta, timezone
import html
import json
import logging
import time
from typing import Any
from ollama import AsyncClient
import os
import re
from telegram import (
    ChatMember,
    ChatMemberUpdated,
    constants, 
    ForceReply,
    helpers,
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    MessageEntity,
    ReplyKeyboardMarkup, 
    ReplyKeyboardRemove, 
    Update,
    WebAppInfo,
    ReactionType
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    MessageReactionHandler,
    filters
)
from hypermindlabs.utils import (
    ChatHistoryManager, 
    CommunityManager, 
    CommunityScoreManager, 
    ConfigManager, 
    CustomFormatter,
    KnowledgeManager, 
    MemberManager, 
    ProposalManager,
    SpamManager,
    UsageManager
)
from hypermindlabs.agents import (
    ConversationOrchestrator,
    ConversationalAgent,
    ImageAgent,
    TweetAgent,
    loadAgentSystemPrompt,
)



###########
# LOGGING #
###########

# Clear any previous logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set the basic config to append logging data to a file
logPath = "logs/"
logFilename = "telegram_log_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
#print(logPath + logFilename)
logging.basicConfig(
    filename=logPath+logFilename,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)", 
    level=logging.DEBUG
)

# Create a stream handler for cli output
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(CustomFormatter())
# add the handler to the root logger
logging.getLogger().addHandler(console)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hypermindlabs.utils").setLevel(logging.INFO)
logging.getLogger("telegram").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Set timezone for time
timezone(-timedelta(hours=7), "Pacific")



###########
# GLOBALS #
###########

chatHistory = ChatHistoryManager()
communities = CommunityManager()
communityScore = CommunityScoreManager()
config = ConfigManager()
knowledge = KnowledgeManager()
members = MemberManager()
proposals = ProposalManager()
spam = SpamManager()
usage = UsageManager()

logger.info(f"Database route status: {config.databaseRoute}")

_TELEGRAM_MESSAGE_GUARD_LOCK = asyncio.Lock()
_TELEGRAM_INFLIGHT_MESSAGE_KEYS: set[str] = set()
_TELEGRAM_COMPLETED_MESSAGE_KEYS: dict[str, float] = {}
_TELEGRAM_INSPECTOR_LOCK = asyncio.Lock()
_TELEGRAM_RUN_INSPECTOR_CACHE: dict[str, dict[str, Any]] = {}
_TELEGRAM_RUN_INSPECTOR_TTL_SECONDS = 6 * 60 * 60
_TELEGRAM_RUN_INSPECTOR_MAX_RECORDS = 512


def _runtime_int(path: str, default: int) -> int:
    return config.runtimeInt(path, default)


def _runtime_float(path: str, default: float) -> float:
    return config.runtimeFloat(path, default)


def _runtime_bool(path: str, default: bool) -> bool:
    return config.runtimeBool(path, default)


def _runtime_str(path: str, default: str) -> str:
    value = config.runtimeValue(path, default)
    text = default if value is None else str(value).strip()
    return text if text else default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_telegram_timestamp(value: datetime | None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return _utc_now()


def _message_temporal_context(message) -> dict[str, datetime]:
    sent_at = _coerce_telegram_timestamp(getattr(message, "date", None))
    received_at = _utc_now()
    return {
        "message_timestamp": sent_at,
        "message_received_timestamp": received_at,
    }


def _default_generate_system_prompt() -> str:
    try:
        prompt = str(loadAgentSystemPrompt("chat_conversation") or "").strip()
        if prompt:
            return prompt
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Falling back to default generate prompt after policy load failure: {error}")

    fallback = config.defaults.get("system_prompt")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return "You are a helpful AI assistant."


def _message_word_threshold() -> int:
    return _runtime_int("conversation.community_score_message_word_threshold", 20)


def _new_member_grace_delta() -> timedelta:
    return timedelta(seconds=_runtime_int("conversation.new_member_grace_period_seconds", 60))


def _password_min_length() -> int:
    return max(8, _runtime_int("security.password_min_length", 12))


def _database_unavailable() -> bool:
    route = config.databaseRoute if isinstance(config.databaseRoute, dict) else {}
    return str(route.get("status", "")).strip().lower() == "failed_all"


def _database_unavailable_message() -> str:
    return (
        "Account services are temporarily unavailable because database "
        "connection/authentication failed. Run setup to fix PostgreSQL "
        "credentials, then send /start again."
    )


def _message_guard_key(chat_id: int, message_id: int) -> str:
    return f"{int(chat_id)}:{int(message_id)}"


async def _claim_message_processing(chat_id: int, message_id: int) -> bool:
    key = _message_guard_key(chat_id, message_id)
    now = time.monotonic()
    ttl_seconds = max(30, _runtime_int("telegram.message_guard_completed_ttl_seconds", 180))
    async with _TELEGRAM_MESSAGE_GUARD_LOCK:
        stale_keys = [
            stale_key
            for stale_key, completed_at in _TELEGRAM_COMPLETED_MESSAGE_KEYS.items()
            if (now - completed_at) > float(ttl_seconds)
        ]
        for stale_key in stale_keys:
            _TELEGRAM_COMPLETED_MESSAGE_KEYS.pop(stale_key, None)

        if key in _TELEGRAM_INFLIGHT_MESSAGE_KEYS:
            return False
        if key in _TELEGRAM_COMPLETED_MESSAGE_KEYS:
            return False

        _TELEGRAM_INFLIGHT_MESSAGE_KEYS.add(key)
        return True


async def _release_message_processing(chat_id: int, message_id: int) -> None:
    key = _message_guard_key(chat_id, message_id)
    async with _TELEGRAM_MESSAGE_GUARD_LOCK:
        _TELEGRAM_INFLIGHT_MESSAGE_KEYS.discard(key)
        _TELEGRAM_COMPLETED_MESSAGE_KEYS[key] = time.monotonic()


def _guard_message_handler(handler):
    async def _wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        chat = update.effective_chat
        message_id = getattr(message, "message_id", None)
        chat_id = getattr(chat, "id", None)
        if message_id is None or chat_id is None:
            return await handler(update, context)

        claimed = await _claim_message_processing(chat_id=int(chat_id), message_id=int(message_id))
        if not claimed:
            logger.info(
                "Skipping duplicate/replayed telegram update (chat_id=%s, message_id=%s).",
                chat_id,
                message_id,
            )
            return

        try:
            return await handler(update, context)
        finally:
            await _release_message_processing(chat_id=int(chat_id), message_id=int(message_id))

    _wrapped.__name__ = f"{getattr(handler, '__name__', 'handler')}_guarded"
    return _wrapped


_ORCHESTRATION_STAGE_LABELS = {
    "orchestrator.start": "Accepted request",
    "orchestrator.fast_path": "Fast path",
    "context.built": "Context built",
    "analysis.start": "Analyzing message",
    "analysis.progress": "Analyzing message",
    "analysis.complete": "Analysis complete",
    "analysis.payload": "Analysis payload",
    "tools.start": "Evaluating tools",
    "tools.model_output": "Tool model output",
    "tools.complete": "Tools complete",
    "response.start": "Generating response",
    "response.progress": "Generating response",
    "response.complete": "Response generated",
    "response.fallback": "Fallback response",
    "response.sanitized": "Sanitized response",
    "orchestrator.complete": "Wrapping up",
}

_TELEGRAM_STAGE_DETAIL_LEVELS = {"minimal", "normal", "debug"}
_TELEGRAM_MAX_MESSAGE_CHARS = 4096
_TELEGRAM_FALLBACK_FINAL_REPLY = "I could not generate a complete reply this turn. Please try again."
_MINIMAL_VISIBLE_STAGES = {
    "orchestrator.start",
    "orchestrator.fast_path",
    "analysis.start",
    "analysis.progress",
    "analysis.complete",
    "tools.start",
    "tools.model_output",
    "tools.complete",
    "response.start",
    "response.progress",
    "response.complete",
}
_NORMAL_HIDDEN_STAGES = {
    "analysis.payload",
}


def _inspector_json_safe(value: Any) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)
    try:
        return json.loads(serialized)
    except Exception:  # noqa: BLE001
        return serialized


def _inspector_trim_text(text: str, max_chars: int = 3900) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    suffix = "\n\n... (truncated)"
    keep = max(1, int(max_chars) - len(suffix))
    return cleaned[:keep].rstrip() + suffix


def _inspector_json_block(payload: Any, max_chars: int = 2800) -> str:
    text = json.dumps(_inspector_json_safe(payload), ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        text = _inspector_trim_text(text, max_chars=max_chars)
    return text


def _stage_event_snapshot(event: dict[str, Any] | None) -> dict[str, Any]:
    payload = event if isinstance(event, dict) else {}
    snapshot = {
        "stage": str(payload.get("stage") or ""),
        "detail": str(payload.get("detail") or ""),
        "timestamp": str(payload.get("timestamp") or _utc_now().isoformat().replace("+00:00", "Z")),
        "meta": _inspector_json_safe(payload.get("meta") if isinstance(payload.get("meta"), dict) else {}),
    }
    return snapshot


def _append_stage_event(stage_events: list[dict[str, Any]], event: dict[str, Any] | None) -> None:
    stage_events.append(_stage_event_snapshot(event))
    if len(stage_events) > 200:
        del stage_events[: len(stage_events) - 200]


def _tool_entries_from_summary(run_summary: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run_summary if isinstance(run_summary, dict) else {}
    tool_summary = summary.get("tool_summary")
    tool_summary = tool_summary if isinstance(tool_summary, dict) else {}
    requested = tool_summary.get("requested_tools")
    requested_tools = requested if isinstance(requested, list) else []
    results = summary.get("tool_results")
    tool_results = results if isinstance(results, list) else []
    max_len = max(len(requested_tools), len(tool_results))
    entries: list[dict[str, Any]] = []
    for idx in range(max_len):
        request_item = requested_tools[idx] if idx < len(requested_tools) and isinstance(requested_tools[idx], dict) else {}
        result_item = tool_results[idx] if idx < len(tool_results) and isinstance(tool_results[idx], dict) else {}
        tool_name = str(request_item.get("name") or result_item.get("tool_name") or f"tool_{idx + 1}")
        entries.append(
            {
                "tool_name": tool_name,
                "arguments": _inspector_json_safe(request_item.get("arguments", {})),
                "status": str(result_item.get("status") or "unknown"),
                "tool_results": _inspector_json_safe(result_item.get("tool_results")),
                "error": _inspector_json_safe(result_item.get("error")),
            }
        )
    return entries


def _render_tool_page_text(
    *,
    tool_entry: dict[str, Any],
    tool_index: int,
    total_tools: int,
) -> str:
    title = f"Tool Call {tool_index + 1}/{max(1, total_tools)}"
    lines = [
        title,
        f"Tool: {tool_entry.get('tool_name', 'unknown')}",
        f"Status: {tool_entry.get('status', 'unknown')}",
        "",
        "Arguments:",
        _inspector_json_block(tool_entry.get("arguments", {}), max_chars=1200),
    ]
    error_payload = tool_entry.get("error")
    if isinstance(error_payload, dict) and error_payload:
        lines.extend(["", "Error:", _inspector_json_block(error_payload, max_chars=1000)])
    else:
        lines.extend(
            [
                "",
                "Response:",
                _inspector_json_block(tool_entry.get("tool_results"), max_chars=1600),
            ]
        )
    return _inspector_trim_text("\n".join(lines), max_chars=_TELEGRAM_MAX_MESSAGE_CHARS - 32)


def _render_stage_page_text(
    *,
    stage_entry: dict[str, Any],
    stage_index: int,
    total_stages: int,
) -> str:
    stage_key = str(stage_entry.get("stage") or "")
    stage_label = _ORCHESTRATION_STAGE_LABELS.get(stage_key, stage_key or "stage")
    stage_detail = str(stage_entry.get("detail") or "")
    timestamp = str(stage_entry.get("timestamp") or "")
    lines = [
        f"Context Stage {stage_index + 1}/{max(1, total_stages)}",
        f"Stage: {stage_label}",
        f"Key: {stage_key}",
        f"Time: {timestamp}",
    ]
    if stage_detail:
        lines.extend(["", "Detail:", stage_detail])

    meta_payload = stage_entry.get("meta")
    if isinstance(meta_payload, dict) and meta_payload:
        lines.extend(["", "Meta:", _inspector_json_block(meta_payload, max_chars=2200)])
    return _inspector_trim_text("\n".join(lines), max_chars=_TELEGRAM_MAX_MESSAGE_CHARS - 32)


async def _cleanup_run_inspector_cache() -> None:
    now = time.time()
    stale_keys = [
        key
        for key, payload in _TELEGRAM_RUN_INSPECTOR_CACHE.items()
        if now - float(payload.get("created_at", now)) > _TELEGRAM_RUN_INSPECTOR_TTL_SECONDS
    ]
    for key in stale_keys:
        _TELEGRAM_RUN_INSPECTOR_CACHE.pop(key, None)
    if len(_TELEGRAM_RUN_INSPECTOR_CACHE) > _TELEGRAM_RUN_INSPECTOR_MAX_RECORDS:
        overflow = len(_TELEGRAM_RUN_INSPECTOR_CACHE) - _TELEGRAM_RUN_INSPECTOR_MAX_RECORDS
        ordered_keys = sorted(
            _TELEGRAM_RUN_INSPECTOR_CACHE.keys(),
            key=lambda item: float(_TELEGRAM_RUN_INSPECTOR_CACHE[item].get("created_at", 0.0)),
        )
        for key in ordered_keys[:overflow]:
            _TELEGRAM_RUN_INSPECTOR_CACHE.pop(key, None)


def _build_run_inspector_keyboard(token: str, payload: dict[str, Any]) -> InlineKeyboardMarkup | None:
    tool_entries = payload.get("tool_entries")
    tools = tool_entries if isinstance(tool_entries, list) else []
    rows: list[list[InlineKeyboardButton]] = []
    current_row: list[InlineKeyboardButton] = []
    for idx, entry in enumerate(tools):
        tool_name = str((entry if isinstance(entry, dict) else {}).get("tool_name") or f"tool_{idx + 1}")
        label = f"Tool {idx + 1}: {tool_name}"
        label = label if len(label) <= 28 else f"Tool {idx + 1}: {tool_name[:24]}..."
        current_row.append(InlineKeyboardButton(label, callback_data=f"diag:{token}:tool:{idx}"))
        if len(current_row) == 2:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)
    rows.append([InlineKeyboardButton("Context", callback_data=f"diag:{token}:ctx:0")])
    if not rows:
        return None
    return InlineKeyboardMarkup(rows)


def _build_tool_view_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Back", callback_data=f"diag:{token}:back"),
                InlineKeyboardButton("Context", callback_data=f"diag:{token}:ctx:0"),
            ]
        ]
    )


def _build_context_view_keyboard(token: str, stage_index: int, total_stages: int) -> InlineKeyboardMarkup:
    left_index = stage_index - 1
    right_index = stage_index + 1
    left_data = f"diag:{token}:ctx:{left_index}" if left_index >= 0 else f"diag:{token}:noop"
    right_data = (
        f"diag:{token}:ctx:{right_index}"
        if right_index < max(0, total_stages)
        else f"diag:{token}:noop"
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("◀", callback_data=left_data),
                InlineKeyboardButton("▶", callback_data=right_data),
            ],
            [InlineKeyboardButton("Back", callback_data=f"diag:{token}:back")],
        ]
    )


async def _register_run_inspector_payload(
    *,
    chat_id: int,
    message_id: int,
    final_text: str,
    run_summary: dict[str, Any] | None,
    stage_events: list[dict[str, Any]] | None,
) -> str | None:
    summary = run_summary if isinstance(run_summary, dict) else {}
    events = stage_events if isinstance(stage_events, list) else []
    tool_entries = _tool_entries_from_summary(summary)
    stage_entries = [entry for entry in events if isinstance(entry, dict)]
    if not tool_entries and not stage_entries:
        return None

    token = os.urandom(6).hex()
    async with _TELEGRAM_INSPECTOR_LOCK:
        await _cleanup_run_inspector_cache()
        _TELEGRAM_RUN_INSPECTOR_CACHE[token] = {
            "created_at": time.time(),
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "final_text": str(final_text or ""),
            "tool_entries": tool_entries,
            "stage_entries": stage_entries,
        }
    return token


async def _lookup_run_inspector_payload(token: str) -> dict[str, Any] | None:
    async with _TELEGRAM_INSPECTOR_LOCK:
        await _cleanup_run_inspector_cache()
        payload = _TELEGRAM_RUN_INSPECTOR_CACHE.get(token)
        if isinstance(payload, dict):
            return payload
    return None


async def _attach_run_inspector_controls(
    *,
    response_message,
    final_text: str,
    run_summary: dict[str, Any] | None,
    stage_events: list[dict[str, Any]] | None,
) -> None:
    if response_message is None:
        return
    chat = getattr(response_message, "chat", None)
    chat_id = getattr(chat, "id", None)
    message_id = getattr(response_message, "message_id", None)
    if chat_id is None or message_id is None:
        return

    token = await _register_run_inspector_payload(
        chat_id=int(chat_id),
        message_id=int(message_id),
        final_text=final_text,
        run_summary=run_summary,
        stage_events=stage_events,
    )
    if not token:
        return

    keyboard = _build_run_inspector_keyboard(token, await _lookup_run_inspector_payload(token) or {})
    if keyboard is None:
        return
    try:
        await response_message.edit_reply_markup(reply_markup=keyboard)
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Unable to attach inspector controls to telegram message {message_id}: {error}")


class TelegramStageStatus:
    def __init__(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        chat_id: int,
        thread_id: int | None = None,
        reply_message=None,
        enabled: bool = True,
        show_json_details: bool = True,
        detail_level: str = "minimal",
        history_limit: int = 8,
        json_char_limit: int = 1400,
        message_char_limit: int = 3800,
        update_min_interval_seconds: float = 1.0,
    ):
        self._context = context
        self._chat_id = chat_id
        self._thread_id = thread_id
        self._reply_message = reply_message
        self._enabled = bool(enabled)
        self._show_json_details = bool(show_json_details)
        detail_level_clean = str(detail_level or "minimal").strip().lower()
        if detail_level_clean not in _TELEGRAM_STAGE_DETAIL_LEVELS:
            detail_level_clean = "minimal"
        self._detail_level = detail_level_clean
        self._history_limit = max(3, int(history_limit))
        self._json_char_limit = max(200, int(json_char_limit))
        self._message_char_limit = max(800, int(message_char_limit))
        self._update_min_interval_seconds = max(0.1, float(update_min_interval_seconds))
        self._message = None
        self._lines: list[str] = []
        self._last_text: str = ""
        self._last_edit_monotonic = 0.0
        self._edit_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._flush_pending = False
        self._flush_shutdown = False

    def _stage_visible(self, stage: str, meta: dict[str, Any]) -> bool:
        if self._detail_level == "debug":
            return True
        if self._detail_level == "normal":
            return stage not in _NORMAL_HIDDEN_STAGES
        if stage not in _MINIMAL_VISIBLE_STAGES:
            return False
        return True

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        text = str(value or "").strip()
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _looks_like_structured_fragment(text: str) -> bool:
        candidate = str(text or "").strip()
        if not candidate:
            return False
        if candidate[0] in {"{", "["}:
            return True
        if "```" in candidate:
            return True
        if ":" in candidate and "{" in candidate:
            return True
        return False

    @staticmethod
    def _extract_from_meta_json(meta: dict[str, Any], *path: str) -> Any:
        cursor: Any = meta.get("json")
        for key in path:
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(key)
        return cursor

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_stage_timestamp(self, raw: Any) -> str:
        raw_text = str(raw or "").strip()
        if raw_text:
            try:
                parsed = datetime.fromisoformat(raw_text.replace("Z", "+00:00"))
                return parsed.astimezone(timezone.utc).strftime("%H:%M:%S UTC")
            except ValueError:
                return self._truncate_text(raw_text, 32)
        return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def _extract_stats_payload(self, meta: dict[str, Any]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_per_second",
            "completion_tokens_per_second",
            "total_tokens_per_second",
        ):
            if key in meta:
                stats[key] = meta.get(key)
        nested_stats = self._extract_from_meta_json(meta, "stats")
        if isinstance(nested_stats, dict):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_tokens_per_second",
                "completion_tokens_per_second",
                "total_tokens_per_second",
            ):
                if key not in stats and key in nested_stats:
                    stats[key] = nested_stats.get(key)
        return stats

    def _stage_meta_summary_line(self, stage: str, meta: dict[str, Any]) -> str:
        if not isinstance(meta, dict):
            return ""

        pieces: list[str] = []
        selected_model = str(meta.get("selected_model") or meta.get("model") or "").strip()
        if not selected_model:
            routing_selected = self._extract_from_meta_json(meta, "routing", "selected_model")
            if isinstance(routing_selected, str) and routing_selected.strip():
                selected_model = routing_selected.strip()
        if selected_model:
            pieces.append(f"model={selected_model}")

        requested = meta.get("requested_tool_calls")
        executed = meta.get("executed_tool_calls")
        tool_calls = meta.get("tool_calls")
        if isinstance(requested, int) or isinstance(executed, int):
            req_text = str(requested) if isinstance(requested, int) else "-"
            exe_text = str(executed) if isinstance(executed, int) else "-"
            pieces.append(f"tools={req_text}->{exe_text}")
        elif isinstance(tool_calls, int):
            pieces.append(f"tools={tool_calls}")

        execution_mode = self._extract_from_meta_json(meta, "summary", "execution_mode")
        if isinstance(execution_mode, str) and execution_mode.strip():
            pieces.append(f"mode={execution_mode.strip()}")

        stats = self._extract_stats_payload(meta)
        total_tps = self._coerce_float(stats.get("total_tokens_per_second"))
        completion_tps = self._coerce_float(stats.get("completion_tokens_per_second"))
        if total_tps and total_tps > 0:
            pieces.append(f"tps={total_tps:.2f}")
        elif completion_tps and completion_tps > 0:
            pieces.append(f"tps={completion_tps:.2f}")

        if self._detail_level == "debug":
            prompt_tokens = stats.get("prompt_tokens")
            completion_tokens = stats.get("completion_tokens")
            total_tokens = stats.get("total_tokens")
            if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
                pieces.append(
                    f"tok={prompt_tokens if isinstance(prompt_tokens, int) else '-'}+"
                    f"{completion_tokens if isinstance(completion_tokens, int) else '-'}"
                )
            elif isinstance(total_tokens, int):
                pieces.append(f"tok={total_tokens}")

        if stage.endswith(".progress"):
            chunks = meta.get("chunks")
            chars = meta.get("chars")
            if isinstance(chunks, int) and chunks >= 0:
                pieces.append(f"chunks={chunks}")
            if isinstance(chars, int) and chars >= 0:
                pieces.append(f"chars={chars}")

        # Keep summaries short in stage stream updates.
        summary = " | ".join(piece for piece in pieces if piece)
        return self._truncate_text(summary, 160)

    def _human_readable_stage_message(self, stage: str, detail: str, meta: dict[str, Any]) -> str:
        snippet = str(meta.get("snippet") or "").strip()
        snippet_structured = self._looks_like_structured_fragment(snippet)
        if snippet and not snippet_structured:
            return self._truncate_text(snippet, 320)

        if stage == "orchestrator.start":
            return "Accepted request and preparing context."

        if stage == "analysis.start":
            return "Running message analysis and model routing."

        if stage == "analysis.progress":
            chunks = meta.get("chunks")
            chars = meta.get("chars")
            if isinstance(chunks, int) and isinstance(chars, int):
                return f"Reasoning over intent/topic and routing model ({chunks} chunks, {chars} chars)."
            return "Reasoning over intent/topic and routing model."

        if stage == "tools.model_output":
            excerpt = self._extract_from_meta_json(meta, "execution_summary", "model_output_excerpt")
            if isinstance(excerpt, str) and excerpt.strip():
                return self._truncate_text(excerpt, 320)

        if stage == "tools.start":
            return "Determining whether tools are needed for this request."

        if stage == "tools.complete":
            model_error = self._extract_from_meta_json(meta, "summary", "model_error")
            if isinstance(model_error, dict):
                error_message = str(model_error.get("message") or "").strip()
                if error_message:
                    return self._truncate_text(f"Tool stage degraded: {error_message}", 320)
            executed = meta.get("executed_tool_calls")
            if isinstance(executed, int):
                tool_names = self._extract_from_meta_json(meta, "summary", "executed_tools")
                if isinstance(tool_names, list) and tool_names:
                    names = ", ".join(str(name) for name in tool_names[:4])
                    return self._truncate_text(f"Executed {executed} tool call(s): {names}", 320)
                return f"Executed {executed} tool call(s)."
            return "No tool calls were executed for this request."

        if stage == "analysis.complete":
            selected_model = str(meta.get("selected_model") or "").strip()
            stats = self._extract_stats_payload(meta)
            tps = self._coerce_float(stats.get("completion_tokens_per_second"))
            completion_tokens = stats.get("completion_tokens")
            if isinstance(completion_tokens, int) and tps and tps > 0:
                return f"Analysis complete ({completion_tokens} tokens @ {tps:.2f} tok/s)."
            if selected_model:
                return f"Analysis model: {selected_model}"
            return "Analysis complete."

        if stage == "response.start":
            return "Generating final response."

        if stage == "response.progress":
            chars = meta.get("chars")
            if isinstance(chars, int):
                return f"Drafting response ({chars} chars generated)."

        if stage == "response.complete":
            stats = self._extract_stats_payload(meta)
            tps = self._coerce_float(stats.get("completion_tokens_per_second"))
            completion_tokens = stats.get("completion_tokens")
            if isinstance(completion_tokens, int) and tps and tps > 0:
                return f"Final response generated ({completion_tokens} tokens @ {tps:.2f} tok/s)."
            return "Final response generated."

        cleaned_detail = str(detail or "").strip()
        if cleaned_detail:
            return self._truncate_text(cleaned_detail, 320)
        return ""

    def _build_stage_entry(
        self,
        *,
        stage: str,
        detail: str,
        meta: dict[str, Any],
        timestamp: Any,
    ) -> str:
        label = _ORCHESTRATION_STAGE_LABELS.get(stage, stage if stage else "Processing")

        message = self._human_readable_stage_message(stage, detail, meta)
        summary = self._stage_meta_summary_line(stage, meta)
        entry = f"<b>{html.escape(label)}</b>"
        if message:
            entry += f"\n<i>{html.escape(message)}</i>"
        if summary and self._detail_level in {"normal", "debug"}:
            entry += f"\n<code>{html.escape(summary)}</code>"
        entry += f"\n<code>{html.escape(self._format_stage_timestamp(timestamp))}</code>"
        return self._truncate_text(entry, self._message_char_limit)

    def _render(self) -> str:
        if not self._lines:
            return "Processing your request..."
        if self._detail_level == "minimal":
            return self._lines[-1]

        lines = self._lines[-self._history_limit :]
        rendered = "\n\n".join(lines)
        while len(lines) > 1 and len(rendered) > self._message_char_limit:
            lines = lines[1:]
            rendered = "\n\n".join(lines)
        if len(rendered) <= self._message_char_limit:
            return rendered
        return lines[-1]

    def _render_json_block(self, payload, *, max_chars: int | None = None) -> str:
        if max_chars is None:
            max_chars = self._json_char_limit
        try:
            json_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            json_text = json.dumps(str(payload), ensure_ascii=False)
        if len(json_text) > max_chars:
            suffix = "\n... (truncated)"
            keep = max(1, int(max_chars) - len(suffix))
            json_text = json_text[:keep].rstrip() + suffix
        return f"<pre>{html.escape(json_text)}</pre>"

    def _normalize_final_reply_text(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            normalized = _TELEGRAM_FALLBACK_FINAL_REPLY
        if len(normalized) > _TELEGRAM_MAX_MESSAGE_CHARS:
            suffix = "\n\n[truncated]"
            keep = max(1, _TELEGRAM_MAX_MESSAGE_CHARS - len(suffix))
            normalized = normalized[:keep].rstrip() + suffix
        return normalized

    async def _safe_edit(self, text: str, *, parse_html: bool = False) -> bool:
        if self._message is None:
            return False
        if text == self._last_text:
            return True
        if str(text or "").strip() == "":
            logger.warning("Failed to update stage status message: Message text is empty")
            return False
        async with self._edit_lock:
            try:
                kwargs: dict[str, Any] = {}
                if parse_html:
                    kwargs["parse_mode"] = constants.ParseMode.HTML
                    kwargs["disable_web_page_preview"] = True
                await self._message.edit_text(text=text, **kwargs)
                self._last_text = text
                self._last_edit_monotonic = time.monotonic()
                return True
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Failed to update stage status message: {error}")
                return False

    async def _flush_stage_loop(self) -> None:
        while not self._flush_shutdown:
            if not self._flush_pending:
                break
            self._flush_pending = False
            elapsed = time.monotonic() - self._last_edit_monotonic
            if elapsed < self._update_min_interval_seconds:
                await asyncio.sleep(self._update_min_interval_seconds - elapsed)
            await self._safe_edit(self._render(), parse_html=True)

    def _schedule_stage_flush(self) -> None:
        if self._flush_shutdown:
            return
        self._flush_pending = True
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = asyncio.create_task(self._flush_stage_loop())

    async def _stop_stage_flush(self) -> None:
        self._flush_shutdown = True
        task = self._flush_task
        self._flush_task = None
        if task is None:
            return
        if task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def start(self) -> None:
        if not self._enabled:
            return
        initial_line = self._build_stage_entry(
            stage="orchestrator.start",
            detail="Accepted request.",
            meta={},
            timestamp=_utc_now().isoformat().replace("+00:00", "Z"),
        )
        self._lines = [initial_line]
        text = self._render()
        try:
            if self._reply_message is not None:
                self._message = await self._reply_message.reply_text(
                    text=text,
                    parse_mode=constants.ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            else:
                self._message = await self._context.bot.send_message(
                    chat_id=self._chat_id,
                    message_thread_id=self._thread_id,
                    text=text,
                    parse_mode=constants.ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            self._last_text = text
            self._last_edit_monotonic = time.monotonic()
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to send stage status message: {error}")
            self._message = None

    async def emit(self, event: dict | None = None) -> None:
        if not self._enabled or self._message is None:
            return
        payload = event if isinstance(event, dict) else {}
        stage = str(payload.get("stage") or "").strip()
        detail = str(payload.get("detail") or "").strip()
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        timestamp = payload.get("timestamp")
        if stage and not self._stage_visible(stage, meta):
            return

        entry = self._build_stage_entry(stage=stage, detail=detail, meta=meta, timestamp=timestamp)
        if self._show_json_details and self._detail_level == "debug" and "json" in meta:
            json_payload = meta.get("json")
            json_block = self._render_json_block(json_payload)
            candidate = f"{entry}\n{json_block}"
            if len(candidate) > self._message_char_limit:
                available = max(180, self._message_char_limit - len(entry) - 24)
                json_block = self._render_json_block(json_payload, max_chars=available)
                candidate = f"{entry}\n{json_block}"
            if len(candidate) > self._message_char_limit:
                candidate = f"{entry}\n<code>json payload omitted (too large)</code>"
            entry = candidate

        if self._lines and self._lines[-1] == entry:
            return
        self._lines.append(entry)
        if len(self._lines) > max(self._history_limit * 4, 24):
            self._lines = self._lines[-max(self._history_limit * 4, 24) :]
        self._schedule_stage_flush()

    async def finalize(self, final_text: str):
        await self._stop_stage_flush()
        final_text_safe = self._normalize_final_reply_text(final_text)
        if self._message is not None:
            edited = await self._safe_edit(final_text_safe, parse_html=False)
            if edited:
                return self._message

        try:
            if self._reply_message is not None:
                return await self._reply_message.reply_text(text=final_text_safe)
            return await self._context.bot.send_message(
                chat_id=self._chat_id,
                message_thread_id=self._thread_id,
                text=final_text_safe,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to send final response message: {error}")
            if self._message is not None:
                # Return existing status message as best effort fallback.
                return self._message
            raise

    async def fail(self, error_text: str) -> None:
        await self._stop_stage_flush()
        if self._message is not None:
            await self._safe_edit(error_text)
            return
        try:
            if self._reply_message is not None:
                await self._reply_message.reply_text(text=error_text)
            else:
                await self._context.bot.send_message(
                    chat_id=self._chat_id,
                    message_thread_id=self._thread_id,
                    text=error_text,
                )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Unable to send failure status message: {error}")



###########################
# DEFINE COMMAND HANDLERS #
###########################

# Define the startBot function to handle "/start commands for PRIVATE chats"
async def startBot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the start command. Registers this private chat with the chatbot. 
    This is required for the bot to be able to send DMs"""
    
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Start command issued by {user.name} (user_id: {user.id}).")

    # Check if the user is already registered
    member = members.getMemberByTelegramID(user.id)

    try:
        if member is None:
            logger.info(f"New user {user.name} (user_id: {user.id}) being registered with the chatbot.")
            # There is no account for this user, begin registration process
            newAccount = {
                "username": user.username,
                "user_id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": None,
                "roles": ["user"],
                "register_date": datetime.now()
            }
            createdMemberID = members.addMemberFromTelegram(newAccount)
            if createdMemberID is None:
                logger.error(
                    "Unable to register %s (user_id: %s) because account persistence failed.",
                    user.name,
                    user.id,
                )
                await message.reply_text(_database_unavailable_message())
                return

            # Re-read the account to ensure registration committed.
            member = members.getMemberByTelegramID(user.id)
            if member is None:
                logger.error(
                    "Registration for %s (user_id: %s) did not persist after create.",
                    user.name,
                    user.id,
                )
                await message.reply_text(_database_unavailable_message())
                return

        
            # TODO get official community chat links from config and insert into the welcome message
            minimumCommunityScore = _runtime_int("telegram.minimum_community_score_private_chat", 50)
            welcomeMessage = f"""Welcome {user.name}, I am the {config.botName} chatbot. 
You will need to have a minimum community score of {minimumCommunityScore} to chat with me in private. 
Engage with the community in one of our group chats to increase your community score. 
            
Use the /help command for more information."""
            
            await message.reply_text(
                text=welcomeMessage
            )
        
        else:
            # A user with this id is already registered
            logger.info(f"User {user.name} (user_id: {user.id}) is already registered.")
            await message.reply_text(f"Welcome back {user.name}, you are already registered with the {config.botName} chatbot.")
        
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")


# Launch the Miniapp Dashboard.
async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with a button that opens a the mini app."""
    
    user = update.effective_user
    if config.webUIUrl is None:
        # TODO message the user / owner account that the config for the dashboard is missing
        #print("missing webUI URL from config")
        return
    
    logger.info(f"Dashboard command issued by {user.name} (user_id: {user.id}).")
    #print(config.webUIUrl + "miniapp/dashboard")
    
    keyboard = [
        [
            InlineKeyboardButton("OPEN", web_app=WebAppInfo(url=config.webUIUrl + "miniapp/dashboard"))
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message with text and appended InlineKeyboard
    await update.message.reply_text(
        text="Click OPEN to open the dashboard.",
        reply_markup=reply_markup
    )


# Display a help menu for the user
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help menu displays the available commands"""

    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Help command issued by {user.name} (user_id: {user.id}).")

    helpMsg = """The following commands are available:

/botid to display the chatbot Telegram ID
/userid to display your Telegram user ID
"""

    if chat.type == "private":
        member = members.getMemberByTelegramID(user.id)

        if member is not None:
            helpMsg = helpMsg + """/info to display your user account info
-------------------------------
Send a message to begin chatting with the chatbot."""
        elif _database_unavailable():
            helpMsg = helpMsg + _database_unavailable_message() + "\n"
        else:
            helpMsg = helpMsg + "Use the /start command to get started.\n"
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)

        if community is not None:
            helpMsg = helpMsg + f"""/info to display group account info 
-------------------------------
Tag @{config.botName} in your message to get a response from the chatbot. The chatbot will also response if you reply to it's message.
"""
        else:
            helpMsg = helpMsg + "The chatbot needs to be added to a group by an owner or admin to register it.\n"
    else:
        # Channels are not supported
        return
    
    try:
        await message.reply_text(text=helpMsg)
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")


# Display user and chat group data
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display account information."""
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    
    logger.info(f"Info command issued by {user.name} (user_id: {user.id}).")

    if chat.type == "private":
        member = members.getMemberByTelegramID(user.id)
        # IMPORTANT info command replies in markdown, make sure to escape markdown characters in all dynamic values
        
        if member is not None:
            infoMsg = f"""*Member ID:*  {member.get("member_id")}
*Telegram User ID:*  {user.id}
*Username:*  {'' if user.username is None else helpers.escape_markdown(user.username)}
*Name:*  {helpers.escape_markdown(member.get('first_name'))} {'' if member.get('last_name') is None else helpers.escape_markdown(member.get('last_name'))}
*Email:*  {'' if member.get('email') is None else helpers.escape_markdown(member.get('email'))}
*Roles:*  {", ".join(member.get('roles'))}
*Created:*  {member.get('register_date')}
*Community Score:*  {member.get('community_score')}"""
        else:
            infoMsg = "User is not registered."
    
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)
        
        # temporary code to leave group chat if the group isn't registered. This is mainly used for development
        if community is None:
            try:
                # Leave the group chat
                await context.bot.leave_chat(chat.id)
            except Exception as err:
                logger.error(f"Exception while leaving the group chat:\n{err}")
            finally:
                # Exit the function
                return
        
        if community is not None:
            infoMsg = f"""*Community ID:*  {community.get("community_id")}
*Telegram Group chat ID:*  {chat.id}
*Chat type:*  {chat.type}
*Community name:*  {helpers.escape_markdown(community.get("community_name"))}
*Roles:*  {'None' if community.get('roles') is None else ', '.join(community.get('roles'))}
*Created:*  {community.get('register_date')}"""     
   
    try:
        await message.reply_markdown(
            text=infoMsg
        )
    except Exception as err:
            logger.error(f"Exception while replying to a telegram message\n{err}")


async def botID(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display this bot's Telegram identifier."""
    message = update.effective_message
    user = update.effective_user
    username = "unknown" if user is None else user.name
    userID = "unknown" if user is None else user.id

    logger.info(f"BotID command issued by {username} (user_id: {userID}).")

    try:
        bot = await context.bot.get_me()
        botUsername = f"@{bot.username}" if bot.username else "not set"
        await message.reply_text(
            text=f"Bot ID: {bot.id}\nBot Username: {botUsername}"
        )
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")


async def userID(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the caller's Telegram user identifier."""
    message = update.effective_message
    user = update.effective_user
    username = "unknown" if user is None else user.name
    userID = "unknown" if user is None else user.id

    logger.info(f"UserID command issued by {username} (user_id: {userID}).")

    if user is None:
        try:
            await message.reply_text(text="Unable to resolve a user id for this update.")
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        return

    try:
        await message.reply_text(text=f"User ID: {user.id}")
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")



########################
# BEGIN COMMAND CHAINS #
########################

# Cancel command is used to cancel most of the command chains
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the command chain."""
    message = update.effective_message
    user = update.message.from_user

    logger.info(f"User {user.name} (user_id: {user.id}) canceled the command chain.")

    try:
        await message.reply_text(
            text="Command chain canceled.", 
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as err:
        logger.warning(f"Exception while replying to a telegram message:\n{err}")
    finally:
        return ConversationHandler.END



#####################################
# Begin the /generate command chain #
#-----------------------------------#

# Define command chain states
SET_SYSTEM_PROMPT, SET_PROMPT = range(2)

# This is the entry point for the /generate command
async def beginGenerate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Generate command chain initiated by {user.name} (user_id: {user.id}).")

    member = members.getMemberByTelegramID(user.id)
    if member is None:
        logger.info(f"An unregistered user attempted to use the /generate command.")
        try:
            await message.reply_text(
                text=f"Only members are able to use the /generate command. To register your telegram account with the {config.botName} chatbot, open a private chat with @{config.botName} and send the /start command."
            )
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        finally:
            return ConversationHandler.END
    else:
        logger.info(f"User {user.name} (user_id: {user.id}) is approved to use the /generate command.")
        
        # Clear any previous temporary generate data
        context.chat_data["generate_data"] = {}
        try:
            await message.reply_text(
                text="Enter your system prompt or /skip to use the default system prompt. /cancel to cancel this command", 
                reply_markup=ForceReply(selective=True)
            )

            return SET_SYSTEM_PROMPT
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
            context.chat_data["generate_data"] = None

            return ConversationHandler.END


async def setSystemPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Store the provided system prompt.")
    message = update.effective_message
    
    # Store the system prompt in telegram storage
    gd = context.chat_data.get("generate_data")
    gd["system_prompt"] = message.text

    try:
        # Get the prompt
        await message.reply_text(
            "Enter your prompt", 
            reply_markup=ForceReply(selective=True)
        )
        
        return SET_PROMPT
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}.")
        # Clear the generate data from telegram storage
        gd = None

        return ConversationHandler.END


async def skip_systemPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Set system prompt to default value.")
    message = update.effective_message
    
    # Store the system prompt
    gd = context.chat_data.get("generate_data")
    gd["system_prompt"] = ""

    try:
        # Get the prompt
        await message.reply_text(
            "Enter your prompt", 
            reply_markup=ForceReply(selective=True)
        )
        
        return SET_PROMPT
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}.")
        gd = None

        return ConversationHandler.END


async def setPrompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Prompt received, generate a response.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None

    gd = context.chat_data.get("generate_data")
    userSystemPrompt = str(gd.get("system_prompt") or "").strip()
    baseSystemPrompt = _default_generate_system_prompt()
    if userSystemPrompt:
        systemPromptText = f"{baseSystemPrompt}\n\nAdditional user instruction:\n{userSystemPrompt}"
    else:
        systemPromptText = baseSystemPrompt

    inferenceGenerate = config.inference.get("generate", {}) if isinstance(config.inference, dict) else {}
    inferenceChat = config.inference.get("chat", {}) if isinstance(config.inference, dict) else {}
    generateHost = str(inferenceGenerate.get("url") or inferenceChat.get("url") or "").strip()
    generateModel = str(inferenceGenerate.get("model") or inferenceChat.get("model") or "").strip()
    if not generateHost or not generateModel:
        logger.error("Generate command cannot resolve model host/model from config.")
        context.chat_data["generate_data"] = None
        return ConversationHandler.END

    generateClient = AsyncClient(host=generateHost)
    statusMessage = None
    responseParts: list[str] = []
    lastEditMonotonic = 0.0
    editIntervalSeconds = 0.75

    try:
        statusMessage = await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text="Generating response...",
        )
    except Exception as err:
        logger.warning(f"Unable to send generate status message:\n{err}")
        statusMessage = None

    try:
        stream = await generateClient.chat(
            model=generateModel,
            stream=True,
            messages=[
                {"role": "system", "content": systemPromptText},
                {"role": "user", "content": message.text},
            ],
        )
        chunk: Any
        async for chunk in stream:
            chunkText = str(getattr(getattr(chunk, "message", None), "content", "") or "")
            if not chunkText:
                continue
            responseParts.append(chunkText)
            if statusMessage is None:
                continue
            now = time.monotonic()
            if (now - lastEditMonotonic) < editIntervalSeconds:
                continue
            previewText = "".join(responseParts).strip()
            if not previewText:
                continue
            previewText = TelegramStageStatus._truncate_text(previewText, 3800)
            try:
                await statusMessage.edit_text(text=previewText)
                lastEditMonotonic = now
            except Exception as err:
                logger.debug(f"Generate stream status update failed:\n{err}")
    except Exception as err:
        logger.error(f"Exception while generating a response from Ollama chat stream:\n{err}")
        context.chat_data["generate_data"] = None
        return ConversationHandler.END
    finally:
        # Delete the generate data from telegram bot storage
        context.chat_data["generate_data"] = None

    responseText = "".join(responseParts).strip()
    if not responseText:
        responseText = _TELEGRAM_FALLBACK_FINAL_REPLY
    if len(responseText) > _TELEGRAM_MAX_MESSAGE_CHARS:
        suffix = "\n\n[truncated]"
        keep = max(1, _TELEGRAM_MAX_MESSAGE_CHARS - len(suffix))
        responseText = responseText[:keep].rstrip() + suffix

    if statusMessage is not None:
        try:
            await statusMessage.delete()
        except Exception:
            pass

    try:
        await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            reply_markup=ReplyKeyboardRemove(),
            text=responseText,
        )
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
    finally:
        gd = None
        return ConversationHandler.END



#####################
# Knowledge Manager #
#-------------------#

# Define command chain states
HANDLE_KNOWLEDGE_TYPE, HANDLE_KNOWLEDGE_TEXT, HANDLE_KNOWLEDGE_SOURCE, HANDLE_KNOWLEDGE_CATEGORY, STORE_KNOWLEDGE = range(5)

# This is the entry point for the /knowledge command chain
async def knowledgeManger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user

    logger.info(f"Knowledge command chain initiated by {user.name} (user_id: {user.id}), begin knowledge manager.")

    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"User is not registered with the chatbot.")
        return ConversationHandler.END
    
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        logger.info(f"Member is authorized to use knowledge command.")
        # Clear any previous data for new knowledge from the telegram bot storage
        context.chat_data["new_knowledge"] = dict()

        keyboard = [
            [
                InlineKeyboardButton("Public", callback_data="public"),
                InlineKeyboardButton("Private", callback_data="private")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            # Send message with text and appended InlineKeyboard
            await message.reply_text(
                text="Let's add some knowledge to our database. First I need to know, is this public or private (confidential) information?\n\nType /cancel to cancel.",
                reply_markup=reply_markup
            )

            return HANDLE_KNOWLEDGE_TYPE
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
            context.chat_data["new_knowledge"] = None

            return ConversationHandler.END
        
    # The follwoing code only runs if the user is not authorized above
    logger.warning(f"User {user.name} (user_id: {user.id}) is not authorized to use knowledge command.")
    try:
        await message.reply_text(text="Only admins are authorized to use the knowledge command.")
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
    finally:
        return ConversationHandler.END


async def setKnowledgeType(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Knowledge visibity received. Get knowledge text.")

    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return
    
    nk = context.chat_data.get("new_knowledge")
    nk["visibility"] = query.data

    try:
        await query.edit_message_text(
            text=f"Knowledge data will be {query.data}. Enter knowledge data text:"
        )

        return HANDLE_KNOWLEDGE_TEXT
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
        nk = None

        return ConversationHandler.END


async def setKnowledgeText(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge text received., get source.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["text"] = message.text

    try:
        await message.reply_text(
            text=f"Ok, heres your document:\n\n{message.text}\n\nDo you want to add any sources?\n/skip this step"
        )

        return HANDLE_KNOWLEDGE_SOURCE
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}.")
        nk = None

        return ConversationHandler.END


async def setKnowledgeSource(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Knowledge source sent, get category tag.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["source"] = message.text

    try:
        await message.reply_text(
            text=f"Ok, heres your source:\n\n{message.text}\n\nDo you want to add a category tag? \n/skip this step"
        )

        return HANDLE_KNOWLEDGE_CATEGORY
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")
        nk = None

        return ConversationHandler.END


async def skip_knowledgeSource(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Skip knowledge source.")
    message = update.effective_message
    user = update.effective_user

    nk = context.chat_data.get("new_knowledge")
    nk["source"] = user.name

    try:
        await message.reply_text(
            text=f"Ok, we will just use your username or full name as the source.\n\nDo you want to add any category tags? Use a comma to separate categroies\n/skip this step"
        )

        return HANDLE_KNOWLEDGE_CATEGORY
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")
        nk = None

        return ConversationHandler.END


async def setKnowledgeCategories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Knowledge category tag received. Call final method to save.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["categories"] = message.text.split(",")

    keyboard = [
        [
            InlineKeyboardButton("Yes", callback_data="yes"),
            InlineKeyboardButton("No", callback_data="no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    responseText = f"""Let's do a final review.

{nk['text']}

Source:  {nk['source']}

Categories:  {message.text}

Do you wish to save?
"""

    try:
        await message.reply_text(
            text=responseText,
            reply_markup=reply_markup
        )

        return STORE_KNOWLEDGE
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}.")
        nk = None
        
        return ConversationHandler.END


async def skip_knowledgeCategories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Skip knowledge category.")
    message = update.effective_message

    nk = context.chat_data.get("new_knowledge")
    nk["categories"] = None

    keyboard = [
        [
            InlineKeyboardButton("Yes", callback_data="yes"),
            InlineKeyboardButton("No", callback_data="no")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    responseText = f"""Ok, no category tag to add.

Let's do a final review.

{nk['text']}

Source:  {nk['source']}

Category:  None

Do you wish to save?
"""

    try:
        await message.reply_text(
            text=responseText,
            reply_markup=reply_markup
        )

        return STORE_KNOWLEDGE
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}.")
        nk = None

        return ConversationHandler.END


async def finalizeKnowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Finalize the add knowledge process.")
    message = update.effective_message
    user = update.effective_user

    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return
    
    nk = context.chat_data.get("new_knowledge")
    
    if query.data == "yes":
        # Get account information
        member = members.getMemberByTelegramID(user.id)

        # If somehow an unregistered user made it to this point
        if member is None:
            logger.warning(f"An unregistered user {user.name} (user_id: {user.id}) attempted to save data to the knowledge database.")
            return ConversationHandler.END
        
        allowedRoles = ["admin", "owner"]
        rolesAvailable = member["roles"]

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to save documents to the knowledge database.")

            document = nk.get("text")
            documentMetadata = dict()
            categories = [] if nk.get("categories") is None else nk.get("categories")
            #print(categories)
            source = nk.get("source")
            if source:
                documentMetadata["source"] = source
            
            record_id = knowledge.addDocument(document, categories=categories, documentMetadata=documentMetadata)
            #print(record_id)

            try:
                # Edit telegram message
                await query.edit_message_text(
                    text=f"Document stored. Record ID:  {record_id}"
                )
            except Exception as err:
                logger.error(f"Exception while editing a telegram message:\n{err}")
            finally:
                # Delete the newKnowledge property in chat_data
                nk = None

                return ConversationHandler.END
        
        # The follwoing code only runs if the user is not authorized in the above for loop
        logger.warning(f"User {user.name} (user_id: {user.id}) is not authorized to use knowledge command.")
        try:
            await message.reply_text(
                text="Only admins are authorized to use the knowledge command."
            )
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        finally:
            return ConversationHandler.END

    else:
        # Text not approved, start over
        logger.info(f"User selected to not store knowledge data, process ended.")
        try:
            await query.edit_message_text(
                text="You've selected to not save the knowledge data. Process ended."
            )
        except Exception as err:
            logger.error(f"Exception while editing a telegram message:\n{err}")
        finally:
            return ConversationHandler.END



#####################
# Promotion Manager #
#-------------------#

# Define command chain states
GET_ACCOUNT, VERIFY_ACCOUNT, VERIFY_PROMOTE = range(3)

async def promoteAccount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    logger.info(f"Promote account command chain initiated by {user.name} (user_id: {user.id}).")

    # Check if the user that issued the promote command is authorized to promote
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.warning(f"A non-registered user {user.name} (user_id: {user.id}) attempted to use the promote command.")
        return

    # First check if user has valid roles. 
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        # Check if message is a reply to another message and get user account from message data
        if message.reply_to_message is not None:
            # Promote command issued as a reply to a user. Get the telegram user information
            userToPromote = message.reply_to_message.from_user
            # Get member from telegram user
            memberToPromote = members.getMemberByTelegramID(userToPromote.id)
            if memberToPromote is not None:
                logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote user {userToPromote.name} (user_id: {userToPromote.id}).")
                context.chat_data["member_to_promote"] = memberToPromote
                
                # TODO build the roles available for promotion from roles list in config
                prettyTitles = {
                    "user": "Users",
                    "tester": "Testers",
                    "marketing": "Marketing",
                    "admin": "Administrators",
                    "owner": "Owner"
                }

                availableRolesForPromotion = [role for role in config.rolesList if role not in memberToPromote.get("roles")]
                print(availableRolesForPromotion)

                # TODO build inline keyboard button pairs from the available roles for promotion
                buttonList = []
                for role in availableRolesForPromotion:
                    buttonText = role if role not in prettyTitles else prettyTitles[role]
                    buttonList.append(InlineKeyboardButton(buttonText, callback_data=role))
                
                # Display role selection
                keyboard = list(pairs(buttonList))
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Send message with text and appended InlineKeyboard
                try:
                    await message.reply_text(
                        text=f"What role would you like to promote {userToPromote.name} to?\n\nType /cancel to cancel.", 
                        reply_markup=reply_markup
                    )
                except Exception as err:
                    logger.error(f"Exception while replying to a telegram message:\n{err}")
                finally:
                    return VERIFY_PROMOTE
                
            else:
                logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote, but user {userToPromote.name} (user_id: {userToPromote.id}) is not registered.")
                try:
                    await message.reply_text(
                        text=f"User {userToPromote.name} is not a registered user.", 
                    )
                except Exception as err:
                    logger.error(f"Exception while sending a telegram message:\n{err}")
                finally:
                    return ConversationHandler.END
                
        elif chat.type == "group" or chat.type == "supergroup":
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to promote group accounts.")
            # Promote command issued in a group chat. Get that group's account information
            communityToPromote = communities.getCommunityByTelegramID(chat.id)
            if communityToPromote is not None:
                context.chat_data["community_to_promote"] = communityToPromote
                keyboard = [
                    [
                        InlineKeyboardButton("Administrator", callback_data="admin"),
                        InlineKeyboardButton("Marketing", callback_data="marketing")
                    ],
                    [
                        InlineKeyboardButton("Tester", callback_data="tester")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                # Send message with text and appended InlineKeyboard
                try:
                    await message.reply_text(
                        text=f"What role would you like to promote the group {communityToPromote['community_name']} to?\n\nType /cancel to cancel.", 
                        reply_markup=reply_markup
                    )
                except Exception as err:
                    logger.error(f"Exception while replying to a telegram message:\n{err}")
                finally:
                    return VERIFY_PROMOTE
        else:
            # TODO look up user by username passed as argument after promote command
            print("who do you want to promote?")

            return ConversationHandler.END

    # Member is registered but not authorized
    else:
        logger.warning(f"A non-authorized member {user.name} (user_id: {user.id}) attempted to use the promote command.")

        return ConversationHandler.END


async def setNewRole(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Update the account with the new role.")
    query = update.callback_query
    chat = update.effective_chat
    user = update.effective_user
    
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return
    
    memberToPromote = context.chat_data.get("member_to_promote")

    if memberToPromote is not None:
        # Get the telegram user from member data
        # TODO The following isn't really necessary, used just for logging
        try:
            userToPromote = await context.bot.get_chat_member(chat_id=chat.id, user_id=memberToPromote["user_id"])
            logger.info(f"User {user.name} (user_id: {user.id}) is promoting {userToPromote.user.name} ({userToPromote.user.id}) to the role of {query.data}.")
        except Exception as err:
            logger.error(f"Exception while getting telegram user information:\n{err}")

        # TODO Verify the role is in the roles list from the config
        if query.data in config.rolesList:
            memberToPromote["roles"].append(query.data)
        
        results = members.updateMemberRoles(memberToPromote.get("member_id"), memberToPromote.get("roles"))
        if results:
            responseText = f"{userToPromote.user.name} has been promoted to {query.data} role."
        else:
            responseText = "An error occured promoting the user."

        # Clear the temporary user to promote value
        context.chat_data['member_to_promote'] = None

    elif "community_to_promote" in context.chat_data:
        logger.info(f"User {user.name} (user_id: {user.id}) is promoting the group {context.chat_data['group_to_promote']['chat_title']} to the role of {query.data}.")

        communityToPromote = context.chat_data["community_to_promote"]
        communityToPromote["roles"].append(query.data)
        #results = accounts.updateRoles(groupToPromote["roles"], groupToPromote["chat_id"], accountType=chat.type)
        results = communities.updateCommunityRoles(communityToPromote.get("community_id"), communityToPromote.get("roles"))
        if results:
            responseText = f"{communityToPromote['chat_title']} has been promoted to {query.data} role."
        else:
            responseText = "An error occured promoting the community chat."

        # Clear the temporary user to promote value
        context.chat_data['community_to_promote'] = None
    else:
        return ConversationHandler.END

    try:
        await query.edit_message_text(
            text=responseText
        )
    except Exception as err:
        logger.error(f"Exception while editing a telegram message:\n{err}")
    finally:
        return ConversationHandler.END



#############################
# Tweet agent command chain #
#---------------------------#

# Define tweet states
CONFIRM_TWEET, MODIFY_TWEET = range(2)

async def tweetStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"The tweet command has been issued.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Need user account for all chat types
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the tweet command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]
    
    if chat.type == "private":
        rolesAvailable = member["roles"]

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to use tweet command in private chat.")
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
            else:
                # No arguments passed check if the command was given as a reply to another message
                if message.reply_to_message is not None:
                    if message.reply_to_message.text is None:
                        # Can only handle replies to text currently
                        return ConversationHandler.END
                    
                    tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                else:
                    # Command was sent without arguments and not as a reply to a message
                    tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
            # call the TwitterAgent
            
            fromUser = {
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": "user",
            }

            messageData = {
                "chat_id": chat.id,
                "topic_id" : topicID,
                "message_id": message.message_id,
                "tweet_prompt": tweetPrompt
            }
            
            ta = TweetAgent(message_data=messageData, from_user=fromUser)
            tweet = await ta.ComposeTweet()

            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END
        
    elif chat.type == "group":
        logger.info(f"Tweet command issued in a group chat.")

        # Get group account data
        #groupAcct = accounts.getAccountByTelegramID(chat.id, chat.type)
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to use the tweet command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        # Combine user and group roles
        rolesAvailable = set(member["roles"] + community["roles"])

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"User {user.name} (user_id: {user.id}) is authorized to use tweet command in the {community['community_name']} group chat.")
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
                knowledge = config.database + "_knowledge"
            else:
                # No arguments passed check if the command was given as a reply to another message
                if message.reply_to_message is not None:
                    if message.reply_to_message.text is None:
                        # Can only handle replies to text currently
                        return ConversationHandler.END
                    
                    tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                    knowledge = config.database + "_knowledge"
                else:
                    # Command was sent without arguments and not as a reply to a message
                    tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
                    knowledge = None
            # call the TwitterAgent
            # Need to update this to the new agent code
            ta = TweetAgent(message=tweetPrompt, chatHistory_db=f"group{community['chat_id']}_chat_history", knowledge_db=knowledge)
            tweet = await ta.ComposeTweet()

            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END

    elif chat.type == "supergroup":
        logger.info(f"Tweet command issued in a supergroup chat.")

        # NOTE in a supergroup, normal messages sent in the general topic 
        # will have message.chat.is_forum return True, but message.is_topic_message return False
        # also will not have a message.reply_to_message value which also means no message.reply_to_message.message_thread_id
        # replies in the general topic area contain a message.reply_to_message that contains a text value of the message being replied to, just like in a normal group message

        # NOTE in a supergroup, normal messages sent in a topic thread 
        # will have message.chat.is_forum return True as well message.is_topic_message return True
        # will have a message.reply_to_message that contains a message_thread_id but does not contain a text property
        # replies sent in a topic thread will contain the above, but will have a text property

        # Get group account data
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.info(f"User {user.name} (user_id: {user.id}) attempted to use the tweet command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        # Combine user and group roles
        rolesAvailable = set(community["roles"] + community["roles"])

        if (any(role in rolesAvailable for role in allowedRoles)):
            logger.info(f"Tweet command issued in a supergroup chat.")
            
            # Check if command sent in a topic thread
            if message.is_topic_message:
                topicID = message.reply_to_message.message_thread_id
            else:
                topicID = None
            
            # Check for "arguments" passed with the tweet command. This will act as a custom prompt for the Twitter Agent
            if len(context.args) > 0:
                tweetPrompt = " ".join(context.args)
                knowledge = config.database + "_knowledge"
            elif message.is_topic_message and message.reply_to_message.text is not None:
                # Command in a topic thread and a reply to a message
                # Messages in a topic thread always have a message.reply_to_message value
                logger.info("Command in a topic thread and a reply to a message")
                tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                knowledge = config.database + "_knowledge"
            elif not message.is_topic_message and message.reply_to_message is not None:
                # Command in General topic and a reply to a message
                logger.info("Command in a general thread and a reply to a message")
                if message.reply_to_message.text is None:
                    # Can only handle replies to text currently
                    return ConversationHandler.END
                
                tweetPrompt = f"Create a tweet based on the following message:\n\n{message.reply_to_message.text}"
                knowledge = config.database + "_knowledge"
            else:
                # Command was sent without arguments and not as a reply to a message
                tweetPrompt = f"Create a tweet based on something you find interesting from our conversation so far."
                knowledge = None
            

            # call the TwitterAgent
            ta = TweetAgent(message=tweetPrompt, chatHistory_db=f"group{community['chat_id']}_chat_history", knowledge_db=knowledge, topicID=topicID)
            tweet = await ta.ComposeTweet()
            context.chat_data["tweet_agent"] = ta

            keyboard = [
                [
                    InlineKeyboardButton("Confirm", callback_data="confirm"),
                    InlineKeyboardButton("Reject", callback_data="reject")
                ],
                [
                    InlineKeyboardButton("Modify", callback_data="modify")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await message.reply_text(
                text=tweet,
                reply_markup=reply_markup
            )

            return CONFIRM_TWEET
        else:
            logger.info(f"User {user.name} (user_id: {user.id}) is not authorized to use tweet command.")
            return ConversationHandler.END
    
    else:
        return ConversationHandler.END


async def modifyTweet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Modify the tweet agent response with a new prompt.")
    message = update.effective_message

    if "tweet_agent" in context.chat_data:
        tweetAgent = context.chat_data["tweet_agent"]
    else:
        # There is no tweet_agent object, exit the function
        logger.warning(f"tweet_agent value missing from context chat_data.")
        return ConversationHandler.END
    
    tweet = await tweetAgent.ModifyTweet(message.text)

    keyboard = [
        [
            InlineKeyboardButton("Confirm", callback_data="confirm"),
            InlineKeyboardButton("Reject", callback_data="reject")
        ],
        [
            InlineKeyboardButton("Modify", callback_data="modify")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(
        text=tweet,
        reply_markup=reply_markup
    )

    return CONFIRM_TWEET


async def confirmTweet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Confirm tweet.")
    user = update.effective_user
    # Temporary area to mock sending tweets
    testChatID = -1002177698730
    testTopicID = 3824
    
    if "tweet_agent" in context.chat_data:
        tweetAgent = context.chat_data["tweet_agent"]
    else:
        # There is no tweet_agent object, exit the function
        logger.warning(f"tweet_agent value missing from context chat_data.")
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    if query.data == "confirm":
        logger.info(f"User confirmed the tweet. Sending...")

        tweetResults = await tweetAgent.SendTweet()
        # TODO add if else condition on the result of the tweet agent
        # Temporarily send the tweet to a specific topic in supergroup chat
        await context.bot.send_message(
            chat_id=testChatID,
            message_thread_id=testTopicID,
            text=f"The following tweet was approved to be sent by {user.name}\n\n{tweetAgent.tweetText}"
        )

        # Edit telegram message
        await query.edit_message_text(
            text=f"The following tweet was sent:\n\n{tweetAgent.tweetText}"
        )

        return ConversationHandler.END
    elif query.data == "modify":
        # Edit telegram message
        await query.edit_message_text(
            text=f"Here is the tweet so far:\n\n{tweetAgent.tweetText}\n\nEnter a new prompt to modify this tweet with."
        )
        return MODIFY_TWEET
    else:
        logger.info(f"User rejected the tweet.")
        # Edit telegram message
        await query.edit_message_text(
            text=f"The following tweet was rejected:\n\n{tweetAgent.tweetText}"
        )
        return ConversationHandler.END



############################
# Newsletter command chain #
#--------------------------#

# Define newsletter states
ROLE_SELECTION, PHOTO_OPTION, ADD_PHOTO, COMPOSE_NEWLETTER, CONFIRM_NEWSLETTER = range(5)

async def newsletterStart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Newsletter command issued.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    # Need user account for all chat types
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.warning(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the newsletter command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]

    # This section allows for this function to inherit roles from the community's roles
    if chat.type == "private":
        rolesAvailable = member["roles"]
    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)

        if community is None:
            logger.warning(f"User {user.name} (user_id: {user.id}) attempted to use the newsletter command in an unregistered group chat.")
            # Exit the function, there is no group account
            return ConversationHandler.END
        
        rolesAvailable = set(member["roles"] + community["roles"])

    else:
        # Exit the function, no other chat types allowed
        return ConversationHandler.END
    
    if (any(role in rolesAvailable for role in allowedRoles)):
        logStr = f"User {user.name} (user_id: {user.id}) (roles:  {', '.join(member['roles'])}) is authorized to use the newsletter command"
        if chat.type != "private":
            logStr = logStr + f" in the {community['chat_title']} group chat (roles:  {', '.join(community['roles'])})."
        else:
            logStr = logStr + f" in private chat."
        logger.info(logStr)

        # Authorized

        # Init the temporary storage for holding newsletter data
        nd = context.chat_data["newsletter_data"] = {
            "text": None,
            "roles": []
        }

        # Check if arguments were passed with the command
        if len(context.args) > 0:
            # Create the text property
            nd["text"] = " ".join(context.args)

        # Check if the command was a reply to a message
        if ((chat.type == "private" or chat.type == "group" or (chat.type == "supergroup" and not message.is_topic_message)) and message.reply_to_message is not None) or (chat.type == "supergroup" and message.is_topic_message and message.reply_to_message.text is not None):
            # Create or add to the text property with the reply text
            nd["text"] = message.reply_to_message.text if nd["text"] is None else nd["text"] + "\n\n" + message.reply_to_message.text

        # Display role selection
        # TODO Build this inline button list based on roles list in config
        keyboard = [
            [
                InlineKeyboardButton("Users", callback_data="user"),
                InlineKeyboardButton("Beta Testers", callback_data="tester")
            ],
            [
                InlineKeyboardButton("Marketing", callback_data="marketing"),
                InlineKeyboardButton("Administrators", callback_data="admin")
            ],
            [
                InlineKeyboardButton("Done", callback_data="done")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await message.reply_text(
                text="Select roles to recieve newsletter.",
                reply_markup=reply_markup
            )
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        finally:
            return ROLE_SELECTION

    else:
        logStr = f"User {user.name} (user_id: {user.id}) (roles:  {', '.join(member['roles'])}) is NOT authorized to use the newsletter command"
        if chat.type != "private":
            logStr = logStr + f" in the {community['chat_title']} group chat (roles:  {', '.join(community['roles'])})."
        else:
            logStr = logStr + f" in private chat."
        logger.info(logStr)

        # Unauthorized

        return ConversationHandler.END


async def selectRole(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("select role")
    message = update.effective_message
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return

    nd = context.chat_data["newsletter_data"]

    prettyTitles = {
        "user": "Users",
        "tester": "Testers",
        "marketing": "Marketing",
        "admin": "Administrators",
        "owner": "Owner"
    }

    if query.data in config.rolesList:
        nd["roles"].append(query.data)

    buttonList = []
    remainingRoles = [role for role in config.rolesList if role not in nd["roles"]]

    for role in remainingRoles:
        buttonText = role if role not in prettyTitles else prettyTitles[role]
        buttonList.append(InlineKeyboardButton(buttonText, callback_data=role))

    buttonList.append(InlineKeyboardButton("Done", callback_data="done"))
    
    # Display role selection
    keyboard = list(pairs(buttonList))
    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await query.edit_message_text(
            text=f"Select roles to recieve newsletter. Roles selected: {', '.join(nd['roles'])}",
            reply_markup=reply_markup
        )
        return ROLE_SELECTION
    
    except Exception as err:
        logger.error(f"Exception while editing a telegram message:\n{err}")
        return


async def roleSelectionDone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Role selection done")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return

    nd = context.chat_data["newsletter_data"]

    if query.data == "done" and len(nd["roles"]) > 0:
        if nd["text"] is None:
            # Ask if they wish to add a photo.

            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data="yes"),
                    InlineKeyboardButton("No", callback_data="no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            try:
                await query.edit_message_text(
                    text=f"Do you want to add an image?",
                    reply_markup=reply_markup
                )
                return PHOTO_OPTION
            except Exception as err:
                logger.error(f"Excpetion while editing a telegram query message:\n{err}")
                return

        else:
            # Already have text body, get confirmation
            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data="yes"),
                    InlineKeyboardButton("No", callback_data="no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            try:
                await query.edit_message_text(
                    text=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )
                return CONFIRM_NEWSLETTER
            except Exception as err:
                logger.error(f"Exception while editing a telegram query message:\n{err}")
                return
    
    return ConversationHandler.END


async def photoOption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add a photo or continue to newsletter text")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Execption while receiving a telegram query response:\n{err}")
        return
    
    nd = context.chat_data["newsletter_data"]

    try:
        if query.data == "yes":
            await query.edit_message_text(
                text="Send the image you wish to add to the newsletter"
            )
            return ADD_PHOTO

        else:
        # Get newsletter body text
            await query.edit_message_text(
                text=f"Compose the text of the newsletter. Newsletter will be sent to the following roles: {', '.join(nd['roles'])}"
            )
            return COMPOSE_NEWLETTER
        
    except Exception as err:
        logger.error(f"Exception while editing a telegram query message:\n{err}")
        return


async def addNewsletterPhoto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add the newsletter photo")
    message = update.effective_message

    nd = context.chat_data["newsletter_data"]

    photo = update.message.photo[-1].file_id

    if photo:
        nd["photo"] = photo

        try:
            await message.reply_text(
                text=f"Compose the text of the newsletter. Newsletter will be sent to the following roles: {', '.join(nd['roles'])}"
            )
            return COMPOSE_NEWLETTER
        except Exception as err:
            logger.error(f"Exception while sending a telegram message:\n{err}")
            return
    else:
        logger.info("There was an issue getting the photo.")
        return COMPOSE_NEWLETTER


async def addNewsletterText(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Add the newsletter text")
    message = update.effective_message

    nd = context.chat_data["newsletter_data"]

    if len(message.text) > 0:
        nd["text"] = message.text

        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data="yes"),
                InlineKeyboardButton("No", callback_data="no")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Check if a photo was added and reply with photo and caption if so
        newsletterPhoto = nd.get("photo")

        try:
            if newsletterPhoto is not None:
                logger.info("Newsletter has a photo.")
                await message.reply_photo(
                    photo=newsletterPhoto,
                    caption=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )

            else:
                logger.info("Newsletter has no photo.")
                await message.reply_text(
                    text=f"Here's your newsletter:\n\n{nd['text']}\n\nSending to the following roles: {', '.join(nd['roles'])}",
                    reply_markup=reply_markup
                )
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
            return

        return CONFIRM_NEWSLETTER
    else:
        # User didn't send text
        return ConversationHandler.END


async def confirmNewsletter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Confirm sending the newsletter")
    query = update.callback_query
    try:
        await query.answer()
    except Exception as err:
        logger.error(f"Exception while receiving a telegram query response:\n{err}")
        return
    
    nd = context.chat_data["newsletter_data"]

    if query.data == "yes":
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption="Sending..."
                )
            else:
                await query.edit_message_text(
                    text="Sending..."
                )
        except Exception as err:
            logger.error(f"Exception while editing a telegram query message:\n{err}")
            return
        
        membersInRoles = members.getMembersByRoles(nd["roles"])
        newsletterPhoto = nd.get("photo")
        for member in membersInRoles:
            try:
                if newsletterPhoto is not None:
                    await context.bot.send_photo(
                        chat_id=member["user_id"],
                        caption=nd["text"],
                        photo=newsletterPhoto
                    )
                else:
                    await context.bot.send_message(
                        chat_id=member["user_id"],
                        text=nd["text"]
                    )
                print(f"sent to {member['username']} ({member['user_id']}).")
            except Exception:
                print(f"Unable to send to {member['username']} ({member['user_id']}).")
        
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption=f"Sent:\n\n{nd['text']}\n\nto the following roles: {', '.join(nd['roles'])}"
                )
            else:
                await query.edit_message_text(
                    text=f"Sent:\n\n{nd['text']}\n\nto the following roles: {', '.join(nd['roles'])}"
                )
        except Exception as err:
            logger.error(f"Exception while editing a telegram query message:\n{err}")
    else:
        try:
            if query.message.caption is not None:
                await query.edit_message_caption(
                    caption=f"Newsletter not sent"
                )
            else:
                await query.edit_message_text(
                    text=f"Newsletter not sent"
                )
        except Exception as err:
            logger.error(f"Exception while editing a telegram query message:\n{err}")
            return
    
    return ConversationHandler.END



SELECT_PROPOSAL, PROPOSAL_NDA, SHOW_PROPOSAL = range(3)

async def proposalsManager(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Proposals Manager started.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None
    user = update.effective_user

    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"An unregistered user {user.name} (user_id: {user.id}) attempted to use the proposals command.")
        # Exit the function if there is no user account
        return ConversationHandler.END

    # Set the allowed roles
    allowedRoles = ["admin", "owner"]
    rolesAvailable = member["roles"]

    if (any(role in rolesAvailable for role in allowedRoles)):
        context.chat_data["proposals"] = proposals = proposals.getProposals()

        if len(proposals) > 0 and len(proposals) < 6:
            logger.info(f"Show proposal list.")
            # Loop and build proposal list
            responseText = "The following proposals are available:\n"
            responseKeyboard = []
            for p in proposals:
                responseText += f"\n*{p['project_title']}* - _{p['submitted_from']}_\n*Description:*  {p['project_description']}"
                # Build reply inline keyboard for each proposal
                btn = [InlineKeyboardButton(p["project_title"], callback_data=p["project_id"])]
                responseKeyboard.append(btn)
            
            reply_markup = InlineKeyboardMarkup(responseKeyboard)
            
            await message.reply_markdown(
                text=responseText,
                reply_markup=reply_markup
            )

            return SELECT_PROPOSAL
        else:
            logger.info(f"There are no proposals to show.")

            await message.reply_text(text="There are no proposals to show.")
    else:
        logger.info(f"An unauthorized user {user.name} (user_id: {user.id}) attempted to use the proposals command.")
    
    return ConversationHandler.END


async def agreeNDA(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Proposal selected, get NDA confirmation.")
    message = update.effective_message
    query = update.callback_query
    await query.answer()
    proposals = context.chat_data.get("proposals")

    if proposals is not None:
        context.chat_data["proposal"] = proposal = next((p for p in proposals if p["project_id"] == int(query.data)), None)

        if proposal is None:
            return ConversationHandler.END

        # Proposal selected, display NDA
        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data="confirm"),
                InlineKeyboardButton("Reject", callback_data="reject")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        confidentialAgreementResponse = f"""*The following proposal ({context.chat_data["proposal"]["project_title"]}) is a confidential document.*
CONFIDENTIALITY AGREEMENT

I acknowledge that the document and its contents provided by Hypermind Labs are confidential. I agree to:

Keep the document and all its contents strictly confidential.

Not share, distribute, or disclose the document or its contents to any third party without prior written consent from Hypermind Labs.

Use the document solely for the intended purpose of evaluating the proposal.

Take all reasonable measures to protect the confidentiality of the document and its contents.

Do you agree to these terms?"""

        await query.edit_message_text(
            text=confidentialAgreementResponse,
            reply_markup=reply_markup,
            parse_mode=constants.ParseMode.MARKDOWN
        )

        return PROPOSAL_NDA
    else:
        return ConversationHandler.END


async def openProposal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"NDA responded to, handle response.")
    message = update.effective_message
    query = update.callback_query
    user = update._effective_user
    await query.answer()

    if query.data != "confirm":
        await query.edit_message_text(
            text="You must agree to the NDA to view the proposal."
        )
        return ConversationHandler.END

    # User selected confirm, load the proposal
    proposal = context.chat_data.get("proposal")
    # Store the NDA acceptance
    proposals.addDisclosureAgreement(user.id, proposal.get("project_id"))

    proposalsPath = "assets/proposals/"
    proposalFile = proposal.get("filename")

    script_dir = os.path.dirname(__file__)
    rel_path = proposalsPath + proposalFile
    abs_file_path = os.path.join(script_dir, rel_path)


    await query.edit_message_text(
        text="Getting the proposal..."
    )

    await message.reply_document(
        document=open(abs_file_path, "rb"),
        protect_content=True
    )
    return ConversationHandler.END



####################
# Message Handlers #
####################

async def catchAllMessages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"A message was captured by the catch all function.")


async def runInspectorView(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    data = str(query.data or "")
    if not data.startswith("diag:"):
        return

    parts = data.split(":", 3)
    if len(parts) < 3:
        await query.answer("Invalid inspector action.", show_alert=False)
        return

    token = parts[1]
    action = parts[2]
    raw_arg = parts[3] if len(parts) > 3 else ""
    if action == "noop":
        await query.answer()
        return

    payload = await _lookup_run_inspector_payload(token)
    if payload is None:
        await query.answer("Inspector session expired.", show_alert=False)
        return

    message = query.message
    message_chat_id = getattr(getattr(message, "chat", None), "id", None)
    message_id = getattr(message, "message_id", None)
    if message_chat_id is None or message_id is None:
        await query.answer("Inspector unavailable for this message.", show_alert=False)
        return

    if int(payload.get("chat_id", -1)) != int(message_chat_id) or int(payload.get("message_id", -1)) != int(message_id):
        await query.answer("Inspector data does not match this message.", show_alert=False)
        return

    try:
        if action == "back":
            final_text = str(payload.get("final_text") or _TELEGRAM_FALLBACK_FINAL_REPLY)
            keyboard = _build_run_inspector_keyboard(token, payload)
            await query.edit_message_text(
                text=_inspector_trim_text(final_text, max_chars=_TELEGRAM_MAX_MESSAGE_CHARS - 8),
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )
            await query.answer()
            return

        if action == "tool":
            tool_entries = payload.get("tool_entries")
            tools = tool_entries if isinstance(tool_entries, list) else []
            try:
                tool_index = int(raw_arg)
            except ValueError:
                tool_index = 0
            if tool_index < 0 or tool_index >= len(tools):
                await query.answer("Tool view is out of range.", show_alert=False)
                return
            tool_entry = tools[tool_index] if isinstance(tools[tool_index], dict) else {}
            tool_text = _render_tool_page_text(
                tool_entry=tool_entry,
                tool_index=tool_index,
                total_tools=len(tools),
            )
            await query.edit_message_text(
                text=tool_text,
                reply_markup=_build_tool_view_keyboard(token),
                disable_web_page_preview=True,
            )
            await query.answer()
            return

        if action == "ctx":
            stage_entries = payload.get("stage_entries")
            stages = stage_entries if isinstance(stage_entries, list) else []
            if not stages:
                await query.answer("No stage context captured for this run.", show_alert=False)
                return
            try:
                stage_index = int(raw_arg)
            except ValueError:
                stage_index = 0
            stage_index = max(0, min(stage_index, len(stages) - 1))
            stage_entry = stages[stage_index] if isinstance(stages[stage_index], dict) else {}
            stage_text = _render_stage_page_text(
                stage_entry=stage_entry,
                stage_index=stage_index,
                total_stages=len(stages),
            )
            await query.edit_message_text(
                text=stage_text,
                reply_markup=_build_context_view_keyboard(token, stage_index, len(stages)),
                disable_web_page_preview=True,
            )
            await query.answer()
            return

        await query.answer("Unknown inspector action.", show_alert=False)
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Inspector callback failed: {error}")
        await query.answer("Unable to update inspector view.", show_alert=False)


async def directChatGroup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Bot messaged in group chat.")
    chat = update.effective_chat
    message = update.effective_message
    topicID = message.message_thread_id if message.is_topic_message else None
    user = update.effective_user

    # Get community data for telegram group
    community = communities.getCommunityByTelegramID(chat.id)

    if community is None:
        logger.info(f"User {user.name} (user_id: {user.id}) attempted to message the chatbot in an unregistered group chat.")
        # No community account, exit the function
        return

    # Get member data
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.warning(f"An unregistered user {user.name} (user_id: {user.id}) attempted to message the chatbot in {community['chat_title']} group chat.")
        try:
            await message.reply_text(text=f"To message {config.botName}, open a private message to the chatbot @{config.botName} and click start.")
        except Exception as err:
            logger.error(f"The following error occurred while sending a telegram message:\n{err}")
        finally:
            # No member account, exit the function
            return  
    
    memberID = member.get("member_id")
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = set(member["roles"] + community["roles"])
    
    # Check user usage
    memberUsage = usage.getUsageForMember(memberID)
    memberUsage = memberUsage if isinstance(memberUsage, list) else list()
    rateLimits = communityScore.getRateLimits(memberID, "community")

    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        messageLimit = rateLimits.get("message", 0)
        if messageLimit > 0 and len(memberUsage) >= messageLimit:
            logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
            try:
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.error(f"The following error occurred while sending a telegram message:\n{err}")
            finally:
                # User has reach rate limit, exit function
                return

    messageContext = {
        "community_id": community.get("community_id"),
        "chat_host_id": community.get("community_id"),
        "chat_type": "community",
        "platform": "telegram",
        "topic_id" : topicID,
        **_message_temporal_context(message),
    }
    
    stageStatus = TelegramStageStatus(
        context,
        chat_id=chat.id,
        thread_id=topicID,
        reply_message=None,
        enabled=_runtime_bool("telegram.show_stage_progress", True),
        show_json_details=_runtime_bool("telegram.show_stage_json_details", True),
        detail_level=_runtime_str("telegram.stage_detail_level", "minimal"),
        update_min_interval_seconds=_runtime_float("telegram.stage_update_min_interval_seconds", 1.0),
    )
    await stageStatus.start()
    stageEvents: list[dict[str, Any]] = []

    async def _stage_callback(event: dict[str, Any]) -> None:
        _append_stage_event(stageEvents, event)
        await stageStatus.emit(event)

    # Create the conversational agent instance
    conversation = ConversationOrchestrator(
        message.text,
        memberID,
        messageContext,
        message.message_id,
        options={"stage_callback": _stage_callback},
    )

    try:
        # Shows the bot as "typing"
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING,
            message_thread_id=topicID
        )
    except Exception as err:
        logger.error(f"Exception while sending a chat action for typing:\n{err}")
        # Non critical to the overall function, continue

    try:
        response = await conversation.runAgents()
    except Exception as err:  # noqa: BLE001
        logger.error(f"Conversation orchestration failed in group chat:\n{err}")
        await stageStatus.fail("I hit an internal error while processing that request.")
        return
    
    try:
        # Send the conversational agent response
        finalText = (
            f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. "
            "Responses may or may not be factually correct."
        )
        responseMessage = await stageStatus.finalize(finalText)
        await _attach_run_inspector_controls(
            response_message=responseMessage,
            final_text=finalText,
            run_summary=conversation.run_summary,
            stage_events=stageEvents,
        )
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
        # Error is critical to the remaining functionality, exit
        return

    # Score the message if the message is greater than the configured threshold
    if len(message.text.split(" ")) > _message_word_threshold():
        communityScore.scoreMessage(conversation.promptHistoryID)

    # Add the response to the chat history
    conversation.storeResponse(responseMessage.message_id)

    # Add the retrieved documents to the retrieval table
    # This had to wait for the prompt and response chat history records to be created
    # TODO Move this code to the conversational orchestrator
    """
    docs = chatAgent._documents
    for doc in docs:
        retrievalID = knowledge.addRetrieval(chatAgent.promptHistoryID, responseHistoryID, doc.get("knowledge_id"), doc.get("distance"))
    """

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context, threadID=topicID)
        except Exception as err:
            logger.error(f"The following error occurred while sending a telegram message:\n{err}")


async def directChatPrivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Bot messaged in private chat.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    minimumCommunityScore = _runtime_int("telegram.minimum_community_score_private_chat", 50)

    # Get account information
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.info(f"Unregistered user {user.name} (user_id: {user.id}) messaged the bot in a private message.")
        try:
            if _database_unavailable():
                await message.reply_text(_database_unavailable_message())
            else:
                await message.reply_text("Use the /start command to begin chatting with the chatbot.")
        except Exception as err:
            logger.error(f"The following error occurred while sending a telegram message:\n{err}")
        finally:
            # No member account, exit the function
            return
    
    memberID = member.get("member_id")
    
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = member.get("roles")
    
    # Check user usage
    memberUsage = usage.getUsageForMember(memberID)
    memberUsage = memberUsage if isinstance(memberUsage, list) else list()
    rateLimits = communityScore.getRateLimits(memberID, chat.type)
    memberCommunityScore = member.get("community_score", 0) or 0

    if (not any(role in rolesAvailable for role in allowedRoles)):
        if memberCommunityScore < minimumCommunityScore:
            try:
                logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to use private chat with the chatbot. Community score:  {memberCommunityScore}/{minimumCommunityScore}")
                await message.reply_text(f"You need a minimum community score of {minimumCommunityScore} to use the private chat features. Please join one of our community chats to build your community score.")
            except Exception as err:
                logger.error(f"The following error occurred while sending a telegram message:\n{err}")
            finally:
                return

        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        messageLimit = rateLimits.get("message", 0)
        if messageLimit > 0 and len(memberUsage) >= messageLimit:
            try:
                logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in private chat.")
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.error(f"The following error occurred while sending a telegram message:\n{err}")
            finally:
                # User has reach rate limit, exit function
                return

    messageData = {
        "community_id": None,
        "chat_host_id": memberID,
        "chat_type": "member",
        "member_id": member.get("member_id"),
        "platform": "telegram",
        "topic_id" : None,
        **_message_temporal_context(message),
    }

    stageStatus = TelegramStageStatus(
        context,
        chat_id=chat.id,
        thread_id=None,
        reply_message=message,
        enabled=_runtime_bool("telegram.show_stage_progress", True),
        show_json_details=_runtime_bool("telegram.show_stage_json_details", True),
        detail_level=_runtime_str("telegram.stage_detail_level", "minimal"),
        update_min_interval_seconds=_runtime_float("telegram.stage_update_min_interval_seconds", 1.0),
    )
    await stageStatus.start()
    stageEvents: list[dict[str, Any]] = []

    async def _stage_callback(event: dict[str, Any]) -> None:
        _append_stage_event(stageEvents, event)
        await stageStatus.emit(event)

    conversation = ConversationOrchestrator(
        message.text,
        memberID,
        messageData,
        message.message_id,
        options={"stage_callback": _stage_callback},
    )

    try:
        # Shows the bot as "typing"
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING
        )
    except Exception as err:
        logger.error(f"Exception while sending a chat action for typing:\n{err}")
        # Non critical to the remaining functionality, continue
    
    try:
        response = await conversation.runAgents()
    except Exception as err:  # noqa: BLE001
        logger.error(f"Conversation orchestration failed in private chat:\n{err}")
        await stageStatus.fail("I hit an internal error while processing that request.")
        return
    
    try:
        finalText = str(response)
        responseMessage = await stageStatus.finalize(finalText)
        await _attach_run_inspector_controls(
            response_message=responseMessage,
            final_text=finalText,
            run_summary=conversation.run_summary,
            stage_events=stageEvents,
        )
    except Exception as err:
        logger.error(f"Exception while replying to a telegram message:\n{err}")
        # Error is critical to the remaining functionality, exit
        return
    
    # Score the message if the message is greater than the configured threshold
    if len(message.text.split(" ")) > _message_word_threshold():
        communityScore.scoreMessage(conversation.promptHistoryID)

    conversation.storeResponse(responseMessage.message_id)

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context)
        except Exception as err:
            logger.error(f"The following error occurred while sending a telegram message:\n{err}")


# TODO Still need to handle chat history and usage storage for images
async def handleImage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Image received.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Need user account regardless of chat type
    member = members.getMemberByTelegramID(user.id)
    if member is None:
        logger.warning(f"Unregistered user {user.name} (user_id: {user.id}) sent an image in a {chat.type} chat.")
        try:
            await message.delete()
            # Ban user if they sent image within 60 seconds of joining the group chat
            userJoined = context.chat_data.get(user.id)
            if userJoined is not None:
                # compare the timestamps
                if userJoined > (datetime.now() - _new_member_grace_delta()):
                    logger.info(f"Banning {user.name} (user_id:  {user.id}) for spam.")
                    await chat.ban_member(user.id)
                    return

            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=topicID,
                text=f"Start a private chat with the @{config.botName} to send images in this chat."
            )
        except Exception as err:
            logger.error(f"Exception while handling potential spam image:\n{err}")
        finally:
            # Exit the function
            return
    
    memberID = member.get("member_id")
    minimumCommunityScore = (
        _runtime_int("telegram.minimum_community_score_private_image", 70)
        if chat.type == "private"
        else _runtime_int("telegram.minimum_community_score_group_image", 20)
    )
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    
    # Get account information
    if chat.type == "private":
        # Set the rolesAvailable
        rolesAvailable = member["roles"]
        # Get rate limits for private chat
        rateLimits = communityScore.getRateLimits(memberID, chat.type)

    elif chat.type == "group" or chat.type == "supergroup":
        community = communities.getCommunityByTelegramID(chat.id)
        if community is None:
            logger.warning(f"User {user.name} (user_id: {user.id}) attempted to send an image in an unregistered group chat.")
            try:
                await message.delete()
            except Exception as err:
                logger.error(f"Exception while deleting an image message in a group chat:\n{err}")
            finally:
                # Exit the function
                return
        
        # Combine the user and group roles into rolesAvailable
        rolesAvailable = set(member["roles"] + community["roles"])
        # Get rate limits for group chat
        rateLimits = communityScore.getRateLimits(memberID, "community") 
    else:
        # Only chat types allowed are private, group, and supergroup
        return
    
    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        memberUsage = usage.getUsageForMember(memberID)
        memberUsage = memberUsage if isinstance(memberUsage, list) else list()
        if len(memberUsage) >= rateLimits["image"]:
            try:
                if member["community_score"] < minimumCommunityScore:
                    logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to send images in a {chat.type} chat.")
                    await message.delete()
                    await context.bot.send_message(
                        chat_id=chat.id,
                        message_thread_id=topicID,
                        text=f"You need a minimum community score of {minimumCommunityScore} to send images in this chat."
                    )
                else:
                    logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
                    await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.error(f"Exception while handling image rate limits in a group chat:\n{err}")
            finally:
                # User has reach rate limit, exit function
                return

    # Convert images to base64
    photoFile = await message.effective_attachment[-1].get_file()
    photoBytes = await photoFile.download_as_bytearray()
    b64_photo = base64.b64encode(photoBytes)
    b64_string = b64_photo.decode()
    imageList = [b64_string]
    imagePrompt = "Describe this image." if not message.caption else message.caption

    messageData = {
        "chat_id": chat.id,
        "topic_id" : topicID,
        "message_id": message.message_id,
        "message_images": imageList,
        "message_text": imagePrompt
    }
    
    # Create the conversational agent instance
    imageAgent = ImageAgent(messageData, memberID)
    
    # Shows the bot as "typing"
    try:
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING,
            message_thread_id=topicID
        )
    except Exception as err:
        logger.error(f"Exception while sending a chat action for typing:\n{err}")
        # Non critical to the remaining functionality, continue
    
    response = await imageAgent.generateResponse()

    # Send the conversational agent response
    try:
        responseMessage = await context.bot.send_message(
            chat_id=chat.id,
            message_thread_id=topicID,
            text=f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. Responses may or may not be factually correct."
        )
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
    # TODO Handle adding prompt and response into the chat history collection.


async def otherGroupChat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Other messages (not directed at the chatbot) from group chats.")
    # Add messages to chat history
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    community = communities.getCommunityByTelegramID(chat.id)
    member = members.getMemberByTelegramID(user.id)
    score = 0 if member is None else member.get("community_score")
    userJoined = context.chat_data.get(user.id)

    minimumCommunityScore = _runtime_int("telegram.minimum_community_score_other_group", 20)
    spamDistanceThreshold = _runtime_float("conversation.spam_distance_threshold", 10.0)
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = list() if member is None else member.get("roles")
    
    # Check message for spam content
    if member is None or (score < minimumCommunityScore and not any(role in rolesAvailable for role in allowedRoles)):
        logger.info("Check message for spam.")
        results = spam.searchSpam(message.text)
        if len(results) > 0:
            distance = results[0].get("distance")
            if distance < spamDistanceThreshold:
                logger.warning("Spam message detected")
                try:
                    # Delete the message
                    await message.delete()

                    # Check how long the user has been in the chat
                    if userJoined is not None:
                        if userJoined > (datetime.now() - _new_member_grace_delta()):
                            logger.info(f"Banning {user.name} (user_id:  {user.id}) for spam.")
                            await chat.ban_member(user.id)
                            return
                    
                    # User not banned, send a requirement message
                    await context.bot.send_message(
                        chat_id=chat.id,
                        message_thread_id=topicID,
                        text=f"Potential spam message deleted."
                    )
                
                except Exception as error:
                    logger.error(f"Exception while handling potential spam message:\n{error}")
                finally:
                    return
    
    if community is not None and member is not None:
        # Update the chat history database with the newest message
        messageHistoryID = chatHistory.addChatHistory(
            messageID=message.message_id, 
            messageText=message.text, 
            platform="telegram", 
            memberID=member.get("member_id"), 
            communityID=community.get("community_id"), 
            topicID=topicID, 
            timestamp=datetime.now()
        )

        # Score the message if the user account exists and the message is above threshold
        if len(message.text.split(" ")) > _message_word_threshold():
            communityScore.scoreMessage(messageHistoryID)


# TODO only include chat history from the reply chain
async def replyToBot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """This function gets called when a user replies to a messages in a group (or supergroup) chat. 
    Replies are filtered to handle replies to the chatbot."""
    logger.info(f"Handle replies in a group (or supergroup) chat.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Get community information
    community = communities.getCommunityByTelegramID(chat.id)

    # Group chat must be registered
    if community is None:
        # This can only occur after an authorizzed user has began adding the chatbot to group chat, but before they are able to finish the registration process
        logger.info(f"User {user.name} (user_id: {user.id}) attempted to message the chatbot in an unregistered group chat.")
        try:
            await message.reply_text(text="Chatbot is not registered with this group chat.")
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        finally:
            return
    
    communityID = community.get("community_id")
    # Determine if the message is an actual reply or a regular message sent in a topic thread in a supergroup
    if (topicID and not message.reply_to_message.text) or (message.reply_to_message.text and message.reply_to_message.from_user.id != config.bot_id):
        logger.info(f"Message is not a reply to the bot but a message sent in a supergroup. Forward message to proper handler function.")
        # If the message is text, forward to other group chat
        try:
            if message.text:
                logger.info(f"Message is text type. Forward message to otherGroupChat function.")
                await otherGroupChat(update=update, context=context)
            elif message.effective_attachment is not None and type(message.effective_attachment) is tuple:
                logger.info(f"Message is image type. Forward message to handleImage function.")
                await handleImage(update=update, context=context)
        except Exception as err:
            logger.error(f"Exception while forwarding a reply update to the appropriate function:\n{err}")
        finally:
            return

    # Message is a reply to the chatbot
    
    # Get user account data
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        logger.warning(f"An unregistered user {user.name} (user_id: {user.id}) attempted to reply to a message from the chatbot in {community['chat_title']} group chat.")
        try:
            await message.reply_text(text=f"To reply to {config.botName}, open a private message to the chatbot @{config.botName} and click start.")
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")
        finally:
            # No user account, exit the function
            return
    
    memberID = member.get("member_id")
    
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]
    rolesAvailable = set(member["roles"] + community["roles"])
    
    if (not any(role in rolesAvailable for role in allowedRoles)):
        # User does not have permission for non rate limited chat
        # Check if user has exceeded hourly rate
        memberUsage = usage.getUsageForMember(memberID)
        memberUsage = memberUsage if isinstance(memberUsage, list) else list()
        rateLimits = communityScore.getRateLimits(memberID, "community")
        messageLimit = rateLimits.get("message", 0)
        if messageLimit > 0 and len(memberUsage) >= messageLimit:
            logger.info(f"User {user.name} (user_id: {user.id}) has reached their hourly message rate in {community['chat_title']} group chat.")
            try:
                await message.reply_text(text=f"You have reached your hourly rate limit.")
            except Exception as err:
                logger.error(f"Exception while replying to a telegram message:\n{err}")
            finally:
                # User has reach rate limit, exit function
                return

    messageData = {
        "community_id": communityID,
        "chat_host_id": communityID,
        "chat_type": "community",
        "member_id": memberID,
        "platform": "telegram",
        "topic_id" : topicID,
        "message_id": message.message_id,
        "message_text": message.text,
        **_message_temporal_context(message),
    }
    
    stageStatus = TelegramStageStatus(
        context,
        chat_id=chat.id,
        thread_id=topicID,
        reply_message=None,
        enabled=_runtime_bool("telegram.show_stage_progress", True),
        show_json_details=_runtime_bool("telegram.show_stage_json_details", True),
        detail_level=_runtime_str("telegram.stage_detail_level", "minimal"),
        update_min_interval_seconds=_runtime_float("telegram.stage_update_min_interval_seconds", 1.0),
    )
    await stageStatus.start()
    stageEvents: list[dict[str, Any]] = []

    async def _stage_callback(event: dict[str, Any]) -> None:
        _append_stage_event(stageEvents, event)
        await stageStatus.emit(event)

    # Create the conversational agent instance
    conversation = ConversationOrchestrator(
        message.text,
        memberID,
        messageData,
        message.message_id,
        options={"stage_callback": _stage_callback},
    )
    # Shows the bot as "typing"
    try:
        await context.bot.send_chat_action(
            chat_id=chat.id, 
            action=constants.ChatAction.TYPING,
            message_thread_id=topicID
        )
    except Exception as err:
        logger.error(f"Exception while sending a chat action for typing:\n{err}")
        # Non critical to the remaining functionality, continue

    try:
        response = await conversation.runAgents()
    except Exception as err:  # noqa: BLE001
        logger.error(f"Conversation orchestration failed while replying in group chat:\n{err}")
        await stageStatus.fail("I hit an internal error while processing that request.")
        return

    # Send the conversational agent response
    try:
        finalText = (
            f"{response}\n\n*Disclaimer*:  Test chatbots are prone to hallucination. "
            "Responses may or may not be factually correct."
        )
        responseMessage = await stageStatus.finalize(finalText)
        await _attach_run_inspector_controls(
            response_message=responseMessage,
            final_text=finalText,
            run_summary=conversation.run_summary,
            stage_events=stageEvents,
        )
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")
        # Critical to the remaining functionality, exit
        return

    # Score the message if the message is greater than the configured threshold
    if len(message.text.split(" ")) > _message_word_threshold():
        communityScore.scoreMessage(conversation.promptHistoryID)

    conversation.storeResponse(responseMessage.message_id)

    # Add the stats to the usage manager
    usage.addUsage(conversation.promptHistoryID, conversation.responseHistoryID, conversation.stats)

    # Check and send if send stats is enabled
    userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]

    if userDefaults.get("send_stats"):
        try:
            await sendStats(stats=conversation.stats, chatID=chat.id, context=context, threadID=topicID)
        except Exception as err:
            logger.error(f"Exception while calling sendStats:\n{err}")



#########################
# Other Update Handlers #
#########################

async def botStatusChanged(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Update handler for when the bot's status has changed.
    Use to register group chat accounts when bot is added to a group."""
    logger.info(f"The bot's status has been updated.")
    chat = update.effective_chat
    fromUser = update.my_chat_member.from_user
    newStatus = update.my_chat_member.new_chat_member

    # Get the group account information
    community = communities.getCommunityByTelegramID(chat.id)
    
    # Check if status change is being added to a group and if the group is not registered with the chatbot
    if newStatus.status == constants.ChatMemberStatus.MEMBER and community is None:
        logger.info(f"Chatbot has been added to a new group chat.")
        # Get the account for the user adding the chatbot to the group
        member = members.getMemberByTelegramID(fromUser.id)

        # Set the allowed roles
        allowedRoles = ["admin", "owner"]
        rolesAvailable = member.get("roles") if member else []

        if (not any(role in rolesAvailable for role in allowedRoles)):
            # User does not have permission to add the bot to the chat
            # Check if user exist
            if member is None:
                logger.warning(f"A non-registered user {fromUser.name} ({fromUser.id}) attempted to add the chatbot to the {chat.title} group chat.")
                responseText="Non registered user's are not authorized to add the chatbot to group chats. To register, start a private conversation with the chatbot."
            else:
                logger.warning(f"A non-authorized user {fromUser.name} ({fromUser.id}) attempted to add the chatbot to the {chat.title} group chat.")
                responseText = "You are not authorized to add the chatbot to group chats."
                
            try:
                # Send response
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=responseText
                )
                # Exit the group chat
                await context.bot.leave_chat(chat.id)
            except Exception as err:
                logger.error(f"Exception while removing chatbot from an unauthorized group:\n{err}")
            finally:
                # Exit the function
                return

        # Chat isn't registered and user is authorized
        logger.info(f"User {fromUser.name} ({fromUser.id}) is authorized to add chatbot to group chats.")

        newCommunityData = {
            "community_name": chat.title,
            "community_link": None,
            "roles": ["user"],
            "created_by": member.get("member_id"),
            "chat_id": chat.id,
            "chat_title": chat.title,
            "has_topics": True if (chat.type) == "supergroup" else False,
            "register_date": datetime.now()
        }

        # Add new individual account via accounts manager
        communities.addCommunityFromTelegram(newCommunityData)
        try:
            await context.bot.send_message(
                chat_id=chat.id,
                text=f"Hello, I am the {config.botName} chatbot. Use the /help command for more information."
            )
        except Exception as err:
            logger.error(f"Exception while sending a telegram message:\n{err}")
        finally:
            # Exit the function
            return
        
    elif newStatus.status != constants.ChatMemberStatus.MEMBER:
        logger.info(f"The chatbot's status change is:  {newStatus.status}.")
        return
        
    elif community is not None:
        logger.info(f"The chatbot has already been registered for this group.")
        return
    else:
        return


async def newChatUser(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Chat member status update, check for new user in group.")
    newUserData = update.chat_member.new_chat_member

    # TODO Update this logic to check for "status" key in the return value of difference, then check for second element of tuple for MEMBER
    print(update.chat_member.difference())

    if newUserData.status == constants.ChatMemberStatus.MEMBER:
        # Store a timestamp for when the telegram user joined the group
        context.chat_data[newUserData.user.id] = datetime.now()


async def linkHandler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("A message was sent with a link")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    member = members.getMemberByTelegramID(user.id)
    if member is None:
        logger.warning(f"Unregistered user {user.name} (user_id: {user.id}) sent a link in a {chat.type} chat.")
        try:
            await message.delete()
            # Ban user if they sent image within 60 seconds of joining the group chat
            userJoined = context.chat_data.get(user.id)
            if userJoined is not None:
                if userJoined > (datetime.now() - _new_member_grace_delta()):
                    logger.info(f"Banning {user.name} (user_id:  {user.id}) for spam.")
                    await chat.ban_member(user.id)
                    return

            # User not banned, send a requirement message
            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=topicID,
                text=f"Start a private chat with the @{config.botName} to send links in this chat."
            )
        except Exception as err:
            logger.error(f"Exception while handling potential spam (message containing a link):\n{err}")
        finally:
            # Exit the function
            return
        
    #memberID = member.get("member_id")
    minimumCommunityScore = _runtime_int("telegram.minimum_community_score_link", 20)
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]

    community = communities.getCommunityByTelegramID(chat.id)
    if community is None:
        # This should technically not be able to happen, but can due to DB error, etc
        logger.warning(f"User {user.name} (user_id: {user.id}) attempted to send an link in an unregistered group chat.")
        try:
            await message.delete()
        except Exception as err:
            logger.error(f"Exception while deleting a potential spam message (containing a link):\n{err}")
        finally:
            # Exit the function
            return
    
    # Combine the user and group roles into rolesAvailable
    rolesAvailable = set(member["roles"] + community["roles"])

    if (not any(role in rolesAvailable for role in allowedRoles)) and member["community_score"] < minimumCommunityScore:
        logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to send links in a {chat.type} chat.")
        try:
            await message.delete()
            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=topicID,
                text=f"You need a minimum community score of {minimumCommunityScore} to send links in the {community.get("community_name")} chat."
            )
        except Exception as err:
            logger.error(f"Exception while handling potential spam message (containing a link):\n{err}")
        finally:
            return


####################
# Helper Functions #
####################

def pairs(l):
    for i in range(0, len(l), 2):
        yield l[i:i + 2]

async def setPassword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User is requesting an new access key.")
    #chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    #topicID = message.message_thread_id if message.is_topic_message else None

    member = members.getMemberByTelegramID(user.id)
    if member is not None:
        memberID = member.get("member_id")
        if len(context.args) == 1:
            password = context.args[0]
            # Do regular expression
            minPasswordLength = _password_min_length()
            pattern = re.compile(
                r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[-+_!@#$%^&*.,?]).{"
                + str(minPasswordLength)
                + r",}"
            )
            validPassword = pattern.search(password)
            if validPassword:
                #password = accounts.setPassword(user.id, password=password)
                password = members.setPassword(memberID, password)
                response = f"Your password {password} has been stored"
            else:
                response = (
                    "You must enter a valid password.\n"
                    f"Hint: passwords must be at least {minPasswordLength} characters, "
                    "contain at least one lowercase, uppercase, digit, and symbol."
                )

        else:
            # Generate a random password
            password = members.setPassword(memberID)
            if password:
                response = f"Your randomly generated password is:  {password}"
            else:
                response = "Something went wrong."
        try:
            await message.reply_text(text=response)
        except Exception as err:
            logger.error(f"Exception while replying to a telegram message:\n{err}")


async def statisticsManager(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # For now this function will just toggle on or off the sending of statistics
    logger.info("Toggling the statistics message")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    member = members.getMemberByTelegramID(user.id)

    if member is not None:
        userDefaults = context.user_data["defaults"] = {} if "defaults" not in context.user_data else context.user_data["defaults"]
        userDefaults["send_stats"] = True if "send_stats" not in userDefaults else not userDefaults["send_stats"]

        try:
            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=message.message_thread_id if message.is_topic_message else None,
                text=f"Show statistics:  {userDefaults['send_stats']}"
            )
        except Exception as err:
            logger.error(f"Exception while sending a telegram message:\n{err}")


async def sendStats(chatID: int, context: ContextTypes.DEFAULT_TYPE, stats: dict, threadID=None) -> None:
    logger.info(f"Send stats for previously generated message.")
    messageText = f"""_Statistics for previous message:_
*Load duration:*  {(stats['load_duration'] / 1000000000):.2f} seconds

*Prompt tokens:*  {stats['prompt_eval_count']} 
*Prompt tokens per second:*  {(stats['prompt_eval_count'] / (stats['prompt_eval_duration'] / 1000000000)):.2f} 

*Response tokens:*  {stats['eval_count']} 
*Response tokens per second:*  {(stats['eval_count'] / (stats['eval_duration'] / 1000000000)):.2f} 

*Total tokens per second:*  {((stats['prompt_eval_count'] + stats['eval_count']) / ((stats['prompt_eval_duration'] + stats['eval_duration']) / 1000000000)):.2f} 
*Total duration:*  {(stats['total_duration'] / 1000000000):.2f} seconds """
    
    try:
        await context.bot.send_message(
            chat_id=chatID,
            message_thread_id=threadID,
            parse_mode=constants.ParseMode.MARKDOWN,
            text=messageText
        )
    except Exception as err:
        logger.error(f"Exception while sending a telegram message:\n{err}")


async def reactionsHandler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Handling message reactions.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user

    # Temporarily only accepting 👍 emoji
    reaction = update.message_reaction.new_reaction[0]
    member = members.getMemberByTelegramID(user.id)
    community = communities.getCommunityByTelegramID(chat.id)
    if member is None or reaction is None or community is None:
        return
    
    memberID = member.get("member_id")
    communityID = community.get("community_id")

    if reaction.emoji == "👍":
        logger.info(f"Registered user {user.name} (user_id: {user.id}) reacted to a message with the 👍 emoji. Process community score.")

        # Get the history ID
        originalMessage = chatHistory.getMessageByMessageID(communityID, "community", "telegram", update.message_reaction.message_id)
        if originalMessage:
            communityScore.scoreMessageFromReaction(memberID, originalMessage.get("history_id"))


async def handleForwardedMessage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """This function is for spam prevention. 
    It is called when forwarded messages are sent in a group chat."""
    logger.info(f"Handle forwarded messages.")
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    topicID = message.message_thread_id if message.is_topic_message else None

    # Prevent spam abuse
    member = members.getMemberByTelegramID(user.id)

    if member is None:
        # Delete the message and give the user information to message the chatbot
        logger.warning(f"An unregistered user {user.name} (user_id: {user.id}) attempted to forward a message into the group chat.")

        try:
            await message.delete()
            # Ban user if they sent image within 60 seconds of joining the group chat
            userJoined = context.chat_data.get(user.id)
            if userJoined is not None:
                # compare the timestamps
                if userJoined > (datetime.now() - _new_member_grace_delta()):
                    logger.info(f"Banning {user.name} (user_id:  {user.id}) for spam.")
                    await chat.ban_member(user.id)
                    return
            
            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=topicID,
                text=f"Start a private chat with the @{config.botName} to forward messages to this chat."
            )
        except Exception as err:
            logger.error(f"Exception while handling potential spam message (forwarded message):\n{err}")
        finally:
            return
    
    minimumCommunityScore = _runtime_int("telegram.minimum_community_score_forward", 20)
    # Set the allowed roles
    allowedRoles = ["tester", "marketing", "admin", "owner"]

    community = communities.getCommunityByTelegramID(chat.id)
    if community is None:
        # This should technically not be able to happen, but can due to DB error, etc
        logger.warning(f"User {user.name} (user_id: {user.id}) attempted to forward a message in an unregistered group chat.")
        try:
            await message.delete()
        except Exception as err:
            logger.error(f"Exception while deleting a potential spam message (forwarded message):\n{err}")
        finally:
            # Exit the function
            return
    
    # Combine the user and group roles into rolesAvailable
    rolesAvailable = set(member["roles"] + community["roles"])

    if (not any(role in rolesAvailable for role in allowedRoles)) and member["community_score"] < minimumCommunityScore:
        logger.info(f"User {user.name} (user_id: {user.id}) does not meet the minimum community score to forward messages in a {chat.type} chat.")
        try:
            await message.delete()
            await context.bot.send_message(
                chat_id=chat.id,
                message_thread_id=topicID,
                text=f"You need a minimum community score of {minimumCommunityScore} to forward messages in the {community.get("community_name")} chat."
            )
        except Exception as err:
            logger.error(f"Exception while handling potential spam message (forwarded message):\n{err}")
        finally:
            return


async def errorHandler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update:\n{context.error}")



#################
# Main Function #
#################

def main() -> None:
    logger.info("RYO - begin telegram ui application.")
    telegramConfigIssues = config.getTelegramConfigIssues()
    if telegramConfigIssues:
        logger.error(
            "Telegram startup blocked: missing/invalid config values: %s",
            ", ".join(telegramConfigIssues),
        )
        logger.error("Run setup wizard to fix Telegram config: python3 scripts/setup_wizard.py --telegram-only")
        return

    # Run the bot
    # Create the Application and pass it your bot's token.
    # .get_updates_write_timeout(100)
    application = (
        Application.builder()
        .token(config.bot_token)
        .concurrent_updates(True)
        .get_updates_write_timeout(_runtime_int("telegram.get_updates_write_timeout", 500))
        .build()
    )

    # Generate command chain
    generateHandler = ConversationHandler(
        entry_points=[CommandHandler("generate", beginGenerate)],
        states={
            SET_SYSTEM_PROMPT : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setSystemPrompt),
                CommandHandler("skip", skip_systemPrompt)
            ],
            SET_PROMPT : [MessageHandler(filters.TEXT & ~filters.COMMAND, setPrompt)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Knowledge command chain
    knowledgeHandler = ConversationHandler(
        entry_points=[CommandHandler("knowledge", knowledgeManger)],
        states={
            HANDLE_KNOWLEDGE_TYPE : [
                CallbackQueryHandler(setKnowledgeType, pattern="^(private|public)$")
            ],
            HANDLE_KNOWLEDGE_TEXT : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeText)
            ],
            HANDLE_KNOWLEDGE_SOURCE : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeSource),
                CommandHandler("skip", skip_knowledgeSource)
            ],
            HANDLE_KNOWLEDGE_CATEGORY : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setKnowledgeCategories),
                CommandHandler("skip", skip_knowledgeCategories)
            ],
            STORE_KNOWLEDGE : [
                CallbackQueryHandler(finalizeKnowledge, pattern="^(yes|no)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Promote account command chain
    promoteHandler = ConversationHandler(
        entry_points=[
            CommandHandler("promote", promoteAccount)
        ],
        states={
            VERIFY_PROMOTE : [
                CallbackQueryHandler(setNewRole, pattern="^(admin|marketing|tester)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Tweet command chain
    tweetHandler = ConversationHandler(
        entry_points=[
            CommandHandler("tweet", tweetStart)
        ],
        states={
            CONFIRM_TWEET : [
                CallbackQueryHandler(confirmTweet, pattern="^(confirm|modify|reject)$")
            ],
            MODIFY_TWEET : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, modifyTweet)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Newsletter command chain
    # Get list of account types from account manager
    rolesList = members.rolesList
    newsletterHandler = ConversationHandler(
        entry_points=[
            CommandHandler("newsletter", newsletterStart)
        ],
        states={
            ROLE_SELECTION : [
                CallbackQueryHandler(selectRole, pattern=f"^({'|'.join(rolesList)})$"),
                CallbackQueryHandler(roleSelectionDone, pattern=f"^done$")
            ],
            PHOTO_OPTION : [
                CallbackQueryHandler(photoOption, pattern=f"^(yes|no)$")
            ],
            ADD_PHOTO : [
                MessageHandler(filters.PHOTO, addNewsletterPhoto)
            ],
            COMPOSE_NEWLETTER : [
                MessageHandler(filters.TEXT & ~filters.COMMAND, addNewsletterText)
            ],
            CONFIRM_NEWSLETTER : [
                CallbackQueryHandler(confirmNewsletter, pattern=f"^(yes|no)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Proposals command chain
    proposalsHandler = ConversationHandler(
        entry_points=[
            CommandHandler("proposals", proposalsManager, filters=filters.ChatType.PRIVATE)
        ],
        states={
            SELECT_PROPOSAL : [
                CallbackQueryHandler(agreeNDA)
            ],
            PROPOSAL_NDA : [
                CallbackQueryHandler(openProposal, pattern="^(confirm|reject)$")
            ],
            SHOW_PROPOSAL : [
                CallbackQueryHandler(openProposal, pattern="^(done)$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Add command handlers
    application.add_handler(CommandHandler("start", startBot, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("dashboard", dashboard, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("botid", botID))
    application.add_handler(CommandHandler("userid", userID))
    application.add_handler(CommandHandler("statistics", statisticsManager))
    application.add_handler(CommandHandler("password", setPassword, filters=filters.ChatType.PRIVATE))
    application.add_handler(CallbackQueryHandler(runInspectorView, pattern=r"^diag:"))

    application.add_handler(MessageReactionHandler(reactionsHandler))
    # Add spam checks
    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & filters.FORWARDED,
            _guard_message_handler(handleForwardedMessage),
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & (filters.Entity("url") | filters.Entity("text_link")),
            _guard_message_handler(linkHandler),
        )
    )
    
    # Add conversational chains
    application.add_handlers([generateHandler, knowledgeHandler, newsletterHandler, promoteHandler, proposalsHandler, tweetHandler])


    # Add message handlers
    application.add_handler(
        MessageHandler(
            filters.Mention(config.botName) & filters.ChatType.GROUPS & ~filters.COMMAND,
            _guard_message_handler(directChatGroup),
        )
    )
    application.add_handler(
        MessageHandler(
            filters.REPLY & filters.ChatType.GROUPS & ~filters.COMMAND,
            _guard_message_handler(replyToBot),
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            _guard_message_handler(directChatPrivate),
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
            _guard_message_handler(otherGroupChat),
        )
    )
    application.add_handler(MessageHandler(filters.PHOTO, _guard_message_handler(handleImage)))
    application.add_handler(MessageHandler(filters.ALL, catchAllMessages))

    # Other update type handlers
    application.add_handler(ChatMemberHandler(botStatusChanged, chat_member_types=ChatMemberHandler.MY_CHAT_MEMBER))
    application.add_handler(ChatMemberHandler(newChatUser, chat_member_types=ChatMemberHandler.CHAT_MEMBER))

    application.add_error_handler(errorHandler)

    # Run the bot until the user presses Ctrl-C
    # poll_interval=5, bootstrap_retries=3, timeout=50
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
    
