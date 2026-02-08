##########################################################################
#                                                                        #
#  Central runtime settings hydration for config.json + .env             #
#                                                                        #
##########################################################################

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RUNTIME_SETTINGS: dict[str, Any] = {
    "inference": {
        "default_ollama_host": "http://127.0.0.1:11434",
        "default_embedding_model": "nomic-embed-text:latest",
        "default_generate_model": "llama3.2:latest",
        "default_chat_model": "llama3.2:latest",
        "default_tool_model": "llama3.2:latest",
        "default_multimodal_model": "llama3.2-vision:latest",
        "model_context_window": 4096,
        "probe_timeout_seconds": 3.0,
    },
    "database": {
        "default_primary_host": "127.0.0.1",
        "default_primary_port": "5432",
        "default_fallback_host": "127.0.0.1",
        "default_fallback_port": "5433",
        "connect_timeout_seconds": 2,
        "bootstrap_connect_retries": 15,
        "bootstrap_retry_delay_seconds": 1.0,
        "bootstrap_port_scan_limit": 100,
    },
    "tool_runtime": {
        "default_timeout_seconds": 8.0,
        "default_max_retries": 1,
        "brave_timeout_seconds": 10.0,
        "chat_history_timeout_seconds": 6.0,
        "knowledge_timeout_seconds": 6.0,
        "skip_tools_timeout_seconds": 2.0,
    },
    "telegram": {
        "minimum_community_score_private_chat": 50,
        "minimum_community_score_private_image": 70,
        "minimum_community_score_group_image": 20,
        "minimum_community_score_other_group": 20,
        "minimum_community_score_link": 20,
        "minimum_community_score_forward": 20,
        "get_updates_write_timeout": 500,
    },
    "security": {
        "password_min_length": 12,
    },
    "vectors": {
        "embedding_dimensions": 768,
    },
    "retrieval": {
        "conversation_short_history_limit": 20,
        "chat_history_window_hours": 12,
        "chat_history_default_limit": 1,
        "chat_history_sender_window_hours": 12,
        "chat_history_sender_default_limit": 0,
        "knowledge_list_limit": 10,
        "knowledge_search_default_limit": 1,
        "spam_list_limit": 10,
        "spam_search_default_limit": 1,
    },
    "conversation": {
        "knowledge_lookup_word_threshold": 6,
        "knowledge_lookup_result_limit": 2,
        "community_score_message_word_threshold": 20,
        "spam_distance_threshold": 10,
        "new_member_grace_period_seconds": 60,
    },
    "community": {
        "score_rules": [
            {
                "private": {"min": 50},
                "community": {"min": 0},
                "message_per_hour": 2,
                "image_per_hour": 0,
            },
            {
                "private": {"min": 55},
                "community": {"min": 5},
                "message_per_hour": 4,
                "image_per_hour": 0,
            },
            {
                "private": {"min": 60},
                "community": {"min": 10},
                "message_per_hour": 6,
                "image_per_hour": 0,
            },
            {
                "private": {"min": 65},
                "community": {"min": 15},
                "message_per_hour": 8,
                "image_per_hour": 0,
            },
            {
                "private": {"min": 70},
                "community": {"min": 20},
                "message_per_hour": 10,
                "image_per_hour": 1,
            },
            {
                "private": {"min": 75},
                "community": {"min": 25},
                "message_per_hour": 12,
                "image_per_hour": 1,
            },
        ],
    },
    "watchdog": {
        "restart_window_seconds": 60,
        "max_restarts_per_window": 5,
        "terminate_timeout_seconds": 8,
        "kill_timeout_seconds": 4,
        "thread_join_timeout_seconds": 2,
    },
}


ENV_OVERRIDES: dict[str, tuple[tuple[str, ...], str]] = {
    "inference.default_ollama_host": (("RYO_DEFAULT_OLLAMA_HOST", "OLLAMA_HOST"), "str"),
    "inference.default_embedding_model": (("RYO_DEFAULT_EMBEDDING_MODEL", "OLLAMA_EMBED_MODEL"), "str"),
    "inference.default_generate_model": (("RYO_DEFAULT_GENERATE_MODEL", "OLLAMA_GENERATE_MODEL"), "str"),
    "inference.default_chat_model": (("RYO_DEFAULT_CHAT_MODEL", "OLLAMA_CHAT_MODEL"), "str"),
    "inference.default_tool_model": (("RYO_DEFAULT_TOOL_MODEL", "OLLAMA_TOOL_MODEL"), "str"),
    "inference.default_multimodal_model": (("RYO_DEFAULT_MULTIMODAL_MODEL", "OLLAMA_MULTIMODAL_MODEL"), "str"),
    "inference.model_context_window": (("RYO_INFERENCE_CONTEXT_WINDOW",), "int"),
    "inference.probe_timeout_seconds": (("RYO_OLLAMA_PROBE_TIMEOUT_SECONDS",), "float"),
    "database.default_primary_host": (("RYO_DB_DEFAULT_PRIMARY_HOST", "POSTGRES_HOST"), "str"),
    "database.default_primary_port": (("RYO_DB_DEFAULT_PRIMARY_PORT", "POSTGRES_PORT"), "str"),
    "database.default_fallback_host": (("RYO_DB_DEFAULT_FALLBACK_HOST", "POSTGRES_FALLBACK_HOST"), "str"),
    "database.default_fallback_port": (("RYO_DB_DEFAULT_FALLBACK_PORT", "POSTGRES_FALLBACK_PORT"), "str"),
    "database.connect_timeout_seconds": (("RYO_DB_CONNECT_TIMEOUT_SECONDS",), "int"),
    "database.bootstrap_connect_retries": (("RYO_BOOTSTRAP_CONNECT_RETRIES",), "int"),
    "database.bootstrap_retry_delay_seconds": (("RYO_BOOTSTRAP_RETRY_DELAY_SECONDS",), "float"),
    "database.bootstrap_port_scan_limit": (("RYO_BOOTSTRAP_PORT_SCAN_LIMIT",), "int"),
    "tool_runtime.default_timeout_seconds": (("RYO_TOOL_RUNTIME_DEFAULT_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.default_max_retries": (("RYO_TOOL_RUNTIME_DEFAULT_MAX_RETRIES",), "int"),
    "tool_runtime.brave_timeout_seconds": (("RYO_TOOL_RUNTIME_BRAVE_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.chat_history_timeout_seconds": (("RYO_TOOL_RUNTIME_CHAT_HISTORY_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.knowledge_timeout_seconds": (("RYO_TOOL_RUNTIME_KNOWLEDGE_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.skip_tools_timeout_seconds": (("RYO_TOOL_RUNTIME_SKIP_TOOLS_TIMEOUT_SECONDS",), "float"),
    "telegram.minimum_community_score_private_chat": (("RYO_TELEGRAM_MIN_SCORE_PRIVATE_CHAT",), "int"),
    "telegram.minimum_community_score_private_image": (("RYO_TELEGRAM_MIN_SCORE_PRIVATE_IMAGE",), "int"),
    "telegram.minimum_community_score_group_image": (("RYO_TELEGRAM_MIN_SCORE_GROUP_IMAGE",), "int"),
    "telegram.minimum_community_score_other_group": (("RYO_TELEGRAM_MIN_SCORE_OTHER_GROUP",), "int"),
    "telegram.minimum_community_score_link": (("RYO_TELEGRAM_MIN_SCORE_LINK",), "int"),
    "telegram.minimum_community_score_forward": (("RYO_TELEGRAM_MIN_SCORE_FORWARD",), "int"),
    "telegram.get_updates_write_timeout": (("RYO_TELEGRAM_GET_UPDATES_WRITE_TIMEOUT",), "int"),
    "security.password_min_length": (("RYO_PASSWORD_MIN_LENGTH",), "int"),
    "vectors.embedding_dimensions": (("RYO_VECTOR_DIMENSIONS",), "int"),
    "retrieval.conversation_short_history_limit": (("RYO_RETRIEVAL_SHORT_HISTORY_LIMIT",), "int"),
    "retrieval.chat_history_window_hours": (("RYO_CHAT_HISTORY_WINDOW_HOURS",), "int"),
    "retrieval.chat_history_default_limit": (("RYO_CHAT_HISTORY_DEFAULT_LIMIT",), "int"),
    "retrieval.chat_history_sender_window_hours": (("RYO_CHAT_HISTORY_SENDER_WINDOW_HOURS",), "int"),
    "retrieval.chat_history_sender_default_limit": (("RYO_CHAT_HISTORY_SENDER_DEFAULT_LIMIT",), "int"),
    "retrieval.knowledge_list_limit": (("RYO_KNOWLEDGE_LIST_LIMIT",), "int"),
    "retrieval.knowledge_search_default_limit": (("RYO_KNOWLEDGE_SEARCH_DEFAULT_LIMIT",), "int"),
    "retrieval.spam_list_limit": (("RYO_SPAM_LIST_LIMIT",), "int"),
    "retrieval.spam_search_default_limit": (("RYO_SPAM_SEARCH_DEFAULT_LIMIT",), "int"),
    "conversation.knowledge_lookup_word_threshold": (("RYO_CONVERSATION_KNOWLEDGE_WORD_THRESHOLD",), "int"),
    "conversation.knowledge_lookup_result_limit": (("RYO_CONVERSATION_KNOWLEDGE_RESULT_LIMIT",), "int"),
    "conversation.community_score_message_word_threshold": (("RYO_CONVERSATION_SCORE_WORD_THRESHOLD",), "int"),
    "conversation.spam_distance_threshold": (("RYO_SPAM_DISTANCE_THRESHOLD",), "float"),
    "conversation.new_member_grace_period_seconds": (("RYO_NEW_MEMBER_GRACE_PERIOD_SECONDS",), "int"),
    "watchdog.restart_window_seconds": (("RYO_WATCHDOG_RESTART_WINDOW_SECONDS",), "int"),
    "watchdog.max_restarts_per_window": (("RYO_WATCHDOG_MAX_RESTARTS_PER_WINDOW",), "int"),
    "watchdog.terminate_timeout_seconds": (("RYO_WATCHDOG_TERMINATE_TIMEOUT_SECONDS",), "float"),
    "watchdog.kill_timeout_seconds": (("RYO_WATCHDOG_KILL_TIMEOUT_SECONDS",), "float"),
    "watchdog.thread_join_timeout_seconds": (("RYO_WATCHDOG_THREAD_JOIN_TIMEOUT_SECONDS",), "float"),
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_env_value(raw_value: str, value_type: str) -> Any:
    if value_type == "str":
        return raw_value
    if value_type == "int":
        return int(raw_value)
    if value_type == "float":
        return float(raw_value)
    if value_type == "bool":
        return _parse_bool(raw_value)
    return raw_value


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _path_parts(path: str) -> list[str]:
    return [part for part in path.split(".") if part]


def get_runtime_setting(settings: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = settings
    for part in _path_parts(path):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor.get(part)
    return cursor


def set_runtime_setting(settings: dict[str, Any], path: str, value: Any) -> None:
    parts = _path_parts(path)
    if not parts:
        return
    cursor: dict[str, Any] = settings
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def load_dotenv_file(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    dotenv_path = Path(path)
    loaded: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded


def build_runtime_settings(
    config_data: dict[str, Any] | None = None,
    env_data: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_RUNTIME_SETTINGS)
    if isinstance(config_data, dict):
        runtime_config = config_data.get("runtime")
        if isinstance(runtime_config, dict):
            _deep_merge(settings, runtime_config)

    env_values = env_data if env_data is not None else os.environ
    for path, (env_keys, value_type) in ENV_OVERRIDES.items():
        raw_value = None
        for env_key in env_keys:
            raw_candidate = env_values.get(env_key)
            if raw_candidate is None or str(raw_candidate).strip() == "":
                continue
            raw_value = raw_candidate
            break
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            parsed = _coerce_env_value(str(raw_value).strip(), value_type)
        except (TypeError, ValueError):
            continue
        set_runtime_setting(settings, path, parsed)

    return settings
