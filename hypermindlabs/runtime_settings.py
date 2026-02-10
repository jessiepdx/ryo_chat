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
        "default_embedding_model": "",
        "default_generate_model": "",
        "default_chat_model": "",
        "default_tool_model": "",
        "default_multimodal_model": "",
        "tool_capable_models": [],
        "model_context_window": 4096,
        "probe_timeout_seconds": 3.0,
        "embedding_timeout_seconds": 12.0,
        "stream_first_token_timeout_seconds": 25.0,
        "stream_chunk_timeout_seconds": 45.0,
        "stream_total_timeout_seconds": 240.0,
        "embedding_max_input_chars": 6000,
        "defer_embeddings_on_write": True,
        "embedding_write_queue_size": 512,
        "prompt_model_selection_on_startup": False,
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
        "auto_bootstrap_on_app_start": True,
        "auto_bootstrap_use_docker": True,
        "auto_migrate_on_app_start": True,
    },
    "tool_runtime": {
        "default_timeout_seconds": 8.0,
        "default_max_retries": 1,
        "auto_model_retry_candidate_limit": 6,
        "pseudo_tool_candidate_limit": 6,
        "enforce_native_tool_capability": True,
        "tool_capability_probe_enabled": True,
        "tool_capability_probe_timeout_seconds": 3.0,
        "tool_capability_probe_cache_ttl_seconds": 21600.0,
        "tool_capability_probe_max_models": 3,
        "brave_timeout_seconds": 10.0,
        "curl_timeout_seconds": 12.0,
        "curl_max_response_chars": 3500,
        "curl_allowlist_domains": [],
        "curl_allow_private_network": False,
        "curl_allow_mutating_methods": False,
        "chat_history_timeout_seconds": 6.0,
        "knowledge_timeout_seconds": 6.0,
        "skip_tools_timeout_seconds": 2.0,
        "enable_human_approval": True,
        "default_approval_timeout_seconds": 45.0,
        "approval_poll_interval_seconds": 0.25,
        "default_dry_run": False,
        "sandbox": {
            "enabled": True,
            "side_effect_class": "read_only",
            "require_approval": False,
            "dry_run": False,
            "approval_timeout_seconds": 45.0,
            "execution_timeout_ceiling": 30.0,
            "max_memory_mb": 512,
            "network": {
                "enabled": True,
                "allowlist_domains": [],
            },
            "filesystem": {
                "mode": "none",
                "allowed_paths": [],
            },
        },
    },
    "telegram": {
        "minimum_community_score_private_chat": 50,
        "minimum_community_score_private_image": 70,
        "minimum_community_score_group_image": 20,
        "minimum_community_score_other_group": 20,
        "minimum_community_score_link": 20,
        "minimum_community_score_forward": 20,
        "admin_claim_enabled": True,
        "admin_claim_accept_raw_message": True,
        "admin_config_commands_enabled": True,
        "group_context_isolation_enabled": True,
        "group_storage_mode": "shared_pg",
        "group_storage_prefix": "community",
        "get_updates_write_timeout": 500,
        "message_guard_completed_ttl_seconds": 180,
        "show_stage_progress": True,
        "show_stage_json_details": False,
        "stage_detail_level": "normal",
        "stage_update_min_interval_seconds": 1.0,
    },
    "orchestrator": {
        "fast_path_small_talk_enabled": True,
        "fast_path_small_talk_max_chars": 96,
        "analysis_bypass_small_talk_enabled": False,
        "analysis_history_limit": 8,
        "analysis_history_limit_on_switch": 4,
        "analysis_history_limit_small_talk": 2,
        "tool_history_limit": 6,
        "response_history_limit": 10,
        "auto_expand_tool_stage_enabled": True,
        "auto_expand_max_rounds": 1,
        "progress_stage_events_enabled": False,
        "analysis_max_output_tokens": 256,
        "analysis_temperature": 0.1,
        "analysis_context_summary_max_chars": 220,
        "control_plane_cache_enabled": True,
        "control_plane_cache_ttl_seconds": 60.0,
        "discovery_layer_enabled": True,
        "discovery_force_tools_on_uncertainty": True,
        "discovery_unknown_threshold": 0.67,
        "discovery_probably_unknown_threshold": 0.4,
        "discovery_default_tool_hints": ["braveSearch", "curlRequest"],
    },
    "personality": {
        "enabled": True,
        "adaptive_enabled": True,
        "narrative_enabled": True,
        "max_injection_chars": 900,
        "narrative_summary_max_chars": 360,
        "narrative_source_history_limit": 8,
        "rollup_turn_threshold": 8,
        "rollup_char_threshold": 2200,
        "max_active_chunks": 6,
        "adaptation_min_turns": 4,
        "adaptation_max_step_per_window": 1,
        "adaptation_window_turns": 4,
        "verbosity_short_token_threshold": 10,
        "verbosity_long_token_threshold": 36,
        "default_tone": "friendly",
        "default_verbosity": "brief",
        "default_reading_level": "moderate",
        "default_format": "plain",
        "default_humor": "low",
        "default_emoji": "off",
        "default_language": "en",
    },
    "temporal": {
        "enabled": True,
        "default_timezone": "UTC",
        "history_limit": 20,
        "excerpt_max_chars": 160,
    },
    "topic_shift": {
        "enabled": True,
        "min_token_count": 3,
        "keyword_terms": 4,
        "jaccard_switch_threshold": 0.18,
        "recent_user_messages": 6,
        "deweight_history_search_on_switch": True,
        "history_window_hours_on_switch": 2,
        "trim_temporal_history_on_small_talk": True,
        "small_talk_history_limit": 2,
        "allow_recent_topic_fallback": False,
        "history_min_overlap_default": 0.08,
        "history_min_overlap_on_switch": 0.22,
        "low_token_switch_enabled": True,
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
        "progressive_history_enabled": True,
        "progressive_history_tool_enabled": True,
        "progressive_history_max_rounds": 5,
        "progressive_history_round_windows_hours": [12, 48, 168, 720],
        "progressive_history_semantic_limit_start": 3,
        "progressive_history_semantic_limit_step": 2,
        "progressive_history_timeline_limit_start": 24,
        "progressive_history_timeline_limit_step": 24,
        "progressive_history_context_radius": 2,
        "progressive_history_match_threshold": 0.42,
        "progressive_history_max_selected": 8,
        "progressive_history_max_message_chars": 220,
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
        "auto_start_routes": True,
    },
    "web": {
        "host": "127.0.0.1",
        "port": 4747,
        "port_scan_limit": 100,
        "debug": False,
        "use_reloader": False,
    },
}


ENV_OVERRIDES: dict[str, tuple[tuple[str, ...], str]] = {
    "inference.default_ollama_host": (("RYO_DEFAULT_OLLAMA_HOST", "OLLAMA_HOST"), "str"),
    "inference.default_embedding_model": (("RYO_DEFAULT_EMBEDDING_MODEL", "OLLAMA_EMBED_MODEL"), "str"),
    "inference.default_generate_model": (("RYO_DEFAULT_GENERATE_MODEL", "OLLAMA_GENERATE_MODEL"), "str"),
    "inference.default_chat_model": (("RYO_DEFAULT_CHAT_MODEL", "OLLAMA_CHAT_MODEL"), "str"),
    "inference.default_tool_model": (("RYO_DEFAULT_TOOL_MODEL", "OLLAMA_TOOL_MODEL"), "str"),
    "inference.default_multimodal_model": (("RYO_DEFAULT_MULTIMODAL_MODEL", "OLLAMA_MULTIMODAL_MODEL"), "str"),
    "inference.tool_capable_models": (("RYO_TOOL_CAPABLE_MODELS",), "csv"),
    "inference.model_context_window": (("RYO_INFERENCE_CONTEXT_WINDOW",), "int"),
    "inference.probe_timeout_seconds": (("RYO_OLLAMA_PROBE_TIMEOUT_SECONDS",), "float"),
    "inference.embedding_timeout_seconds": (("RYO_OLLAMA_EMBED_TIMEOUT_SECONDS",), "float"),
    "inference.stream_first_token_timeout_seconds": (("RYO_OLLAMA_STREAM_FIRST_TOKEN_TIMEOUT_SECONDS",), "float"),
    "inference.stream_chunk_timeout_seconds": (("RYO_OLLAMA_STREAM_CHUNK_TIMEOUT_SECONDS",), "float"),
    "inference.stream_total_timeout_seconds": (("RYO_OLLAMA_STREAM_TOTAL_TIMEOUT_SECONDS",), "float"),
    "inference.embedding_max_input_chars": (("RYO_EMBEDDING_MAX_INPUT_CHARS",), "int"),
    "inference.defer_embeddings_on_write": (("RYO_DEFER_EMBEDDINGS_ON_WRITE",), "bool"),
    "inference.embedding_write_queue_size": (("RYO_EMBEDDING_WRITE_QUEUE_SIZE",), "int"),
    "inference.prompt_model_selection_on_startup": (("RYO_PROMPT_MODEL_SELECTION_ON_STARTUP",), "bool"),
    "database.default_primary_host": (("RYO_DB_DEFAULT_PRIMARY_HOST", "POSTGRES_HOST"), "str"),
    "database.default_primary_port": (("RYO_DB_DEFAULT_PRIMARY_PORT", "POSTGRES_PORT"), "str"),
    "database.default_fallback_host": (("RYO_DB_DEFAULT_FALLBACK_HOST", "POSTGRES_FALLBACK_HOST"), "str"),
    "database.default_fallback_port": (("RYO_DB_DEFAULT_FALLBACK_PORT", "POSTGRES_FALLBACK_PORT"), "str"),
    "database.connect_timeout_seconds": (("RYO_DB_CONNECT_TIMEOUT_SECONDS",), "int"),
    "database.bootstrap_connect_retries": (("RYO_BOOTSTRAP_CONNECT_RETRIES",), "int"),
    "database.bootstrap_retry_delay_seconds": (("RYO_BOOTSTRAP_RETRY_DELAY_SECONDS",), "float"),
    "database.bootstrap_port_scan_limit": (("RYO_BOOTSTRAP_PORT_SCAN_LIMIT",), "int"),
    "database.auto_bootstrap_on_app_start": (("RYO_DB_AUTO_BOOTSTRAP_ON_START",), "bool"),
    "database.auto_bootstrap_use_docker": (("RYO_DB_AUTO_BOOTSTRAP_USE_DOCKER",), "bool"),
    "database.auto_migrate_on_app_start": (("RYO_DB_AUTO_MIGRATE_ON_START",), "bool"),
    "tool_runtime.default_timeout_seconds": (("RYO_TOOL_RUNTIME_DEFAULT_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.default_max_retries": (("RYO_TOOL_RUNTIME_DEFAULT_MAX_RETRIES",), "int"),
    "tool_runtime.auto_model_retry_candidate_limit": (("RYO_TOOL_RUNTIME_AUTO_MODEL_RETRY_CANDIDATE_LIMIT",), "int"),
    "tool_runtime.pseudo_tool_candidate_limit": (("RYO_TOOL_RUNTIME_PSEUDO_TOOL_CANDIDATE_LIMIT",), "int"),
    "tool_runtime.enforce_native_tool_capability": (("RYO_TOOL_RUNTIME_ENFORCE_NATIVE_CAPABILITY",), "bool"),
    "tool_runtime.tool_capability_probe_enabled": (("RYO_TOOL_RUNTIME_CAPABILITY_PROBE_ENABLED",), "bool"),
    "tool_runtime.tool_capability_probe_timeout_seconds": (("RYO_TOOL_RUNTIME_CAPABILITY_PROBE_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.tool_capability_probe_cache_ttl_seconds": (("RYO_TOOL_RUNTIME_CAPABILITY_PROBE_CACHE_TTL_SECONDS",), "float"),
    "tool_runtime.tool_capability_probe_max_models": (("RYO_TOOL_RUNTIME_CAPABILITY_PROBE_MAX_MODELS",), "int"),
    "tool_runtime.brave_timeout_seconds": (("RYO_TOOL_RUNTIME_BRAVE_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.curl_timeout_seconds": (("RYO_TOOL_RUNTIME_CURL_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.curl_max_response_chars": (("RYO_TOOL_RUNTIME_CURL_MAX_RESPONSE_CHARS",), "int"),
    "tool_runtime.curl_allowlist_domains": (("RYO_TOOL_RUNTIME_CURL_ALLOWLIST_DOMAINS",), "csv"),
    "tool_runtime.curl_allow_private_network": (("RYO_TOOL_RUNTIME_CURL_ALLOW_PRIVATE_NETWORK",), "bool"),
    "tool_runtime.curl_allow_mutating_methods": (("RYO_TOOL_RUNTIME_CURL_ALLOW_MUTATING_METHODS",), "bool"),
    "tool_runtime.chat_history_timeout_seconds": (("RYO_TOOL_RUNTIME_CHAT_HISTORY_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.knowledge_timeout_seconds": (("RYO_TOOL_RUNTIME_KNOWLEDGE_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.skip_tools_timeout_seconds": (("RYO_TOOL_RUNTIME_SKIP_TOOLS_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.enable_human_approval": (("RYO_TOOL_RUNTIME_ENABLE_HUMAN_APPROVAL",), "bool"),
    "tool_runtime.default_approval_timeout_seconds": (("RYO_TOOL_RUNTIME_DEFAULT_APPROVAL_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.approval_poll_interval_seconds": (("RYO_TOOL_RUNTIME_APPROVAL_POLL_INTERVAL_SECONDS",), "float"),
    "tool_runtime.default_dry_run": (("RYO_TOOL_RUNTIME_DEFAULT_DRY_RUN",), "bool"),
    "tool_runtime.sandbox.enabled": (("RYO_TOOL_SANDBOX_ENABLED",), "bool"),
    "tool_runtime.sandbox.require_approval": (("RYO_TOOL_SANDBOX_REQUIRE_APPROVAL",), "bool"),
    "tool_runtime.sandbox.dry_run": (("RYO_TOOL_SANDBOX_DRY_RUN",), "bool"),
    "tool_runtime.sandbox.approval_timeout_seconds": (("RYO_TOOL_SANDBOX_APPROVAL_TIMEOUT_SECONDS",), "float"),
    "tool_runtime.sandbox.execution_timeout_ceiling": (("RYO_TOOL_SANDBOX_EXEC_TIMEOUT_CEILING",), "float"),
    "tool_runtime.sandbox.max_memory_mb": (("RYO_TOOL_SANDBOX_MAX_MEMORY_MB",), "int"),
    "tool_runtime.sandbox.network.enabled": (("RYO_TOOL_SANDBOX_NETWORK_ENABLED",), "bool"),
    "tool_runtime.sandbox.filesystem.mode": (("RYO_TOOL_SANDBOX_FILESYSTEM_MODE",), "str"),
    "telegram.minimum_community_score_private_chat": (("RYO_TELEGRAM_MIN_SCORE_PRIVATE_CHAT",), "int"),
    "telegram.minimum_community_score_private_image": (("RYO_TELEGRAM_MIN_SCORE_PRIVATE_IMAGE",), "int"),
    "telegram.minimum_community_score_group_image": (("RYO_TELEGRAM_MIN_SCORE_GROUP_IMAGE",), "int"),
    "telegram.minimum_community_score_other_group": (("RYO_TELEGRAM_MIN_SCORE_OTHER_GROUP",), "int"),
    "telegram.minimum_community_score_link": (("RYO_TELEGRAM_MIN_SCORE_LINK",), "int"),
    "telegram.minimum_community_score_forward": (("RYO_TELEGRAM_MIN_SCORE_FORWARD",), "int"),
    "telegram.admin_claim_enabled": (("RYO_TELEGRAM_ADMIN_CLAIM_ENABLED",), "bool"),
    "telegram.admin_claim_accept_raw_message": (("RYO_TELEGRAM_ADMIN_CLAIM_ACCEPT_RAW_MESSAGE",), "bool"),
    "telegram.admin_config_commands_enabled": (("RYO_TELEGRAM_ADMIN_CONFIG_COMMANDS_ENABLED",), "bool"),
    "telegram.group_context_isolation_enabled": (("RYO_TELEGRAM_GROUP_CONTEXT_ISOLATION_ENABLED",), "bool"),
    "telegram.group_storage_mode": (("RYO_TELEGRAM_GROUP_STORAGE_MODE",), "str"),
    "telegram.group_storage_prefix": (("RYO_TELEGRAM_GROUP_STORAGE_PREFIX",), "str"),
    "telegram.get_updates_write_timeout": (("RYO_TELEGRAM_GET_UPDATES_WRITE_TIMEOUT",), "int"),
    "telegram.message_guard_completed_ttl_seconds": (("RYO_TELEGRAM_MESSAGE_GUARD_TTL_SECONDS",), "int"),
    "telegram.show_stage_progress": (("RYO_TELEGRAM_SHOW_STAGE_PROGRESS",), "bool"),
    "telegram.show_stage_json_details": (("RYO_TELEGRAM_SHOW_STAGE_JSON_DETAILS",), "bool"),
    "telegram.stage_detail_level": (("RYO_TELEGRAM_STAGE_DETAIL_LEVEL",), "str"),
    "telegram.stage_update_min_interval_seconds": (("RYO_TELEGRAM_STAGE_UPDATE_MIN_INTERVAL_SECONDS",), "float"),
    "orchestrator.fast_path_small_talk_enabled": (
        ("RYO_ORCHESTRATOR_FAST_PATH_BREVITY_ENABLED", "RYO_ORCHESTRATOR_FAST_PATH_SMALL_TALK_ENABLED"),
        "bool",
    ),
    "orchestrator.fast_path_small_talk_max_chars": (
        ("RYO_ORCHESTRATOR_FAST_PATH_BREVITY_MAX_CHARS", "RYO_ORCHESTRATOR_FAST_PATH_SMALL_TALK_MAX_CHARS"),
        "int",
    ),
    "orchestrator.analysis_bypass_small_talk_enabled": (
        ("RYO_ORCHESTRATOR_ANALYSIS_BYPASS_SMALL_TALK_ENABLED",),
        "bool",
    ),
    "orchestrator.analysis_history_limit": (
        ("RYO_ORCHESTRATOR_ANALYSIS_HISTORY_LIMIT",),
        "int",
    ),
    "orchestrator.analysis_history_limit_on_switch": (
        ("RYO_ORCHESTRATOR_ANALYSIS_HISTORY_LIMIT_ON_SWITCH",),
        "int",
    ),
    "orchestrator.analysis_history_limit_small_talk": (
        ("RYO_ORCHESTRATOR_ANALYSIS_HISTORY_LIMIT_SMALL_TALK",),
        "int",
    ),
    "orchestrator.tool_history_limit": (
        ("RYO_ORCHESTRATOR_TOOL_HISTORY_LIMIT",),
        "int",
    ),
    "orchestrator.response_history_limit": (
        ("RYO_ORCHESTRATOR_RESPONSE_HISTORY_LIMIT",),
        "int",
    ),
    "orchestrator.auto_expand_tool_stage_enabled": (
        ("RYO_ORCHESTRATOR_AUTO_EXPAND_TOOL_STAGE_ENABLED",),
        "bool",
    ),
    "orchestrator.auto_expand_max_rounds": (
        ("RYO_ORCHESTRATOR_AUTO_EXPAND_MAX_ROUNDS",),
        "int",
    ),
    "orchestrator.progress_stage_events_enabled": (
        ("RYO_ORCHESTRATOR_PROGRESS_STAGE_EVENTS_ENABLED",),
        "bool",
    ),
    "orchestrator.analysis_max_output_tokens": (
        ("RYO_ORCHESTRATOR_ANALYSIS_MAX_OUTPUT_TOKENS",),
        "int",
    ),
    "orchestrator.analysis_temperature": (
        ("RYO_ORCHESTRATOR_ANALYSIS_TEMPERATURE",),
        "float",
    ),
    "orchestrator.analysis_context_summary_max_chars": (
        ("RYO_ORCHESTRATOR_ANALYSIS_CONTEXT_SUMMARY_MAX_CHARS",),
        "int",
    ),
    "orchestrator.control_plane_cache_enabled": (
        ("RYO_ORCHESTRATOR_CONTROL_PLANE_CACHE_ENABLED",),
        "bool",
    ),
    "orchestrator.control_plane_cache_ttl_seconds": (
        ("RYO_ORCHESTRATOR_CONTROL_PLANE_CACHE_TTL_SECONDS",),
        "float",
    ),
    "orchestrator.discovery_layer_enabled": (
        ("RYO_ORCHESTRATOR_DISCOVERY_LAYER_ENABLED",),
        "bool",
    ),
    "orchestrator.discovery_force_tools_on_uncertainty": (
        ("RYO_ORCHESTRATOR_DISCOVERY_FORCE_TOOLS",),
        "bool",
    ),
    "orchestrator.discovery_unknown_threshold": (
        ("RYO_ORCHESTRATOR_DISCOVERY_UNKNOWN_THRESHOLD",),
        "float",
    ),
    "orchestrator.discovery_probably_unknown_threshold": (
        ("RYO_ORCHESTRATOR_DISCOVERY_PROBABLY_UNKNOWN_THRESHOLD",),
        "float",
    ),
    "orchestrator.discovery_default_tool_hints": (
        ("RYO_ORCHESTRATOR_DISCOVERY_DEFAULT_TOOL_HINTS",),
        "csv",
    ),
    "personality.enabled": (("RYO_PERSONALITY_ENABLED",), "bool"),
    "personality.adaptive_enabled": (("RYO_PERSONALITY_ADAPTIVE_ENABLED",), "bool"),
    "personality.narrative_enabled": (("RYO_PERSONALITY_NARRATIVE_ENABLED",), "bool"),
    "personality.max_injection_chars": (("RYO_PERSONALITY_MAX_INJECTION_CHARS",), "int"),
    "personality.narrative_summary_max_chars": (("RYO_PERSONALITY_NARRATIVE_SUMMARY_MAX_CHARS",), "int"),
    "personality.narrative_source_history_limit": (("RYO_PERSONALITY_NARRATIVE_SOURCE_HISTORY_LIMIT",), "int"),
    "personality.rollup_turn_threshold": (("RYO_PERSONALITY_ROLLUP_TURN_THRESHOLD",), "int"),
    "personality.rollup_char_threshold": (("RYO_PERSONALITY_ROLLUP_CHAR_THRESHOLD",), "int"),
    "personality.max_active_chunks": (("RYO_PERSONALITY_MAX_ACTIVE_CHUNKS",), "int"),
    "personality.adaptation_min_turns": (("RYO_PERSONALITY_ADAPTATION_MIN_TURNS",), "int"),
    "personality.adaptation_max_step_per_window": (("RYO_PERSONALITY_ADAPTATION_MAX_STEP_PER_WINDOW",), "int"),
    "personality.adaptation_window_turns": (("RYO_PERSONALITY_ADAPTATION_WINDOW_TURNS",), "int"),
    "personality.verbosity_short_token_threshold": (("RYO_PERSONALITY_VERBOSITY_SHORT_TOKEN_THRESHOLD",), "int"),
    "personality.verbosity_long_token_threshold": (("RYO_PERSONALITY_VERBOSITY_LONG_TOKEN_THRESHOLD",), "int"),
    "personality.default_tone": (("RYO_PERSONALITY_DEFAULT_TONE",), "str"),
    "personality.default_verbosity": (("RYO_PERSONALITY_DEFAULT_VERBOSITY",), "str"),
    "personality.default_reading_level": (("RYO_PERSONALITY_DEFAULT_READING_LEVEL",), "str"),
    "personality.default_format": (("RYO_PERSONALITY_DEFAULT_FORMAT",), "str"),
    "personality.default_humor": (("RYO_PERSONALITY_DEFAULT_HUMOR",), "str"),
    "personality.default_emoji": (("RYO_PERSONALITY_DEFAULT_EMOJI",), "str"),
    "personality.default_language": (("RYO_PERSONALITY_DEFAULT_LANGUAGE",), "str"),
    "temporal.enabled": (("RYO_TEMPORAL_CONTEXT_ENABLED",), "bool"),
    "temporal.default_timezone": (("RYO_TEMPORAL_DEFAULT_TIMEZONE",), "str"),
    "temporal.history_limit": (("RYO_TEMPORAL_HISTORY_LIMIT",), "int"),
    "temporal.excerpt_max_chars": (("RYO_TEMPORAL_EXCERPT_MAX_CHARS",), "int"),
    "topic_shift.enabled": (("RYO_TOPIC_SHIFT_ENABLED",), "bool"),
    "topic_shift.min_token_count": (("RYO_TOPIC_SHIFT_MIN_TOKEN_COUNT",), "int"),
    "topic_shift.keyword_terms": (("RYO_TOPIC_SHIFT_KEYWORD_TERMS",), "int"),
    "topic_shift.jaccard_switch_threshold": (("RYO_TOPIC_SHIFT_JACCARD_THRESHOLD",), "float"),
    "topic_shift.recent_user_messages": (("RYO_TOPIC_SHIFT_RECENT_USER_MESSAGES",), "int"),
    "topic_shift.deweight_history_search_on_switch": (("RYO_TOPIC_SHIFT_DEWEIGHT_HISTORY_ON_SWITCH",), "bool"),
    "topic_shift.history_window_hours_on_switch": (("RYO_TOPIC_SHIFT_HISTORY_WINDOW_HOURS_ON_SWITCH",), "int"),
    "topic_shift.trim_temporal_history_on_small_talk": (("RYO_TOPIC_SHIFT_TRIM_TEMPORAL_ON_SMALL_TALK",), "bool"),
    "topic_shift.small_talk_history_limit": (("RYO_TOPIC_SHIFT_SMALL_TALK_HISTORY_LIMIT",), "int"),
    "topic_shift.allow_recent_topic_fallback": (("RYO_TOPIC_SHIFT_ALLOW_RECENT_TOPIC_FALLBACK",), "bool"),
    "topic_shift.history_min_overlap_default": (("RYO_TOPIC_SHIFT_HISTORY_MIN_OVERLAP_DEFAULT",), "float"),
    "topic_shift.history_min_overlap_on_switch": (("RYO_TOPIC_SHIFT_HISTORY_MIN_OVERLAP_ON_SWITCH",), "float"),
    "topic_shift.low_token_switch_enabled": (("RYO_TOPIC_SHIFT_LOW_TOKEN_SWITCH_ENABLED",), "bool"),
    "security.password_min_length": (("RYO_PASSWORD_MIN_LENGTH",), "int"),
    "vectors.embedding_dimensions": (("RYO_VECTOR_DIMENSIONS",), "int"),
    "retrieval.conversation_short_history_limit": (("RYO_RETRIEVAL_SHORT_HISTORY_LIMIT",), "int"),
    "retrieval.chat_history_window_hours": (("RYO_CHAT_HISTORY_WINDOW_HOURS",), "int"),
    "retrieval.chat_history_default_limit": (("RYO_CHAT_HISTORY_DEFAULT_LIMIT",), "int"),
    "retrieval.chat_history_sender_window_hours": (("RYO_CHAT_HISTORY_SENDER_WINDOW_HOURS",), "int"),
    "retrieval.chat_history_sender_default_limit": (("RYO_CHAT_HISTORY_SENDER_DEFAULT_LIMIT",), "int"),
    "retrieval.progressive_history_enabled": (("RYO_PROGRESSIVE_HISTORY_ENABLED",), "bool"),
    "retrieval.progressive_history_tool_enabled": (("RYO_PROGRESSIVE_HISTORY_TOOL_ENABLED",), "bool"),
    "retrieval.progressive_history_max_rounds": (("RYO_PROGRESSIVE_HISTORY_MAX_ROUNDS",), "int"),
    "retrieval.progressive_history_round_windows_hours": (("RYO_PROGRESSIVE_HISTORY_ROUND_WINDOWS_HOURS",), "csv"),
    "retrieval.progressive_history_semantic_limit_start": (("RYO_PROGRESSIVE_HISTORY_SEMANTIC_LIMIT_START",), "int"),
    "retrieval.progressive_history_semantic_limit_step": (("RYO_PROGRESSIVE_HISTORY_SEMANTIC_LIMIT_STEP",), "int"),
    "retrieval.progressive_history_timeline_limit_start": (("RYO_PROGRESSIVE_HISTORY_TIMELINE_LIMIT_START",), "int"),
    "retrieval.progressive_history_timeline_limit_step": (("RYO_PROGRESSIVE_HISTORY_TIMELINE_LIMIT_STEP",), "int"),
    "retrieval.progressive_history_context_radius": (("RYO_PROGRESSIVE_HISTORY_CONTEXT_RADIUS",), "int"),
    "retrieval.progressive_history_match_threshold": (("RYO_PROGRESSIVE_HISTORY_MATCH_THRESHOLD",), "float"),
    "retrieval.progressive_history_max_selected": (("RYO_PROGRESSIVE_HISTORY_MAX_SELECTED",), "int"),
    "retrieval.progressive_history_max_message_chars": (("RYO_PROGRESSIVE_HISTORY_MAX_MESSAGE_CHARS",), "int"),
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
    "watchdog.auto_start_routes": (("RYO_WATCHDOG_AUTO_START_ROUTES",), "bool"),
    "web.host": (("RYO_WEB_HOST",), "str"),
    "web.port": (("RYO_WEB_PORT",), "int"),
    "web.port_scan_limit": (("RYO_WEB_PORT_SCAN_LIMIT",), "int"),
    "web.debug": (("RYO_WEB_DEBUG",), "bool"),
    "web.use_reloader": (("RYO_WEB_USE_RELOADER",), "bool"),
}

COMMUNITY_SCORE_REQUIREMENT_MAP: dict[str, str] = {
    "private_chat": "minimum_community_score_private_chat",
    "private_image": "minimum_community_score_private_image",
    "group_image": "minimum_community_score_group_image",
    "other_group": "minimum_community_score_other_group",
    "link_sharing": "minimum_community_score_link",
    "message_forwarding": "minimum_community_score_forward",
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
    if value_type == "csv":
        values: list[str] = []
        for item in raw_value.split(","):
            cleaned = item.strip()
            if not cleaned or cleaned in values:
                continue
            values.append(cleaned)
        return values
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
    community_requirements_from_config: dict[str, Any] | None = None
    if isinstance(config_data, dict):
        runtime_config = config_data.get("runtime")
        if isinstance(runtime_config, dict):
            _deep_merge(settings, runtime_config)

        community_requirements = config_data.get("community_score_requirements")
        if isinstance(community_requirements, dict):
            community_requirements_from_config = dict(community_requirements)
            for public_key, runtime_key in COMMUNITY_SCORE_REQUIREMENT_MAP.items():
                raw_value = community_requirements.get(public_key)
                if raw_value is None:
                    continue
                try:
                    parsed = int(str(raw_value).strip())
                except (TypeError, ValueError):
                    continue
                if parsed < 0:
                    parsed = 0
                set_runtime_setting(settings, f"telegram.{runtime_key}", parsed)

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

    # Keep visible community-score requirements authoritative when explicitly configured.
    # This prevents .env defaults from unintentionally overriding config.json values.
    if isinstance(community_requirements_from_config, dict):
        for public_key, runtime_key in COMMUNITY_SCORE_REQUIREMENT_MAP.items():
            raw_value = community_requirements_from_config.get(public_key)
            if raw_value is None:
                continue
            try:
                parsed = int(str(raw_value).strip())
            except (TypeError, ValueError):
                continue
            if parsed < 0:
                parsed = 0
            set_runtime_setting(settings, f"telegram.{runtime_key}", parsed)

    return settings
