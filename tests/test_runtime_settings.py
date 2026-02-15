import tempfile
import unittest
from pathlib import Path

from hypermindlabs.runtime_settings import (
    build_runtime_settings,
    load_dotenv_file,
)


class TestRuntimeSettings(unittest.TestCase):
    def test_build_runtime_settings_uses_config_runtime_overrides(self):
        config_data = {
            "runtime": {
                "conversation": {
                    "knowledge_lookup_word_threshold": 11,
                },
                "telegram": {
                    "minimum_community_score_private_chat": 77,
                },
            }
        }
        settings = build_runtime_settings(config_data=config_data, env_data={})
        self.assertEqual(settings["conversation"]["knowledge_lookup_word_threshold"], 11)
        self.assertEqual(settings["telegram"]["minimum_community_score_private_chat"], 77)

    def test_build_runtime_settings_supports_legacy_and_ryo_env_keys(self):
        settings = build_runtime_settings(
            config_data={},
            env_data={
                "OLLAMA_HOST": "http://legacy-ollama:11434",
                "OLLAMA_CHAT_MODEL": "legacy-chat-model",
            },
        )
        self.assertEqual(settings["inference"]["default_ollama_host"], "http://legacy-ollama:11434")
        self.assertEqual(settings["inference"]["default_chat_model"], "legacy-chat-model")

        # RYO-prefixed keys should take precedence when both are present.
        overridden = build_runtime_settings(
            config_data={},
            env_data={
                "OLLAMA_HOST": "http://legacy-ollama:11434",
                "RYO_DEFAULT_OLLAMA_HOST": "http://ryo-ollama:11434",
            },
        )
        self.assertEqual(overridden["inference"]["default_ollama_host"], "http://ryo-ollama:11434")

    def test_build_runtime_settings_supports_startup_bool_overrides(self):
        settings = build_runtime_settings(
            config_data={},
            env_data={
                "RYO_PROMPT_MODEL_SELECTION_ON_STARTUP": "true",
                "RYO_WATCHDOG_AUTO_START_ROUTES": "false",
            },
        )
        self.assertTrue(settings["inference"]["prompt_model_selection_on_startup"])
        self.assertFalse(settings["watchdog"]["auto_start_routes"])

    def test_build_runtime_settings_supports_temporal_overrides(self):
        settings = build_runtime_settings(
            config_data={
                "runtime": {
                    "temporal": {
                        "default_timezone": "America/New_York",
                        "history_limit": 44,
                    }
                }
            },
            env_data={
                "RYO_TEMPORAL_CONTEXT_ENABLED": "false",
                "RYO_TEMPORAL_EXCERPT_MAX_CHARS": "220",
            },
        )
        self.assertFalse(settings["temporal"]["enabled"])
        self.assertEqual(settings["temporal"]["default_timezone"], "America/New_York")
        self.assertEqual(settings["temporal"]["history_limit"], 44)
        self.assertEqual(settings["temporal"]["excerpt_max_chars"], 220)

    def test_build_runtime_settings_supports_discovery_overrides(self):
        settings = build_runtime_settings(
            config_data={
                "runtime": {
                    "orchestrator": {
                        "discovery_unknown_threshold": 0.71,
                        "discovery_default_tool_hints": ["knowledgeSearch", "curlRequest"],
                    }
                }
            },
            env_data={
                "RYO_ORCHESTRATOR_DISCOVERY_FORCE_TOOLS": "false",
            },
        )
        self.assertEqual(settings["orchestrator"]["discovery_unknown_threshold"], 0.71)
        self.assertEqual(
            settings["orchestrator"]["discovery_default_tool_hints"],
            ["knowledgeSearch", "curlRequest"],
        )
        self.assertFalse(settings["orchestrator"]["discovery_force_tools_on_uncertainty"])

    def test_build_runtime_settings_supports_progressive_history_overrides(self):
        settings = build_runtime_settings(
            config_data={
                "runtime": {
                    "retrieval": {
                        "progressive_history_max_rounds": 6,
                        "progressive_history_context_radius": 3,
                    }
                }
            },
            env_data={
                "RYO_PROGRESSIVE_HISTORY_ENABLED": "false",
                "RYO_PROGRESSIVE_HISTORY_ROUND_WINDOWS_HOURS": "6,24,72",
                "RYO_PROGRESSIVE_HISTORY_MATCH_THRESHOLD": "0.51",
            },
        )
        self.assertFalse(settings["retrieval"]["progressive_history_enabled"])
        self.assertEqual(settings["retrieval"]["progressive_history_max_rounds"], 6)
        self.assertEqual(settings["retrieval"]["progressive_history_context_radius"], 3)
        self.assertEqual(settings["retrieval"]["progressive_history_round_windows_hours"], ["6", "24", "72"])
        self.assertEqual(settings["retrieval"]["progressive_history_match_threshold"], 0.51)

    def test_build_runtime_settings_supports_document_taxonomy_overrides(self):
        settings = build_runtime_settings(
            config_data={
                "runtime": {
                    "documents": {
                        "taxonomy_enabled": False,
                        "taxonomy_topic_limit": 6,
                        "taxonomy_topic_min_confidence": 0.4,
                    }
                }
            },
            env_data={
                "RYO_DOCUMENT_TAXONOMY_ENABLED": "true",
                "RYO_DOCUMENT_TAXONOMY_TOPIC_LIMIT": "3",
                "RYO_DOCUMENT_TAXONOMY_TOPIC_MIN_CONFIDENCE": "0.15",
            },
        )
        self.assertTrue(settings["documents"]["taxonomy_enabled"])
        self.assertEqual(settings["documents"]["taxonomy_topic_limit"], 3)
        self.assertEqual(settings["documents"]["taxonomy_topic_min_confidence"], 0.15)

    def test_build_runtime_settings_supports_personality_overrides(self):
        settings = build_runtime_settings(
            config_data={
                "runtime": {
                    "personality": {
                        "rollup_turn_threshold": 12,
                        "default_tone": "professional",
                    }
                }
            },
            env_data={
                "RYO_PERSONALITY_ENABLED": "false",
                "RYO_PERSONALITY_DEFAULT_VERBOSITY": "detailed",
            },
        )
        self.assertFalse(settings["personality"]["enabled"])
        self.assertEqual(settings["personality"]["rollup_turn_threshold"], 12)
        self.assertEqual(settings["personality"]["default_tone"], "professional")
        self.assertEqual(settings["personality"]["default_verbosity"], "detailed")

    def test_build_runtime_settings_hydrates_from_public_community_requirements(self):
        config_data = {
            "runtime": {
                "telegram": {
                    "minimum_community_score_private_chat": 33,
                }
            },
            "community_score_requirements": {
                "private_chat": 90,
                "link_sharing": 55,
            },
        }
        settings = build_runtime_settings(config_data=config_data, env_data={})
        self.assertEqual(settings["telegram"]["minimum_community_score_private_chat"], 90)
        self.assertEqual(settings["telegram"]["minimum_community_score_link"], 55)

    def test_load_dotenv_file_parses_basic_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dotenv_path = Path(tmp_dir) / ".env"
            dotenv_path.write_text(
                "export RYO_PASSWORD_MIN_LENGTH=16\n"
                "RYO_TELEGRAM_MIN_SCORE_PRIVATE_CHAT='66'\n",
                encoding="utf-8",
            )
            loaded = load_dotenv_file(dotenv_path, override=True)

        self.assertEqual(loaded["RYO_PASSWORD_MIN_LENGTH"], "16")
        self.assertEqual(loaded["RYO_TELEGRAM_MIN_SCORE_PRIVATE_CHAT"], "66")


if __name__ == "__main__":
    unittest.main()
