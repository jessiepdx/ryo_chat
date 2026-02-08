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
