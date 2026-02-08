import argparse
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


_SETUP_WIZARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_wizard.py"
_SETUP_WIZARD_SPEC = importlib.util.spec_from_file_location("setup_wizard", _SETUP_WIZARD_PATH)
setup_wizard = importlib.util.module_from_spec(_SETUP_WIZARD_SPEC)
assert _SETUP_WIZARD_SPEC.loader is not None
_SETUP_WIZARD_SPEC.loader.exec_module(setup_wizard)


def _telegram_args(**overrides):
    values = {
        "bot_name": None,
        "bot_id": None,
        "bot_token": None,
        "web_ui_url": None,
        "owner_first_name": None,
        "owner_last_name": None,
        "owner_user_id": None,
        "owner_username": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _telegram_state():
    return {
        "bot_name": "new_bot",
        "bot_id": "54321",
        "bot_token": "new-token",
        "web_ui_url": "http://127.0.0.1:4747/",
        "owner_first_name": "New",
        "owner_last_name": "Owner",
        "owner_user_id": "999",
        "owner_username": "new_owner",
    }


class TestTelegramConfigIngress(unittest.TestCase):
    def test_apply_telegram_state_preserves_inference_and_database(self):
        base_config = {
            "bot_name": "old_bot",
            "bot_id": 12345,
            "bot_token": "old-token",
            "web_ui_url": "http://old/",
            "owner_info": {
                "first_name": "Old",
                "last_name": "Owner",
                "user_id": 100,
                "username": "old_owner",
            },
            "database": {
                "db_name": "ryo_chat",
                "user": "postgres",
                "password": "secret",
                "host": "db.internal",
                "port": "5432",
            },
            "inference": {
                key: {"url": "http://custom-ollama:11434", "model": "model-a"}
                for key in setup_wizard.INFERENCE_KEYS
            },
        }
        merged = setup_wizard.apply_telegram_state(copy.deepcopy(base_config), _telegram_state())

        self.assertEqual(merged["bot_name"], "new_bot")
        self.assertEqual(merged["bot_id"], 54321)
        self.assertEqual(merged["bot_token"], "new-token")
        self.assertEqual(merged["owner_info"]["user_id"], 999)
        self.assertEqual(merged["database"], base_config["database"])
        for key in setup_wizard.INFERENCE_KEYS:
            self.assertEqual(merged["inference"][key]["url"], "http://custom-ollama:11434")

    def test_state_to_env_updates_telegram_only_excludes_ollama_and_postgres(self):
        state = _telegram_state()
        updates = setup_wizard.state_to_env_updates(state, telegram_only=True)

        self.assertEqual(updates["TELEGRAM_BOT_TOKEN"], "new-token")
        self.assertEqual(updates["TELEGRAM_BOT_ID"], "54321")
        self.assertIn("WEB_UI_URL", updates)
        self.assertNotIn("OLLAMA_HOST", updates)
        self.assertNotIn("POSTGRES_DB", updates)

    def test_write_env_telegram_only_preserves_existing_ollama_host(self):
        state = _telegram_state()
        updates = setup_wizard.state_to_env_updates(state, telegram_only=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            template_path = Path(tmp_dir) / ".env.example"
            env_path.write_text(
                "OLLAMA_HOST=http://custom-ollama:11434\n"
                "TELEGRAM_BOT_TOKEN=old-token\n"
                "POSTGRES_DB=ryo_chat\n",
                encoding="utf-8",
            )
            template_path.write_text("", encoding="utf-8")

            setup_wizard.write_env_with_updates(env_path, template_path, updates)
            _, values = setup_wizard.parse_env_file(env_path)

            self.assertEqual(values["OLLAMA_HOST"], "http://custom-ollama:11434")
            self.assertEqual(values["TELEGRAM_BOT_TOKEN"], "new-token")
            self.assertEqual(values["POSTGRES_DB"], "ryo_chat")

    def test_validate_required_telegram_config_reports_invalid_fields(self):
        invalid = {
            "bot_name": "",
            "bot_id": "abc",
            "bot_token": "",
            "web_ui_url": "not-a-url",
            "owner_info": {
                "first_name": "",
                "last_name": "",
                "user_id": "bad",
                "username": "",
            },
        }
        missing = setup_wizard.validate_required_telegram_config(invalid)

        self.assertIn("bot_name", missing)
        self.assertIn("bot_id", missing)
        self.assertIn("bot_token", missing)
        self.assertIn("web_ui_url", missing)
        self.assertIn("owner_info.user_id", missing)

    def test_build_state_non_interactive_telegram_uses_existing_with_overrides(self):
        args = _telegram_args(bot_token="override-token", owner_username="override_owner")
        config_data = {
            "bot_name": "existing_bot",
            "bot_id": 100,
            "bot_token": "existing-token",
            "web_ui_url": "http://existing/",
            "owner_info": {
                "first_name": "Existing",
                "last_name": "Owner",
                "user_id": 200,
                "username": "existing_owner",
            },
        }
        state = setup_wizard.build_state_non_interactive_telegram(args, config_data)

        self.assertEqual(state["bot_name"], "existing_bot")
        self.assertEqual(state["bot_id"], 100)
        self.assertEqual(state["bot_token"], "override-token")
        self.assertEqual(state["owner_username"], "override_owner")


if __name__ == "__main__":
    unittest.main()
