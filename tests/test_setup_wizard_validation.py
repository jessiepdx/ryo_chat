import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_SETUP_WIZARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_wizard.py"
_SETUP_WIZARD_SPEC = importlib.util.spec_from_file_location("setup_wizard", _SETUP_WIZARD_PATH)
setup_wizard = importlib.util.module_from_spec(_SETUP_WIZARD_SPEC)
assert _SETUP_WIZARD_SPEC.loader is not None
_SETUP_WIZARD_SPEC.loader.exec_module(setup_wizard)


def _host_args(**overrides):
    values = {
        "ollama_host": None,
        "non_interactive": True,
        "default_host": setup_wizard.DEFAULT_OLLAMA_HOST,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestSetupWizardValidation(unittest.TestCase):
    def test_is_valid_http_url(self):
        self.assertTrue(setup_wizard.is_valid_http_url("http://127.0.0.1:11434"))
        self.assertTrue(setup_wizard.is_valid_http_url("https://example.com"))
        self.assertFalse(setup_wizard.is_valid_http_url("127.0.0.1:11434"))
        self.assertFalse(setup_wizard.is_valid_http_url("ftp://example.com"))
        self.assertFalse(setup_wizard.is_valid_http_url(""))

    def test_choose_ollama_host_precedence_non_interactive(self):
        args = _host_args(ollama_host="http://custom:11434")
        selected = setup_wizard.choose_ollama_host(args, existing_host="http://existing:11434")
        self.assertEqual(selected, "http://custom:11434")

        args = _host_args(ollama_host=None)
        selected = setup_wizard.choose_ollama_host(args, existing_host="http://existing:11434")
        self.assertEqual(selected, "http://existing:11434")

        selected = setup_wizard.choose_ollama_host(args, existing_host=None)
        self.assertEqual(selected, setup_wizard.DEFAULT_OLLAMA_HOST)

    def test_choose_ollama_host_rejects_invalid_explicit_value(self):
        args = _host_args(ollama_host="not-a-url")
        with self.assertRaises(ValueError):
            setup_wizard.choose_ollama_host(args, existing_host=None)

    def test_probe_models_gracefully_handles_missing_client_dependency(self):
        original_client = setup_wizard.Client
        setup_wizard.Client = None
        try:
            models, error = setup_wizard.probe_ollama_models(setup_wizard.DEFAULT_OLLAMA_HOST)
            self.assertEqual(models, [])
            self.assertIn("not installed", error)
        finally:
            setup_wizard.Client = original_client

    def test_apply_setup_state_populates_required_sections(self):
        state = {
            "ollama_host": "http://127.0.0.1:11434",
            "model_map": {
                "embedding": "nomic-embed-text:latest",
                "generate": "llama3.2:latest",
                "chat": "llama3.2:latest",
                "tool": "llama3.2:latest",
                "multimodal": "llama3.2-vision:latest",
            },
            "bot_name": "ryo_bot",
            "bot_id": "12345",
            "bot_token": "token",
            "web_ui_url": "http://127.0.0.1:4747",
            "owner_first_name": "First",
            "owner_last_name": "Last",
            "owner_user_id": "67890",
            "owner_username": "owner_user",
            "db_name": "ryo_chat",
            "db_user": "postgres_user",
            "db_password": "postgres_password",
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "fallback_enabled": True,
            "fallback_mode": "local",
            "fallback_db_name": "ryo_chat_fallback",
            "fallback_db_user": "postgres_user",
            "fallback_db_password": "postgres_password",
            "fallback_db_host": "127.0.0.1",
            "fallback_db_port": "5433",
            "brave_search_key": "brave-key",
            "twitter_consumer_key": "",
            "twitter_consumer_secret": "",
            "twitter_access_token": "",
            "twitter_access_token_secret": "",
        }

        merged = setup_wizard.apply_setup_state({}, state)
        self.assertEqual(merged["bot_id"], 12345)
        self.assertEqual(merged["owner_info"]["user_id"], 67890)
        self.assertEqual(merged["database"]["db_name"], "ryo_chat")
        self.assertTrue(merged["database_fallback"]["enabled"])
        self.assertEqual(merged["api_keys"]["brave_search"], "brave-key")
        for key in setup_wizard.INFERENCE_KEYS:
            self.assertEqual(merged["inference"][key]["url"], "http://127.0.0.1:11434")
            self.assertTrue(merged["inference"][key]["model"])

    def test_validate_required_config_reports_missing_fields(self):
        missing = setup_wizard.validate_required_config({})
        self.assertIn("bot_name", missing)
        self.assertIn("database.db_name", missing)
        self.assertIn("inference.chat.model", missing)

    def test_partial_state_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "wizard_state.json"
            payload = {"bot_name": "ryo", "fallback_enabled": True}

            setup_wizard.save_partial_state(state_path, payload)
            loaded = setup_wizard.load_partial_state(state_path)
            self.assertEqual(loaded, payload)

            setup_wizard.clear_partial_state(state_path)
            self.assertFalse(state_path.exists())

    def test_load_json_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "broken.json"
            config_path.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(ValueError):
                setup_wizard.load_json(config_path)

    def test_resolve_bootstrap_target_prefers_explicit_then_fallback_state(self):
        config_data = {
            "database_fallback": {"enabled": True}
        }
        self.assertEqual(
            setup_wizard.resolve_bootstrap_target(config_data, explicit_target="primary"),
            "primary",
        )
        self.assertEqual(
            setup_wizard.resolve_bootstrap_target(config_data, explicit_target=None),
            "both",
        )
        self.assertEqual(
            setup_wizard.resolve_bootstrap_target({"database_fallback": {"enabled": False}}, explicit_target=None),
            "primary",
        )

    def test_run_postgres_bootstrap_invokes_script_with_expected_flags(self):
        completed = mock.Mock(returncode=0, stdout="ok", stderr="")
        with mock.patch.object(setup_wizard.subprocess, "run", return_value=completed) as run_mock:
            ok, output = setup_wizard.run_postgres_bootstrap(
                config_path=Path("config.json"),
                target="both",
                use_docker=True,
            )

        self.assertTrue(ok)
        self.assertIn("ok", output)
        called = run_mock.call_args.args[0]
        self.assertTrue(any(str(item).endswith("bootstrap_postgres.py") for item in called))
        self.assertIn("--config", called)
        self.assertIn("--target", called)
        self.assertIn("both", called)
        self.assertIn("--docker", called)

    def test_is_local_database_setup_true_for_local_hosts(self):
        state = {
            "db_host": "127.0.0.1",
            "fallback_enabled": True,
            "fallback_mode": "local",
            "fallback_db_host": "localhost",
        }
        self.assertTrue(setup_wizard.is_local_database_setup(state))

    def test_is_local_database_setup_false_for_remote_primary(self):
        state = {
            "db_host": "db.example.com",
            "fallback_enabled": False,
        }
        self.assertFalse(setup_wizard.is_local_database_setup(state))

    def test_build_state_non_interactive_enables_auto_local_bootstrap(self):
        args = argparse.Namespace(
            ollama_host=None,
            non_interactive=True,
            default_host=setup_wizard.DEFAULT_OLLAMA_HOST,
            embedding_model=None,
            generate_model=None,
            chat_model=None,
            tool_model=None,
            multimodal_model=None,
            bot_name=None,
            bot_id=None,
            bot_token=None,
            web_ui_url=None,
            owner_first_name=None,
            owner_last_name=None,
            owner_user_id=None,
            owner_username=None,
            db_name=None,
            db_user=None,
            db_password=None,
            db_host=None,
            db_port=None,
            fallback_enabled=False,
            fallback_disabled=False,
            fallback_mode=None,
            fallback_db_name=None,
            fallback_db_user=None,
            fallback_db_password=None,
            fallback_db_host=None,
            fallback_db_port=None,
            brave_search_key=None,
            twitter_consumer_key=None,
            twitter_consumer_secret=None,
            twitter_access_token=None,
            twitter_access_token_secret=None,
            bootstrap_postgres=False,
            bootstrap_docker=False,
        )
        config_data = {
            "bot_name": "ryo_bot",
            "bot_id": 123,
            "bot_token": "token",
            "web_ui_url": "http://127.0.0.1:4747",
            "owner_info": {
                "first_name": "First",
                "last_name": "Last",
                "user_id": 456,
                "username": "owner",
            },
            "database": {
                "db_name": "ryo_chat",
                "user": "postgres_user",
                "password": "postgres_password",
                "host": "127.0.0.1",
                "port": "5432",
            },
            "database_fallback": {"enabled": False},
            "twitter_keys": {},
            "api_keys": {},
        }
        with mock.patch.object(setup_wizard, "probe_ollama_models", return_value=([], None)):
            state = setup_wizard.build_state_non_interactive(args, config_data)
        self.assertTrue(state["bootstrap_postgres"])
        self.assertTrue(state["bootstrap_docker"])
        self.assertTrue(state["write_env"])

    def test_sync_db_state_from_config_prefers_persisted_ports(self):
        state = {
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "fallback_enabled": True,
            "fallback_db_host": "127.0.0.1",
            "fallback_db_port": "5433",
        }
        config_data = {
            "database": {
                "db_name": "ryo_chat",
                "user": "postgres_user",
                "password": "postgres_password",
                "host": "127.0.0.1",
                "port": "5542",
            },
            "database_fallback": {
                "enabled": True,
                "mode": "local",
                "db_name": "ryo_chat_fallback",
                "user": "postgres_user",
                "password": "postgres_password",
                "host": "127.0.0.1",
                "port": "5543",
            },
        }
        synced = setup_wizard.sync_db_state_from_config(state, config_data)
        self.assertEqual(synced["db_port"], "5542")
        self.assertEqual(synced["fallback_db_port"], "5543")


if __name__ == "__main__":
    unittest.main()
