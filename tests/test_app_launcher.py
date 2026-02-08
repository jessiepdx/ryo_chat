import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app


class TestAppLauncher(unittest.TestCase):
    class _FakeHTTPResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def test_is_active_target_venv_from_prefix(self):
        self.assertTrue(
            app.is_active_target_venv(
                app.PROJECT_ROOT / ".venv",
                current_prefix=app.PROJECT_ROOT / ".venv",
                virtual_env_var=None,
            )
        )

    def test_is_active_target_venv_from_env_var(self):
        self.assertTrue(
            app.is_active_target_venv(
                app.PROJECT_ROOT / ".venv",
                current_prefix="/usr",
                virtual_env_var=str(app.PROJECT_ROOT / ".venv"),
            )
        )

    def test_is_active_target_venv_false_for_system_prefix(self):
        self.assertFalse(
            app.is_active_target_venv(
                app.PROJECT_ROOT / ".venv",
                current_prefix="/usr",
                virtual_env_var=None,
            )
        )

    def test_should_run_setup_when_config_was_created(self):
        state = {"setup_completed": True}
        artifacts = {"created_config": True}
        self.assertTrue(app.should_run_setup(state, artifacts, config_data={}))

    def test_should_run_setup_when_prior_setup_failed(self):
        state = {"setup_completed": False}
        artifacts = {"created_config": False}
        self.assertTrue(app.should_run_setup(state, artifacts, config_data={"bot_name": "ok"}))

    def test_should_not_run_setup_when_state_is_complete(self):
        state = {"setup_completed": True}
        artifacts = {"created_config": False}
        config_data = {"bot_name": "real_bot", "bot_token": "real_token"}
        self.assertFalse(app.should_run_setup(state, artifacts, config_data))

    def test_resolve_ollama_host_uses_inference_url_then_default(self):
        config_data = {
            "inference": {
                "chat": {
                    "url": "http://custom-ollama:11434",
                    "model": "llama3.2:latest",
                }
            }
        }
        self.assertEqual(app.resolve_ollama_host(config_data), "http://custom-ollama:11434")
        self.assertEqual(app.resolve_ollama_host({}), app.DEFAULT_OLLAMA_HOST)

    def test_collect_required_models_dedupes(self):
        config_data = {
            "inference": {
                "chat": {"model": "llama3.2:latest"},
                "tool": {"model": "llama3.2:latest"},
            }
        }
        required = app.collect_required_models(config_data)
        self.assertIn("llama3.2:latest", required)
        self.assertEqual(required.count("llama3.2:latest"), 1)
        self.assertNotIn("llama3.2-vision:latest", required)

    def test_collect_required_models_uses_configured_values_only(self):
        config_data = {
            "inference": {
                "embedding": {"model": "nomic-embed-text:latest"},
                "generate": {"model": "qwen2.5:latest"},
                "chat": {"model": "qwen2.5:latest"},
                "tool": {"model": "qwen2.5:latest"},
                "multimodal": {"model": "llava:latest"},
            }
        }
        required = app.collect_required_models(config_data)
        self.assertEqual(
            required,
            ["nomic-embed-text:latest", "qwen2.5:latest", "llava:latest"],
        )

    def test_apply_default_text_model_sets_chat_generate_tool(self):
        source = {
            "inference": {
                "embedding": {"url": "http://127.0.0.1:11434", "model": "nomic-embed-text:latest"},
                "multimodal": {"url": "http://127.0.0.1:11434", "model": "llama3.2-vision:latest"},
            }
        }
        updated = app.apply_default_text_model(
            copy.deepcopy(source),
            host="http://127.0.0.1:11434",
            model_name="qwen2.5:latest",
        )

        self.assertEqual(updated["inference"]["chat"]["model"], "qwen2.5:latest")
        self.assertEqual(updated["inference"]["generate"]["model"], "qwen2.5:latest")
        self.assertEqual(updated["inference"]["tool"]["model"], "qwen2.5:latest")
        for key in app.INFERENCE_KEYS:
            self.assertEqual(updated["inference"][key]["url"], "http://127.0.0.1:11434")

    def test_should_prompt_model_selection_defaults_to_false_when_model_exists(self):
        runtime_settings = {"inference": {"prompt_model_selection_on_startup": False}}
        self.assertFalse(app.should_prompt_model_selection(runtime_settings, "qwen3-vl:latest"))

    def test_should_prompt_model_selection_when_model_missing(self):
        runtime_settings = {"inference": {"prompt_model_selection_on_startup": False}}
        self.assertTrue(app.should_prompt_model_selection(runtime_settings, None))

    def test_apply_community_score_requirements_writes_public_and_runtime_sections(self):
        runtime_settings = {"telegram": {"minimum_community_score_private_chat": 50}}
        config_data = {}
        updated = app.apply_community_score_requirements(
            config_data,
            {
                "private_chat": 88,
                "private_image": 66,
                "group_image": 44,
                "other_group": 22,
                "link_sharing": 33,
                "message_forwarding": 11,
            },
            runtime_settings=runtime_settings,
        )
        self.assertEqual(updated["community_score_requirements"]["private_chat"], 88)
        self.assertEqual(
            updated["runtime"]["telegram"]["minimum_community_score_private_chat"],
            88,
        )
        self.assertEqual(
            updated["runtime"]["telegram"]["minimum_community_score_forward"],
            11,
        )

    def test_read_community_score_requirements_prefers_public_section(self):
        runtime_settings = {
            "telegram": {
                "minimum_community_score_private_chat": 10,
                "minimum_community_score_private_image": 20,
                "minimum_community_score_group_image": 30,
                "minimum_community_score_other_group": 40,
                "minimum_community_score_link": 50,
                "minimum_community_score_forward": 60,
            }
        }
        config_data = {
            "runtime": {"telegram": {"minimum_community_score_private_chat": 77}},
            "community_score_requirements": {"private_chat": 99},
        }
        values = app.read_community_score_requirements(
            config_data,
            runtime_settings=runtime_settings,
        )
        self.assertEqual(values["private_chat"], 99)
        self.assertEqual(values["private_image"], 20)

    def test_community_score_requirements_configured_requires_all_keys(self):
        self.assertFalse(app.community_score_requirements_configured({}))
        self.assertTrue(
            app.community_score_requirements_configured(
                {
                    "community_score_requirements": {
                        "private_chat": 1,
                        "private_image": 1,
                        "group_image": 1,
                        "other_group": 1,
                        "link_sharing": 1,
                        "message_forwarding": 1,
                    }
                }
            )
        )

    def test_bootstrap_community_score_requirements_non_interactive_hydrates(self):
        runtime_settings = {
            "telegram": {
                "minimum_community_score_private_chat": 77,
                "minimum_community_score_private_image": 66,
                "minimum_community_score_group_image": 55,
                "minimum_community_score_other_group": 44,
                "minimum_community_score_link": 33,
                "minimum_community_score_forward": 22,
            }
        }
        updated = app.bootstrap_community_score_requirements(
            {},
            runtime_settings=runtime_settings,
            non_interactive=True,
            prompt_on_startup=True,
        )
        self.assertEqual(updated["community_score_requirements"]["private_chat"], 77)
        self.assertEqual(updated["community_score_requirements"]["message_forwarding"], 22)

    def test_ensure_requirements_installed_uses_venv_python(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            requirements_file = tmp_root / "requirements.txt"
            requirements_file.write_text("pytest==8.4.0\n", encoding="utf-8")
            venv_dir = tmp_root / ".venv"
            venv_python = venv_dir / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            requirements_stamp = venv_dir / ".requirements.sha256"

            with (
                mock.patch.object(app, "REQUIREMENTS_FILE", requirements_file),
                mock.patch.object(app, "REQUIREMENTS_STAMP", requirements_stamp),
                mock.patch.object(app, "VENV_DIR", venv_dir),
                mock.patch.object(app, "run_command") as run_mock,
            ):
                app.ensure_requirements_installed()

            self.assertTrue(run_mock.called)
            invoked_command = run_mock.call_args.args[0]
            self.assertEqual(invoked_command[0], str(venv_python))
            self.assertEqual(invoked_command[1:4], ["-m", "pip", "install"])

    def test_telegram_bot_link_normalizes_username(self):
        config_data = {"bot_name": "@ryo_test_bot"}
        self.assertEqual(app._telegram_bot_link(config_data), "https://t.me/ryo_test_bot")

    def test_telegram_bot_link_uses_get_me_from_bot_token(self):
        config_data = {"bot_token": "123:abc", "bot_name": "stale_name"}
        fake_payload = {
            "ok": True,
            "result": {"id": 123, "is_bot": True, "username": "resolved_from_token"},
        }
        with mock.patch.dict(app._TELEGRAM_BOT_USERNAME_CACHE, {}, clear=True):
            with mock.patch.object(app, "urlopen", return_value=self._FakeHTTPResponse(fake_payload)) as open_mock:
                link = app._telegram_bot_link(config_data)
        self.assertEqual(link, "https://t.me/resolved_from_token")
        self.assertEqual(open_mock.call_count, 1)

    def test_telegram_bot_link_falls_back_to_bot_name_when_get_me_fails(self):
        config_data = {"bot_token": "123:abc", "bot_name": "@fallback_name"}
        with mock.patch.dict(app._TELEGRAM_BOT_USERNAME_CACHE, {}, clear=True):
            with mock.patch.object(app, "urlopen", side_effect=app.URLError("down")):
                link = app._telegram_bot_link(config_data)
        self.assertEqual(link, "https://t.me/fallback_name")

    def test_telegram_bot_link_caches_get_me_result(self):
        config_data = {"bot_token": "123:abc"}
        fake_payload = {
            "ok": True,
            "result": {"id": 123, "is_bot": True, "username": "cached_bot"},
        }
        with mock.patch.dict(app._TELEGRAM_BOT_USERNAME_CACHE, {}, clear=True):
            with mock.patch.object(app, "urlopen", return_value=self._FakeHTTPResponse(fake_payload)) as open_mock:
                first = app._telegram_bot_link(config_data)
                second = app._telegram_bot_link(config_data)
        self.assertEqual(first, "https://t.me/cached_bot")
        self.assertEqual(second, "https://t.me/cached_bot")
        self.assertEqual(open_mock.call_count, 1)

    def test_telegram_bot_link_missing_when_placeholder(self):
        config_data = {"bot_name": "telegram_bot_username"}
        self.assertIsNone(app._telegram_bot_link(config_data))

    def test_route_access_summary_for_web_when_missing(self):
        summary = app._route_access_summary("web", {}, {"web": {"host": "127.0.0.1", "port": 4747}})
        self.assertIn("local web url:", summary)

    def test_route_open_action_web_uses_local_endpoint(self):
        with mock.patch.object(app, "_open_url_with_system_handler", return_value=(True, "opened")):
            ok, lines = app._route_open_action(
                "web",
                {},
                {},
                {"web": {"host": "127.0.0.1", "port": 4747}},
            )
        self.assertTrue(ok)
        self.assertTrue(any("Local Web UI URL: http://127.0.0.1:4747/" in line for line in lines))

    def test_route_open_action_cli_uses_terminal_launcher(self):
        with mock.patch.object(app, "_launch_script_in_transient_terminal", return_value=(True, "ok")) as launch_mock:
            ok, lines = app._route_open_action("cli", {"script": "cli_ui.py"}, {}, {})
        self.assertTrue(ok)
        self.assertTrue(launch_mock.called)
        self.assertTrue(any("cli_ui.py" in line for line in lines))

    def test_find_available_port_increments(self):
        with mock.patch.object(app, "_is_tcp_port_available", side_effect=[False, True]):
            selected = app._find_available_port("127.0.0.1", 4747, 10)
        self.assertEqual(selected, 4748)

    def test_web_public_host_normalizes_wildcard_bind(self):
        self.assertEqual(app._web_public_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(app._web_public_host("127.0.0.1"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
