import copy
import unittest

import app


class TestAppLauncher(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
