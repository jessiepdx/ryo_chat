import json
import tempfile
import unittest
from pathlib import Path

from hypermindlabs.policy_manager import (
    FALLBACK_SYSTEM_PROMPT,
    PolicyManager,
    PolicyValidationError,
)


def _write_policy(base_dir: Path, policy_name: str, payload: dict) -> None:
    policy_path = base_dir / "policies" / "agent" / f"{policy_name}_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_prompt(base_dir: Path, policy_name: str, text: str = "System prompt") -> None:
    prompt_path = base_dir / "policies" / "agent" / "system_prompt" / f"{policy_name}_sp.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(text, encoding="utf-8")


class TestPolicyManager(unittest.TestCase):
    def test_resolve_host_precedence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={"chat": {"url": "http://configured:11434"}},
                endpoint_override="http://custom:11434",
            )
            self.assertEqual(
                manager.resolve_host("chat_conversation"),
                "http://custom:11434",
            )

            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={"chat": {"url": "http://configured:11434"}},
                endpoint_override="not-a-url",
            )
            self.assertEqual(
                manager.resolve_host("chat_conversation"),
                "http://configured:11434",
            )

            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={},
            )
            self.assertEqual(
                manager.resolve_host("chat_conversation"),
                "http://127.0.0.1:11434",
            )

    def test_validate_policy_reports_missing_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            _write_policy(
                base,
                "chat_conversation",
                {
                    "allow_custom_system_prompt": False,
                    "allowed_models": ["llama3.2:latest"],
                },
            )
            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={},
            )
            manager.discover_models = lambda host: ([], "probe skipped in tests")

            report = manager.validate_policy("chat_conversation")
            self.assertTrue(any("Missing system prompt file" in error for error in report.errors))

    def test_validate_policy_schema_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            _write_policy(
                base,
                "chat_conversation",
                {
                    "allow_custom_system_prompt": "false",
                    "allowed_models": "llama3.2:latest",
                },
            )
            _write_prompt(base, "chat_conversation")
            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={},
            )
            manager.discover_models = lambda host: ([], "probe skipped in tests")

            report = manager.validate_policy("chat_conversation")
            self.assertTrue(any("allow_custom_system_prompt" in error for error in report.errors))
            self.assertTrue(any("allowed_models" in error for error in report.errors))

    def test_model_mismatch_warn_vs_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            _write_policy(
                base,
                "chat_conversation",
                {
                    "allow_custom_system_prompt": False,
                    "allowed_models": ["missing-model", "existing-model"],
                },
            )
            _write_prompt(base, "chat_conversation")
            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={"chat": {"url": "http://127.0.0.1:11434"}},
            )

            manager.discover_models = lambda host: (["existing-model"], None)
            warn_report = manager.validate_policy("chat_conversation", strict_model_check=False)
            self.assertEqual(len(warn_report.errors), 0)
            self.assertTrue(any("not found on endpoint" in warning for warning in warn_report.warnings))

            strict_report = manager.validate_policy("chat_conversation", strict_model_check=True)
            self.assertTrue(any("not found on endpoint" in error for error in strict_report.errors))

    def test_save_policy_rolls_back_on_post_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            policy_name = "chat_conversation"
            policy_path = base / "policies" / "agent" / f"{policy_name}_policy.json"
            original_policy = {
                "allow_custom_system_prompt": False,
                "allowed_models": ["llama3.2:latest"],
            }
            _write_policy(base, policy_name, original_policy)
            _write_prompt(base, policy_name)

            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={},
            )
            manager.discover_models = lambda host: ([], "probe skipped in tests")

            original_validate = manager.validate_policy

            def forced_invalid(policy_name: str, strict_model_check: bool = False):
                report = original_validate(policy_name, strict_model_check=strict_model_check)
                report.errors.append("forced post-save validation failure")
                return report

            manager.validate_policy = forced_invalid
            save_result = manager.save_policy(
                policy_name=policy_name,
                updates={"allow_custom_system_prompt": True},
                strict_model_check=False,
            )

            self.assertFalse(save_result.saved)
            self.assertTrue(save_result.rollback_performed)
            restored = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertEqual(restored, original_policy)

    def test_load_policy_and_prompt_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            policy_name = "chat_conversation"
            _write_policy(
                base,
                policy_name,
                {
                    "allow_custom_system_prompt": "bad",
                    "allowed_models": [],
                },
            )
            manager = PolicyManager(
                policies_dir=base / "policies" / "agent",
                inference_config={},
            )
            manager.discover_models = lambda host: ([], "probe skipped in tests")

            fallback_policy = manager.load_policy(policy_name, strict=False)
            self.assertIn("allowed_models", fallback_policy)
            self.assertTrue(isinstance(fallback_policy.get("allow_custom_system_prompt"), bool))

            with self.assertRaises(PolicyValidationError):
                manager.load_system_prompt(policy_name=policy_name, strict=True)

            prompt = manager.load_system_prompt(policy_name=policy_name, strict=False)
            self.assertEqual(prompt, FALLBACK_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
