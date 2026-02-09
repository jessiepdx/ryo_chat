import tempfile
import unittest
from pathlib import Path

from hypermindlabs.tool_sandbox import (
    ToolSandboxEnforcer,
    ToolSandboxPolicyStore,
    normalize_tool_sandbox_policy,
)


class ToolSandboxTests(unittest.TestCase):
    def test_network_disabled_blocks_url_argument(self):
        enforcer = ToolSandboxEnforcer()
        decision = enforcer.evaluate(
            tool_name="httpFetch",
            prepared_args={"url": "https://example.com/resource"},
            tool_policy={
                "tool_name": "httpFetch",
                "network": {"enabled": False},
            },
        )
        self.assertEqual(decision["status"], "deny")
        self.assertEqual(decision["code"], "sandbox_block_network")

    def test_allowlist_blocks_unapproved_hosts(self):
        enforcer = ToolSandboxEnforcer()
        decision = enforcer.evaluate(
            tool_name="webTool",
            prepared_args={"url": "https://not-allowed.example/path"},
            tool_policy={
                "tool_name": "webTool",
                "network": {"enabled": True, "allowlist_domains": ["api.search.brave.com"]},
            },
        )
        self.assertEqual(decision["status"], "deny")
        self.assertEqual(decision["code"], "sandbox_block_network_allowlist")

    def test_filesystem_mode_none_blocks_path_args(self):
        enforcer = ToolSandboxEnforcer()
        decision = enforcer.evaluate(
            tool_name="fileTool",
            prepared_args={"file_path": "/tmp/output.txt"},
            tool_policy={
                "tool_name": "fileTool",
                "filesystem": {"mode": "none"},
            },
        )
        self.assertEqual(decision["status"], "deny")
        self.assertEqual(decision["code"], "sandbox_block_filesystem")

    def test_filesystem_read_only_blocks_write_intent(self):
        enforcer = ToolSandboxEnforcer()
        decision = enforcer.evaluate(
            tool_name="fileTool",
            prepared_args={"path": "/tmp/output.txt", "write": True},
            tool_policy={
                "tool_name": "fileTool",
                "filesystem": {"mode": "read_only", "allowed_paths": ["/tmp"]},
            },
        )
        self.assertEqual(decision["status"], "deny")
        self.assertEqual(decision["code"], "sandbox_block_filesystem_write")

    def test_policy_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            store = ToolSandboxPolicyStore(storage_path=Path(tempdir) / "tool_sandbox.json")
            stored = store.upsert_policy(
                {
                    "tool_name": "customEcho",
                    "side_effect_class": "mutating",
                    "approval_required": True,
                    "sandbox_policy": {
                        "network": {"enabled": False},
                        "filesystem": {"mode": "read_write", "allowed_paths": ["/tmp/safe"]},
                    },
                },
                actor_member_id=5,
            )
            self.assertEqual(stored["tool_name"], "customEcho")
            self.assertEqual(stored["side_effect_class"], "mutating")
            self.assertTrue(stored["require_approval"])

            fetched = store.get_policy("customEcho")
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched["tool_name"], "customEcho")
            self.assertEqual(fetched["filesystem"]["mode"], "read_write")

    def test_normalization_sets_approval_for_sensitive_tools(self):
        normalized = normalize_tool_sandbox_policy(
            {
                "tool_name": "paymentTool",
                "side_effect_class": "sensitive",
            }
        )
        self.assertTrue(normalized["require_approval"])


if __name__ == "__main__":
    unittest.main()
