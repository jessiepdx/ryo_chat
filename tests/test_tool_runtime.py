import time
import tempfile
import unittest

from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.tool_runtime import ToolDefinition, ToolRuntime


def _echo_tool(queryString: str, count: int = 5):
    return {"query": queryString, "count": count}


def _slow_tool(queryString: str, count: int = 1):
    time.sleep(0.2)
    return {"query": queryString, "count": count}


def _error_tool(queryString: str):
    raise RuntimeError("boom")


class TestToolRuntime(unittest.TestCase):
    def test_unknown_tool_returns_structured_error(self):
        runtime = ToolRuntime()
        result = runtime.execute("missingTool", {"queryString": "hello"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "tool_not_registered")

    def test_missing_api_key_returns_structured_error(self):
        runtime = ToolRuntime(api_keys={"brave_search": ""})
        runtime.register_tool(
            ToolDefinition(
                name="braveSearch",
                function=_echo_tool,
                required_args=("queryString",),
                required_api_key="brave_search",
            )
        )
        result = runtime.execute("braveSearch", {"queryString": "hello"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "missing_api_key")

    def test_invalid_arguments_returns_structured_error(self):
        runtime = ToolRuntime()
        runtime.register_tool(
            ToolDefinition(
                name="knowledgeSearch",
                function=_echo_tool,
                required_args=("queryString",),
            )
        )
        result = runtime.execute("knowledgeSearch", {"count": 2})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertIn("queryString", result["error"]["details"]["missing_required"])

    def test_timeout_returns_structured_error(self):
        runtime = ToolRuntime()
        runtime.register_tool(
            ToolDefinition(
                name="slowTool",
                function=_slow_tool,
                required_args=("queryString",),
                timeout_seconds=0.01,
                max_retries=1,
            )
        )
        result = runtime.execute("slowTool", {"queryString": "hello"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "tool_timeout")
        self.assertEqual(result["error"]["attempts"], 2)

    def test_execution_exception_returns_structured_error(self):
        runtime = ToolRuntime()
        runtime.register_tool(
            ToolDefinition(
                name="errorTool",
                function=_error_tool,
                required_args=("queryString",),
                timeout_seconds=1.0,
                max_retries=0,
            )
        )
        result = runtime.execute("errorTool", {"queryString": "hello"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "tool_execution_failed")

    def test_success_returns_output_with_defaults(self):
        runtime = ToolRuntime()
        runtime.register_tool(
            ToolDefinition(
                name="chatHistorySearch",
                function=_echo_tool,
                required_args=("queryString",),
                optional_args={"count": 2},
            )
        )
        result = runtime.execute("chatHistorySearch", {"queryString": "hello"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_results"]["query"], "hello")
        self.assertEqual(result["tool_results"]["count"], 2)

    def test_sandbox_blocks_network_when_disabled(self):
        runtime = ToolRuntime()
        runtime.register_tool(
            ToolDefinition(
                name="httpFetch",
                function=_echo_tool,
                required_args=("queryString",),
                sandbox_policy={
                    "tool_name": "httpFetch",
                    "network": {"enabled": False},
                },
            )
        )
        result = runtime.execute("httpFetch", {"queryString": "https://example.com"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "sandbox_blocked")

    def test_dry_run_returns_mock_without_execution(self):
        runtime = ToolRuntime(default_dry_run=True)
        runtime.register_tool(
            ToolDefinition(
                name="mutableTool",
                function=_error_tool,
                required_args=("queryString",),
                mock_result={"status": "mocked"},
            )
        )
        result = runtime.execute("mutableTool", {"queryString": "ignored"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_results"]["status"], "mocked")
        self.assertEqual(result["attempts"], 0)

    def test_approval_timeout_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            approval_manager = ApprovalManager(storage_path=f"{tempdir}/approvals.json")
            runtime = ToolRuntime(
                approval_manager=approval_manager,
                default_approval_timeout_seconds=1,
                approval_poll_interval_seconds=0.05,
            )
            runtime.set_runtime_context({"run_id": "run-1", "member_id": 1})
            runtime.register_tool(
                ToolDefinition(
                    name="dangerousTool",
                    function=_echo_tool,
                    required_args=("queryString",),
                    side_effect_class="mutating",
                    sandbox_policy={
                        "tool_name": "dangerousTool",
                        "require_approval": True,
                        "approval_timeout_seconds": 1,
                    },
                )
            )
            result = runtime.execute("dangerousTool", {"queryString": "hello"})
            self.assertEqual(result["status"], "error")
            self.assertIn(result["error"]["code"], {"approval_denied", "approval_timeout"})


if __name__ == "__main__":
    unittest.main()
