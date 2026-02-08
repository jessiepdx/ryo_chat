import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
