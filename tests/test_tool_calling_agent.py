import unittest

from hypermindlabs.tool_registry import build_tool_specs, register_runtime_tools
from hypermindlabs.tool_runtime import ToolRuntime


class _FuncCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _FuncCall(name, arguments)


def _echo_tool(queryString: str, count: int = 1):
    return {"query": queryString, "count": count}


def _skip_tool():
    return {"skipped": True}


class TestToolCallingAgentStack(unittest.TestCase):
    def _build_runtime(self, reject_unknown_args: bool = False) -> ToolRuntime:
        runtime = ToolRuntime(api_keys={"brave_search": "x"})
        specs = build_tool_specs(
            brave_search_fn=_echo_tool,
            chat_history_search_fn=_echo_tool,
            knowledge_search_fn=_echo_tool,
            skip_tools_fn=_skip_tool,
            knowledge_domains=["General", "Project"],
        )
        register_runtime_tools(
            runtime=runtime,
            specs=specs,
            tool_policy={},
            default_timeout_seconds=8.0,
            default_max_retries=1,
            reject_unknown_args=reject_unknown_args,
        )
        return runtime

    def test_tool_call_argument_coercion_from_string(self):
        runtime = self._build_runtime()
        tool_call = _ToolCall(
            name="knowledgeSearch",
            arguments={"queryString": "   vector index health   ", "count": "3"},
        )
        result = runtime.execute_tool_call(tool_call)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_results"]["query"], "vector index health")
        self.assertEqual(result["tool_results"]["count"], 3)

    def test_nested_properties_arguments_are_parsed(self):
        runtime = self._build_runtime()
        tool_call = {
            "function": {
                "name": "chatHistorySearch",
                "arguments": {"properties": {"queryString": "recent failures", "count": "2"}},
            }
        }
        result = runtime.execute_tool_call(tool_call)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_results"]["count"], 2)

    def test_unknown_tool_is_structured_error(self):
        runtime = self._build_runtime()
        tool_call = _ToolCall(name="totallyUnknownTool", arguments={"queryString": "x"})
        result = runtime.execute_tool_call(tool_call)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "tool_not_registered")
        self.assertEqual(result["tool_name"], "totallyUnknownTool")

    def test_missing_tool_name_is_invalid_tool_call(self):
        runtime = self._build_runtime()
        tool_call = {"function": {"name": "", "arguments": {"queryString": "x"}}}
        result = runtime.execute_tool_call(tool_call)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_tool_call")

    def test_reject_unknown_args_when_enabled(self):
        runtime = self._build_runtime(reject_unknown_args=True)
        tool_call = _ToolCall(
            name="knowledgeSearch",
            arguments={"queryString": "schema", "count": 2, "extra": "bad"},
        )
        result = runtime.execute_tool_call(tool_call)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertIn("unknown_args", result["error"]["details"])


if __name__ == "__main__":
    unittest.main()
