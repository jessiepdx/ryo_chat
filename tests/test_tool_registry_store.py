import tempfile
import unittest
from pathlib import Path

from hypermindlabs.tool_registry import (
    ToolRegistryStore,
    build_tool_specs,
    model_tool_definitions,
    register_runtime_tools,
    tool_catalog_entries,
)
from hypermindlabs.tool_runtime import ToolRuntime


def _echo_tool(queryString: str, count: int = 1):
    return {"query": queryString, "count": count}


def _skip_tool():
    return {"skipped": True}


class ToolRegistryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tempdir.name) / "tool_registry.json"
        self.store = ToolRegistryStore(storage_path=self.store_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_upsert_and_remove_custom_tool(self):
        created = self.store.upsert_custom_tool(
            {
                "name": "customEcho",
                "description": "Echo custom values",
                "enabled": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "count": {"type": "integer", "default": 1},
                    },
                    "required": ["topic"],
                },
                "handler_mode": "echo",
            },
            actor_member_id=7,
        )
        self.assertEqual(created["name"], "customEcho")
        self.assertTrue(created["enabled"])

        custom_tools = self.store.list_custom_tools(include_disabled=True)
        self.assertEqual(len(custom_tools), 1)
        self.assertEqual(custom_tools[0]["name"], "customEcho")

        removed = self.store.remove_custom_tool("customEcho")
        self.assertTrue(removed)
        self.assertEqual(self.store.list_custom_tools(include_disabled=True), [])

    def test_custom_tool_flows_into_model_defs_and_runtime(self):
        self.store.upsert_custom_tool(
            {
                "name": "customStatic",
                "description": "Static response",
                "enabled": True,
                "handler_mode": "static",
                "static_result": {"ok": True},
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            actor_member_id=8,
        )
        custom_tools = self.store.list_custom_tools(include_disabled=False)
        specs = build_tool_specs(
            brave_search_fn=_echo_tool,
            chat_history_search_fn=_echo_tool,
            knowledge_search_fn=_echo_tool,
            skip_tools_fn=_skip_tool,
            knowledge_domains=["General"],
            custom_tool_entries=custom_tools,
        )

        model_defs = model_tool_definitions(specs)
        names = [item.get("function", {}).get("name") for item in model_defs]
        self.assertIn("customStatic", names)

        runtime = ToolRuntime(api_keys={"brave_search": "x"})
        register_runtime_tools(runtime, specs)
        result = runtime.execute("customStatic", {"query": "hello"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_results"]["ok"], True)

        catalog = tool_catalog_entries(specs, custom_entries=custom_tools)
        custom_rows = [item for item in catalog if item.get("name") == "customStatic"]
        self.assertEqual(len(custom_rows), 1)
        self.assertEqual(custom_rows[0]["source"], "custom")


if __name__ == "__main__":
    unittest.main()
