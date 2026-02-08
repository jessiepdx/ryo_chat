import tempfile
import unittest
from pathlib import Path

from hypermindlabs.agent_definitions import AgentDefinitionStore, runtime_options_from_agent_definition


class AgentDefinitionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tempdir.name) / "agent_definitions.json"
        self.store = AgentDefinitionStore(storage_path=self.store_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_create_version_and_rollback(self):
        created = self.store.create_definition(
            {
                "identity": {"name": "Planner Agent", "description": "v1"},
                "model_policy": {"default_model": "qwen3-vl:latest"},
            },
            author_member_id=9,
            change_summary="initial",
        )
        definition_id = created["definition_id"]
        self.assertEqual(created["active_version"], 1)
        self.assertEqual(created["name"], "Planner Agent")

        updated = self.store.create_version(
            definition_id,
            {
                "identity": {"name": "Planner Agent", "description": "v2"},
                "model_policy": {"default_model": "gemma3:4b"},
                "tool_access_policy": {"enabled_tools": ["skipTools"]},
            },
            author_member_id=9,
            change_summary="v2 update",
        )
        self.assertEqual(updated["active_version"], 2)
        self.assertEqual(updated["selected_version"], 2)
        self.assertEqual(updated["definition"]["model_policy"]["default_model"], "gemma3:4b")

        rolled = self.store.rollback(
            definition_id,
            target_version=1,
            author_member_id=9,
            change_summary="rollback",
        )
        self.assertEqual(rolled["active_version"], 3)
        self.assertEqual(rolled["definition"]["model_policy"]["default_model"], "qwen3-vl:latest")
        self.assertEqual(len(rolled["versions"]), 3)

    def test_export_import_roundtrip_json(self):
        created = self.store.create_definition(
            {
                "identity": {"name": "Importer Agent"},
                "model_policy": {"default_model": "llama3.2:latest"},
            },
            author_member_id=4,
            change_summary="seed",
        )
        definition_id = created["definition_id"]
        exported = self.store.export_definition(definition_id, fmt="json")
        self.assertIn("definition_id", exported)
        self.assertIn("Importer Agent", exported)

        imported = self.store.import_definition(
            raw_payload=exported,
            fmt="json",
            author_member_id=4,
            change_summary="import",
        )
        self.assertNotEqual(imported["definition_id"], definition_id)
        self.assertEqual(imported["definition"]["identity"]["name"], "Importer Agent")

    def test_runtime_options_map_from_definition(self):
        options = runtime_options_from_agent_definition(
            {
                "identity": {"name": "Runtime Agent"},
                "model_policy": {
                    "default_model": "qwen3-vl:latest",
                    "allowed_models": ["qwen3-vl:latest", "gemma3:4b"],
                    "temperature": 0.4,
                },
                "guardrail_hooks": {"allow_internal_diagnostics": True},
                "tool_access_policy": {
                    "enabled_tools": ["skipTools"],
                    "per_tool": {"skipTools": {"timeout_seconds": 1.5, "max_retries": 0}},
                },
            }
        )
        self.assertEqual(options["model_requested"], "qwen3-vl:latest")
        self.assertIn("skipTools", options["enabled_tools"])
        self.assertTrue(options["allow_internal_diagnostics"])
        self.assertIn("skipTools", options["tool_policy"])


if __name__ == "__main__":
    unittest.main()
