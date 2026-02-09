import tempfile
import unittest
from pathlib import Path

from hypermindlabs.agent_definitions import AgentDefinitionStore
from hypermindlabs.approval_manager import ApprovalManager
from hypermindlabs.replay_manager import ReplayManager
from hypermindlabs.run_manager import RunManager
from hypermindlabs.tool_registry import ToolRegistryStore
from hypermindlabs.tool_sandbox import ToolSandboxPolicyStore
from hypermindlabs.tool_test_harness import ToolTestHarnessStore

try:
    import web_ui
    _WEB_UI_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    web_ui = None
    _WEB_UI_IMPORT_ERROR = error


@unittest.skipIf(web_ui is None, f"web_ui dependencies unavailable: {_WEB_UI_IMPORT_ERROR}")
class PlaygroundRegistryAPITests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.agent_store = AgentDefinitionStore(storage_path=Path(self.tempdir.name) / "agent_definitions.json")
        self.tool_store = ToolRegistryStore(storage_path=Path(self.tempdir.name) / "tool_registry.json")

        self.original_agent_store = web_ui.agentDefinitions
        self.original_tool_store = web_ui.toolRegistry
        self.original_sandbox_store = getattr(web_ui, "toolSandboxPolicies", None)
        self.original_approvals = getattr(web_ui, "toolApprovals", None)
        self.original_tool_harness = getattr(web_ui, "toolHarness", None)
        self.original_run_manager = web_ui.playgroundRuns
        self.original_replay_manager = web_ui.playgroundReplay
        self.original_get_member = web_ui.members.getMemberByID

        web_ui.agentDefinitions = self.agent_store
        web_ui.toolRegistry = self.tool_store
        web_ui.toolSandboxPolicies = ToolSandboxPolicyStore(storage_path=Path(self.tempdir.name) / "tool_sandbox.json")
        web_ui.toolApprovals = ApprovalManager(storage_path=Path(self.tempdir.name) / "tool_approvals.json")
        web_ui.toolHarness = ToolTestHarnessStore(storage_path=Path(self.tempdir.name) / "tool_harness.json")
        web_ui.playgroundRuns = RunManager(enable_db=False)
        web_ui.playgroundReplay = ReplayManager(web_ui.playgroundRuns)
        web_ui.members.getMemberByID = lambda member_id: {
            "member_id": int(member_id),
            "username": "tester",
            "first_name": "Test",
            "last_name": "User",
            "roles": ["owner"],
        }

        web_ui.app.config["TESTING"] = True
        self.client = web_ui.app.test_client()
        with self.client.session_transaction() as session:
            session["member_id"] = 777

    def tearDown(self):
        web_ui.agentDefinitions = self.original_agent_store
        web_ui.toolRegistry = self.original_tool_store
        if self.original_sandbox_store is not None:
            web_ui.toolSandboxPolicies = self.original_sandbox_store
        if self.original_approvals is not None:
            web_ui.toolApprovals = self.original_approvals
        web_ui.toolHarness = self.original_tool_harness
        web_ui.playgroundRuns = self.original_run_manager
        web_ui.playgroundReplay = self.original_replay_manager
        web_ui.members.getMemberByID = self.original_get_member
        self.tempdir.cleanup()

    def test_agent_definition_crud_and_run_binding(self):
        create_response = self.client.post(
            "/api/agent-playground/agent-definitions",
            json={
                "definition": {
                    "identity": {"name": "API Agent"},
                    "model_policy": {"default_model": "qwen3-vl:latest"},
                },
                "change_summary": "seed",
            },
        )
        self.assertEqual(create_response.status_code, 201)
        created = create_response.get_json()["definition"]
        definition_id = created["definition_id"]

        version_response = self.client.post(
            f"/api/agent-playground/agent-definitions/{definition_id}/versions",
            json={
                "definition": {
                    "identity": {"name": "API Agent"},
                    "model_policy": {"default_model": "gemma3:4b"},
                },
                "change_summary": "switch model",
            },
        )
        self.assertEqual(version_response.status_code, 200)
        updated = version_response.get_json()["definition"]
        self.assertEqual(updated["active_version"], 2)

        run_response = self.client.post(
            "/api/agent-playground/runs",
            json={
                "mode": "chat",
                "message": "hello",
                "auto_start": False,
                "agent_definition_id": definition_id,
                "agent_definition_version": 2,
            },
        )
        self.assertEqual(run_response.status_code, 201)
        run_payload = run_response.get_json()["run"]
        request_payload = run_payload.get("request", {})
        self.assertEqual(
            request_payload.get("agent_definition", {}).get("model_policy", {}).get("default_model"),
            "gemma3:4b",
        )
        self.assertEqual(
            request_payload.get("agent_definition_ref", {}).get("definition_id"),
            definition_id,
        )

    def test_tool_registry_write_and_list(self):
        upsert_response = self.client.post(
            "/api/agent-playground/tools",
            json={
                "tool": {
                    "name": "customLookup",
                    "description": "Custom lookup helper",
                    "enabled": True,
                    "handler_mode": "echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                }
            },
        )
        self.assertEqual(upsert_response.status_code, 200)
        self.assertEqual(upsert_response.get_json()["tool"]["name"], "customLookup")

        list_response = self.client.get("/api/agent-playground/tools")
        self.assertEqual(list_response.status_code, 200)
        tools = list_response.get_json()["tools"]
        names = [item.get("name") for item in tools]
        self.assertIn("customLookup", names)

        delete_response = self.client.delete("/api/agent-playground/tools/customLookup")
        self.assertEqual(delete_response.status_code, 200)
        tools_after = delete_response.get_json()["tools"]
        names_after = [item.get("name") for item in tools_after]
        self.assertNotIn("customLookup", names_after)

    def test_sandbox_policy_and_approval_queue_api(self):
        sandbox_response = self.client.post(
            "/api/agent-playground/tools/sandbox-policies",
            json={
                "policy": {
                    "tool_name": "knowledgeSearch",
                    "side_effect_class": "read_only",
                    "sandbox_policy": {
                        "network": {"enabled": False},
                        "filesystem": {"mode": "none"},
                    },
                }
            },
        )
        self.assertEqual(sandbox_response.status_code, 200)
        policy = sandbox_response.get_json()["policy"]
        self.assertEqual(policy["tool_name"], "knowledgeSearch")

        list_response = self.client.get("/api/agent-playground/tools/sandbox-policies")
        self.assertEqual(list_response.status_code, 200)
        policies = list_response.get_json()["policies"]
        names = [item.get("tool_name") for item in policies]
        self.assertIn("knowledgeSearch", names)

        approval = web_ui.toolApprovals.request_approval(
            run_id="run-xyz",
            tool_name="dangerousTool",
            requested_by_member_id=777,
            run_owner_member_id=777,
            timeout_seconds=30,
        )
        approval_id = approval["request_id"]

        queue_response = self.client.get("/api/agent-playground/tool-approvals?status=pending")
        self.assertEqual(queue_response.status_code, 200)
        queue = queue_response.get_json()["approvals"]
        self.assertTrue(any(item.get("request_id") == approval_id for item in queue))

        decision_response = self.client.post(
            f"/api/agent-playground/tool-approvals/{approval_id}/decision",
            json={"decision": "approve", "reason": "approved by test"},
        )
        self.assertEqual(decision_response.status_code, 200)
        self.assertEqual(decision_response.get_json()["approval"]["status"], "approved")

    def test_tool_harness_case_run_and_baselines(self):
        tool_response = self.client.post(
            "/api/agent-playground/tools",
            json={
                "tool": {
                    "name": "customHarnessTool",
                    "description": "Harness custom tool",
                    "enabled": True,
                    "handler_mode": "static",
                    "static_result": {"ok": True, "version": 1},
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "queryString": {"type": "string"},
                        },
                        "required": ["queryString"],
                    },
                }
            },
        )
        self.assertEqual(tool_response.status_code, 200)

        create_case = self.client.post(
            "/api/agent-playground/tools/harness/cases",
            json={
                "case": {
                    "tool_name": "customHarnessTool",
                    "fixture_name": "Static output check",
                    "execution_mode": "real",
                    "input_args": {"queryString": "hello"},
                }
            },
        )
        self.assertEqual(create_case.status_code, 200)
        case = create_case.get_json()["case"]
        case_id = case["case_id"]

        first_run = self.client.post(
            "/api/agent-playground/tools/harness/run",
            json={"case_id": case_id, "persist_run": True},
        )
        self.assertEqual(first_run.status_code, 200)
        first_report = first_run.get_json()["report"]
        self.assertEqual(first_report["result"]["status"], "success")
        self.assertEqual(first_report["regression"]["status"], "missing_golden")

        save_contract = self.client.post(f"/api/agent-playground/tools/harness/cases/{case_id}/contract", json={})
        self.assertEqual(save_contract.status_code, 200)
        self.assertEqual(save_contract.get_json()["case"]["contract_snapshot"]["tool_name"], "customHarnessTool")

        save_golden = self.client.post(f"/api/agent-playground/tools/harness/cases/{case_id}/golden", json={})
        self.assertEqual(save_golden.status_code, 200)
        self.assertTrue(save_golden.get_json()["case"]["golden_hash"])

        second_run = self.client.post(
            "/api/agent-playground/tools/harness/run",
            json={"case_id": case_id, "persist_run": True},
        )
        self.assertEqual(second_run.status_code, 200)
        second_report = second_run.get_json()["report"]
        self.assertEqual(second_report["regression"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
