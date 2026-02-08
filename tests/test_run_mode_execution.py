import inspect
import sys
import time
import types
import unittest

from hypermindlabs.run_manager import RunManager


def _wait_for_terminal(manager: RunManager, run_id: str, timeout_seconds: float = 4.0) -> dict:
    start = time.time()
    while time.time() - start < timeout_seconds:
        run = manager.get_run(run_id)
        if run and run.get("status") in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for run to reach terminal state")


class _FakeConversationOrchestrator:
    calls: list["_FakeConversationOrchestrator"] = []

    def __init__(self, message, memberID, context, messageID=None, options=None):
        self.message = str(message or "")
        self.memberID = int(memberID)
        self.context = context if isinstance(context, dict) else {}
        self.messageID = messageID
        self.options = options if isinstance(options, dict) else {}
        self.stats = {"token_count": max(1, len(self.message.split()))}
        self.promptHistoryID = f"prompt-{self.memberID}"
        self.streamingResponse = lambda _chunk: None
        _FakeConversationOrchestrator.calls.append(self)

    async def runAgents(self):
        stage_callback = self.options.get("stage_callback")
        if callable(stage_callback):
            maybe_awaitable = stage_callback(
                {
                    "stage": "analysis",
                    "detail": f"processed {self.message}",
                    "meta": {"member_id": self.memberID},
                }
            )
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

        self.streamingResponse(f"{self.message[:32]}")
        model_name = str(self.options.get("model_requested") or "router-default")
        return f"{self.message}::{model_name}"


class RunModeExecutionTests(unittest.TestCase):
    def setUp(self):
        self._original_agents_module = sys.modules.get("hypermindlabs.agents")
        fake_agents_module = types.ModuleType("hypermindlabs.agents")
        fake_agents_module.ConversationOrchestrator = _FakeConversationOrchestrator
        sys.modules["hypermindlabs.agents"] = fake_agents_module
        _FakeConversationOrchestrator.calls = []

    def tearDown(self):
        if self._original_agents_module is None:
            sys.modules.pop("hypermindlabs.agents", None)
        else:
            sys.modules["hypermindlabs.agents"] = self._original_agents_module

    def test_workflow_mode_emits_stages_and_workflow_artifact(self):
        manager = RunManager(enable_db=False)
        run = manager.create_run(
            member_id=41,
            mode="workflow",
            request_payload={
                "message": "seed",
                "workflow_steps": [
                    {"id": "plan", "prompt": "create plan"},
                    {"id": "verify", "prompt": "verify plan"},
                ],
                "options": {},
            },
            auto_start=True,
        )
        terminal = _wait_for_terminal(manager, run["run_id"])
        self.assertEqual(terminal["status"], "completed")

        events = manager.get_events(run["run_id"], after_seq=0, limit=200)
        stages = {str(item.get("stage")) for item in events}
        self.assertIn("workflow.plan.start", stages)
        self.assertIn("workflow.plan.complete", stages)
        self.assertIn("workflow.verify.start", stages)
        self.assertIn("workflow.verify.complete", stages)

        artifacts = manager.get_artifacts(run["run_id"], limit=50)
        workflow_artifacts = [item for item in artifacts if item.get("artifact_type") == "workflow"]
        self.assertEqual(len(workflow_artifacts), 1)
        steps = workflow_artifacts[0].get("artifact", {}).get("steps", [])
        self.assertEqual(len(steps), 2)
        self.assertEqual([step.get("step_id") for step in steps], ["plan", "verify"])

    def test_batch_mode_creates_row_results(self):
        manager = RunManager(enable_db=False)
        run = manager.create_run(
            member_id=42,
            mode="batch",
            request_payload={
                "batch_inputs": ["alpha", "beta", "gamma"],
                "options": {},
            },
            auto_start=True,
        )
        terminal = _wait_for_terminal(manager, run["run_id"])
        self.assertEqual(terminal["status"], "completed")

        artifacts = manager.get_artifacts(run["run_id"], limit=50)
        batch_artifacts = [item for item in artifacts if item.get("artifact_type") == "batch"]
        self.assertEqual(len(batch_artifacts), 1)
        results = batch_artifacts[0].get("artifact", {}).get("results", [])
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.get("status") == "completed" for result in results))

    def test_compare_mode_applies_each_requested_model(self):
        manager = RunManager(enable_db=False)
        run = manager.create_run(
            member_id=43,
            mode="compare",
            request_payload={
                "message": "compare this",
                "compare_models": ["model-a:latest", "model-b:latest"],
                "options": {},
            },
            auto_start=True,
        )
        terminal = _wait_for_terminal(manager, run["run_id"])
        self.assertEqual(terminal["status"], "completed")

        requested_models = [call.options.get("model_requested") for call in _FakeConversationOrchestrator.calls]
        self.assertEqual(requested_models, ["model-a:latest", "model-b:latest"])

        artifacts = manager.get_artifacts(run["run_id"], limit=50)
        compare_artifacts = [item for item in artifacts if item.get("artifact_type") == "compare"]
        self.assertEqual(len(compare_artifacts), 1)
        result_models = [row.get("model") for row in compare_artifacts[0].get("artifact", {}).get("results", [])]
        self.assertEqual(result_models, ["model-a:latest", "model-b:latest"])

    def test_replay_mode_uses_source_run_and_state_plan(self):
        manager = RunManager(enable_db=False)
        source = manager.create_run(
            member_id=44,
            mode="chat",
            request_payload={
                "message": "source prompt",
                "options": {"temperature": 0.2},
            },
            auto_start=False,
        )
        manager.append_snapshot(
            source["run_id"],
            step_seq=3,
            stage="chat.final",
            state={"memory": {"summary": "before"}},
        )

        replay = manager.create_run(
            member_id=44,
            mode="replay",
            request_payload={
                "source_run_id": source["run_id"],
                "replay_from_seq": 3,
                "state_overrides": {"memory": {"summary": "after"}},
                "options": {},
            },
            auto_start=True,
        )
        terminal = _wait_for_terminal(manager, replay["run_id"])
        self.assertEqual(terminal["status"], "completed")

        artifacts = manager.get_artifacts(replay["run_id"], limit=100)
        replay_artifacts = [item for item in artifacts if item.get("artifact_type") == "replay"]
        self.assertEqual(len(replay_artifacts), 1)
        replay_state = replay_artifacts[0].get("artifact", {})
        self.assertEqual(replay_state.get("selected_snapshot_seq"), 3)
        self.assertEqual(replay_state.get("merged_state", {}).get("memory", {}).get("summary"), "after")

        events = manager.get_events(replay["run_id"], after_seq=0, limit=200)
        stages = {str(item.get("stage")) for item in events}
        self.assertIn("replay.prepare", stages)

        self.assertGreaterEqual(len(_FakeConversationOrchestrator.calls), 1)
        replay_context = _FakeConversationOrchestrator.calls[-1].context.get("replay_context", {})
        self.assertEqual(replay_context.get("source_run_id"), source["run_id"])
        self.assertEqual(replay_context.get("selected_snapshot_seq"), 3)
        self.assertIn("memory", replay_context.get("state_override_keys", []))


if __name__ == "__main__":
    unittest.main()
