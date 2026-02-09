import tempfile
import threading
import time
import unittest
from pathlib import Path

from hypermindlabs.approval_manager import ApprovalManager, ApprovalValidationError


class ApprovalManagerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.manager = ApprovalManager(storage_path=Path(self.tempdir.name) / "tool_approvals.json")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_request_and_decide(self):
        request = self.manager.request_approval(
            run_id="run-123",
            tool_name="dangerTool",
            tool_args={"id": 1},
            requested_by_member_id=7,
            run_owner_member_id=7,
            timeout_seconds=30,
        )
        request_id = request["request_id"]
        self.assertEqual(request["status"], "pending")

        pending = self.manager.list_requests(status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["request_id"], request_id)

        decided = self.manager.decide_request(
            request_id,
            decision="approve",
            actor_member_id=9,
            reason="Looks safe.",
        )
        self.assertEqual(decided["status"], "approved")
        self.assertEqual(decided["decided_by_member_id"], 9)

    def test_wait_for_decision_receives_async_approval(self):
        request = self.manager.request_approval(
            run_id="run-abc",
            tool_name="mutatingTool",
            timeout_seconds=15,
        )
        request_id = request["request_id"]

        def decide_later():
            time.sleep(0.2)
            self.manager.decide_request(request_id, decision="deny", reason="blocked")

        worker = threading.Thread(target=decide_later, daemon=True)
        worker.start()

        resolved = self.manager.wait_for_decision(request_id, timeout_seconds=5, poll_interval_seconds=0.05)
        self.assertEqual(resolved["status"], "denied")

    def test_expired_request_cannot_be_decided(self):
        request = self.manager.request_approval(
            run_id="run-expire",
            tool_name="mutatingTool",
            timeout_seconds=1,
        )
        request_id = request["request_id"]
        time.sleep(1.1)
        fetched = self.manager.get_request(request_id)
        self.assertEqual(fetched["status"], "expired")
        with self.assertRaises(ApprovalValidationError):
            self.manager.decide_request(request_id, decision="approve")


if __name__ == "__main__":
    unittest.main()
