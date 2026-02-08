import time
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


class RunManagerTests(unittest.TestCase):
    def test_create_run_executes_and_completes(self):
        def fake_execute(manager: RunManager, run_id: str):
            manager.append_event(
                run_id,
                event_type="run.stage",
                stage="fake.start",
                status="running",
                payload={"detail": "started"},
            )
            manager.append_snapshot(
                run_id,
                step_seq=2,
                stage="fake.start",
                state={"step": "fake.start"},
            )
            manager.append_artifact(
                run_id,
                artifact_type="text",
                artifact_name="fake_output",
                artifact={"value": "ok"},
                step_seq=2,
            )
            return {"response": "hello world"}

        manager = RunManager(execute_fn=fake_execute, enable_db=False)
        run = manager.create_run(
            member_id=7,
            mode="chat",
            request_payload={"message": "test"},
            auto_start=True,
        )

        terminal = _wait_for_terminal(manager, run["run_id"])
        self.assertEqual(terminal["status"], "completed")

        events = manager.get_events(run["run_id"], after_seq=0, limit=100)
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[0]["event_type"], "run.created")
        self.assertTrue(any(item["event_type"] == "run.completed" for item in events))

        snapshots = manager.get_snapshots(run["run_id"], limit=20)
        self.assertEqual(len(snapshots), 1)

        artifacts = manager.get_artifacts(run["run_id"], limit=20)
        self.assertEqual(len(artifacts), 1)

    def test_cancel_queued_run_sets_cancelled(self):
        manager = RunManager(execute_fn=lambda _m, _r: {"response": "ok"}, enable_db=False)
        run = manager.create_run(
            member_id=3,
            mode="chat",
            request_payload={"message": "cancel me"},
            auto_start=False,
        )

        cancelled = manager.cancel_run(run["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")

    def test_replay_run_creates_child_lineage(self):
        manager = RunManager(execute_fn=lambda _m, _r: {"response": "ok"}, enable_db=False)
        source = manager.create_run(
            member_id=9,
            mode="chat",
            request_payload={"message": "source"},
            auto_start=False,
        )

        replay = manager.replay_run(source["run_id"], replay_from_seq=2, auto_start=False)
        self.assertEqual(replay["mode"], "replay")
        self.assertEqual(replay["parent_run_id"], source["run_id"])
        self.assertEqual(replay["lineage"]["type"], "replay")

    def test_metrics_summary_reports_status_counts(self):
        manager = RunManager(execute_fn=lambda _m, _r: {"response": "ok"}, enable_db=False)
        run = manager.create_run(
            member_id=11,
            mode="chat",
            request_payload={"message": "metrics"},
            auto_start=True,
        )
        _wait_for_terminal(manager, run["run_id"])

        metrics = manager.metrics_summary()
        self.assertGreaterEqual(metrics["total_runs"], 1)
        self.assertIn("completed", metrics["status_counts"])


if __name__ == "__main__":
    unittest.main()
