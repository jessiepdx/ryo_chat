import unittest

from hypermindlabs.replay_manager import ReplayManager


class _FakeRunManager:
    def __init__(self):
        self.calls = []

    def replay_run(self, run_id, replay_from_seq=None, state_overrides=None, auto_start=True):
        self.calls.append(
            {
                "run_id": run_id,
                "replay_from_seq": replay_from_seq,
                "state_overrides": state_overrides,
                "auto_start": auto_start,
            }
        )
        return {"run_id": "child-run", "parent_run_id": run_id}


class ReplayManagerTests(unittest.TestCase):
    def test_replay_from_start_forwards_arguments(self):
        fake = _FakeRunManager()
        manager = ReplayManager(fake)
        result = manager.replay_from_start("run-1", auto_start=False)

        self.assertEqual(result["parent_run_id"], "run-1")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["run_id"], "run-1")
        self.assertIsNone(fake.calls[0]["replay_from_seq"])
        self.assertFalse(fake.calls[0]["auto_start"])

    def test_replay_with_state_forwards_step_and_overrides(self):
        fake = _FakeRunManager()
        manager = ReplayManager(fake)
        manager.replay_with_state(
            "run-2",
            step_seq=7,
            state_overrides={"memory": {"summary": "edited"}},
            auto_start=True,
        )

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["run_id"], "run-2")
        self.assertEqual(fake.calls[0]["replay_from_seq"], 7)
        self.assertIn("memory", fake.calls[0]["state_overrides"])
        self.assertTrue(fake.calls[0]["auto_start"])


if __name__ == "__main__":
    unittest.main()
