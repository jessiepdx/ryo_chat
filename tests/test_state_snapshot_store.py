import unittest

from hypermindlabs.state_snapshot_store import build_replay_state_plan, select_snapshot_for_seq


class StateSnapshotStoreTests(unittest.TestCase):
    def test_select_snapshot_uses_closest_prior_step(self):
        snapshots = [
            {"step_seq": 2, "state": {"value": "early"}},
            {"step_seq": 5, "state": {"value": "mid"}},
            {"step_seq": 9, "state": {"value": "late"}},
        ]
        selected = select_snapshot_for_seq(snapshots, 7)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["step_seq"], 5)

    def test_select_snapshot_defaults_to_latest(self):
        snapshots = [
            {"step_seq": 1, "state": {"value": "one"}},
            {"step_seq": 3, "state": {"value": "three"}},
        ]
        selected = select_snapshot_for_seq(snapshots, None)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["step_seq"], 3)

    def test_build_replay_state_plan_merges_nested_overrides(self):
        snapshots = [
            {"step_seq": 4, "state": {"memory": {"summary": "base", "confidence": 0.2}, "mode": "chat"}},
        ]
        plan = build_replay_state_plan(
            snapshots,
            replay_from_seq=4,
            state_overrides={"memory": {"summary": "edited"}},
        )
        self.assertEqual(plan["selected_snapshot_seq"], 4)
        self.assertEqual(plan["merged_state"]["memory"]["summary"], "edited")
        self.assertEqual(plan["merged_state"]["memory"]["confidence"], 0.2)
        self.assertEqual(plan["merged_state"]["mode"], "chat")
        self.assertIn("memory", plan["override_keys"])


if __name__ == "__main__":
    unittest.main()
