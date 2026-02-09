import tempfile
import unittest
from pathlib import Path

from hypermindlabs.tool_test_harness import ToolTestHarnessStore


class ToolHarnessStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ToolTestHarnessStore(storage_path=Path(self.tempdir.name) / "tool_harness.json")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_upsert_case_and_round_trip(self):
        created = self.store.upsert_case(
            {
                "tool_name": "customEcho",
                "fixture_name": "echo baseline",
                "execution_mode": "real",
                "input_args": {"queryString": "hello"},
            },
            actor_member_id=9,
        )
        self.assertEqual(created["tool_name"], "customEcho")
        self.assertEqual(created["fixture_name"], "echo baseline")
        self.assertEqual(created["input_args"]["queryString"], "hello")

        fetched = self.store.get_case(created["case_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["case_id"], created["case_id"])

    def test_record_run_and_save_baselines(self):
        created = self.store.upsert_case(
            {
                "tool_name": "customEcho",
                "fixture_name": "echo run",
                "input_args": {"queryString": "hello"},
            },
            actor_member_id=5,
        )
        case_id = created["case_id"]
        report = {
            "schema": "ryo.tool_harness_report.v1",
            "case_id": case_id,
            "tool_name": "customEcho",
            "result": {"status": "success", "tool_results": {"ok": True}},
            "contract": {"status": "missing_baseline"},
            "regression": {"status": "missing_golden"},
        }

        updated = self.store.record_run(case_id, report, actor_member_id=5)
        self.assertEqual(updated["last_report"]["result"]["status"], "success")
        self.assertEqual(len(updated["run_history"]), 1)

        contract = {
            "schema": "ryo.tool_contract.v1",
            "tool_name": "customEcho",
            "required": ["queryString"],
            "properties": {"queryString": {"type": ["string"]}},
            "hash": "abc123",
        }
        updated = self.store.set_contract_snapshot(case_id, contract, actor_member_id=5)
        self.assertEqual(updated["contract_snapshot"]["tool_name"], "customEcho")

        updated = self.store.set_golden_output(case_id, {"ok": True}, actor_member_id=5)
        self.assertEqual(updated["golden_output"]["ok"], True)
        self.assertTrue(updated["golden_hash"])


if __name__ == "__main__":
    unittest.main()
