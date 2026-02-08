import unittest

from hypermindlabs.tool_test_harness import (
    build_contract_snapshot,
    compare_contract_snapshots,
    compare_golden_outputs,
)


class ToolContractTests(unittest.TestCase):
    def test_contract_snapshot_and_compatible_additive_change(self):
        baseline = build_contract_snapshot(
            "customEcho",
            {
                "type": "object",
                "properties": {
                    "queryString": {"type": "string"},
                },
                "required": ["queryString"],
            },
        )
        current = build_contract_snapshot(
            "customEcho",
            {
                "type": "object",
                "properties": {
                    "queryString": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["queryString"],
            },
        )
        result = compare_contract_snapshots(baseline, current)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["compatible"])
        self.assertIn("count", result["added_properties"])

    def test_contract_detects_required_and_type_breaks(self):
        baseline = build_contract_snapshot(
            "customEcho",
            {
                "type": "object",
                "properties": {
                    "queryString": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["queryString"],
            },
        )
        current = build_contract_snapshot(
            "customEcho",
            {
                "type": "object",
                "properties": {
                    "queryString": {"type": "integer"},
                },
                "required": ["queryString", "topic"],
            },
        )
        result = compare_contract_snapshots(baseline, current)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["compatible"])
        self.assertIn("topic", result["added_required"])
        self.assertIn("count", result["removed_properties"])
        self.assertTrue(any(change.get("property") == "queryString" for change in result["type_changes"]))

    def test_golden_comparison_states(self):
        missing = compare_golden_outputs(None, {"ok": True})
        self.assertEqual(missing["status"], "missing_golden")

        match = compare_golden_outputs({"ok": True}, {"ok": True})
        self.assertEqual(match["status"], "pass")
        self.assertTrue(match["match"])

        drift = compare_golden_outputs({"ok": True, "count": 1}, {"ok": False, "count": 2})
        self.assertEqual(drift["status"], "fail")
        self.assertGreater(drift["diff_count"], 0)


if __name__ == "__main__":
    unittest.main()
