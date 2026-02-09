import unittest

from hypermindlabs.run_events import event_schema, make_event, normalize_event_type


class RunEventsTests(unittest.TestCase):
    def test_make_event_builds_expected_envelope(self):
        event = make_event(
            run_id="run-123",
            seq=4,
            event_type="run.stage",
            stage="analysis.start",
            status="running",
            payload={"detail": "ok"},
        ).to_dict()

        self.assertEqual(event["run_id"], "run-123")
        self.assertEqual(event["seq"], 4)
        self.assertEqual(event["event_type"], "run.stage")
        self.assertEqual(event["stage"], "analysis.start")
        self.assertEqual(event["status"], "running")
        self.assertEqual(event["payload"]["detail"], "ok")
        self.assertIn("timestamp", event)

    def test_normalize_event_type_defaults_when_empty(self):
        self.assertEqual(normalize_event_type(None), "run.metric")
        self.assertEqual(normalize_event_type(""), "run.metric")

    def test_event_schema_contains_required_keys(self):
        schema = event_schema()
        required = set(schema.get("required", []))
        self.assertIn("run_id", required)
        self.assertIn("seq", required)
        self.assertIn("event_type", required)


if __name__ == "__main__":
    unittest.main()
