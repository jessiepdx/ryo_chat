from datetime import datetime, timezone
import unittest

from hypermindlabs.temporal_context import build_temporal_context, coerce_datetime_utc


class TestTemporalContext(unittest.TestCase):
    def test_coerce_datetime_utc_supports_iso_and_naive(self):
        parsed_iso = coerce_datetime_utc("2026-02-08T10:00:00Z")
        self.assertIsNotNone(parsed_iso)
        self.assertEqual(parsed_iso.isoformat(), "2026-02-08T10:00:00+00:00")

        parsed_naive = coerce_datetime_utc("2026-02-08 10:00:00")
        self.assertIsNotNone(parsed_naive)
        self.assertEqual(parsed_naive.isoformat(), "2026-02-08T10:00:00+00:00")

    def test_build_temporal_context_orders_and_limits_history(self):
        now_utc = datetime(2026, 2, 8, 10, 10, 0, tzinfo=timezone.utc)
        history = [
            {
                "history_id": 2,
                "message_id": 200,
                "member_id": 99,
                "message_text": "Second user message in timeline order.",
                "message_timestamp": "2026-02-08 10:05:00",
            },
            {
                "history_id": 1,
                "message_id": 100,
                "member_id": 98,
                "message_text": "First user message in timeline order.",
                "message_timestamp": "2026-02-08 10:00:00",
            },
            {
                "history_id": 3,
                "message_id": 300,
                "member_id": None,
                "message_text": "Assistant response that is intentionally very long for excerpt truncation checks.",
                "message_timestamp": datetime(2026, 2, 8, 10, 6, 0, tzinfo=timezone.utc),
            },
        ]
        context = build_temporal_context(
            platform="telegram",
            chat_type="member",
            chat_host_id=123,
            topic_id=None,
            timezone_name="UTC",
            now_utc=now_utc,
            inbound_sent_at="2026-02-08T10:09:00Z",
            inbound_received_at="2026-02-08T10:09:01Z",
            history_messages=history,
            history_limit=2,
            excerpt_max_chars=24,
        )

        recent = context["timeline"]["recent"]
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["history_id"], 2)
        self.assertEqual(recent[1]["history_id"], 3)
        self.assertEqual(recent[1]["role"], "assistant")
        self.assertTrue(recent[1]["excerpt"].endswith("..."))
        self.assertEqual(context["clock"]["now_utc"], "2026-02-08T10:10:00Z")
        self.assertEqual(context["inbound"]["message_sent_at_utc"], "2026-02-08T10:09:00Z")
        self.assertEqual(context["inbound"]["message_received_at_utc"], "2026-02-08T10:09:01Z")

    def test_build_temporal_context_falls_back_to_utc_for_invalid_timezone(self):
        now_utc = datetime(2026, 2, 8, 20, 0, 0, tzinfo=timezone.utc)
        context = build_temporal_context(
            platform="telegram",
            chat_type="community",
            chat_host_id=10,
            topic_id=11,
            timezone_name="Invalid/Zone",
            now_utc=now_utc,
            inbound_sent_at=None,
            inbound_received_at=None,
            history_messages=[],
            history_limit=20,
            excerpt_max_chars=160,
        )

        self.assertEqual(context["clock"]["timezone"], "UTC")
        self.assertEqual(context["clock"]["now_local"], "2026-02-08T20:00:00+00:00")
        self.assertEqual(context["inbound"]["message_received_at_utc"], "2026-02-08T20:00:00Z")


if __name__ == "__main__":
    unittest.main()

