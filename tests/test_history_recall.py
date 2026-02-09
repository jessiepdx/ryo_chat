from datetime import datetime, timedelta, timezone
import unittest

from hypermindlabs.history_recall import ProgressiveHistoryExplorer, ProgressiveHistoryExplorerConfig


class _FakeChatHistoryManager:
    def __init__(self) -> None:
        self._now = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)

    def searchChatHistory(  # noqa: N802
        self,
        text: str,
        limit: int,
        chatHostID: int,
        chatType: str,
        platform: str,
        topicID: int | None,
        scopeTopic: bool,
        timeInHours: int,
    ) -> list[dict]:
        if timeInHours <= 12:
            return []
        return [
            {
                "history_id": 100,
                "message_id": 501,
                "message_text": "You asked about the alpha release timeline and rollout plan.",
                "message_timestamp": self._now - timedelta(hours=5),
                "distance": 0.12,
            }
        ][:limit]

    def getChatHistory(  # noqa: N802
        self,
        chatHostID: int,
        chatType: str,
        platform: str,
        topicID: int | None = None,
        timeInHours: int | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        if (timeInHours or 0) <= 12:
            return []
        rows = [
            {
                "history_id": 99,
                "member_id": 2,
                "message_id": 500,
                "message_text": "Can you remind me what we said about alpha release?",
                "message_timestamp": self._now - timedelta(hours=5, minutes=2),
            },
            {
                "history_id": 100,
                "member_id": None,
                "message_id": 501,
                "message_text": "We covered the alpha release timeline and rollout plan.",
                "message_timestamp": self._now - timedelta(hours=5),
            },
            {
                "history_id": 101,
                "member_id": 2,
                "message_id": 502,
                "message_text": "Thanks, that was helpful.",
                "message_timestamp": self._now - timedelta(hours=4, minutes=58),
            },
        ]
        return rows[: (limit or len(rows))]


class TestProgressiveHistoryExplorer(unittest.TestCase):
    def test_explore_expands_rounds_and_finds_target(self):
        manager = _FakeChatHistoryManager()
        config = ProgressiveHistoryExplorerConfig(
            enabled=True,
            max_rounds=3,
            round_windows_hours=(6, 24, 72),
            semantic_limit_start=2,
            semantic_limit_step=1,
            timeline_limit_start=8,
            timeline_limit_step=4,
            context_radius=1,
            match_threshold=0.35,
            max_selected=6,
            max_message_chars=240,
        )
        explorer = ProgressiveHistoryExplorer(manager, config)

        result = explorer.explore(
            query_text="what did we say about alpha release timeline",
            chat_host_id=2,
            chat_type="member",
            platform="telegram",
            topic_id=None,
            history_recall_requested=True,
            switched=False,
            allow_history_search=True,
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["found_round"], 2)
        self.assertEqual(result["target_history_id"], 100)
        self.assertGreaterEqual(result["selected_count"], 1)
        self.assertEqual(result["selected"][0]["history_id"], 99)

    def test_explore_returns_no_selection_when_disabled(self):
        manager = _FakeChatHistoryManager()
        config = ProgressiveHistoryExplorerConfig(enabled=False)
        explorer = ProgressiveHistoryExplorer(manager, config)
        result = explorer.explore(
            query_text="alpha release",
            chat_host_id=2,
            chat_type="member",
            platform="telegram",
        )
        self.assertFalse(result["found"])
        self.assertEqual(result["selected_count"], 0)


if __name__ == "__main__":
    unittest.main()
