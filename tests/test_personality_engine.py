import unittest

from hypermindlabs.personality_engine import PersonalityEngine, PersonalityRuntimeConfig
from hypermindlabs.personality_injector import PersonalityInjector
from hypermindlabs.personality_rollup import NarrativeRollupEngine


class TestPersonalityEngine(unittest.TestCase):
    def test_adaptive_update_respects_locked_fields(self):
        engine = PersonalityEngine()
        config = PersonalityRuntimeConfig(
            adaptation_min_turns=1,
            adaptation_window_turns=1,
            adaptation_max_step_per_window=1,
            default_verbosity="brief",
        )
        profile = engine.resolve_profile(
            member_id=2,
            stored_profile={
                "explicit_directive_json": {"locked_fields": ["verbosity"]},
                "adaptive_state_json": {"verbosity": "brief", "reading_level": "moderate", "turns_observed": 0},
                "effective_profile_json": {},
                "locked_fields_json": ["verbosity"],
            },
            runtime_config=config,
        )

        result = engine.adapt_after_turn(
            profile=profile,
            user_message="Please provide a deep and comprehensive multi-part explanation with detail.",
            assistant_message="Sure.",
            analysis_payload={"response_style": {"length": "detailed"}},
            runtime_config=config,
        )
        self.assertNotIn("verbosity", result["changed_fields"])

    def test_apply_analysis_style_maps_verbosity(self):
        engine = PersonalityEngine()
        payload = {"response_style": {"tone": "friendly", "length": "concise"}}
        profile = {
            "effective": {
                "tone": "professional",
                "verbosity": "detailed",
                "reading_level": "advanced",
            }
        }
        merged = engine.apply_analysis_style(analysis_payload=payload, profile=profile)
        self.assertEqual(merged["response_style"]["tone"], "professional")
        self.assertEqual(merged["response_style"]["length"], "detailed")


class TestPersonalityInjectorAndRollup(unittest.TestCase):
    def test_injector_builds_payload(self):
        injector = PersonalityInjector()
        payload = injector.build_payload(
            profile={
                "member_id": 3,
                "effective": {
                    "tone": "friendly",
                    "verbosity": "brief",
                    "reading_level": "simple",
                    "format": "plain",
                    "humor": "low",
                    "emoji": "off",
                    "language": "en",
                },
                "adaptive": {"confidence": 0.4, "turns_observed": 5, "last_reason": "test"},
                "explicit": {"locked_fields": []},
                "narrative": {"active_summary": "User prefers concise replies."},
            },
            max_injection_chars=500,
        )
        self.assertEqual(payload["schema"], "ryo.personality_injection.v1")
        self.assertEqual(payload["effective_style"]["verbosity"], "brief")

    def test_rollup_threshold_and_build(self):
        rollup = NarrativeRollupEngine()
        should = rollup.should_rollup(
            profile={"narrative": {"turns_since_rollup": 9, "chars_since_rollup": 100}},
            turn_threshold=8,
            char_threshold=2000,
        )
        self.assertTrue(should)
        result = rollup.build_rollup(
            profile={"narrative": {"last_chunk_index": 2}},
            history_messages=[
                {"member_id": 2, "message_text": "We talked about deployment and testing."},
                {"member_id": None, "message_text": "I suggested a staged rollout."},
            ],
            user_message="What did we agree to do next?",
            assistant_message="We agreed to run staged rollout with validation checks.",
            analysis_payload={"topic": "deployment", "intent": "recall_plan"},
            max_source_messages=6,
            max_summary_chars=280,
        )
        self.assertEqual(result["chunk_index"], 3)
        self.assertTrue(result["summary_text"])


if __name__ == "__main__":
    unittest.main()

