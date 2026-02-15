import unittest

from hypermindlabs.document_taxonomy import (
    CONTROLLED_DOMAIN_LABELS,
    CONTROLLED_FORMAT_LABELS,
    CONTROLLED_TOPIC_LABELS,
    detect_domain_signals,
    detect_format_signals,
    extract_topic_signals,
    flatten_signal_labels,
    normalize_domain_label,
    normalize_format_label,
    normalize_topic_label,
)


class DocumentTaxonomyTests(unittest.TestCase):
    def test_normalizers_map_synonyms_to_controlled_vocab(self):
        self.assertEqual(normalize_topic_label("authn"), "authentication")
        self.assertEqual(normalize_topic_label("RBAC"), "authorization")
        self.assertEqual(normalize_format_label("to-do"), "checklist")
        self.assertEqual(normalize_domain_label("ops"), "operations")

        self.assertEqual(normalize_topic_label("unknown-topic"), "")
        self.assertEqual(normalize_format_label("diagram"), "")
        self.assertEqual(normalize_domain_label("unknown-domain"), "")

    def test_extract_topic_signals_returns_bounded_controlled_labels(self):
        text = "API authentication authn authz RBAC deployment monitoring reliability incident response"
        signals = extract_topic_signals(
            text,
            max_topics=5,
            min_confidence=0.1,
            synonym_expansion=True,
        )
        labels = [str(item.get("label")) for item in signals]
        confidences = [float(item.get("confidence", 0.0)) for item in signals]

        self.assertGreaterEqual(len(signals), 3)
        self.assertLessEqual(len(signals), 5)
        self.assertIn("authentication", labels)
        self.assertIn("authorization", labels)
        for label in labels:
            self.assertIn(label, CONTROLLED_TOPIC_LABELS)
        for confidence in confidences:
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_detect_format_signals_uses_node_type_and_text_patterns(self):
        signals = detect_format_signals(
            "Step 1: do thing\n- item one\n- item two",
            node_type="list",
            max_labels=4,
            min_confidence=0.1,
        )
        labels = flatten_signal_labels(signals, max_items=4)

        self.assertIn("checklist", labels)
        self.assertIn("procedure", labels)
        for label in labels:
            self.assertIn(label, CONTROLLED_FORMAT_LABELS)

    def test_detect_domain_signals_combines_text_and_topic_mappings(self):
        topic_signals = [
            {"label": "authentication", "confidence": 1.0},
            {"label": "api", "confidence": 0.6},
        ]
        signals = detect_domain_signals(
            "security vulnerability pii authentication api",
            topic_signals=topic_signals,
            max_domains=3,
            min_confidence=0.1,
        )
        labels = flatten_signal_labels(signals, max_items=3)

        self.assertGreaterEqual(len(labels), 1)
        self.assertIn("security", labels)
        for label in labels:
            self.assertIn(label, CONTROLLED_DOMAIN_LABELS)

    def test_flatten_signal_labels_deduplicates_and_applies_limit(self):
        flattened = flatten_signal_labels(
            [
                {"label": "security", "confidence": 1.0},
                {"label": "security", "confidence": 0.8},
                {"label": "operations", "confidence": 0.6},
            ],
            max_items=1,
        )
        self.assertEqual(flattened, ["security"])


if __name__ == "__main__":
    unittest.main()
