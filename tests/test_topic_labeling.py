from __future__ import annotations

import unittest
from typing import Any

from hypermindlabs.document_metadata import enrich_document_metadata
from hypermindlabs.document_taxonomy import (
    CONTROLLED_DOMAIN_LABELS,
    CONTROLLED_FORMAT_LABELS,
    CONTROLLED_TOPIC_LABELS,
)

try:
    import web_ui

    _WEB_UI_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    web_ui = None
    _WEB_UI_IMPORT_ERROR = error


class _Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = dict(values or {})

    def runtimeValue(self, path: str, default: Any) -> Any:  # noqa: N802
        return self._values.get(path, default)


def _sample_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_output = {
        "sections": [
            {
                "section_id": "s1",
                "text": "Security policy for API authentication and incident response.",
            },
            {
                "section_id": "s2",
                "text": "Step 1: collect billing data and pricing metrics.",
            },
        ]
    }
    tree_payload = {
        "nodes": [
            {
                "node_key": "n1",
                "node_type": "paragraph",
                "node_title": "Security Overview",
                "node_metadata": {"section_id": "s1"},
            },
            {
                "node_key": "n2",
                "node_type": "list",
                "node_title": "Billing Steps",
                "node_metadata": {"section_id": "s2"},
            },
        ],
        "edges": [],
    }
    chunk_payload = {
        "chunks": [
            {
                "chunk_key": "c1",
                "chunk_text": "Authentication policy must enforce RBAC and security controls.",
                "chunk_metadata": {"node_key": "n1", "node_type": "paragraph"},
            },
            {
                "chunk_key": "c2",
                "chunk_text": "",
                "chunk_metadata": {"node_key": "n2", "node_type": "list"},
            },
        ]
    }
    return canonical_output, tree_payload, chunk_payload


class TopicLabelingTests(unittest.TestCase):
    def test_enrichment_attaches_taxonomy_to_nodes_chunks_and_summary(self):
        canonical_output, tree_payload, chunk_payload = _sample_payloads()
        enriched = enrich_document_metadata(
            canonical_output=canonical_output,
            tree_payload=tree_payload,
            chunk_payload=chunk_payload,
            config_manager=_Config(),
        )

        summary = enriched["summary"]
        self.assertTrue(summary.get("enabled"))
        self.assertGreaterEqual(len(list(summary.get("topic_tags") or [])), 1)
        self.assertGreaterEqual(len(list(summary.get("domain_labels") or [])), 1)
        self.assertGreaterEqual(len(list(summary.get("format_labels") or [])), 1)
        self.assertEqual(summary.get("record_count", {}).get("nodes"), 2)
        self.assertEqual(summary.get("record_count", {}).get("chunks"), 2)

        for topic in list(summary.get("topic_tags") or []):
            self.assertIn(topic, CONTROLLED_TOPIC_LABELS)
        for domain in list(summary.get("domain_labels") or []):
            self.assertIn(domain, CONTROLLED_DOMAIN_LABELS)
        for fmt in list(summary.get("format_labels") or []):
            self.assertIn(fmt, CONTROLLED_FORMAT_LABELS)

        nodes = list(enriched["tree_payload"].get("nodes") or [])
        chunks = list(enriched["chunk_payload"].get("chunks") or [])
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(chunks), 2)

        for node in nodes:
            node_meta = node.get("node_metadata") if isinstance(node.get("node_metadata"), dict) else {}
            taxonomy = node_meta.get("taxonomy") if isinstance(node_meta.get("taxonomy"), dict) else {}
            self.assertIn("topic_tags", taxonomy)
            self.assertIn("domain_labels", taxonomy)
            self.assertIn("format_labels", taxonomy)

        fallback_chunk_meta = chunks[1].get("chunk_metadata") if isinstance(chunks[1].get("chunk_metadata"), dict) else {}
        fallback_topic_tags = list(fallback_chunk_meta.get("topic_tags") or [])
        self.assertGreaterEqual(len(fallback_topic_tags), 1)

    def test_enrichment_respects_runtime_limits(self):
        canonical_output, tree_payload, chunk_payload = _sample_payloads()
        enriched = enrich_document_metadata(
            canonical_output=canonical_output,
            tree_payload=tree_payload,
            chunk_payload=chunk_payload,
            config_manager=_Config(
                {
                    "documents.taxonomy_topic_limit": 2,
                    "documents.taxonomy_max_labels_per_record": 2,
                }
            ),
        )

        summary = enriched["summary"]
        self.assertLessEqual(len(list(summary.get("topic_tags") or [])), 2)
        self.assertLessEqual(len(list(summary.get("domain_labels") or [])), 2)
        self.assertLessEqual(len(list(summary.get("format_labels") or [])), 2)

    def test_enrichment_can_be_disabled(self):
        canonical_output, tree_payload, chunk_payload = _sample_payloads()
        enriched = enrich_document_metadata(
            canonical_output=canonical_output,
            tree_payload=tree_payload,
            chunk_payload=chunk_payload,
            config_manager=_Config({"documents.taxonomy_enabled": False}),
        )

        summary = enriched["summary"]
        self.assertFalse(summary.get("enabled"))
        self.assertEqual(summary.get("topic_tags"), [])
        self.assertEqual(summary.get("domain_labels"), [])
        self.assertEqual(summary.get("format_labels"), [])


@unittest.skipIf(web_ui is None, f"web_ui dependencies unavailable: {_WEB_UI_IMPORT_ERROR}")
class TopicLabelingApiTests(unittest.TestCase):
    def setUp(self):
        self.original_document_manager = web_ui.DocumentManager
        self.original_get_member = web_ui.members.getMemberByID
        self.chunk_calls: list[dict[str, Any]] = []

        class _FakeDocumentManager:
            def __init__(self, outer: "TopicLabelingApiTests"):
                self._outer = outer

            def getDocumentChunks(self, scope_payload: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
                payload = {"scope_payload": dict(scope_payload), "kwargs": dict(kwargs)}
                self._outer.chunk_calls.append(payload)
                return [
                    {
                        "document_chunk_id": 1,
                        "chunk_key": "c1",
                        "chunk_metadata": {
                            "topic_tags": ["security"],
                            "domain_labels": ["engineering"],
                            "format_labels": ["policy"],
                        },
                    }
                ]

        web_ui.DocumentManager = lambda: _FakeDocumentManager(self)
        web_ui.members.getMemberByID = lambda member_id: {
            "member_id": int(member_id),
            "username": "tester",
            "first_name": "Test",
            "last_name": "User",
            "roles": ["owner"],
        }
        web_ui.app.config["TESTING"] = True
        self.client = web_ui.app.test_client()
        with self.client.session_transaction() as session:
            session["member_id"] = 777

    def tearDown(self):
        web_ui.DocumentManager = self.original_document_manager
        web_ui.members.getMemberByID = self.original_get_member

    def test_chunks_endpoint_forwards_taxonomy_filters(self):
        response = self.client.get(
            "/api/documents/chunks?"
            "document_version_id=22&"
            "topic_tags=security,policy&topic_tags=api&"
            "domain_labels=engineering&"
            "format_labels=policy"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("document_version_id"), 22)
        self.assertEqual(payload.get("filters", {}).get("topic_tags"), ["security", "policy", "api"])
        self.assertEqual(payload.get("filters", {}).get("domain_labels"), ["engineering"])
        self.assertEqual(payload.get("filters", {}).get("format_labels"), ["policy"])

        self.assertEqual(len(self.chunk_calls), 1)
        kwargs = self.chunk_calls[0]["kwargs"]
        self.assertEqual(kwargs.get("document_version_id"), 22)
        self.assertEqual(kwargs.get("topic_tags"), ["security", "policy", "api"])
        self.assertEqual(kwargs.get("domain_labels"), ["engineering"])
        self.assertEqual(kwargs.get("format_labels"), ["policy"])


if __name__ == "__main__":
    unittest.main()
