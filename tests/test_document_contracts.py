import unittest

from hypermindlabs.document_contracts import (
    DocumentContractValidationError,
    normalize_document_ingest_request,
    normalize_document_retrieval_request,
    schema_version_compatibility_policy,
    validate_schema_version,
)


def _base_scope() -> dict:
    return {
        "owner_member_id": 7,
        "chat_host_id": 42,
        "chat_type": "group",
        "community_id": 2,
        "topic_id": 9,
        "platform": "telegram",
    }


class DocumentContractTests(unittest.TestCase):
    def test_ingest_normalizes_source_and_optional_version(self):
        payload = {
            "schema_version": 1,
            "scope": _base_scope(),
            "source": {
                "source_name": "report.pdf",
                "source_mime": "application/pdf",
                "source_size_bytes": 4096,
                "source_state": "received",
                "source_metadata": {"language": "en"},
            },
            "version": {
                "version_number": 1,
                "parser_name": "pdf-layout",
                "parser_version": "0.1.0",
                "parser_status": "queued",
                "parse_artifact": {"pages": 10},
            },
        }

        normalized = normalize_document_ingest_request(
            payload,
            authenticated_member_id=7,
        )
        self.assertEqual(normalized["schema_version"], 1)
        self.assertEqual(normalized["scope"]["owner_member_id"], 7)
        self.assertEqual(normalized["source"]["source_name"], "report.pdf")
        self.assertEqual(normalized["source"]["source_size_bytes"], 4096)
        self.assertEqual(normalized["version"]["version_number"], 1)
        self.assertEqual(
            normalized["version"]["parse_artifact"]["artifact"]["pages"],
            10,
        )

    def test_ingest_rejects_missing_scope_keys(self):
        payload = {
            "schema_version": 1,
            "scope": {
                "owner_member_id": 7,
                "chat_host_id": 42,
                "community_id": 2,
                "topic_id": 9,
                "platform": "telegram",
            },
            "source": {"source_name": "missing-chat-type.pdf"},
        }
        with self.assertRaises(DocumentContractValidationError):
            normalize_document_ingest_request(payload, authenticated_member_id=7)

    def test_ingest_rejects_owner_mismatch(self):
        payload = {
            "schema_version": 1,
            "scope": _base_scope(),
            "source": {"source_name": "owner-mismatch.pdf"},
        }
        with self.assertRaises(DocumentContractValidationError):
            normalize_document_ingest_request(payload, authenticated_member_id=8)

    def test_schema_version_policy_and_validation(self):
        policy = schema_version_compatibility_policy()
        self.assertEqual(policy["current_schema_version"], 1)
        self.assertIn(1, policy["supported_schema_versions"])
        self.assertEqual(validate_schema_version(1), 1)
        with self.assertRaises(DocumentContractValidationError):
            validate_schema_version(99)

    def test_retrieval_normalizes_request_and_generates_request_id(self):
        payload = {
            "scope": _base_scope(),
            "event": {
                "query_text": "what are the retention requirements?",
                "result_count": 3,
                "citations": [
                    {
                        "document_chunk_id": 14,
                        "start_char": 5,
                        "end_char": 20,
                        "quote_text": "retention period is 7 years",
                    }
                ],
            },
        }
        normalized = normalize_document_retrieval_request(
            payload,
            authenticated_member_id=7,
        )
        self.assertTrue(normalized["event"]["request_id"])
        self.assertEqual(normalized["event"]["result_count"], 3)
        self.assertEqual(len(normalized["event"]["citations"]), 1)

    def test_retrieval_rejects_invalid_citation_bounds(self):
        payload = {
            "scope": _base_scope(),
            "event": {
                "query_text": "bad citation",
                "citations": [
                    {
                        "start_char": 20,
                        "end_char": 5,
                    }
                ],
            },
        }
        with self.assertRaises(DocumentContractValidationError):
            normalize_document_retrieval_request(payload, authenticated_member_id=7)


if __name__ == "__main__":
    unittest.main()
