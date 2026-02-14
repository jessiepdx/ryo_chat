from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import DocumentParseProfile, DocumentParserAdapter
from hypermindlabs.document_parser.contracts import (
    CANONICAL_PARSE_SCHEMA,
    build_document_parse_artifact_patch,
    normalize_canonical_parse_output,
    validate_canonical_parse_output,
)
from hypermindlabs.document_parser.registry import DocumentParserRegistry
from hypermindlabs.document_parser.router import DocumentParserRouter


class _Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = dict(values or {})

    def runtimeValue(self, path: str, default: Any) -> Any:  # noqa: N802
        return self._values.get(path, default)


def _profile_payload(path: Path, *, file_name: str, file_mime: str) -> dict[str, Any]:
    return {
        "document_source_id": 10,
        "document_version_id": 20,
        "storage_object_id": 30,
        "file_name": file_name,
        "file_mime": file_mime,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "scope": {
            "owner_member_id": 5,
            "chat_host_id": 55,
            "chat_type": "member",
            "community_id": None,
            "topic_id": None,
            "platform": "web",
        },
    }


class _FailingAdapter(DocumentParserAdapter):
    adapter_name = "failing-adapter"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        return bool(profile.file_path)

    def confidence(self, profile: DocumentParseProfile) -> float:
        return 0.92

    def cost(self, profile: DocumentParseProfile) -> float:
        return 1.0

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        raise RuntimeError("simulated parser failure")


class _LowConfidenceAdapter(DocumentParserAdapter):
    adapter_name = "low-confidence-adapter"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        return True

    def confidence(self, profile: DocumentParseProfile) -> float:
        return 0.1

    def cost(self, profile: DocumentParseProfile) -> float:
        return 0.5

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        return {
            "status": "parsed",
            "content_text": "weak parse",
            "sections": [{"section_id": "s1", "text": "weak parse"}],
        }


class _SuccessAdapter(DocumentParserAdapter):
    adapter_name = "success-adapter"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        return bool(profile.file_path)

    def confidence(self, profile: DocumentParseProfile) -> float:
        return 0.88

    def cost(self, profile: DocumentParseProfile) -> float:
        return 1.5

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        return {
            "status": "parsed",
            "content_text": "successful parse",
            "sections": [{"section_id": "s1", "text": "successful parse"}],
            "metadata": {"adapter": self.adapter_name},
        }


class DocumentParserRouterTests(unittest.TestCase):
    def test_router_prefers_pdf_adapter_for_pdf_magic_probe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n")
            router = DocumentParserRouter(
                config_manager=_Config(
                    {
                        "documents.parser_preferred_adapters": [
                            "pdf-text-layout",
                            "pdf-ocr",
                            "pdf-basic",
                            "text-plain",
                            "binary-fallback",
                        ],
                        "documents.parser_min_confidence": 0.2,
                    }
                )
            )
            canonical = router.parse_document(
                _profile_payload(pdf_path, file_name="sample.pdf", file_mime="application/octet-stream")
            )
            provenance = dict(canonical.get("provenance") or {})
            self.assertEqual(provenance.get("selected_adapter"), "pdf-basic")
            self.assertIn("pdf-basic", provenance.get("adapter_chain") or [])

    def test_contract_normalization_and_parse_artifact_patch(self):
        canonical = normalize_canonical_parse_output(
            {"text": "hello world"},
            parser_name="unit-parser",
            parser_version="1.0.0",
            adapter_chain=["unit-parser"],
            confidence=0.77,
            cost=0.3,
            duration_ms=17,
            route_debug={"selected_chain": ["unit-parser"]},
            profile_summary={"file_name": "note.txt"},
        )
        self.assertEqual(canonical["canonical_schema"], CANONICAL_PARSE_SCHEMA)
        self.assertEqual(canonical["status"], "parsed")
        self.assertEqual(len(canonical["sections"]), 1)

        validated = validate_canonical_parse_output(canonical)
        patch = build_document_parse_artifact_patch(validated)
        self.assertEqual(patch["parse_mode"], "adapter_router")
        self.assertEqual(patch["status"], "parsed")
        self.assertEqual(patch["parser_name"], "unit-parser")
        self.assertIn("canonical", patch["artifact"])

    def test_fallback_chain_uses_success_adapter_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "payload.bin"
            path.write_bytes(b"binary payload")

            registry = DocumentParserRegistry()
            registry.register(_FailingAdapter())
            registry.register(_SuccessAdapter())
            router = DocumentParserRouter(
                registry=registry,
                config_manager=_Config(
                    {
                        "documents.parser_preferred_adapters": ["failing-adapter", "success-adapter"],
                        "documents.parser_fallback_enabled": True,
                        "documents.parser_min_confidence": 0.2,
                    }
                ),
            )
            canonical = router.parse_document(
                _profile_payload(path, file_name="payload.bin", file_mime="application/octet-stream")
            )
            provenance = dict(canonical.get("provenance") or {})
            self.assertEqual(provenance.get("selected_adapter"), "success-adapter")
            self.assertTrue(bool(provenance.get("fallback_used")))
            attempts = provenance.get("attempts") if isinstance(provenance.get("attempts"), list) else []
            self.assertTrue(any(item.get("adapter") == "failing-adapter" for item in attempts))

    def test_low_confidence_adapter_falls_back_to_higher_confidence_adapter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "payload.txt"
            path.write_text("text payload", encoding="utf-8")

            registry = DocumentParserRegistry()
            registry.register(_LowConfidenceAdapter())
            registry.register(_SuccessAdapter())
            router = DocumentParserRouter(
                registry=registry,
                config_manager=_Config(
                    {
                        "documents.parser_preferred_adapters": ["low-confidence-adapter", "success-adapter"],
                        "documents.parser_fallback_enabled": True,
                        "documents.parser_min_confidence": 0.5,
                    }
                ),
            )
            canonical = router.parse_document(
                _profile_payload(path, file_name="payload.txt", file_mime="text/plain")
            )
            provenance = dict(canonical.get("provenance") or {})
            self.assertEqual(provenance.get("selected_adapter"), "success-adapter")
            attempts = provenance.get("attempts") if isinstance(provenance.get("attempts"), list) else []
            self.assertTrue(any(item.get("status") == "low_confidence_fallback" for item in attempts))


if __name__ == "__main__":
    unittest.main()
