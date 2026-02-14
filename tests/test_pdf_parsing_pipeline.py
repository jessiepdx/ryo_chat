from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import build_parse_profile
from hypermindlabs.document_parser.adapters.pdf_ocr import PdfOcrParserAdapter
from hypermindlabs.document_parser.adapters.pdf_text import PdfTextLayoutParserAdapter
from hypermindlabs.document_parser.registry import BinaryFallbackParserAdapter, DocumentParserRegistry
from hypermindlabs.document_parser.router import DocumentParserRouter


_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "documents" / "pdf"


def _profile_payload(path: Path, *, file_mime: str = "application/pdf") -> dict[str, Any]:
    return {
        "document_source_id": 901,
        "document_version_id": 902,
        "storage_object_id": 903,
        "file_name": path.name,
        "file_mime": file_mime,
        "file_extension": path.suffix,
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


class _Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = dict(values or {})

    def runtimeValue(self, path: str, default: Any) -> Any:  # noqa: N802
        return self._values.get(path, default)


class PdfParsingPipelineTests(unittest.TestCase):
    def test_pdf_text_layout_adapter_parses_digital_fixture(self):
        path = _FIXTURE_ROOT / "digital-layout.pdf"
        adapter = PdfTextLayoutParserAdapter(config_manager=_Config())

        output = adapter.parse(build_parse_profile(_profile_payload(path)))

        self.assertEqual(output.get("status"), "parsed")
        self.assertIn("Quarterly Operations Report", str(output.get("content_text") or ""))
        metadata = dict(output.get("metadata") or {})
        diagnostics = metadata.get("page_diagnostics") if isinstance(metadata.get("page_diagnostics"), list) else []
        self.assertGreaterEqual(len(diagnostics), 1)
        section_metadata = [dict((section or {}).get("metadata") or {}) for section in output.get("sections") or []]
        element_types = {str(item.get("element_type") or "") for item in section_metadata}
        self.assertIn("heading", element_types)
        self.assertIn("list", element_types)

    def test_router_falls_back_to_pdf_ocr_on_scanned_fixture(self):
        path = _FIXTURE_ROOT / "scanned-blank.pdf"

        def _ocr_provider(*_: Any, **kwargs: Any) -> dict[str, Any]:
            page_number = int(kwargs.get("page_number") or 0)
            return {
                "text": f"OCR content page {page_number}",
                "confidence": 0.92,
                "available": True,
                "engine": "test-ocr",
            }

        registry = DocumentParserRegistry()
        registry.register(PdfTextLayoutParserAdapter(config_manager=_Config()))
        registry.register(PdfOcrParserAdapter(config_manager=_Config(), ocr_provider=_ocr_provider))
        registry.register(BinaryFallbackParserAdapter())

        router = DocumentParserRouter(
            registry=registry,
            config_manager=_Config(
                {
                    "documents.pdf_ocr_enabled": True,
                    "documents.parser_preferred_adapters": ["pdf-text-layout", "pdf-ocr", "binary-fallback"],
                    "documents.parser_fallback_enabled": True,
                    "documents.parser_min_confidence": 0.4,
                }
            ),
        )
        canonical = router.parse_document(_profile_payload(path))
        provenance = dict(canonical.get("provenance") or {})

        self.assertEqual(provenance.get("selected_adapter"), "pdf-ocr")
        self.assertTrue(bool(provenance.get("fallback_used")))
        self.assertIn("OCR content page 1", str(canonical.get("content_text") or ""))
        attempts = provenance.get("attempts") if isinstance(provenance.get("attempts"), list) else []
        self.assertTrue(any(item.get("status") == "low_confidence_fallback" for item in attempts))

    def test_pdf_ocr_adapter_reports_unavailable_engine(self):
        path = _FIXTURE_ROOT / "scanned-blank.pdf"

        def _unavailable_provider(*_: Any, **__: Any) -> dict[str, Any]:
            return {
                "text": "",
                "confidence": 0.0,
                "available": False,
                "engine": "none",
                "warning": "ocr_engine_unavailable",
            }

        adapter = PdfOcrParserAdapter(
            config_manager=_Config(
                {
                    "documents.pdf_ocr_enabled": True,
                    "documents.pdf_ocr_force": True,
                }
            ),
            ocr_provider=_unavailable_provider,
        )
        output = adapter.parse(build_parse_profile(_profile_payload(path)))

        self.assertEqual(output.get("status"), "partial")
        warnings = [str(item) for item in output.get("warnings") or []]
        self.assertIn("ocr_engine_unavailable", warnings)
        self.assertIn("ocr_no_text_extracted", warnings)
        metadata = dict(output.get("metadata") or {})
        self.assertFalse(bool(metadata.get("ocr_available")))
        diagnostics = metadata.get("page_diagnostics") if isinstance(metadata.get("page_diagnostics"), list) else []
        self.assertTrue(diagnostics)
        self.assertTrue(all(bool(item.get("ocr_attempted")) for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
