from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import (
    DocumentParseProfile,
    DocumentParserAdapter,
    clamp_confidence,
    clamp_cost,
)
from hypermindlabs.document_parser.adapters.html import HtmlWebParserAdapter
from hypermindlabs.document_parser.adapters.office import OfficeDocumentParserAdapter
from hypermindlabs.document_parser.adapters.pdf_ocr import PdfOcrParserAdapter
from hypermindlabs.document_parser.adapters.pdf_text import PdfTextLayoutParserAdapter
from hypermindlabs.document_parser.adapters.structured_data import StructuredDataParserAdapter
from hypermindlabs.document_parser.adapters.text_markdown import TextMarkdownParserAdapter


_PRINTABLE_RUN_PATTERN = re.compile(rb"[A-Za-z0-9][A-Za-z0-9 \t,.;:()\-_/]{18,}")


def _runtime_value(config_manager: Any | None, path: str, default: Any) -> Any:
    if config_manager is None:
        return default
    try:
        return config_manager.runtimeValue(path, default)
    except Exception:  # noqa: BLE001
        return default


def _runtime_csv(config_manager: Any | None, path: str) -> list[str]:
    value = _runtime_value(config_manager, path, [])
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value or "").split(",")
    normalized: list[str] = []
    for item in parts:
        text = str(item).strip().lower()
        if text:
            normalized.append(text)
    return normalized


class DocumentParserRegistry:
    """In-memory parser adapter registry with deterministic iteration order."""

    def __init__(self):
        self._adapters: dict[str, DocumentParserAdapter] = {}
        self._order: list[str] = []

    def register(self, adapter: DocumentParserAdapter) -> None:
        name = str(getattr(adapter, "adapter_name", "") or "").strip().lower()
        if not name:
            raise ValueError("Parser adapter must define a non-empty adapter_name.")
        if name not in self._adapters:
            self._order.append(name)
        self._adapters[name] = adapter

    def get(self, adapter_name: str) -> DocumentParserAdapter | None:
        key = str(adapter_name or "").strip().lower()
        return self._adapters.get(key)

    def list_adapters(self) -> list[DocumentParserAdapter]:
        return [self._adapters[name] for name in self._order if name in self._adapters]


class PlainTextParserAdapter(DocumentParserAdapter):
    adapter_name = "text-plain"
    adapter_version = "v1"
    _TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".log",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".sql",
    }

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        mime = str(profile.file_mime or "").lower()
        if mime.startswith("text/"):
            return True
        if profile.file_extension in self._TEXT_EXTENSIONS:
            return True
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        return bool(probes.get("utf8_text"))

    def confidence(self, profile: DocumentParseProfile) -> float:
        mime = str(profile.file_mime or "").lower()
        score = 0.25
        if mime.startswith("text/"):
            score += 0.6
        if profile.file_extension in self._TEXT_EXTENSIONS:
            score += 0.2
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if probes.get("utf8_text"):
            score += 0.15
        if probes.get("has_nul"):
            score -= 0.4
        score -= max(0.0, float(profile.complexity_score or 0.0) - 0.6) * 0.3
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(0.25 + (size_mb / 6.0), default=0.25)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")
        raw_bytes = path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        warnings: list[str] = []
        if not normalized.strip():
            warnings.append("empty_text_after_decode")
        return {
            "status": "parsed",
            "content_text": normalized,
            "sections": [
                {
                    "section_id": "s1",
                    "title": "",
                    "level": 1,
                    "text": normalized,
                    "start_char": 0,
                    "end_char": len(normalized),
                    "metadata": {
                        "line_count": normalized.count("\n") + 1 if normalized else 0,
                    },
                }
            ],
            "metadata": {
                "parser_kind": "plain_text",
                "bytes_read": len(raw_bytes),
            },
            "warnings": warnings,
            "errors": [],
        }


class PdfBasicParserAdapter(DocumentParserAdapter):
    adapter_name = "pdf-basic"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        mime = str(profile.file_mime or "").lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if mime == "application/pdf":
            return True
        if profile.file_extension == ".pdf":
            return True
        return bool(probes.get("pdf_header"))

    def confidence(self, profile: DocumentParseProfile) -> float:
        mime = str(profile.file_mime or "").lower()
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        score = 0.25
        if mime == "application/pdf":
            score += 0.35
        if profile.file_extension == ".pdf":
            score += 0.2
        if probes.get("pdf_header"):
            score += 0.2
        if probes.get("has_nul"):
            score += 0.05
        score -= max(0.0, float(profile.complexity_score or 0.0) - 0.7) * 0.2
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(1.25 + (size_mb / 3.0), default=1.25)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")
        raw_bytes = path.read_bytes()
        runs: list[str] = []
        for match in _PRINTABLE_RUN_PATTERN.findall(raw_bytes[: 2 * 1024 * 1024]):
            runs.append(match.decode("latin-1", errors="ignore").strip())
            if len(runs) >= 32:
                break
        extracted_text = "\n".join(item for item in runs if item).strip()
        warnings = ["pdf_heuristic_text_extraction"]
        if not extracted_text:
            warnings.append("no_extractable_pdf_text")
        return {
            "status": "partial",
            "content_text": extracted_text,
            "sections": [
                {
                    "section_id": "s1",
                    "title": "",
                    "level": 1,
                    "text": extracted_text,
                    "start_char": 0,
                    "end_char": len(extracted_text),
                    "metadata": {
                        "extraction": "heuristic-printable-runs",
                    },
                }
            ],
            "metadata": {
                "parser_kind": "pdf_basic",
                "bytes_read": len(raw_bytes),
            },
            "warnings": warnings,
            "errors": [],
        }


class BinaryFallbackParserAdapter(DocumentParserAdapter):
    adapter_name = "binary-fallback"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        return bool(profile.file_path)

    def confidence(self, profile: DocumentParseProfile) -> float:
        score = 0.12
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if probes.get("utf8_text"):
            score += 0.1
        return clamp_confidence(score, default=0.1)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(4.0 + (size_mb / 2.0), default=4.0)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")
        sample_bytes = path.read_bytes()[:2048]
        return {
            "status": "partial",
            "content_text": "",
            "sections": [],
            "metadata": {
                "parser_kind": "binary_fallback",
                "sample_hex": sample_bytes[:24].hex(),
                "bytes_sampled": len(sample_bytes),
            },
            "warnings": [
                "fallback_metadata_only_parse",
                "manual_parser_upgrade_recommended",
            ],
            "errors": [],
        }


def build_default_parser_registry(*, config_manager: Any | None = None) -> DocumentParserRegistry:
    registry = DocumentParserRegistry()
    adapters: list[DocumentParserAdapter] = [
        PdfTextLayoutParserAdapter(config_manager=config_manager),
        PdfOcrParserAdapter(config_manager=config_manager),
        OfficeDocumentParserAdapter(config_manager=config_manager),
        HtmlWebParserAdapter(config_manager=config_manager),
        TextMarkdownParserAdapter(),
        StructuredDataParserAdapter(config_manager=config_manager),
        PdfBasicParserAdapter(),
        PlainTextParserAdapter(),
        BinaryFallbackParserAdapter(),
    ]
    enabled = set(_runtime_csv(config_manager, "documents.parser_enabled_adapters"))
    for adapter in adapters:
        name = str(adapter.adapter_name).strip().lower()
        if enabled and name not in enabled:
            continue
        registry.register(adapter)
    return registry
