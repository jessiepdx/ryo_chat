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


_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".rst"}
_TEXT_EXTENSIONS = {".txt", ".log", ".text"}
_LIST_PATTERN = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _detect_markdown(text: str, extension: str) -> bool:
    if extension in _MARKDOWN_EXTENSIONS:
        return True
    sample = text[:2000]
    if _HEADING_PATTERN.search(sample):
        return True
    if re.search(r"\[[^\]]+\]\([^\)]+\)", sample):
        return True
    if re.search(r"(^|\n)```", sample):
        return True
    return False


def _append_section(
    sections: list[dict[str, Any]],
    *,
    text: str,
    element_type: str,
    level: int,
    cursor: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    value = _safe_text(text)
    if not value:
        return cursor
    start_char = int(cursor)
    end_char = start_char + len(value)
    section = {
        "section_id": f"s{len(sections) + 1}",
        "title": value if element_type == "heading" else "",
        "level": max(1, int(level)),
        "text": value,
        "start_char": start_char,
        "end_char": end_char,
        "page_start": None,
        "page_end": None,
        "metadata": {
            "element_type": element_type,
            "source": "text_markdown",
        },
    }
    if isinstance(metadata, dict):
        section["metadata"].update(dict(metadata))
    sections.append(section)
    return end_char + 2


def _markdown_sections(text: str) -> list[tuple[str, int, str, dict[str, Any]]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    entries: list[tuple[str, int, str, dict[str, Any]]] = []

    in_code_block = False
    code_lines: list[str] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        joined = " ".join(line.strip() for line in paragraph_lines if line.strip())
        if joined:
            entries.append(("paragraph", 2, joined, {"line_start": paragraph_start + 1}))
        paragraph_lines = []

    def flush_code() -> None:
        nonlocal code_lines
        if not code_lines:
            return
        joined = "\n".join(code_lines).strip()
        if joined:
            entries.append(("code", 2, joined, {}))
        code_lines = []

    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                flush_paragraph()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            continue

        heading_match = _HEADING_PATTERN.match(stripped)
        if heading_match:
            flush_paragraph()
            marks = heading_match.group(1)
            heading_text = heading_match.group(2)
            entries.append(("heading", len(marks), heading_text, {"markdown_level": len(marks)}))
            continue

        if _LIST_PATTERN.match(stripped):
            flush_paragraph()
            entries.append(("list", 2, stripped, {"list_item": True}))
            continue

        if not paragraph_lines:
            paragraph_start = index
        paragraph_lines.append(stripped)

    if in_code_block:
        flush_code()
    flush_paragraph()
    return entries


class TextMarkdownParserAdapter(DocumentParserAdapter):
    adapter_name = "text-markdown"
    adapter_version = "v1"

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        mime = str(profile.file_mime or "").strip().lower()
        ext = str(profile.file_extension or "").strip().lower()
        if ext in {".csv", ".tsv", ".json", ".xml", ".yml", ".yaml"}:
            return False
        if mime in {
            "application/json",
            "text/json",
            "application/xml",
            "text/xml",
            "text/csv",
            "application/csv",
            "text/tab-separated-values",
        }:
            return False
        if ext in {".html", ".htm", ".xhtml"}:
            return False
        if mime in {"text/html", "application/xhtml+xml"}:
            return False
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        sample = str(probes.get("sample_text") or "").lower()
        if "<html" in sample or ("<body" in sample and ">" in sample):
            return False
        if ext in _MARKDOWN_EXTENSIONS or ext in _TEXT_EXTENSIONS:
            return True
        if mime in {"text/plain", "text/markdown", "text/x-markdown", "application/markdown"}:
            return True
        return bool(probes.get("utf8_text")) and not bool(probes.get("pdf_header"))

    def confidence(self, profile: DocumentParseProfile) -> float:
        mime = str(profile.file_mime or "").strip().lower()
        ext = str(profile.file_extension or "").strip().lower()
        score = 0.3
        if ext in _MARKDOWN_EXTENSIONS:
            score += 0.45
        elif ext in _TEXT_EXTENSIONS:
            score += 0.3
        if mime in {"text/plain", "text/markdown", "text/x-markdown", "application/markdown"}:
            score += 0.25
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        if probes.get("utf8_text"):
            score += 0.15
        sample = str(probes.get("sample_text") or "").lower()
        if "<html" in sample or ("<body" in sample and ">" in sample):
            score -= 0.35
        if probes.get("has_nul"):
            score -= 0.4
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(0.22 + (size_mb / 8.0), default=0.22)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")

        text, encoding = _read_text(path)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        is_markdown = _detect_markdown(normalized, str(profile.file_extension or "").lower())

        sections: list[dict[str, Any]] = []
        cursor = 0
        if is_markdown:
            for element_type, level, section_text, metadata in _markdown_sections(normalized):
                cursor = _append_section(
                    sections,
                    text=section_text,
                    element_type=element_type,
                    level=level,
                    cursor=cursor,
                    metadata=metadata,
                )
        else:
            paragraph_blocks = [part.strip() for part in normalized.split("\n\n") if part.strip()]
            for block in paragraph_blocks:
                cursor = _append_section(
                    sections,
                    text=block,
                    element_type="paragraph",
                    level=2,
                    cursor=cursor,
                    metadata={},
                )

        content_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))
        warnings: list[str] = []
        if not content_text:
            warnings.append("empty_text_after_decode")

        status = "parsed" if content_text else "partial"
        return {
            "status": status,
            "content_text": content_text,
            "sections": sections,
            "metadata": {
                "parser_kind": "text_markdown",
                "encoding": encoding,
                "markdown_detected": is_markdown,
                "line_count": normalized.count("\n") + (1 if normalized else 0),
            },
            "warnings": warnings,
            "errors": [],
            "confidence": self.confidence(profile),
            "cost": self.cost(profile),
        }
