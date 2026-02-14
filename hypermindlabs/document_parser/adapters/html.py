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

try:
    from bs4 import BeautifulSoup
except Exception:  # noqa: BLE001
    BeautifulSoup = None  # type: ignore[assignment]


_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_TEXT_PATTERN = re.compile(r"\s+")


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
    for part in parts:
        text = str(part).strip().lower()
        if text:
            normalized.append(text)
    return normalized


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _normalize_ws(text: str) -> str:
    return _TEXT_PATTERN.sub(" ", str(text or "")).strip()


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _append_section(
    sections: list[dict[str, Any]],
    *,
    text: str,
    element_type: str,
    level: int,
    cursor: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    value = _normalize_ws(text)
    if not value:
        return cursor
    start_char = int(cursor)
    end_char = start_char + len(value)
    item = {
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
            "source": "html_web",
        },
    }
    if isinstance(metadata, dict):
        item["metadata"].update(dict(metadata))
    sections.append(item)
    return end_char + 2


def _fallback_html_text_sections(html_text: str) -> tuple[list[dict[str, Any]], str]:
    stripped = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
    stripped = re.sub(r"<style[\s\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    plain = _normalize_ws(stripped)
    sections: list[dict[str, Any]] = []
    if not plain:
        return sections, ""
    _append_section(
        sections,
        text=plain,
        element_type="paragraph",
        level=2,
        cursor=0,
        metadata={"fallback_parser": True},
    )
    return sections, plain


class HtmlWebParserAdapter(DocumentParserAdapter):
    adapter_name = "html-web"
    adapter_version = "v1"

    def __init__(self, *, config_manager: Any | None = None):
        self._config = config_manager

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        mime = str(profile.file_mime or "").strip().lower()
        ext = str(profile.file_extension or "").strip().lower()
        if ext in _HTML_EXTENSIONS:
            return True
        if mime in {"text/html", "application/xhtml+xml"}:
            return True
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        sample = str(probes.get("sample_text") or "").lower()
        return "<html" in sample or ("<body" in sample and "<" in sample and ">" in sample)

    def confidence(self, profile: DocumentParseProfile) -> float:
        score = 0.25
        ext = str(profile.file_extension or "").strip().lower()
        mime = str(profile.file_mime or "").strip().lower()
        if ext in _HTML_EXTENSIONS:
            score += 0.35
        if mime in {"text/html", "application/xhtml+xml"}:
            score += 0.35
        probes = profile.probes if isinstance(profile.probes, dict) else {}
        sample = str(probes.get("sample_text") or "").lower()
        if "<html" in sample or "<!doctype html" in sample:
            score += 0.2
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(0.6 + (size_mb / 4.0), default=0.6)

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")

        html_text, encoding = _read_text(path)
        warnings: list[str] = []
        sections: list[dict[str, Any]] = []
        title = ""

        if BeautifulSoup is None:
            warnings.append("bs4_unavailable_fallback_html_strip")
            sections, content_text = _fallback_html_text_sections(html_text)
        else:
            drop_tags = _runtime_csv(
                self._config,
                "documents.html_drop_tags",
            ) or ["script", "style", "noscript", "template", "svg", "canvas", "nav", "header", "footer"]
            soup = BeautifulSoup(html_text, "lxml")

            for tag_name in drop_tags:
                for node in soup.find_all(tag_name):
                    node.decompose()

            title_node = soup.find("title")
            title = _normalize_ws(title_node.get_text(" ", strip=True)) if title_node else ""

            cursor = 0
            if title:
                cursor = _append_section(
                    sections,
                    text=title,
                    element_type="heading",
                    level=1,
                    cursor=cursor,
                    metadata={"source_tag": "title"},
                )

            for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]):
                tag = str(node.name or "").lower()
                if tag == "table":
                    rows: list[str] = []
                    for row in node.find_all("tr"):
                        cells = [_normalize_ws(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                        cells = [cell for cell in cells if cell]
                        if cells:
                            rows.append(" | ".join(cells))
                    table_text = "\n".join(rows).strip()
                    cursor = _append_section(
                        sections,
                        text=table_text,
                        element_type="table",
                        level=2,
                        cursor=cursor,
                        metadata={"source_tag": "table", "row_count": len(rows)},
                    )
                    continue

                text = _normalize_ws(node.get_text(" ", strip=True))
                if not text:
                    continue
                if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
                    level = max(1, int(tag[1]))
                    element_type = "heading"
                elif tag == "li":
                    level = 2
                    element_type = "list"
                    text = f"- {text}"
                elif tag in {"pre", "code"}:
                    level = 2
                    element_type = "code"
                else:
                    level = 2
                    element_type = "paragraph"
                cursor = _append_section(
                    sections,
                    text=text,
                    element_type=element_type,
                    level=level,
                    cursor=cursor,
                    metadata={"source_tag": tag},
                )

            content_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))

        if not content_text:
            warnings.append("no_extractable_html_text")
        status = "parsed" if content_text else "partial"

        return {
            "status": status,
            "content_text": content_text,
            "sections": sections,
            "metadata": {
                "parser_kind": "html_web",
                "encoding": encoding,
                "title": title,
                "section_count": len(sections),
            },
            "warnings": warnings,
            "errors": [],
            "confidence": self.confidence(profile),
            "cost": self.cost(profile),
        }
