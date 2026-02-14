from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from hypermindlabs.document_parser.adapters.base import (
    DocumentParseProfile,
    DocumentParserAdapter,
    clamp_confidence,
    clamp_cost,
)


_STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".json", ".xml"}


def _runtime_value(config_manager: Any | None, path: str, default: Any) -> Any:
    if config_manager is None:
        return default
    try:
        return config_manager.runtimeValue(path, default)
    except Exception:  # noqa: BLE001
        return default


def _runtime_int(config_manager: Any | None, path: str, default: int) -> int:
    try:
        return int(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return int(default)


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


def _clip_text(text: str, max_chars: int) -> str:
    value = _safe_text(text)
    if max_chars <= 0:
        return value
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


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
    item = {
        "section_id": f"s{len(sections) + 1}",
        "title": "",
        "level": max(1, int(level)),
        "text": value,
        "start_char": start_char,
        "end_char": end_char,
        "page_start": None,
        "page_end": None,
        "metadata": {
            "element_type": element_type,
            "source": "structured_data",
        },
    }
    if isinstance(metadata, dict):
        item["metadata"].update(dict(metadata))
    sections.append(item)
    return end_char + 2


def _flatten_json_items(
    value: Any,
    *,
    path: str,
    max_depth: int,
    max_items: int,
    value_max_chars: int,
    items: list[tuple[str, str]],
) -> None:
    if len(items) >= max_items:
        return
    if max_depth < 0:
        return

    if isinstance(value, dict):
        for key, child in value.items():
            child_key = _safe_text(key) or "key"
            child_path = f"{path}.{child_key}" if path else child_key
            _flatten_json_items(
                child,
                path=child_path,
                max_depth=max_depth - 1,
                max_items=max_items,
                value_max_chars=value_max_chars,
                items=items,
            )
            if len(items) >= max_items:
                break
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            _flatten_json_items(
                child,
                path=child_path,
                max_depth=max_depth - 1,
                max_items=max_items,
                value_max_chars=value_max_chars,
                items=items,
            )
            if len(items) >= max_items:
                break
        return

    rendered = _clip_text(_safe_text(value), value_max_chars)
    if rendered:
        items.append((path or "$", rendered))


def _flatten_xml_items(
    element: ET.Element,
    *,
    current_path: str,
    max_depth: int,
    max_items: int,
    value_max_chars: int,
    items: list[tuple[str, str]],
) -> None:
    if len(items) >= max_items or max_depth < 0:
        return

    node_name = _safe_text(element.tag) or "node"
    node_path = f"{current_path}/{node_name}" if current_path else f"/{node_name}"

    for attr_key, attr_value in dict(element.attrib or {}).items():
        if len(items) >= max_items:
            break
        attr_name = _safe_text(attr_key) or "attr"
        attr_text = _clip_text(_safe_text(attr_value), value_max_chars)
        if attr_text:
            items.append((f"{node_path}@{attr_name}", attr_text))

    node_text = _clip_text(_safe_text(element.text), value_max_chars)
    if node_text and len(items) < max_items:
        items.append((node_path, node_text))

    for child in list(element):
        _flatten_xml_items(
            child,
            current_path=node_path,
            max_depth=max_depth - 1,
            max_items=max_items,
            value_max_chars=value_max_chars,
            items=items,
        )
        if len(items) >= max_items:
            break


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


class StructuredDataParserAdapter(DocumentParserAdapter):
    adapter_name = "structured-data"
    adapter_version = "v1"

    def __init__(self, *, config_manager: Any | None = None):
        self._config = config_manager

    def can_parse(self, profile: DocumentParseProfile) -> bool:
        ext = str(profile.file_extension or "").strip().lower()
        mime = str(profile.file_mime or "").strip().lower()
        if ext in _STRUCTURED_EXTENSIONS:
            return True
        if mime in {
            "application/json",
            "text/json",
            "application/xml",
            "text/xml",
            "text/csv",
            "application/csv",
            "text/tab-separated-values",
        }:
            return True
        return False

    def confidence(self, profile: DocumentParseProfile) -> float:
        ext = str(profile.file_extension or "").strip().lower()
        score = 0.3
        if ext == ".json":
            score += 0.45
        elif ext == ".xml":
            score += 0.42
        elif ext in {".csv", ".tsv"}:
            score += 0.4
        mime = str(profile.file_mime or "").strip().lower()
        if any(token in mime for token in ("json", "xml", "csv", "tab-separated")):
            score += 0.2
        return clamp_confidence(score, default=0.0)

    def cost(self, profile: DocumentParseProfile) -> float:
        size_mb = max(0.0, float(profile.file_size_bytes or 0) / (1024.0 * 1024.0))
        return clamp_cost(0.85 + (size_mb / 2.8), default=0.85)

    def _parse_json(self, path: Path) -> dict[str, Any]:
        value_max_chars = max(24, _runtime_int(self._config, "documents.structured_value_max_chars", 400))
        max_depth = max(1, _runtime_int(self._config, "documents.structured_max_depth", 8))
        max_items = max(1, _runtime_int(self._config, "documents.structured_max_items", 4000))

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        items: list[tuple[str, str]] = []
        _flatten_json_items(
            payload,
            path="",
            max_depth=max_depth,
            max_items=max_items,
            value_max_chars=value_max_chars,
            items=items,
        )

        sections: list[dict[str, Any]] = []
        cursor = 0
        for key_path, value in items:
            cursor = _append_section(
                sections,
                text=f"{key_path}: {value}",
                element_type="structured_field",
                level=2,
                cursor=cursor,
                metadata={"path": key_path, "format": "json"},
            )

        content_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))
        warnings: list[str] = []
        if len(items) >= max_items:
            warnings.append("structured_item_limit_applied")

        return {
            "status": "parsed" if content_text else "partial",
            "content_text": content_text,
            "sections": sections,
            "metadata": {
                "parser_kind": "structured_data",
                "structured_format": "json",
                "item_count": len(items),
                "max_depth": max_depth,
                "max_items": max_items,
            },
            "warnings": warnings,
            "errors": [],
        }

    def _parse_xml(self, path: Path) -> dict[str, Any]:
        value_max_chars = max(24, _runtime_int(self._config, "documents.structured_value_max_chars", 400))
        max_depth = max(1, _runtime_int(self._config, "documents.structured_max_depth", 8))
        max_items = max(1, _runtime_int(self._config, "documents.structured_max_items", 4000))

        root = ET.parse(str(path)).getroot()
        items: list[tuple[str, str]] = []
        _flatten_xml_items(
            root,
            current_path="",
            max_depth=max_depth,
            max_items=max_items,
            value_max_chars=value_max_chars,
            items=items,
        )

        sections: list[dict[str, Any]] = []
        cursor = 0
        for key_path, value in items:
            cursor = _append_section(
                sections,
                text=f"{key_path}: {value}",
                element_type="structured_field",
                level=2,
                cursor=cursor,
                metadata={"path": key_path, "format": "xml"},
            )

        content_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))
        warnings: list[str] = []
        if len(items) >= max_items:
            warnings.append("structured_item_limit_applied")

        return {
            "status": "parsed" if content_text else "partial",
            "content_text": content_text,
            "sections": sections,
            "metadata": {
                "parser_kind": "structured_data",
                "structured_format": "xml",
                "item_count": len(items),
                "max_depth": max_depth,
                "max_items": max_items,
            },
            "warnings": warnings,
            "errors": [],
        }

    def _parse_delimited(self, path: Path, *, delimiter_hint: str | None = None) -> dict[str, Any]:
        max_rows = max(1, _runtime_int(self._config, "documents.structured_csv_max_rows", 2000))
        max_cols = max(1, _runtime_int(self._config, "documents.structured_csv_max_cols", 32))
        low_signal_headers = set(
            _runtime_csv(
                self._config,
                "documents.structured_csv_low_signal_headers",
            )
            or ["id", "idx", "index", "row", "row_id", "number", "no"]
        )

        text, encoding = _read_text(path)
        sample = text[:2048]
        if delimiter_hint:
            delimiter = delimiter_hint
        else:
            try:
                delimiter = csv.Sniffer().sniff(sample).delimiter
            except Exception:  # noqa: BLE001
                delimiter = ","

        rows: list[list[str]] = []
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        for row_index, row in enumerate(reader, start=1):
            if row_index > max_rows:
                break
            values = [_safe_text(cell) for cell in list(row)[:max_cols]]
            rows.append(values)

        headers = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []

        sections: list[dict[str, Any]] = []
        cursor = 0
        for row_index, row in enumerate(body, start=2):
            pieces: list[str] = []
            for column_index, value in enumerate(row):
                if not value:
                    continue
                header = headers[column_index] if column_index < len(headers) else f"col_{column_index + 1}"
                header_text = _safe_text(header) or f"col_{column_index + 1}"
                if header_text.lower() in low_signal_headers and value.replace(".", "", 1).isdigit():
                    continue
                pieces.append(f"{header_text}={value}")
            if not pieces:
                continue
            row_text = "; ".join(pieces)
            cursor = _append_section(
                sections,
                text=row_text,
                element_type="table_row",
                level=2,
                cursor=cursor,
                metadata={
                    "row_index": row_index,
                    "column_count": len(row),
                    "format": "tsv" if delimiter == "\t" else "csv",
                },
            )

        content_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))
        warnings: list[str] = []
        if len(rows) > max_rows:
            warnings.append("structured_csv_row_limit_applied")
        if len(rows) <= 1:
            warnings.append("structured_csv_no_body_rows")

        return {
            "status": "parsed" if content_text else "partial",
            "content_text": content_text,
            "sections": sections,
            "metadata": {
                "parser_kind": "structured_data",
                "structured_format": "tsv" if delimiter == "\t" else "csv",
                "row_count": max(0, len(rows) - 1),
                "column_count": len(headers),
                "encoding": encoding,
                "delimiter": delimiter,
                "max_rows": max_rows,
                "max_cols": max_cols,
            },
            "warnings": warnings,
            "errors": [],
        }

    def parse(self, profile: DocumentParseProfile) -> dict[str, Any]:
        path = Path(profile.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parser input file does not exist: {path}")

        ext = str(profile.file_extension or "").strip().lower()
        try:
            if ext == ".json":
                output = self._parse_json(path)
            elif ext == ".xml":
                output = self._parse_xml(path)
            elif ext == ".tsv":
                output = self._parse_delimited(path, delimiter_hint="\t")
            elif ext == ".csv":
                output = self._parse_delimited(path, delimiter_hint=None)
            else:
                return {
                    "status": "failed",
                    "content_text": "",
                    "sections": [],
                    "metadata": {
                        "parser_kind": "structured_data",
                        "structured_format": ext,
                    },
                    "warnings": [],
                    "errors": [f"unsupported_structured_format:{ext or 'unknown'}"],
                    "confidence": 0.0,
                    "cost": self.cost(profile),
                }
        except Exception as error:  # noqa: BLE001
            return {
                "status": "failed",
                "content_text": "",
                "sections": [],
                "metadata": {
                    "parser_kind": "structured_data",
                    "structured_format": ext,
                },
                "warnings": [],
                "errors": [f"structured_parse_error:{error}"],
                "confidence": 0.0,
                "cost": self.cost(profile),
            }

        output["confidence"] = self.confidence(profile)
        output["cost"] = self.cost(profile)
        return output
