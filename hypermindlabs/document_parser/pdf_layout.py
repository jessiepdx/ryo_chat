from __future__ import annotations

import math
import re
from statistics import median
from typing import Any


_BULLET_PREFIX_PATTERN = re.compile(r"^(?:[-*•]|\d+[.)]|\([a-z0-9]+\))\s+")
_CAPTION_PREFIX_PATTERN = re.compile(r"^(?:figure|fig\.|table|chart|diagram)\b", re.IGNORECASE)
_FOOTNOTE_PATTERN = re.compile(r"^(?:\[\d+\]|\d{1,2}[.)])\s+")


def _as_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return float(parsed)


def _line_font_size(words: list[dict[str, Any]]) -> float:
    sizes = [_safe_float(word.get("size"), 0.0) for word in words if _safe_float(word.get("size"), 0.0) > 0.0]
    if not sizes:
        return 0.0
    return float(median(sizes))


def group_pdf_words_into_lines(
    words: list[dict[str, Any]],
    *,
    vertical_tolerance: float = 3.0,
) -> list[dict[str, Any]]:
    if not isinstance(words, list) or not words:
        return []

    normalized = [dict(item) for item in words if isinstance(item, dict) and _as_text(item.get("text"))]
    normalized.sort(key=lambda item: (_safe_float(item.get("top"), 0.0), _safe_float(item.get("x0"), 0.0)))

    groups: list[list[dict[str, Any]]] = []
    anchors: list[float] = []
    for word in normalized:
        top = _safe_float(word.get("top"), 0.0)
        placed = False
        for index, anchor in enumerate(anchors):
            if abs(top - anchor) <= max(0.5, float(vertical_tolerance)):
                groups[index].append(word)
                anchors[index] = (anchor + top) / 2.0
                placed = True
                break
        if not placed:
            groups.append([word])
            anchors.append(top)

    lines: list[dict[str, Any]] = []
    for words_in_line in groups:
        words_in_line.sort(key=lambda item: _safe_float(item.get("x0"), 0.0))
        text = " ".join(_as_text(item.get("text")) for item in words_in_line if _as_text(item.get("text")))
        if not text:
            continue
        x0_values = [_safe_float(item.get("x0"), 0.0) for item in words_in_line]
        x1_values = [_safe_float(item.get("x1"), 0.0) for item in words_in_line]
        top_values = [_safe_float(item.get("top"), 0.0) for item in words_in_line]
        bottom_values = [_safe_float(item.get("bottom"), 0.0) for item in words_in_line]
        lines.append(
            {
                "text": text,
                "x0": min(x0_values) if x0_values else 0.0,
                "x1": max(x1_values) if x1_values else 0.0,
                "top": min(top_values) if top_values else 0.0,
                "bottom": max(bottom_values) if bottom_values else 0.0,
                "font_size": _line_font_size(words_in_line),
                "word_count": len(words_in_line),
            }
        )

    lines.sort(key=lambda item: (_safe_float(item.get("top"), 0.0), _safe_float(item.get("x0"), 0.0)))
    return lines


def detect_multicolumn_layout(lines: list[dict[str, Any]], *, page_width: float) -> bool:
    width = max(1.0, float(page_width or 0.0))
    if not isinstance(lines, list) or len(lines) < 6:
        return False
    left_count = 0
    right_count = 0
    for line in lines:
        x0 = _safe_float(line.get("x0"), 0.0)
        if x0 <= width * 0.45:
            left_count += 1
        elif x0 >= width * 0.55:
            right_count += 1
    return left_count >= 3 and right_count >= 3


def classify_layout_line(
    line: dict[str, Any],
    *,
    heading_word_limit: int = 14,
    page_height: float = 0.0,
) -> tuple[str, int, float]:
    text = _as_text(line.get("text"))
    if not text:
        return "paragraph", 1, 0.3

    word_count = len([part for part in text.split(" ") if part])
    font_size = _safe_float(line.get("font_size"), 0.0)
    top = _safe_float(line.get("top"), 0.0)

    if _BULLET_PREFIX_PATTERN.match(text):
        return "list", 2, 0.92
    if _CAPTION_PREFIX_PATTERN.match(text):
        return "caption", 2, 0.9

    near_bottom = page_height > 0.0 and top > (0.83 * page_height)
    if _FOOTNOTE_PATTERN.match(text) and near_bottom:
        return "footnote", 3, 0.78

    upper_ratio = 0.0
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if alpha_chars:
        upper_ratio = len([ch for ch in alpha_chars if ch.isupper()]) / float(len(alpha_chars))

    title_like = text == text.title()
    likely_heading = (
        word_count <= max(2, int(heading_word_limit))
        and len(text) <= 100
        and (upper_ratio >= 0.58 or title_like or text.endswith(":"))
    )
    if likely_heading:
        level = 1 if font_size >= 14.0 else 2
        return "heading", level, 0.84

    return "paragraph", 2, 0.72


def _line_to_element(
    line: dict[str, Any],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    multi_column: bool,
    order_index: int,
) -> dict[str, Any]:
    element_type, level, confidence = classify_layout_line(
        line,
        heading_word_limit=14,
        page_height=page_height,
    )
    x0 = _safe_float(line.get("x0"), 0.0)
    column_index = 1
    if multi_column and page_width > 0.0:
        column_index = 1 if x0 <= page_width * 0.5 else 2
    return {
        "element_type": element_type,
        "level": level,
        "text": _as_text(line.get("text")),
        "page_number": int(page_number),
        "column_index": int(column_index),
        "multi_column": bool(multi_column),
        "bbox": {
            "x0": x0,
            "y0": _safe_float(line.get("top"), 0.0),
            "x1": _safe_float(line.get("x1"), 0.0),
            "y1": _safe_float(line.get("bottom"), 0.0),
        },
        "confidence": confidence,
        "order_index": int(order_index),
        "source": "pdf_text_layout",
    }


def _append_element_to_sections(
    sections: list[dict[str, Any]],
    element: dict[str, Any],
    *,
    section_index: int,
    cursor: int,
) -> int:
    text = _as_text(element.get("text"))
    start_char = int(cursor)
    end_char = start_char + len(text)
    metadata = {
        "element_type": str(element.get("element_type") or "paragraph"),
        "column_index": int(element.get("column_index") or 1),
        "multi_column": bool(element.get("multi_column")),
        "confidence": _safe_float(element.get("confidence"), 0.0),
        "bbox": dict(element.get("bbox") or {}),
        "source": str(element.get("source") or "pdf_text_layout"),
    }
    extra_metadata = element.get("metadata")
    if isinstance(extra_metadata, dict):
        metadata.update(dict(extra_metadata))
    sections.append(
        {
            "section_id": f"s{section_index}",
            "title": text if element.get("element_type") == "heading" else "",
            "level": int(element.get("level") or 1),
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "page_start": int(element.get("page_number") or 0),
            "page_end": int(element.get("page_number") or 0),
            "metadata": metadata,
        }
    )
    return end_char + 2


def build_layout_sections(
    page_layouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    elements: list[dict[str, Any]] = []
    if not isinstance(page_layouts, list):
        return [], "", elements

    for page in page_layouts:
        page_number = int(page.get("page_number") or 0)
        page_width = _safe_float(page.get("page_width"), 0.0)
        page_height = _safe_float(page.get("page_height"), 0.0)
        lines = list(page.get("lines") or [])
        tables = list(page.get("tables") or [])
        multi_column = detect_multicolumn_layout(lines, page_width=page_width)

        for line in lines:
            if multi_column:
                x0 = _safe_float(line.get("x0"), 0.0)
                line["column_index"] = 1 if x0 <= page_width * 0.5 else 2
            else:
                line["column_index"] = 1

        if multi_column:
            lines.sort(
                key=lambda item: (
                    int(item.get("column_index") or 1),
                    _safe_float(item.get("top"), 0.0),
                    _safe_float(item.get("x0"), 0.0),
                )
            )
        else:
            lines.sort(key=lambda item: (_safe_float(item.get("top"), 0.0), _safe_float(item.get("x0"), 0.0)))

        for line_index, line in enumerate(lines, start=1):
            element = _line_to_element(
                dict(line),
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                multi_column=multi_column,
                order_index=line_index,
            )
            elements.append(element)

        for table in tables:
            table_element = dict(table)
            table_element.setdefault("element_type", "table")
            table_element.setdefault("level", 2)
            table_element.setdefault("page_number", page_number)
            table_element.setdefault("column_index", 1)
            table_element.setdefault("multi_column", multi_column)
            table_element.setdefault("confidence", 0.88)
            table_element.setdefault("source", "pdf_table_extractor")
            table_element.setdefault("order_index", 100000 + len(elements))
            if not isinstance(table_element.get("bbox"), dict):
                table_element["bbox"] = {}
            elements.append(table_element)

    elements.sort(
        key=lambda item: (
            int(item.get("page_number") or 0),
            int(item.get("column_index") or 1),
            int(item.get("order_index") or 0),
            _safe_float((item.get("bbox") or {}).get("y0"), 0.0),
        )
    )

    sections: list[dict[str, Any]] = []
    content_parts: list[str] = []
    cursor = 0
    for index, element in enumerate(elements, start=1):
        text = _as_text(element.get("text"))
        if not text:
            continue
        cursor = _append_element_to_sections(sections, element, section_index=index, cursor=cursor)
        content_parts.append(text)

    content_text = "\n\n".join(content_parts)
    return sections, content_text, elements


def build_page_diagnostics(
    page_layouts: list[dict[str, Any]],
    *,
    min_text_density: float = 0.015,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for page in list(page_layouts or []):
        page_number = int(page.get("page_number") or 0)
        page_width = max(1.0, _safe_float(page.get("page_width"), 1.0))
        page_height = max(1.0, _safe_float(page.get("page_height"), 1.0))
        lines = list(page.get("lines") or [])
        tables = list(page.get("tables") or [])
        line_char_count = sum(len(_as_text(line.get("text"))) for line in lines)
        page_area = max(1.0, page_width * page_height)
        text_density = float(line_char_count) / page_area
        element_counts: dict[str, int] = {}
        for line in lines:
            element_type, _, _ = classify_layout_line(
                dict(line),
                heading_word_limit=14,
                page_height=page_height,
            )
            element_counts[element_type] = int(element_counts.get(element_type, 0)) + 1
        element_counts["table"] = int(element_counts.get("table", 0)) + len(tables)
        has_extractable_text = line_char_count > 0
        diagnostics.append(
            {
                "page_number": page_number,
                "char_count": line_char_count,
                "line_count": len(lines),
                "table_count": len(tables),
                "text_density": text_density,
                "has_extractable_text": has_extractable_text,
                "scanned_candidate": bool(
                    not has_extractable_text
                    or (text_density < max(0.000001, float(min_text_density)) and line_char_count < 24)
                ),
                "element_counts": element_counts,
            }
        )
    return diagnostics
