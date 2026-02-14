from __future__ import annotations

from typing import Any


def _as_text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _normalize_row(row: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if not isinstance(row, (list, tuple)):
        return []
    return [_as_text(cell) for cell in row]


def _drop_empty_rows(rows: list[list[str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        if any(cell for cell in row):
            normalized.append(row)
    return normalized


def _table_to_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    line_rows = [" | ".join(cell if cell else " " for cell in row) for row in rows]
    return "\n".join(line_rows).strip()


def _table_dimensions(rows: list[list[str]]) -> tuple[int, int, int]:
    if not rows:
        return 0, 0, 0
    row_count = len(rows)
    col_count = max((len(row) for row in rows), default=0)
    cell_count = sum(len(row) for row in rows)
    return row_count, col_count, cell_count


def _extract_table_objects(page: Any, table_settings: dict[str, Any]) -> list[Any]:
    try:
        found = page.find_tables(table_settings=table_settings)
    except Exception:  # noqa: BLE001
        return []
    return list(found or [])


def extract_tables_from_pdfplumber_page(
    page: Any,
    *,
    page_number: int,
    max_tables: int = 6,
    min_cells: int = 4,
    table_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if page is None:
        return []

    table_settings = dict(table_settings or {})
    tables: list[dict[str, Any]] = []
    limit = max(0, int(max_tables))
    min_cells = max(1, int(min_cells))
    if limit == 0:
        return tables

    table_objects = _extract_table_objects(page, table_settings)
    for index, table_object in enumerate(table_objects, start=1):
        if len(tables) >= limit:
            break
        try:
            raw_rows = table_object.extract() or []
        except Exception:  # noqa: BLE001
            raw_rows = []
        rows = _drop_empty_rows([_normalize_row(row) for row in raw_rows])
        row_count, col_count, cell_count = _table_dimensions(rows)
        if cell_count < min_cells:
            continue
        text = _table_to_text(rows)
        if not text:
            continue
        bbox_source = getattr(table_object, "bbox", None)
        bbox = {
            "x0": float(bbox_source[0]) if isinstance(bbox_source, (tuple, list)) and len(bbox_source) > 0 else 0.0,
            "y0": float(bbox_source[1]) if isinstance(bbox_source, (tuple, list)) and len(bbox_source) > 1 else 0.0,
            "x1": float(bbox_source[2]) if isinstance(bbox_source, (tuple, list)) and len(bbox_source) > 2 else 0.0,
            "y1": float(bbox_source[3]) if isinstance(bbox_source, (tuple, list)) and len(bbox_source) > 3 else 0.0,
        }
        tables.append(
            {
                "element_type": "table",
                "level": 2,
                "text": text,
                "page_number": int(page_number),
                "bbox": bbox,
                "confidence": 0.88,
                "source": "pdfplumber_table",
                "metadata": {
                    "table_index": index,
                    "rows": row_count,
                    "columns": col_count,
                    "cells": cell_count,
                },
            }
        )

    if tables:
        return tables

    try:
        raw_tables = page.extract_tables(table_settings=table_settings) or []
    except Exception:  # noqa: BLE001
        raw_tables = []

    for index, raw_table in enumerate(raw_tables, start=1):
        if len(tables) >= limit:
            break
        rows = _drop_empty_rows([_normalize_row(row) for row in list(raw_table or [])])
        row_count, col_count, cell_count = _table_dimensions(rows)
        if cell_count < min_cells:
            continue
        text = _table_to_text(rows)
        if not text:
            continue
        tables.append(
            {
                "element_type": "table",
                "level": 2,
                "text": text,
                "page_number": int(page_number),
                "bbox": {},
                "confidence": 0.8,
                "source": "pdfplumber_table_extract",
                "metadata": {
                    "table_index": index,
                    "rows": row_count,
                    "columns": col_count,
                    "cells": cell_count,
                },
            }
        )

    return tables
