from __future__ import annotations

import hashlib
from typing import Any

from hypermindlabs.document_graph import repair_document_graph
from hypermindlabs.document_parser.contracts import validate_canonical_parse_output


DOCUMENT_TREE_SCHEMA_VERSION = 1

_FIGURE_TYPES = {"figure", "caption", "image", "diagram"}
_TABLE_TYPES = {"table", "table_row"}
_LIST_TYPES = {"list", "list_item"}
_CODE_TYPES = {"code", "pre"}
_FOOTNOTE_TYPES = {"footnote", "note"}
_HEADING_TYPES = {"heading", "title"}


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if parsed < 0:
        parsed = int(default)
    return max(0, parsed)


def _word_count(text: str) -> int:
    value = _as_text(text)
    if not value:
        return 0
    return len([part for part in value.split() if part])


def _stable_node_key(*parts: Any) -> str:
    seed = "|".join(_as_text(part) for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"nd_{digest[:32]}"


def _node_type_for_section(section: dict[str, Any]) -> str:
    metadata = section.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
    element_type = _as_text(metadata_dict.get("element_type"), "paragraph").lower()
    level = max(1, _non_negative_int(section.get("level"), 1))
    if element_type in _HEADING_TYPES:
        return "section" if level <= 1 else "subsection"
    if element_type in _LIST_TYPES:
        return "list"
    if element_type in _TABLE_TYPES:
        return "table"
    if element_type in _CODE_TYPES:
        return "code"
    if element_type in _FIGURE_TYPES:
        return "figure"
    if element_type in _FOOTNOTE_TYPES:
        return "footnote"
    return "paragraph"


def _node_title_for_section(section: dict[str, Any], *, node_type: str) -> str:
    title = _as_text(section.get("title"))
    text = _as_text(section.get("text"))
    if node_type in {"section", "subsection"}:
        return title or text[:160]
    if node_type in {"figure", "table"}:
        first_line = text.split("\n", 1)[0].strip()
        return first_line[:160]
    return ""


def _section_span(section: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    char_start = _optional_int(section.get("start_char"))
    char_end = _optional_int(section.get("end_char"))
    if (
        char_start is not None
        and char_end is not None
        and char_end < char_start
    ):
        char_end = char_start
    page_start = _optional_int(section.get("page_start"))
    page_end = _optional_int(section.get("page_end"))
    if (
        page_start is not None
        and page_end is not None
        and page_end < page_start
    ):
        page_end = page_start
    return page_start, page_end, char_start, char_end


def build_canonical_document_tree(
    canonical_output: dict[str, Any],
    *,
    document_version_id: int,
) -> dict[str, Any]:
    canonical = validate_canonical_parse_output(canonical_output if isinstance(canonical_output, dict) else {})
    sections = list(canonical.get("sections") or [])
    content_text = _as_text(canonical.get("content_text"))
    canonical_metadata = dict(canonical.get("metadata") or {})

    page_values: list[int] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        page_start, page_end, _, _ = _section_span(section)
        if page_start is not None:
            page_values.append(page_start)
        if page_end is not None:
            page_values.append(page_end)
    page_start = min(page_values) if page_values else None
    page_end = max(page_values) if page_values else None

    root_title = _as_text(canonical_metadata.get("title"))
    root_key = _stable_node_key(
        "document",
        root_title,
        _as_text(canonical.get("canonical_schema")),
        hashlib.sha256(content_text.encode("utf-8")).hexdigest()[:24],
    )
    root_node: dict[str, Any] = {
        "node_key": root_key,
        "node_type": "document",
        "parent_node_key": None,
        "node_title": root_title,
        "ordinal": 0,
        "token_count": _word_count(content_text),
        "page_start": page_start,
        "page_end": page_end,
        "char_start": 0 if content_text else None,
        "char_end": len(content_text) if content_text else None,
        "path": "/document",
        "node_metadata": {
            "canonical_schema": _as_text(canonical.get("canonical_schema")),
            "status": _as_text(canonical.get("status"), "parsed"),
            "warnings_count": len([item for item in list(canonical.get("warnings") or []) if _as_text(item)]),
            "errors_count": len([item for item in list(canonical.get("errors") or []) if _as_text(item)]),
            "metadata": canonical_metadata,
        },
    }

    nodes: list[dict[str, Any]] = [root_node]
    path_by_key: dict[str, str] = {root_key: "/document"}
    child_ordinals: dict[str, int] = {root_key: 0}
    heading_stack: list[tuple[int, str]] = []

    for section_index, raw_section in enumerate(sections, start=1):
        section = dict(raw_section) if isinstance(raw_section, dict) else {}
        text = _as_text(section.get("text"))
        if not text:
            continue

        level = max(1, _non_negative_int(section.get("level"), 1))
        node_type = _node_type_for_section(section)
        node_title = _node_title_for_section(section, node_type=node_type)

        if node_type in {"section", "subsection"}:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_key = heading_stack[-1][1] if heading_stack else root_key
        else:
            parent_key = heading_stack[-1][1] if heading_stack else root_key

        parent_path = _as_text(path_by_key.get(parent_key), "/document")
        ordinal = child_ordinals.get(parent_key, 0)
        child_ordinals[parent_key] = ordinal + 1
        path = f"{parent_path}/{node_type}:{ordinal + 1}"

        section_id = _as_text(section.get("section_id"), f"s{section_index}")
        metadata = section.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        page_start, page_end, char_start, char_end = _section_span(section)
        node_key = _stable_node_key(
            node_type,
            path,
            node_title,
            section_id,
            _as_text(char_start),
            _as_text(char_end),
            _as_text(page_start),
            _as_text(page_end),
            hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
        node_payload = {
            "node_key": node_key,
            "node_type": node_type,
            "parent_node_key": parent_key,
            "node_title": node_title,
            "ordinal": ordinal,
            "token_count": _word_count(text),
            "page_start": page_start,
            "page_end": page_end,
            "char_start": char_start,
            "char_end": char_end,
            "path": path,
            "node_metadata": {
                "section_id": section_id,
                "section_level": level,
                "element_type": _as_text(metadata_dict.get("element_type"), "paragraph"),
                "source_metadata": metadata_dict,
            },
        }
        nodes.append(node_payload)
        path_by_key[node_key] = path
        child_ordinals.setdefault(node_key, 0)
        if node_type in {"section", "subsection"}:
            heading_stack.append((level, node_key))

    repaired = repair_document_graph(nodes, root_node_key=root_key)
    repaired_nodes = list(repaired.get("nodes") or [])
    repaired_edges = list(repaired.get("edges") or [])
    diagnostics = dict(repaired.get("diagnostics") or {})

    node_type_counts: dict[str, int] = {}
    for node in repaired_nodes:
        node_type = _as_text((node or {}).get("node_type"), "paragraph")
        node_type_counts[node_type] = int(node_type_counts.get(node_type, 0)) + 1

    return {
        "schema_version": DOCUMENT_TREE_SCHEMA_VERSION,
        "document_version_id": int(max(1, int(document_version_id))),
        "root_node_key": _as_text(diagnostics.get("root_node_key"), root_key),
        "nodes": repaired_nodes,
        "edges": repaired_edges,
        "integrity": diagnostics,
        "stats": {
            "node_count": len(repaired_nodes),
            "edge_count": len(repaired_edges),
            "node_type_counts": node_type_counts,
        },
    }


def build_tree_artifact_summary(tree_payload: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(tree_payload) if isinstance(tree_payload, dict) else {}
    stats = dict(source.get("stats") or {})
    integrity = dict(source.get("integrity") or {})
    return {
        "schema_version": DOCUMENT_TREE_SCHEMA_VERSION,
        "root_node_key": _as_text(source.get("root_node_key")),
        "node_count": _non_negative_int(stats.get("node_count"), len(list(source.get("nodes") or []))),
        "edge_count": _non_negative_int(stats.get("edge_count"), len(list(source.get("edges") or []))),
        "node_type_counts": dict(stats.get("node_type_counts") or {}),
        "integrity": {
            "is_valid": bool(integrity.get("is_valid")),
            "was_repaired": bool(integrity.get("was_repaired")),
            "repaired_orphan_count": _non_negative_int(integrity.get("repaired_orphan_count"), 0),
            "repaired_cycle_count": _non_negative_int(integrity.get("repaired_cycle_count"), 0),
            "reordered_parent_count": _non_negative_int(integrity.get("reordered_parent_count"), 0),
        },
    }
