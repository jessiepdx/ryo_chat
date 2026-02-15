from __future__ import annotations

import hashlib
import re
from typing import Any

from hypermindlabs.document_chunk_rules import (
    chunk_runtime_settings,
    resolve_chunk_rule,
)


DOCUMENT_CHUNK_SCHEMA_VERSION = 1
_TOKEN_PATTERN = re.compile(r"\S+")


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _word_count(text: str) -> int:
    value = _as_text(text)
    if not value:
        return 0
    return len([part for part in value.split() if part])


def _chunk_key(
    *,
    document_version_id: int,
    node_key: str,
    local_start: int | None,
    local_end: int | None,
    segment_index: int,
    text: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{document_version_id}|{node_key}|{local_start}|{local_end}|"
            f"{segment_index}|{hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]}"
        ).encode("utf-8")
    ).hexdigest()
    return f"ck_{digest[:32]}"


def _section_text_map(canonical_output: dict[str, Any]) -> dict[str, str]:
    section_map: dict[str, str] = {}
    sections = list(canonical_output.get("sections") or []) if isinstance(canonical_output, dict) else []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = _as_text(section.get("section_id"))
        if not section_id:
            continue
        section_map[section_id] = _as_text(section.get("text"))
    return section_map


def _token_segments(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    token_matches = list(_TOKEN_PATTERN.finditer(text))
    if not token_matches:
        return []
    window_tokens = max(1, min(int(max_tokens), int(target_tokens)))
    overlap = max(0, min(int(overlap_tokens), window_tokens - 1))
    step = max(1, window_tokens - overlap)
    segments: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for segment_index, start_token in enumerate(range(0, len(token_matches), step), start=1):
        end_token = min(len(token_matches), start_token + window_tokens)
        start_char = int(token_matches[start_token].start())
        end_char = int(token_matches[end_token - 1].end())
        token_range = (start_token, end_token)
        if token_range in seen_ranges:
            continue
        seen_ranges.add(token_range)
        segment_text = text[start_char:end_char].strip()
        if not segment_text:
            continue
        segments.append(
            {
                "text": segment_text,
                "local_start": start_char,
                "local_end": end_char,
                "token_start": start_token,
                "token_end": end_token,
                "token_count": max(0, end_token - start_token),
            }
        )
        if len(segments) >= max_segments or end_token >= len(token_matches):
            break
    return segments


def _line_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        start = cursor
        end = cursor + len(line)
        cursor += len(raw_line)
        if not line.strip():
            continue
        spans.append({"text": line, "start": start, "end": end})
    if not spans and _as_text(text):
        spans.append({"text": _as_text(text), "start": 0, "end": len(text)})
    return spans


def _line_window_segments(
    text: str,
    *,
    lines_per_chunk: int,
    overlap_lines: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    lines = _line_spans(text)
    if not lines:
        return []
    if len(lines) <= lines_per_chunk:
        segment_text = "\n".join(line["text"] for line in lines).strip()
        return [
            {
                "text": segment_text,
                "local_start": int(lines[0]["start"]),
                "local_end": int(lines[-1]["end"]),
                "line_start": 0,
                "line_end": len(lines),
                "token_count": _word_count(segment_text),
            }
        ]
    overlap = max(0, min(overlap_lines, max(0, lines_per_chunk - 1)))
    step = max(1, lines_per_chunk - overlap)
    segments: list[dict[str, Any]] = []
    for line_start in range(0, len(lines), step):
        line_end = min(len(lines), line_start + lines_per_chunk)
        window = lines[line_start:line_end]
        if not window:
            continue
        segment_text = "\n".join(line["text"] for line in window).strip()
        if not segment_text:
            continue
        segments.append(
            {
                "text": segment_text,
                "local_start": int(window[0]["start"]),
                "local_end": int(window[-1]["end"]),
                "line_start": line_start,
                "line_end": line_end,
                "token_count": _word_count(segment_text),
            }
        )
        if len(segments) >= max_segments or line_end >= len(lines):
            break
    return segments


def _table_segments(
    text: str,
    *,
    rows_per_chunk: int,
    overlap_rows: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    lines = _line_spans(text)
    if not lines:
        return []
    if len(lines) == 1:
        segment_text = _as_text(lines[0]["text"])
        return [
            {
                "text": segment_text,
                "local_start": int(lines[0]["start"]),
                "local_end": int(lines[0]["end"]),
                "row_start": 0,
                "row_end": 1,
                "token_count": _word_count(segment_text),
                "header_repeated": False,
            }
        ]
    header = lines[0]
    rows = lines[1:]
    overlap = max(0, min(overlap_rows, max(0, rows_per_chunk - 1)))
    step = max(1, rows_per_chunk - overlap)
    segments: list[dict[str, Any]] = []
    for row_start in range(0, len(rows), step):
        row_end = min(len(rows), row_start + rows_per_chunk)
        window_rows = rows[row_start:row_end]
        if not window_rows:
            continue
        include_header = True
        row_text = "\n".join(row["text"] for row in window_rows)
        if include_header:
            segment_text = f"{header['text']}\n{row_text}".strip()
        else:
            segment_text = row_text.strip()
        local_start = int(window_rows[0]["start"])
        local_end = int(window_rows[-1]["end"])
        if row_start == 0:
            local_start = int(header["start"])
        segments.append(
            {
                "text": segment_text,
                "local_start": local_start,
                "local_end": local_end,
                "row_start": row_start,
                "row_end": row_end,
                "token_count": _word_count(segment_text),
                "header_repeated": bool(row_start > 0),
            }
        )
        if len(segments) >= max_segments or row_end >= len(rows):
            break
    return segments


def _node_depth(node_key: str, nodes_by_key: dict[str, dict[str, Any]]) -> int:
    depth = 0
    current_key = str(node_key)
    visited: set[str] = set()
    while current_key and current_key in nodes_by_key and current_key not in visited:
        visited.add(current_key)
        parent_key = _as_text(nodes_by_key[current_key].get("parent_node_key"))
        if not parent_key:
            break
        depth += 1
        current_key = parent_key
    return depth


def _heading_trail(node_key: str, nodes_by_key: dict[str, dict[str, Any]]) -> list[str]:
    trail: list[str] = []
    current_key = str(node_key)
    visited: set[str] = set()
    while current_key and current_key in nodes_by_key and current_key not in visited:
        visited.add(current_key)
        node = nodes_by_key[current_key]
        node_type = _as_text(node.get("node_type")).lower()
        node_title = _as_text(node.get("node_title"))
        if node_title and node_type in {"document", "section", "subsection"}:
            trail.append(node_title)
        current_key = _as_text(node.get("parent_node_key"))
    trail.reverse()
    return trail


def build_document_chunks(
    *,
    tree_payload: dict[str, Any] | None,
    canonical_output: dict[str, Any] | None,
    document_version_id: int,
    config_manager: Any | None = None,
) -> dict[str, Any]:
    settings = chunk_runtime_settings(config_manager)
    max_total_chunks = int(settings.get("chunk_max_total_chunks", 6000))
    section_text_by_id = _section_text_map(canonical_output if isinstance(canonical_output, dict) else {})

    source_tree = dict(tree_payload) if isinstance(tree_payload, dict) else {}
    nodes = list(source_tree.get("nodes") or [])
    nodes_by_key = {
        _as_text(node.get("node_key")): dict(node)
        for node in nodes
        if isinstance(node, dict) and _as_text(node.get("node_key"))
    }

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    by_node_type: dict[str, int] = {}
    truncated = False

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_key = _as_text(node.get("node_key"))
        node_type = _as_text(node.get("node_type"), "paragraph").lower()
        if not node_key or node_type == "document":
            continue
        if len(chunks) >= max_total_chunks:
            truncated = True
            break

        node_metadata = node.get("node_metadata")
        metadata = dict(node_metadata) if isinstance(node_metadata, dict) else {}
        section_id = _as_text(metadata.get("section_id"))
        text = _as_text(section_text_by_id.get(section_id))
        if not text:
            text = _as_text(metadata.get("section_text"))
        if not text:
            continue

        depth = _node_depth(node_key, nodes_by_key)
        rule = resolve_chunk_rule(node_type, depth=depth, settings=settings)
        if rule.split_mode == "none":
            local_segments = [
                {
                    "text": text,
                    "local_start": 0,
                    "local_end": len(text),
                    "token_count": _word_count(text),
                }
            ]
        elif rule.split_mode == "lines":
            local_segments = _line_window_segments(
                text,
                lines_per_chunk=max(1, int(settings.get("chunk_code_lines_per_chunk", 20))),
                overlap_lines=2 if rule.overlap_tokens > 0 else 0,
                max_segments=rule.max_segments,
            )
        elif rule.split_mode == "list_items":
            local_segments = _line_window_segments(
                text,
                lines_per_chunk=max(1, int(settings.get("chunk_list_items_per_chunk", 12))),
                overlap_lines=1 if rule.overlap_tokens > 0 else 0,
                max_segments=rule.max_segments,
            )
        elif rule.split_mode == "table_rows":
            local_segments = _table_segments(
                text,
                rows_per_chunk=max(1, int(settings.get("chunk_table_rows_per_chunk", 20))),
                overlap_rows=1 if rule.overlap_tokens > 0 else 0,
                max_segments=rule.max_segments,
            )
        else:
            local_segments = _token_segments(
                text,
                target_tokens=rule.target_tokens,
                overlap_tokens=rule.overlap_tokens,
                max_tokens=rule.max_tokens,
                max_segments=rule.max_segments,
            )

        heading_trail = _heading_trail(node_key, nodes_by_key)
        parent_node_key = _as_text(node.get("parent_node_key"))
        parent_node_path = ""
        if parent_node_key and parent_node_key in nodes_by_key:
            parent_node_path = _as_text(nodes_by_key[parent_node_key].get("path"))

        node_char_start = _optional_int(node.get("char_start"))
        node_char_end = _optional_int(node.get("char_end"))
        node_page_start = _optional_int(node.get("page_start"))
        node_page_end = _optional_int(node.get("page_end"))

        for segment_index, segment in enumerate(local_segments):
            if len(chunks) >= max_total_chunks:
                truncated = True
                break
            chunk_text = _as_text(segment.get("text"))
            if not chunk_text:
                continue

            local_start = _optional_int(segment.get("local_start"))
            local_end = _optional_int(segment.get("local_end"))
            start_char = None
            end_char = None
            if node_char_start is not None and local_start is not None:
                start_char = node_char_start + int(local_start)
            if node_char_start is not None and local_end is not None:
                end_char = node_char_start + int(local_end)
            if node_char_end is not None and end_char is not None and end_char > node_char_end:
                end_char = node_char_end

            token_count = int(segment.get("token_count", _word_count(chunk_text)) or _word_count(chunk_text))
            chunk_record = {
                "chunk_key": _chunk_key(
                    document_version_id=document_version_id,
                    node_key=node_key,
                    local_start=local_start,
                    local_end=local_end,
                    segment_index=segment_index,
                    text=chunk_text,
                ),
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "document_node_key": node_key,
                "token_count": token_count,
                "start_char": start_char,
                "end_char": end_char,
                "start_page": node_page_start,
                "end_page": node_page_end,
                "chunk_digest": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "chunk_metadata": {
                    "node_key": node_key,
                    "node_type": node_type,
                    "node_path": _as_text(node.get("path")),
                    "parent_node_key": parent_node_key,
                    "parent_node_path": parent_node_path,
                    "heading_trail": heading_trail,
                    "depth": depth,
                    "split_mode": rule.split_mode,
                    "rule": {
                        "target_tokens": rule.target_tokens,
                        "overlap_tokens": rule.overlap_tokens,
                        "max_tokens": rule.max_tokens,
                        "max_segments": rule.max_segments,
                    },
                    "node_section_id": section_id,
                    "segment_index": segment_index,
                    "segment_count": len(local_segments),
                    "header_repeated": bool(segment.get("header_repeated")),
                    "char_count": len(chunk_text),
                    "token_count": token_count,
                },
            }
            chunks.append(chunk_record)
            chunk_index += 1
            by_node_type[node_type] = int(by_node_type.get(node_type, 0)) + 1

    return {
        "schema_version": DOCUMENT_CHUNK_SCHEMA_VERSION,
        "document_version_id": int(max(1, int(document_version_id))),
        "chunks": chunks,
        "stats": {
            "chunk_count": len(chunks),
            "chunk_count_by_node_type": by_node_type,
            "max_total_chunks": max_total_chunks,
            "truncated": truncated,
        },
    }


def build_chunk_artifact_summary(chunk_payload: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(chunk_payload) if isinstance(chunk_payload, dict) else {}
    stats = dict(source.get("stats") or {})
    chunks = list(source.get("chunks") or [])
    return {
        "schema_version": DOCUMENT_CHUNK_SCHEMA_VERSION,
        "chunk_count": int(stats.get("chunk_count", len(chunks)) or 0),
        "chunk_count_by_node_type": dict(stats.get("chunk_count_by_node_type") or {}),
        "max_total_chunks": int(stats.get("max_total_chunks", 0) or 0),
        "truncated": bool(stats.get("truncated")),
    }
