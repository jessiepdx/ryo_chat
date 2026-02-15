from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentChunkRule:
    node_type: str
    split_mode: str
    target_tokens: int
    overlap_tokens: int
    max_tokens: int
    max_segments: int


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


def _positive(value: int, default: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        return int(default)
    return parsed


def _non_negative(value: int, default: int = 0) -> int:
    parsed = int(value)
    if parsed < 0:
        return int(default)
    return parsed


def chunk_runtime_settings(config_manager: Any | None = None) -> dict[str, int]:
    target_tokens = _positive(_runtime_int(config_manager, "documents.chunk_target_tokens", 220), 220)
    overlap_tokens = _non_negative(_runtime_int(config_manager, "documents.chunk_overlap_tokens", 40), 40)
    max_tokens = _positive(_runtime_int(config_manager, "documents.chunk_max_tokens", 320), 320)
    max_chunks_per_node = _positive(
        _runtime_int(config_manager, "documents.chunk_max_chunks_per_node", 16),
        16,
    )
    max_total_chunks = _positive(
        _runtime_int(config_manager, "documents.chunk_max_total_chunks", 6000),
        6000,
    )
    list_items_per_chunk = _positive(
        _runtime_int(config_manager, "documents.chunk_list_items_per_chunk", 12),
        12,
    )
    table_rows_per_chunk = _positive(
        _runtime_int(config_manager, "documents.chunk_table_rows_per_chunk", 20),
        20,
    )
    code_lines_per_chunk = _positive(
        _runtime_int(config_manager, "documents.chunk_code_lines_per_chunk", 20),
        20,
    )
    return {
        "chunk_target_tokens": target_tokens,
        "chunk_overlap_tokens": overlap_tokens,
        "chunk_max_tokens": max(max_tokens, target_tokens),
        "chunk_max_chunks_per_node": max_chunks_per_node,
        "chunk_max_total_chunks": max_total_chunks,
        "chunk_list_items_per_chunk": list_items_per_chunk,
        "chunk_table_rows_per_chunk": table_rows_per_chunk,
        "chunk_code_lines_per_chunk": code_lines_per_chunk,
    }


def resolve_chunk_rule(
    node_type: str,
    *,
    depth: int,
    settings: dict[str, int],
) -> DocumentChunkRule:
    normalized_type = str(node_type or "paragraph").strip().lower() or "paragraph"
    target = int(settings.get("chunk_target_tokens", 220))
    overlap = int(settings.get("chunk_overlap_tokens", 40))
    max_tokens = int(settings.get("chunk_max_tokens", 320))
    max_segments = int(settings.get("chunk_max_chunks_per_node", 16))

    split_mode = "tokens"
    if normalized_type in {"list"}:
        split_mode = "list_items"
        max_segments = max(1, int(settings.get("chunk_max_chunks_per_node", 16)))
    elif normalized_type in {"table"}:
        split_mode = "table_rows"
        overlap = 0
    elif normalized_type in {"code"}:
        split_mode = "lines"
        target = max(120, int(target * 0.9))
    elif normalized_type in {"figure", "footnote"}:
        split_mode = "none"
        target = max(64, int(target * 0.6))
        overlap = 0
    elif normalized_type in {"section", "subsection"}:
        split_mode = "tokens"
        target = int(target * 1.15)
        max_tokens = int(max_tokens * 1.2)

    if depth >= 3 and split_mode == "tokens":
        target = max(96, int(target * 0.9))

    return DocumentChunkRule(
        node_type=normalized_type,
        split_mode=split_mode,
        target_tokens=max(32, _positive(target, 220)),
        overlap_tokens=max(0, _non_negative(overlap, 0)),
        max_tokens=max(64, _positive(max_tokens, 320)),
        max_segments=max(1, _positive(max_segments, 16)),
    )
