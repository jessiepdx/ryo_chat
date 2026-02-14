from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


DOCUMENT_GRAPH_PARENT_EDGE = "parent_child"
DOCUMENT_GRAPH_SIBLING_EDGE = "next_sibling"


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if parsed < 0:
        parsed = int(default)
    return max(0, parsed)


def _normalize_node(raw: Any, *, fallback_index: int) -> dict[str, Any]:
    node = dict(raw) if isinstance(raw, dict) else {}
    node_key = _as_text(node.get("node_key"))
    if not node_key:
        node_key = f"auto-node-{fallback_index}"
    node_type = _as_text(node.get("node_type"), "paragraph").lower()
    parent_node_key = _as_text(node.get("parent_node_key"))
    node_metadata = node.get("node_metadata")
    path = _as_text(node.get("path"))
    return {
        **copy.deepcopy(node),
        "node_key": node_key,
        "node_type": node_type or "paragraph",
        "parent_node_key": parent_node_key or None,
        "ordinal": _non_negative_int(node.get("ordinal"), 0),
        "node_metadata": dict(node_metadata) if isinstance(node_metadata, dict) else {},
        "path": path,
    }


def _find_cycle(by_key: dict[str, dict[str, Any]], *, root_key: str) -> list[str]:
    for start_key in by_key:
        if start_key == root_key:
            continue
        path: list[str] = []
        seen_at: dict[str, int] = {}
        current_key = start_key
        while current_key and current_key in by_key and current_key != root_key:
            if current_key in seen_at:
                cycle_start = seen_at[current_key]
                return path[cycle_start:]
            seen_at[current_key] = len(path)
            path.append(current_key)
            next_key = _as_text(by_key[current_key].get("parent_node_key"))
            if not next_key:
                break
            current_key = next_key
    return []


def _ordered_unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _as_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def repair_document_graph(
    nodes: list[dict[str, Any]] | None,
    *,
    root_node_key: str | None = None,
) -> dict[str, Any]:
    normalized_input = list(nodes) if isinstance(nodes, list) else []
    if not normalized_input:
        return {
            "nodes": [],
            "edges": [],
            "diagnostics": {
                "is_valid": True,
                "was_repaired": False,
                "root_node_key": "",
                "node_count": 0,
                "edge_count": 0,
                "duplicate_node_keys": [],
                "orphan_node_keys": [],
                "cycle_node_keys": [],
                "reordered_parent_keys": [],
                "repaired_orphan_count": 0,
                "repaired_cycle_count": 0,
                "reordered_parent_count": 0,
            },
        }

    by_key: dict[str, dict[str, Any]] = {}
    node_order: dict[str, int] = {}
    duplicate_node_keys: list[str] = []
    for index, raw_node in enumerate(normalized_input):
        node = _normalize_node(raw_node, fallback_index=index + 1)
        node_key = str(node["node_key"])
        if node_key in by_key:
            duplicate_node_keys.append(node_key)
            continue
        by_key[node_key] = node
        node_order[node_key] = index

    if not by_key:
        return {
            "nodes": [],
            "edges": [],
            "diagnostics": {
                "is_valid": False,
                "was_repaired": True,
                "root_node_key": "",
                "node_count": 0,
                "edge_count": 0,
                "duplicate_node_keys": _ordered_unique(duplicate_node_keys),
                "orphan_node_keys": [],
                "cycle_node_keys": [],
                "reordered_parent_keys": [],
                "repaired_orphan_count": 0,
                "repaired_cycle_count": 0,
                "reordered_parent_count": 0,
            },
        }

    resolved_root_key = _as_text(root_node_key)
    if not resolved_root_key or resolved_root_key not in by_key:
        resolved_root_key = ""
        for node_key, node in by_key.items():
            if str(node.get("node_type", "")).lower() == "document":
                resolved_root_key = node_key
                break
        if not resolved_root_key:
            resolved_root_key = next(iter(by_key.keys()))

    root_node = by_key[resolved_root_key]
    root_node["parent_node_key"] = None
    root_node["ordinal"] = 0

    orphan_node_keys: list[str] = []
    for node_key, node in by_key.items():
        if node_key == resolved_root_key:
            continue
        parent_node_key = _as_text(node.get("parent_node_key"))
        if not parent_node_key or parent_node_key not in by_key or parent_node_key == node_key:
            node["parent_node_key"] = resolved_root_key
            orphan_node_keys.append(node_key)

    cycle_node_keys: list[str] = []
    while True:
        cycle = _find_cycle(by_key, root_key=resolved_root_key)
        if not cycle:
            break
        cycle_node_keys.extend(cycle)
        breaker_key = cycle[-1]
        by_key[breaker_key]["parent_node_key"] = resolved_root_key

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for node_key, node in by_key.items():
        if node_key == resolved_root_key:
            continue
        parent_node_key = _as_text(node.get("parent_node_key"), resolved_root_key)
        if parent_node_key not in by_key:
            parent_node_key = resolved_root_key
            node["parent_node_key"] = resolved_root_key
        children_by_parent[parent_node_key].append(node_key)

    reordered_parent_keys: list[str] = []
    for parent_node_key, child_node_keys in children_by_parent.items():
        sorted_children = sorted(
            child_node_keys,
            key=lambda key: (
                _non_negative_int(by_key[key].get("ordinal"), 0),
                node_order.get(key, 10**9),
                key,
            ),
        )
        for new_ordinal, child_node_key in enumerate(sorted_children):
            if _non_negative_int(by_key[child_node_key].get("ordinal"), 0) != new_ordinal:
                reordered_parent_keys.append(parent_node_key)
            by_key[child_node_key]["ordinal"] = new_ordinal
        children_by_parent[parent_node_key] = sorted_children

    root_path = _as_text(root_node.get("path")) or f"/{_as_text(root_node.get('node_type'), 'document')}"
    root_node["path"] = root_path
    ordered_node_keys: list[str] = []
    seen_node_keys: set[str] = set()

    stack: list[str] = [resolved_root_key]
    while stack:
        current_key = stack.pop()
        if current_key in seen_node_keys:
            continue
        seen_node_keys.add(current_key)
        ordered_node_keys.append(current_key)
        current_path = _as_text(by_key[current_key].get("path"))
        for child_key in reversed(children_by_parent.get(current_key, [])):
            child = by_key[child_key]
            child_segment = f"{_as_text(child.get('node_type'), 'paragraph')}:{_non_negative_int(child.get('ordinal'), 0) + 1}"
            child["path"] = f"{current_path}/{child_segment}" if current_path else f"/{child_segment}"
            stack.append(child_key)

    for node_key in sorted(by_key.keys(), key=lambda key: (node_order.get(key, 10**9), key)):
        if node_key in seen_node_keys:
            continue
        node = by_key[node_key]
        node["parent_node_key"] = resolved_root_key
        node["ordinal"] = len(children_by_parent.get(resolved_root_key, []))
        node["path"] = f"{root_path}/{_as_text(node.get('node_type'), 'paragraph')}:{node['ordinal'] + 1}"
        children_by_parent[resolved_root_key].append(node_key)
        ordered_node_keys.append(node_key)

    parent_iteration_order = [key for key in ordered_node_keys if key in children_by_parent]
    edges: list[dict[str, Any]] = []
    for parent_node_key in parent_iteration_order:
        for child_node_key in children_by_parent.get(parent_node_key, []):
            child = by_key[child_node_key]
            edges.append(
                {
                    "source_node_key": parent_node_key,
                    "target_node_key": child_node_key,
                    "edge_type": DOCUMENT_GRAPH_PARENT_EDGE,
                    "ordinal": _non_negative_int(child.get("ordinal"), 0),
                    "edge_metadata": {},
                }
            )
        siblings = children_by_parent.get(parent_node_key, [])
        for sibling_index in range(1, len(siblings)):
            prev_key = siblings[sibling_index - 1]
            curr_key = siblings[sibling_index]
            edges.append(
                {
                    "source_node_key": prev_key,
                    "target_node_key": curr_key,
                    "edge_type": DOCUMENT_GRAPH_SIBLING_EDGE,
                    "ordinal": sibling_index,
                    "edge_metadata": {},
                }
            )

    ordered_nodes = [copy.deepcopy(by_key[key]) for key in ordered_node_keys]
    orphan_unique = _ordered_unique(orphan_node_keys)
    cycle_unique = _ordered_unique(cycle_node_keys)
    reordered_unique = _ordered_unique(reordered_parent_keys)
    duplicate_unique = _ordered_unique(duplicate_node_keys)
    was_repaired = bool(orphan_unique or cycle_unique or reordered_unique or duplicate_unique)
    diagnostics = {
        "is_valid": not was_repaired,
        "was_repaired": was_repaired,
        "root_node_key": resolved_root_key,
        "node_count": len(ordered_nodes),
        "edge_count": len(edges),
        "duplicate_node_keys": duplicate_unique,
        "orphan_node_keys": orphan_unique,
        "cycle_node_keys": cycle_unique,
        "reordered_parent_keys": reordered_unique,
        "repaired_orphan_count": len(orphan_unique),
        "repaired_cycle_count": len(cycle_unique),
        "reordered_parent_count": len(reordered_unique),
    }
    return {
        "nodes": ordered_nodes,
        "edges": edges,
        "diagnostics": diagnostics,
    }
