from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from hypermindlabs.document_taxonomy import (
    detect_domain_signals,
    detect_format_signals,
    extract_topic_signals,
    flatten_signal_labels,
)


DOCUMENT_METADATA_SCHEMA_VERSION = 1


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


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


def _runtime_float(config_manager: Any | None, path: str, default: float) -> float:
    try:
        return float(_runtime_value(config_manager, path, default))
    except (TypeError, ValueError):
        return float(default)


def _runtime_bool(config_manager: Any | None, path: str, default: bool) -> bool:
    value = _runtime_value(config_manager, path, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def metadata_runtime_settings(config_manager: Any | None = None) -> dict[str, Any]:
    return {
        "enabled": _runtime_bool(config_manager, "documents.taxonomy_enabled", True),
        "topic_limit": max(1, _runtime_int(config_manager, "documents.taxonomy_topic_limit", 8)),
        "topic_min_confidence": max(0.0, min(1.0, _runtime_float(config_manager, "documents.taxonomy_topic_min_confidence", 0.2))),
        "domain_min_confidence": max(0.0, min(1.0, _runtime_float(config_manager, "documents.taxonomy_domain_min_confidence", 0.2))),
        "format_min_confidence": max(0.0, min(1.0, _runtime_float(config_manager, "documents.taxonomy_format_min_confidence", 0.2))),
        "synonym_expansion": _runtime_bool(config_manager, "documents.taxonomy_synonym_expansion", True),
        "max_labels_per_record": max(1, _runtime_int(config_manager, "documents.taxonomy_max_labels_per_record", 6)),
    }


def _section_text_map(canonical_output: dict[str, Any] | None) -> dict[str, str]:
    section_map: dict[str, str] = {}
    source = dict(canonical_output) if isinstance(canonical_output, dict) else {}
    for section in list(source.get("sections") or []):
        if not isinstance(section, dict):
            continue
        section_id = _as_text(section.get("section_id"))
        if not section_id:
            continue
        section_map[section_id] = _as_text(section.get("text"))
    return section_map


def _aggregate_signals(
    signal_sets: list[list[dict[str, Any]]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    confidence_by_label: dict[str, float] = defaultdict(float)
    count_by_label: dict[str, int] = defaultdict(int)
    for signals in signal_sets:
        for item in list(signals or []):
            label = _as_text((item or {}).get("label")).lower()
            if not label:
                continue
            confidence = max(0.0, min(1.0, float((item or {}).get("confidence", 0.0) or 0.0)))
            confidence_by_label[label] += confidence
            count_by_label[label] += 1
    aggregated: list[dict[str, Any]] = []
    for label, total_confidence in confidence_by_label.items():
        count = max(1, count_by_label.get(label, 1))
        aggregated.append({"label": label, "confidence": round(total_confidence / float(count), 4)})
    aggregated.sort(key=lambda item: (-float(item["confidence"]), str(item["label"])))
    return aggregated[: max(1, int(max_items))]


def _build_taxonomy_payload(
    *,
    text: str,
    node_type: str,
    topic_limit: int,
    topic_min_confidence: float,
    domain_min_confidence: float,
    format_min_confidence: float,
    synonym_expansion: bool,
    max_labels_per_record: int,
) -> dict[str, Any]:
    topic_signals = extract_topic_signals(
        text,
        max_topics=topic_limit,
        min_confidence=topic_min_confidence,
        synonym_expansion=synonym_expansion,
    )
    format_signals = detect_format_signals(
        text,
        node_type=node_type,
        max_labels=max_labels_per_record,
        min_confidence=format_min_confidence,
    )
    domain_signals = detect_domain_signals(
        text,
        topic_signals=topic_signals,
        max_domains=max(1, min(3, max_labels_per_record)),
        min_confidence=domain_min_confidence,
    )
    topic_tags = flatten_signal_labels(topic_signals, max_items=topic_limit)
    domain_labels = flatten_signal_labels(domain_signals, max_items=max_labels_per_record)
    format_labels = flatten_signal_labels(format_signals, max_items=max_labels_per_record)
    return {
        "schema_version": DOCUMENT_METADATA_SCHEMA_VERSION,
        "topics": topic_signals,
        "topic_tags": topic_tags,
        "domains": domain_signals,
        "domain_labels": domain_labels,
        "formats": format_signals,
        "format_labels": format_labels,
        "primary_domain": domain_labels[0] if domain_labels else "general",
    }


def enrich_document_metadata(
    *,
    canonical_output: dict[str, Any] | None,
    tree_payload: dict[str, Any] | None,
    chunk_payload: dict[str, Any] | None,
    config_manager: Any | None = None,
) -> dict[str, Any]:
    settings = metadata_runtime_settings(config_manager)
    tree = copy.deepcopy(tree_payload) if isinstance(tree_payload, dict) else {"nodes": [], "edges": []}
    chunks = copy.deepcopy(chunk_payload) if isinstance(chunk_payload, dict) else {"chunks": []}
    section_map = _section_text_map(canonical_output)

    if not settings["enabled"]:
        summary = {
            "schema_version": DOCUMENT_METADATA_SCHEMA_VERSION,
            "enabled": False,
            "topic_tags": [],
            "domain_labels": [],
            "format_labels": [],
            "primary_domain": "general",
            "record_count": {
                "nodes": len(list(tree.get("nodes") or [])),
                "chunks": len(list(chunks.get("chunks") or [])),
            },
        }
        return {
            "tree_payload": tree,
            "chunk_payload": chunks,
            "source_taxonomy": dict(summary),
            "version_taxonomy": dict(summary),
            "summary": summary,
        }

    node_signal_topics: list[list[dict[str, Any]]] = []
    node_signal_domains: list[list[dict[str, Any]]] = []
    node_signal_formats: list[list[dict[str, Any]]] = []
    node_taxonomy_by_key: dict[str, dict[str, Any]] = {}

    enriched_nodes: list[dict[str, Any]] = []
    for node in list(tree.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        node_copy = dict(node)
        node_key = _as_text(node_copy.get("node_key"))
        node_type = _as_text(node_copy.get("node_type"), "paragraph").lower()
        node_metadata = node_copy.get("node_metadata")
        metadata = dict(node_metadata) if isinstance(node_metadata, dict) else {}
        section_id = _as_text(metadata.get("section_id"))
        text = _as_text(section_map.get(section_id))
        if not text:
            text = _as_text(node_copy.get("node_title"))
        taxonomy = _build_taxonomy_payload(
            text=text,
            node_type=node_type,
            topic_limit=int(settings["topic_limit"]),
            topic_min_confidence=float(settings["topic_min_confidence"]),
            domain_min_confidence=float(settings["domain_min_confidence"]),
            format_min_confidence=float(settings["format_min_confidence"]),
            synonym_expansion=bool(settings["synonym_expansion"]),
            max_labels_per_record=int(settings["max_labels_per_record"]),
        )
        metadata["taxonomy"] = taxonomy
        metadata["topic_tags"] = list(taxonomy.get("topic_tags") or [])
        metadata["domain_labels"] = list(taxonomy.get("domain_labels") or [])
        metadata["format_labels"] = list(taxonomy.get("format_labels") or [])
        node_copy["node_metadata"] = metadata
        if node_key:
            node_taxonomy_by_key[node_key] = taxonomy
        node_signal_topics.append(list(taxonomy.get("topics") or []))
        node_signal_domains.append(list(taxonomy.get("domains") or []))
        node_signal_formats.append(list(taxonomy.get("formats") or []))
        enriched_nodes.append(node_copy)
    tree["nodes"] = enriched_nodes

    chunk_signal_topics: list[list[dict[str, Any]]] = []
    chunk_signal_domains: list[list[dict[str, Any]]] = []
    chunk_signal_formats: list[list[dict[str, Any]]] = []
    enriched_chunks: list[dict[str, Any]] = []
    for chunk in list(chunks.get("chunks") or []):
        if not isinstance(chunk, dict):
            continue
        chunk_copy = dict(chunk)
        chunk_metadata = chunk_copy.get("chunk_metadata")
        metadata = dict(chunk_metadata) if isinstance(chunk_metadata, dict) else {}
        chunk_text = _as_text(chunk_copy.get("chunk_text"))
        node_key = _as_text(metadata.get("node_key") or chunk_copy.get("document_node_key"))
        node_taxonomy = node_taxonomy_by_key.get(node_key, {})
        node_type = _as_text(metadata.get("node_type") or node_taxonomy.get("node_type"), "paragraph").lower()
        taxonomy = _build_taxonomy_payload(
            text=chunk_text,
            node_type=node_type,
            topic_limit=int(settings["topic_limit"]),
            topic_min_confidence=float(settings["topic_min_confidence"]),
            domain_min_confidence=float(settings["domain_min_confidence"]),
            format_min_confidence=float(settings["format_min_confidence"]),
            synonym_expansion=bool(settings["synonym_expansion"]),
            max_labels_per_record=int(settings["max_labels_per_record"]),
        )
        if not taxonomy.get("topic_tags"):
            taxonomy["topic_tags"] = list(node_taxonomy.get("topic_tags") or [])
            taxonomy["topics"] = list(node_taxonomy.get("topics") or [])
        if not taxonomy.get("domain_labels"):
            taxonomy["domain_labels"] = list(node_taxonomy.get("domain_labels") or [])
            taxonomy["domains"] = list(node_taxonomy.get("domains") or [])
            taxonomy["primary_domain"] = _as_text(node_taxonomy.get("primary_domain"), "general")
        if not taxonomy.get("format_labels"):
            taxonomy["format_labels"] = list(node_taxonomy.get("format_labels") or [])
            taxonomy["formats"] = list(node_taxonomy.get("formats") or [])

        metadata["taxonomy"] = taxonomy
        metadata["topic_tags"] = list(taxonomy.get("topic_tags") or [])
        metadata["domain_labels"] = list(taxonomy.get("domain_labels") or [])
        metadata["format_labels"] = list(taxonomy.get("format_labels") or [])
        chunk_copy["chunk_metadata"] = metadata

        chunk_signal_topics.append(list(taxonomy.get("topics") or []))
        chunk_signal_domains.append(list(taxonomy.get("domains") or []))
        chunk_signal_formats.append(list(taxonomy.get("formats") or []))
        enriched_chunks.append(chunk_copy)
    chunks["chunks"] = enriched_chunks

    summary_topics = _aggregate_signals(
        chunk_signal_topics if any(chunk_signal_topics) else node_signal_topics,
        max_items=int(settings["topic_limit"]),
    )
    summary_domains = _aggregate_signals(
        chunk_signal_domains if any(chunk_signal_domains) else node_signal_domains,
        max_items=max(1, min(3, int(settings["max_labels_per_record"]))),
    )
    summary_formats = _aggregate_signals(
        chunk_signal_formats if any(chunk_signal_formats) else node_signal_formats,
        max_items=int(settings["max_labels_per_record"]),
    )

    summary = {
        "schema_version": DOCUMENT_METADATA_SCHEMA_VERSION,
        "enabled": True,
        "topics": summary_topics,
        "topic_tags": flatten_signal_labels(summary_topics, max_items=int(settings["topic_limit"])),
        "domains": summary_domains,
        "domain_labels": flatten_signal_labels(summary_domains, max_items=int(settings["max_labels_per_record"])),
        "formats": summary_formats,
        "format_labels": flatten_signal_labels(summary_formats, max_items=int(settings["max_labels_per_record"])),
        "primary_domain": (
            flatten_signal_labels(summary_domains, max_items=1)[0]
            if flatten_signal_labels(summary_domains, max_items=1)
            else "general"
        ),
        "record_count": {
            "nodes": len(enriched_nodes),
            "chunks": len(enriched_chunks),
        },
    }

    return {
        "tree_payload": tree,
        "chunk_payload": chunks,
        "source_taxonomy": dict(summary),
        "version_taxonomy": dict(summary),
        "summary": summary,
    }
