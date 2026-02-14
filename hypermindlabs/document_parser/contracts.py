from __future__ import annotations

import copy
from typing import Any


CANONICAL_PARSE_SCHEMA = "document.parse.canonical.v1"
CANONICAL_PARSE_STATUSES: tuple[str, ...] = ("parsed", "partial", "failed")


class DocumentParserContractError(ValueError):
    """Raised when a parser adapter output violates the canonical parse contract."""


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return []


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
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


def _positive_int(value: Any, default: int = 1) -> int:
    parsed = _non_negative_int(value, default=max(1, default))
    if parsed <= 0:
        return max(1, int(default))
    return parsed


def _clamp_float(value: Any, *, minimum: float = 0.0, maximum: float = 1.0, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if parsed < minimum:
        parsed = minimum
    if parsed > maximum:
        parsed = maximum
    return parsed


def _normalize_sections(payload: dict[str, Any], *, content_text: str) -> list[dict[str, Any]]:
    raw_sections = _coerce_list(payload.get("sections"))
    if not raw_sections:
        return [
            {
                "section_id": "s1",
                "title": "",
                "level": 1,
                "text": str(content_text),
                "start_char": 0,
                "end_char": len(content_text),
                "page_start": None,
                "page_end": None,
                "metadata": {},
            }
        ]

    normalized: list[dict[str, Any]] = []
    running_start = 0
    for index, item in enumerate(raw_sections, start=1):
        section = _coerce_dict(item)
        text_value = _as_text(section.get("text"), "")
        start_char = _optional_non_negative_int(section.get("start_char"))
        if start_char is None:
            start_char = running_start
        end_char = _optional_non_negative_int(section.get("end_char"))
        if end_char is None or end_char < start_char:
            end_char = start_char + len(text_value)
        normalized_section = {
            "section_id": _as_text(section.get("section_id"), f"s{index}"),
            "title": _as_text(section.get("title"), ""),
            "level": _positive_int(section.get("level"), 1),
            "text": text_value,
            "start_char": start_char,
            "end_char": end_char,
            "page_start": _optional_non_negative_int(section.get("page_start")),
            "page_end": _optional_non_negative_int(section.get("page_end")),
            "metadata": _coerce_dict(section.get("metadata")),
        }
        normalized.append(normalized_section)
        running_start = int(normalized_section.get("end_char", running_start))
    return normalized


def normalize_canonical_parse_output(
    payload: dict[str, Any] | None,
    *,
    parser_name: str,
    parser_version: str,
    adapter_chain: list[str] | tuple[str, ...] | None = None,
    confidence: float = 0.0,
    cost: float = 0.0,
    duration_ms: int = 0,
    route_debug: dict[str, Any] | None = None,
    profile_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = _coerce_dict(payload)
    status = _as_text(source.get("status"), "parsed").lower()
    if status not in CANONICAL_PARSE_STATUSES:
        raise DocumentParserContractError(
            f"Invalid canonical parse status '{status}'. Allowed: {', '.join(CANONICAL_PARSE_STATUSES)}."
        )
    content_text = _as_text(source.get("content_text"), _as_text(source.get("text"), ""))
    warnings = [str(item).strip() for item in _coerce_list(source.get("warnings")) if str(item).strip()]
    errors = [str(item).strip() for item in _coerce_list(source.get("errors")) if str(item).strip()]
    if status == "failed" and not errors:
        errors.append("parser_failed_without_error_details")
    sections = _normalize_sections(source, content_text=content_text)

    canonical: dict[str, Any] = {
        "schema_version": _positive_int(source.get("schema_version"), 1),
        "canonical_schema": CANONICAL_PARSE_SCHEMA,
        "status": status,
        "content_text": content_text,
        "sections": sections,
        "metadata": _coerce_dict(source.get("metadata")),
        "warnings": warnings,
        "errors": errors,
        "provenance": {
            "selected_adapter": str(parser_name),
            "selected_adapter_version": str(parser_version),
            "adapter_chain": [str(item).strip() for item in (adapter_chain or []) if str(item).strip()],
            "confidence": _clamp_float(confidence, minimum=0.0, maximum=1.0, default=0.0),
            "cost": max(0.0, float(cost)),
            "duration_ms": _non_negative_int(duration_ms, 0),
            "route_debug": _coerce_dict(route_debug),
            "profile_summary": _coerce_dict(profile_summary),
        },
    }
    return canonical


def validate_canonical_parse_output(payload: dict[str, Any]) -> dict[str, Any]:
    source = _coerce_dict(payload)
    provenance = _coerce_dict(source.get("provenance"))
    return normalize_canonical_parse_output(
        source,
        parser_name=_as_text(provenance.get("selected_adapter"), "unknown-parser"),
        parser_version=_as_text(provenance.get("selected_adapter_version"), ""),
        adapter_chain=provenance.get("adapter_chain"),
        confidence=_clamp_float(provenance.get("confidence"), minimum=0.0, maximum=1.0, default=0.0),
        cost=max(0.0, float(provenance.get("cost", 0.0) or 0.0)),
        duration_ms=_non_negative_int(provenance.get("duration_ms"), 0),
        route_debug=_coerce_dict(provenance.get("route_debug")),
        profile_summary=_coerce_dict(provenance.get("profile_summary")),
    )


def canonical_status_to_document_state(status: str) -> str:
    normalized = _as_text(status, "failed").lower()
    if normalized in {"parsed", "partial"}:
        return "parsed"
    return "failed"


def build_document_parse_artifact_patch(canonical_output: dict[str, Any]) -> dict[str, Any]:
    canonical = validate_canonical_parse_output(canonical_output)
    provenance = _coerce_dict(canonical.get("provenance"))
    return {
        "schema_version": int(canonical.get("schema_version", 1)),
        "parser_name": _as_text(provenance.get("selected_adapter"), "unknown-parser"),
        "parser_version": _as_text(provenance.get("selected_adapter_version"), ""),
        "parse_mode": "adapter_router",
        "status": canonical_status_to_document_state(str(canonical.get("status", "failed"))),
        "artifact": {
            "canonical": canonical,
        },
        "warnings": [str(item) for item in _coerce_list(canonical.get("warnings"))],
        "errors": [str(item) for item in _coerce_list(canonical.get("errors"))],
    }
