from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


DOCUMENT_SCOPE_KEYS: tuple[str, ...] = (
    "owner_member_id",
    "chat_host_id",
    "chat_type",
    "community_id",
    "topic_id",
    "platform",
)


class DocumentScopeValidationError(ValueError):
    """Raised when a document scope payload is missing required scope context."""


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _as_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _required_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentScopeValidationError(f"{field_name} must be an integer.") from None
    if parsed <= 0:
        raise DocumentScopeValidationError(f"{field_name} must be greater than zero.")
    return parsed


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DocumentScopeValidationError(f"{field_name} must be an integer or null.") from None
    if parsed < 0:
        raise DocumentScopeValidationError(f"{field_name} cannot be negative.")
    return parsed


def _scope_source(payload: dict[str, Any]) -> dict[str, Any]:
    source = {}
    nested = payload.get("scope")
    if isinstance(nested, dict):
        source.update(_coerce_dict(nested))
    for key in DOCUMENT_SCOPE_KEYS:
        if key in payload:
            source[key] = payload.get(key)
    return source


@dataclass(frozen=True)
class DocumentScopeContext:
    owner_member_id: int
    chat_host_id: int
    chat_type: str
    community_id: int | None
    topic_id: int | None
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_member_id": int(self.owner_member_id),
            "chat_host_id": int(self.chat_host_id),
            "chat_type": str(self.chat_type),
            "community_id": self.community_id,
            "topic_id": self.topic_id,
            "platform": str(self.platform),
        }


def resolve_document_scope(
    payload: dict[str, Any] | None,
    *,
    authenticated_member_id: int | None = None,
    fallback_scope: dict[str, Any] | None = None,
) -> DocumentScopeContext:
    source = _scope_source(_coerce_dict(payload))
    if isinstance(fallback_scope, dict):
        fallback = _scope_source(_coerce_dict(fallback_scope))
        for key, value in fallback.items():
            source.setdefault(key, value)

    missing = [key for key in DOCUMENT_SCOPE_KEYS if key not in source]
    if missing:
        raise DocumentScopeValidationError(
            f"Missing required scope keys: {', '.join(missing)}."
        )

    owner_member_id = _required_int(source.get("owner_member_id"), "owner_member_id")
    if authenticated_member_id is not None and owner_member_id != int(authenticated_member_id):
        raise DocumentScopeValidationError(
            "owner_member_id does not match the authenticated member."
        )

    chat_host_id = _required_int(source.get("chat_host_id"), "chat_host_id")
    chat_type = _as_text(source.get("chat_type")).lower()
    if not chat_type:
        raise DocumentScopeValidationError("chat_type is required.")
    platform = _as_text(source.get("platform")).lower()
    if not platform:
        raise DocumentScopeValidationError("platform is required.")

    return DocumentScopeContext(
        owner_member_id=owner_member_id,
        chat_host_id=chat_host_id,
        chat_type=chat_type,
        community_id=_optional_int(source.get("community_id"), "community_id"),
        topic_id=_optional_int(source.get("topic_id"), "topic_id"),
        platform=platform,
    )


def scope_matches_record(
    record: dict[str, Any] | None,
    scope: DocumentScopeContext | dict[str, Any],
) -> bool:
    if not isinstance(record, dict):
        return False
    resolved_scope = (
        scope
        if isinstance(scope, DocumentScopeContext)
        else resolve_document_scope(scope)
    )
    required_pairs = {
        "owner_member_id": resolved_scope.owner_member_id,
        "chat_host_id": resolved_scope.chat_host_id,
        "chat_type": resolved_scope.chat_type,
        "platform": resolved_scope.platform,
    }
    for key, expected in required_pairs.items():
        if str(record.get(key)).strip().lower() != str(expected).strip().lower():
            return False

    if resolved_scope.community_id is not None:
        if str(record.get("community_id")).strip() != str(resolved_scope.community_id):
            return False
    if resolved_scope.topic_id is not None:
        if str(record.get("topic_id")).strip() != str(resolved_scope.topic_id):
            return False
    return True


def build_scope_where_clause(
    scope: DocumentScopeContext | dict[str, Any],
    *,
    table_alias: str = "",
) -> tuple[str, list[Any]]:
    resolved_scope = (
        scope
        if isinstance(scope, DocumentScopeContext)
        else resolve_document_scope(scope)
    )
    prefix = f"{table_alias}." if table_alias else ""
    clauses: list[str] = [
        f"{prefix}owner_member_id = %s",
        f"{prefix}chat_host_id = %s",
        f"{prefix}chat_type = %s",
        f"{prefix}platform = %s",
    ]
    params: list[Any] = [
        resolved_scope.owner_member_id,
        resolved_scope.chat_host_id,
        resolved_scope.chat_type,
        resolved_scope.platform,
    ]
    if resolved_scope.community_id is not None:
        clauses.append(f"{prefix}community_id = %s")
        params.append(resolved_scope.community_id)
    if resolved_scope.topic_id is not None:
        clauses.append(f"{prefix}topic_id = %s")
        params.append(resolved_scope.topic_id)
    return " AND ".join(clauses), params


def scope_session_settings(scope: DocumentScopeContext | dict[str, Any]) -> dict[str, str]:
    resolved_scope = (
        scope
        if isinstance(scope, DocumentScopeContext)
        else resolve_document_scope(scope)
    )
    return {
        "app.scope_bypass": "0",
        "app.owner_member_id": str(resolved_scope.owner_member_id),
        "app.chat_host_id": str(resolved_scope.chat_host_id),
        "app.chat_type": str(resolved_scope.chat_type),
        "app.community_id": "" if resolved_scope.community_id is None else str(resolved_scope.community_id),
        "app.topic_id": "" if resolved_scope.topic_id is None else str(resolved_scope.topic_id),
        "app.platform": str(resolved_scope.platform),
    }


def apply_pg_scope_settings(
    cursor: Any,
    scope: DocumentScopeContext | dict[str, Any],
) -> None:
    for key, value in scope_session_settings(scope).items():
        cursor.execute("SELECT set_config(%s, %s, true);", (key, value))
