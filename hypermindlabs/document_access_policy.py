from __future__ import annotations

from typing import Any

from hypermindlabs.document_scope import (
    DocumentScopeContext,
    resolve_document_scope,
    scope_matches_record,
)


class DocumentScopeAccessError(PermissionError):
    """Raised when an actor tries to access a scope they do not own."""


def _normalize_roles(roles: list[str] | tuple[str, ...] | None) -> set[str]:
    if not isinstance(roles, (list, tuple)):
        return set()
    return {str(role).strip().lower() for role in roles if str(role).strip()}


class DocumentAccessPolicy:
    def __init__(self, privileged_roles: tuple[str, ...] = ("admin", "owner")):
        self._privileged_roles = _normalize_roles(privileged_roles)

    def can_access_scope(
        self,
        *,
        actor_member_id: int | None,
        scope: DocumentScopeContext | dict[str, Any],
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        if actor_member_id is None:
            return False
        resolved_scope = (
            scope
            if isinstance(scope, DocumentScopeContext)
            else resolve_document_scope(scope)
        )
        if int(actor_member_id) == int(resolved_scope.owner_member_id):
            return True
        roles = _normalize_roles(actor_roles)
        return bool(self._privileged_roles.intersection(roles))

    def assert_scope_access(
        self,
        *,
        actor_member_id: int | None,
        scope: DocumentScopeContext | dict[str, Any],
        actor_roles: list[str] | tuple[str, ...] | None = None,
    ) -> DocumentScopeContext:
        resolved_scope = (
            scope
            if isinstance(scope, DocumentScopeContext)
            else resolve_document_scope(scope)
        )
        if not self.can_access_scope(
            actor_member_id=actor_member_id,
            scope=resolved_scope,
            actor_roles=actor_roles,
        ):
            raise DocumentScopeAccessError(
                "Cross-scope access is not allowed for this actor."
            )
        return resolved_scope

    def filter_records(
        self,
        records: list[dict[str, Any]] | None,
        *,
        scope: DocumentScopeContext | dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            return []
        resolved_scope = (
            scope
            if isinstance(scope, DocumentScopeContext)
            else resolve_document_scope(scope)
        )
        output: list[dict[str, Any]] = []
        for item in records:
            if scope_matches_record(item, resolved_scope):
                output.append(item)
        return output
