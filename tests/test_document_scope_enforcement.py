import unittest

from hypermindlabs.document_access_policy import DocumentAccessPolicy, DocumentScopeAccessError
from hypermindlabs.document_scope import (
    DocumentScopeValidationError,
    build_scope_where_clause,
    resolve_document_scope,
    scope_matches_record,
)


def _base_scope() -> dict:
    return {
        "owner_member_id": 12,
        "chat_host_id": 901,
        "chat_type": "group",
        "community_id": 77,
        "topic_id": 5,
        "platform": "telegram",
    }


class DocumentScopeEnforcementTests(unittest.TestCase):
    def test_resolve_scope_requires_required_keys(self):
        with self.assertRaises(DocumentScopeValidationError):
            resolve_document_scope({"scope": {"owner_member_id": 1}})

    def test_resolve_scope_enforces_authenticated_owner(self):
        payload = {"scope": _base_scope()}
        resolved = resolve_document_scope(payload, authenticated_member_id=12)
        self.assertEqual(resolved.owner_member_id, 12)
        with self.assertRaises(DocumentScopeValidationError):
            resolve_document_scope(payload, authenticated_member_id=13)

    def test_access_policy_blocks_cross_scope_without_privilege(self):
        policy = DocumentAccessPolicy()
        scope = resolve_document_scope({"scope": _base_scope()})
        self.assertFalse(
            policy.can_access_scope(
                actor_member_id=99,
                scope=scope,
                actor_roles=["user"],
            )
        )
        with self.assertRaises(DocumentScopeAccessError):
            policy.assert_scope_access(
                actor_member_id=99,
                scope=scope,
                actor_roles=["user"],
            )

    def test_access_policy_allows_cross_scope_for_admin(self):
        policy = DocumentAccessPolicy()
        scope = resolve_document_scope({"scope": _base_scope()})
        self.assertTrue(
            policy.can_access_scope(
                actor_member_id=99,
                scope=scope,
                actor_roles=["admin"],
            )
        )

    def test_scope_sql_and_record_match(self):
        scope = resolve_document_scope({"scope": _base_scope()})
        where_sql, params = build_scope_where_clause(scope)
        self.assertIn("owner_member_id = %s", where_sql)
        self.assertIn("chat_host_id = %s", where_sql)
        self.assertIn("platform = %s", where_sql)
        self.assertGreaterEqual(len(params), 4)

        matching = {
            "owner_member_id": 12,
            "chat_host_id": 901,
            "chat_type": "group",
            "community_id": 77,
            "topic_id": 5,
            "platform": "telegram",
        }
        off_scope = dict(matching)
        off_scope["chat_host_id"] = 500
        self.assertTrue(scope_matches_record(matching, scope))
        self.assertFalse(scope_matches_record(off_scope, scope))


if __name__ == "__main__":
    unittest.main()
