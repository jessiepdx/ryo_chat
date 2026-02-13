# WO-DOC-002: Storage Tenancy, Access Boundaries, and RLS

Source item: User request (February 10, 2026) requiring per-user and per-chat file storage and retrieval isolation.

## 1. Verbose Description of Work Order
Implement strong tenancy boundaries for all document and retrieval data using row-level filters, policy checks, and scoped query interfaces.

This work order ensures that retrieval never crosses user/chat/community boundaries unless explicitly authorized by policy.

Scope includes:
1. Row-level security strategy for document tables.
2. Query-layer scope enforcement in all manager methods.
3. Consistent scope-derivation from runtime context.
4. Isolation audit checks and negative tests.

## 2. Expression of Affected Files
Primary files:
1. New migrations: `db/migrations/095_document_rls_policies.sql`
2. New: `hypermindlabs/document_scope.py`
3. New: `hypermindlabs/document_access_policy.py`
4. `hypermindlabs/utils.py` (scope-aware query methods)

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (scope propagation into retrieval calls)
2. `web_ui.py` (scope checks for document endpoints)
3. `tests/test_document_scope_enforcement.py`
4. `tests/test_document_rls.py`

## 3. Success and Validation Metrics
Valid outcome:
1. All read/write paths require scope context.
2. Cross-scope retrieval attempts are blocked and logged.
3. RLS policies match application-level scope rules.
4. Automated tests verify both allow and deny paths.

Partial outcome:
1. Scope checks implemented in app code but not in SQL policy.
2. Some retrieval endpoints still run unscoped queries.
3. Access denials are not auditable.

Validation method:
1. Integration tests for member chat, group chat, and topic-scoped isolation.
2. SQL-level policy tests using role/session simulation.
3. Retrieval replay checks confirming no cross-tenant chunk returns.

## 4. Potential Failure Modes
1. Policy and application filters disagree and create bypass windows.
2. Topic scope is inconsistently treated as nullable vs strict.
3. Missing indexes cause RLS queries to become slow at scale.
4. Operational users receive broad access without explicit review controls.
