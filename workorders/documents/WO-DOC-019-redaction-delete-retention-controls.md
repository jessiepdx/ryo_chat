# WO-DOC-019: Redaction, Delete Semantics, and Retention Controls

Source item: User request (February 10, 2026) requiring safe storage/retrieval and control over persisted document information.

## 1. Verbose Description of Work Order
Implement data governance controls for document content, including redaction workflows, soft/hard delete semantics, retention windows, and audit trails.

This work order ensures sensitive document content can be managed without breaking lineage and citation integrity.

Scope includes:
1. Reversible redaction at source/node/chunk levels.
2. Soft delete with tombstones and hard-delete lifecycle windows.
3. Retention policy configuration per scope.
4. Audit logging for all governance actions.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_governance.py`
2. New migration: `db/migrations/111_document_redactions.sql`
3. New migration: `db/migrations/112_document_retention_policies.sql`
4. New migration: `db/migrations/113_document_governance_audit.sql`

Secondary/API-surface files:
1. `web_ui.py` (redact/delete/restore endpoints)
2. `hypermindlabs/agents.py` (redaction-aware retrieval filtering)
3. `tests/test_document_governance.py`
4. `tests/test_document_retention.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Redaction and delete operations are auditable and policy-controlled.
2. Retrieval excludes redacted/deleted artifacts by default.
3. Restore operations preserve lineage references.
4. Retention jobs execute idempotently.

Partial outcome:
1. Delete exists but no retention or audit trail.
2. Redacted content is hidden in UI but still retrievable via API.
3. Restore flow breaks citation references.

Validation method:
1. Governance workflow integration tests.
2. Policy enforcement tests by role and scope.
3. Retrieval regression tests after redaction/delete actions.

## 4. Potential Failure Modes
1. Hard-delete removes data still referenced by historical runs.
2. Redaction masks content but leaks through debug events.
3. Retention policy conflicts across user/chat/community scopes.
4. Audit logs are mutable or incomplete.
