# WO-DOC-001: Document Data Contracts and Scope Keys

Source item: User request (February 10, 2026) for arbitrarily long and arbitrarily formatted document parsing, storage, and hierarchical retrieval.

## 1. Verbose Description of Work Order
Define the canonical data contracts for document ingestion, parsing outputs, tree nodes, chunks, retrieval events, and citations.

This work order establishes the non-negotiable scope keys that must exist on all persisted document records so retrieval can be isolated per user/chat/community/topic.

Scope includes:
1. Canonical JSON contracts for source file, parse artifact, logical node, chunk, embedding, retrieval event, and citation span.
2. Mandatory scope fields: `owner_member_id`, `chat_host_id`, `chat_type`, `community_id`, `topic_id`, `platform`.
3. Contract versioning (`schema_version`) and backward-compatible change policy.
4. Validation layer for ingest and retrieval requests.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_models.py`
2. New: `hypermindlabs/document_contracts.py`
3. New migrations: `db/migrations/090_document_sources.sql`
4. New migrations: `db/migrations/091_document_versions.sql`
5. New migrations: `db/migrations/092_document_nodes.sql`
6. New migrations: `db/migrations/093_document_chunks.sql`
7. New migrations: `db/migrations/094_document_retrieval_events.sql`

Secondary/API-surface files:
1. `hypermindlabs/utils.py` (`KnowledgeManager` extension path)
2. `web_ui.py` (request validation for ingest/retrieval endpoints)
3. `tests/test_document_contracts.py`
4. `docs/master-engineering.md` (status linkage)

## 3. Success and Validation Metrics
Valid outcome:
1. All new document tables enforce scope fields and `schema_version`.
2. Contract validators reject missing or inconsistent scope values.
3. Retrieval and ingestion payloads serialize to canonical contract shapes.
4. Contract tests cover required/optional field behavior.

Partial outcome:
1. Tables exist but scope fields are optional.
2. Contract schema exists but is not enforced in API.
3. Version field exists without compatibility policy.

Validation method:
1. Unit tests for contract parsing and validation.
2. DB migration tests verifying non-null and foreign-key constraints.
3. API tests ensuring invalid payload rejection.

## 4. Potential Failure Modes
1. Scope-key drift across modules causes cross-tenant data leakage.
2. Contract versioning is added but never used by ingestion workers.
3. Weak validation allows malformed parse artifacts into storage.
4. Overly rigid schema blocks legitimate parser variants.
