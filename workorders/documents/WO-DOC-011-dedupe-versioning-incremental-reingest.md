# WO-DOC-011: Deduplication, Versioning, and Incremental Re-Ingestion

Source item: User request (February 10, 2026) requiring reliable long-document storage and modern best-practice lifecycle controls.

## 1. Verbose Description of Work Order
Implement source-level and chunk-level deduplication with explicit version lineage and incremental re-ingestion to avoid unnecessary recompute.

This work order defines how document updates are tracked and what parts of the index are rebuilt.

Scope includes:
1. Source/version lineage model (`supersedes_version_id`, semantic version labels, timestamps).
2. Digest-based duplicate detection for file, node, and chunk levels.
3. Incremental re-ingest that only rebuilds changed subtrees/chunks.
4. Version pinning and active-version selection per scope.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_versioning.py`
2. New: `hypermindlabs/document_dedupe.py`
3. New migration: `db/migrations/102_document_version_lineage.sql`
4. New migration: `db/migrations/103_document_chunk_digests.sql`

Secondary/API-surface files:
1. `hypermindlabs/document_ingestion_worker.py` (delta planner)
2. `web_ui.py` (version management endpoints)
3. `tests/test_document_versioning.py`
4. `tests/test_incremental_reingest.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Re-uploaded identical files are linked without full reprocessing.
2. Changed versions trigger targeted subtree/chunk rebuilds.
3. Retrieval can pin to active or historical versions explicitly.
4. Version lineage remains queryable and auditable.

Partial outcome:
1. Duplicate detection exists only at full-file level.
2. Versioning exists but retrieval always uses latest implicitly.
3. Incremental path exists but still recomputes full embeddings.

Validation method:
1. Version diff tests with synthetic edits.
2. Re-ingest performance comparison full vs incremental.
3. Retrieval tests across pinned historical versions.

## 4. Potential Failure Modes
1. Hash collisions or weak digest strategy produce false dedupe.
2. Version pointers drift and orphan historical artifacts.
3. Partial re-index invalidates citation references.
4. Active-version selection causes non-deterministic retrieval output.
