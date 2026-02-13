# WO-DOC-012: Embedding Pipeline, Multi-Vector Strategy, and Backfills

Source item: User request (February 10, 2026) requiring effective retrieval over arbitrarily long and complex documents.

## 1. Verbose Description of Work Order
Build a robust embedding pipeline for document chunks, including asynchronous execution, model version tracking, and optional multi-vector representations.

This work order upgrades beyond single-vector naive retrieval and supports model upgrades without full downtime.

Scope includes:
1. Embedding jobs with retry and backoff.
2. Embedding model/version metadata per vector row.
3. Optional multi-vector records (content vector + heading/path vector).
4. Backfill plan for embedding model rotation.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_embeddings.py`
2. New migration: `db/migrations/104_document_chunk_embeddings.sql`
3. New migration: `db/migrations/105_document_embedding_jobs.sql`
4. `hypermindlabs/utils.py` (`getEmbeddings` reuse or extraction into dedicated module)

Secondary/API-surface files:
1. `hypermindlabs/runtime_settings.py` (embedding budgets, concurrency, model ids)
2. `web_ui.py` (embedding job status and reindex controls)
3. `tests/test_document_embeddings.py`
4. `tests/test_embedding_backfill.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Embeddings are generated asynchronously without blocking user flow.
2. Embedding rows include model/version metadata.
3. Re-index flow supports safe model upgrade/backfill.
4. Embedding failures are visible and recoverable.

Partial outcome:
1. Embedding generation works but only synchronously.
2. Model metadata is not persisted, preventing safe backfill.
3. Multi-vector support exists in schema but not in retrieval.

Validation method:
1. Worker tests for retry/cancel/backoff.
2. Model migration tests with mixed-version vectors.
3. Query-level checks ensuring model-consistent comparisons.

## 4. Potential Failure Modes
1. Embedding queue starves parser queue under high load.
2. Model mismatch across vectors degrades distance quality.
3. Backfill tasks overwrite active vectors prematurely.
4. Large chunk payloads silently truncate before embedding.
