# WO-DOC-004: Ingestion Queue, Workers, and Idempotency

Source item: User request (February 10, 2026) requiring robust parsing/storage for arbitrarily long documents.

## 1. Verbose Description of Work Order
Build an asynchronous ingestion execution model so document parsing and embedding can run reliably with retries, cancellation, and idempotency.

This work order provides the orchestration plane for all downstream parser and indexing stages.

Scope includes:
1. Persistent ingestion jobs table and state machine.
2. Worker loop with lease/heartbeat and retry backoff.
3. Idempotent execution by `(source_id, version_id, pipeline_version)`.
4. Dead-letter capture and operator requeue controls.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_ingestion_jobs.py`
2. New: `hypermindlabs/document_ingestion_worker.py`
3. New migration: `db/migrations/097_document_ingestion_jobs.sql`
4. New migration: `db/migrations/098_document_ingestion_attempts.sql`

Secondary/API-surface files:
1. `app.py` or startup worker bootstrap path
2. `web_ui.py` (job status, retry, cancel endpoints)
3. `hypermindlabs/run_events.py` (ingestion stage event emission)
4. `tests/test_document_ingestion_worker.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Ingestion jobs survive process restarts.
2. Same document version is not reprocessed unnecessarily.
3. Retry logic handles transient parser/embedding errors.
4. Dead-lettered jobs preserve full error context.

Partial outcome:
1. Queue works only in-memory.
2. Idempotency exists but does not cover embedding stage.
3. Retries happen without bounded backoff.

Validation method:
1. Integration tests with forced worker crash/restart.
2. Duplicate enqueue tests.
3. Retry and dead-letter behavior tests.

## 4. Potential Failure Modes
1. Job lease recovery races trigger duplicate processing.
2. Retry policy hammers unavailable parser services.
3. Cancellation semantics leave half-written chunk trees.
4. Ingestion jobs are scoped incorrectly and cross tenant boundaries.
