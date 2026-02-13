# WO-DOC-003: File Ingress and Object Lifecycle Management

Source item: User request (February 10, 2026) requiring reliable storage of long and arbitrarily formatted source files.

## 1. Verbose Description of Work Order
Implement the source-file ingress layer for document upload, file identity, integrity checks, and lifecycle transitions from raw file to parsed artifact.

This work order handles raw file persistence and identity before parsing and chunking.

Scope includes:
1. Upload path supporting large files with streaming write.
2. SHA256 digest, MIME detection, size limits, and duplicate detection.
3. Source state machine (`received`, `queued`, `parsed`, `failed`, `archived`, `deleted`).
4. Retention metadata and reversible soft-delete states.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_storage.py`
2. New: `hypermindlabs/document_ingress.py`
3. New migration: `db/migrations/096_document_storage_objects.sql`
4. `web_ui.py` (upload endpoints and job creation)

Secondary/API-surface files:
1. `hypermindlabs/run_artifacts.py` or equivalent artifact persistence path
2. `config.empty.json` (document storage backend settings)
3. `hypermindlabs/runtime_settings.py` (storage limits and retention defaults)
4. `tests/test_document_ingress.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Large files ingest without loading full payload into memory.
2. Duplicate uploads can be detected by digest and versioned.
3. Lifecycle state transitions are persisted and auditable.
4. Upload API returns deterministic IDs and state info.

Partial outcome:
1. Upload works but digest and dedupe are missing.
2. Files are persisted but lifecycle status is opaque.
3. Large files succeed only for local low-latency environments.

Validation method:
1. Upload tests for small and large files.
2. Duplicate-file tests for digest collisions and same-source versioning.
3. Failure injection tests for partial writes and timeout behavior.

## 4. Potential Failure Modes
1. Non-streaming upload path causes memory pressure.
2. Digest computed after write without atomicity guarantees.
3. Delete operations remove files before audit references are preserved.
4. MIME spoofing bypasses parser safety checks.
