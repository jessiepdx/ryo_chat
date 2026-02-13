# WO-DOC-018: Retrieval Debugger, Trace Events, and Explainability Surfaces

Source item: User request (February 10, 2026) requiring highly effective recall with transparent retrieval behavior.

## 1. Verbose Description of Work Order
Expose full retrieval observability for operators and developers, including query rewrites, candidate scores, reranker decisions, and final context packing decisions.

This work order turns retrieval into an inspectable system instead of a black box.

Scope includes:
1. Structured retrieval event schema (`query`, `filters`, `candidates`, `scores`, `selected`).
2. Debug endpoints for ingestion and retrieval state.
3. Redacted operator views by role.
4. Deterministic event IDs linked to runs.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/retrieval_debug.py`
2. New migration: `db/migrations/110_document_retrieval_debug_events.sql`
3. `hypermindlabs/run_events.py` (retrieval event emission)
4. `web_ui.py` (debug APIs)

Secondary/API-surface files:
1. `static/agent-playground/rag/retrieval-debugger.js`
2. `templates/knowledge-tools.html`
3. `tests/test_retrieval_debug.py`
4. `tests/test_retrieval_event_schema.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Each retrieval round emits complete debug metadata.
2. Operators can inspect candidate and selection rationale.
3. Debug payloads are role-filtered and auditable.
4. Debug views align with actual retrieval behavior.

Partial outcome:
1. Candidate scores are available but reranker/packer steps are missing.
2. Debug metadata exists only in logs, not via API.
3. Events are emitted but not linked to run IDs.

Validation method:
1. End-to-end retrieval trace assertions in tests.
2. Access-control tests for debug payload redaction.
3. Event schema contract tests.

## 4. Potential Failure Modes
1. Debug mode exposes sensitive chunk text without redaction.
2. Event volume creates storage pressure.
3. Missing event ordering causes confusing replay timelines.
4. Debug endpoint latency impacts production calls.
