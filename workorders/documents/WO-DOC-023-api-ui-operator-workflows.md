# WO-DOC-023: API and Operator Workflow Surfaces for Document RAG

Source item: User request (February 10, 2026) requiring practical storage and retrieval operations per user/chat with recall controls.

## 1. Verbose Description of Work Order
Expose robust API and operator workflow surfaces for upload, ingest status, retrieval testing, curiosity-policy inspection, and governance actions.

This work order translates backend capability into usable operational flows.

Scope includes:
1. Upload and ingest-status APIs.
2. Retrieval-debug and citation inspection APIs.
3. Curiosity-policy explain endpoints.
4. Operator controls for re-index, redaction, restore, and retention.

## 2. Expression of Affected Files
Primary files:
1. `web_ui.py` (document API route group)
2. New: `hypermindlabs/document_api.py`
3. New: `static/agent-playground/rag/ingestion-ui.js`
4. `templates/knowledge-tools.html` (document workflow UI)

Secondary/API-surface files:
1. `static/agent-playground/workspace.js` (integration hooks)
2. `hypermindlabs/capability_manifest.py` (document capability schemas)
3. `tests/test_document_api.py`
4. `tests/test_document_operator_workflows.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Operators can execute full ingest -> retrieve -> debug -> govern workflow via API/UI.
2. Scope boundaries are preserved in all operator workflows.
3. Capability manifests expose document features for schema-driven UI.
4. API error handling is consistent and actionable.

Partial outcome:
1. Upload UI exists but retrieval debug is disconnected.
2. APIs exist but no capability-manifest integration.
3. Governance actions require direct DB access.

Validation method:
1. API integration tests for each operator workflow.
2. Auth and scope tests on all endpoints.
3. UI smoke tests in `knowledge-tools` surfaces.

## 4. Potential Failure Modes
1. UI shows stale job/retrieval state due to missing event polling.
2. Endpoint auth checks diverge across route handlers.
3. Operator tools permit destructive actions without safeguards.
4. Capability manifests drift from actual backend behavior.
