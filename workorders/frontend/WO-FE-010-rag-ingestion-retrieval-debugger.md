# WO-FE-010: RAG Ingestion and Retrieval Debugger

Source item: `docs/master-engineering.md` section 4.4 and TODO-FE-018 through TODO-FE-020.

## 1. Verbose Description of Work Order
Implement first-class RAG operations and debugging workflows in the web playground.

Scope:
1. Build ingestion UI for documents with controls for:
- chunking
- metadata extraction
- deduplication
- versioning
2. Build retrieval debugger showing:
- retrieved chunks
- vector distance/scores
- query rewrite details
- reranker decisions
3. Add citation-required mode and provenance evidence rendering in responses.
4. Add reversible redaction/delete operations for ingested artifacts.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/utils.py` (knowledge manager extension points)
2. New: `hypermindlabs/rag_ingestion.py`
3. New: `hypermindlabs/retrieval_debug.py`
4. `web_ui.py` (ingestion/retrieval/provenance endpoints)
5. New: `static/agent-playground/rag/ingestion-ui.js`
6. New: `static/agent-playground/rag/retrieval-debugger.js`
7. `templates/knowledge-tools.html` (convert placeholder editor to operational UI)

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (emit retrieval event metadata)
2. `hypermindlabs/run_events.py` (retrieval/citation events)
3. New: `tests/test_rag_ingestion.py`
4. New: `tests/test_retrieval_debug.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Users can ingest/version documents from web UI.
2. Retrieval debugger shows chunk-level evidence and scoring details for each run.
3. Citation-required mode blocks non-evidenced claims where policy requires provenance.
4. Redaction/delete workflows are auditable and reversible where policy allows.

Partial outcome:
1. Ingestion exists but retrieval debugger is not wired to run traces.
2. Chunk scores shown without source metadata/citation links.
3. Citation-required mode exists but is not enforceable.

Validation method:
1. Integration tests across ingest -> run -> retrieve -> cite flow.
2. Negative tests for missing/invalid metadata.
3. Provenance integrity tests verifying source references map to stored docs.

## 4. Potential Failure Modes
1. Duplicate or stale embeddings from ingestion re-runs without version controls.
2. Retrieval debugging leaks sensitive raw content to unauthorized users.
3. Reranker decision data omitted, limiting diagnostic value.
4. Citation enforcement over-constrains legitimate conversational answers.
