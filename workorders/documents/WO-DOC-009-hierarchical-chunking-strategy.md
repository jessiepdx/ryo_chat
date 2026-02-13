# WO-DOC-009: Hierarchical Chunking Strategy and Formatting-Aware Segmentation

Source item: User request (February 10, 2026) requiring formatting-based and information-tree-based storage and retrieval.

## 1. Verbose Description of Work Order
Implement chunk generation that respects document hierarchy and formatting semantics instead of fixed-size naive slicing.

This work order converts tree nodes into retrieval chunks with boundary rules tuned for headings, lists, tables, code blocks, and citations.

Scope includes:
1. Chunk policies by node type and depth.
2. Sliding overlap and windowing constraints for long sections.
3. Table and list chunking that preserves row/item context.
4. Chunk metadata: parent node path, heading trail, and token/char stats.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_chunker.py`
2. New: `hypermindlabs/document_chunk_rules.py`
3. `hypermindlabs/document_ingestion_worker.py` (chunk stage)
4. New migration: `db/migrations/100_document_chunk_metadata.sql`

Secondary/API-surface files:
1. `hypermindlabs/runtime_settings.py` (chunk size/overlap/budget settings)
2. `config.empty.json` (chunking defaults)
3. `tests/test_document_chunker.py`
4. `tests/test_chunk_rule_edge_cases.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Chunks preserve logical boundaries for major node types.
2. Long sections split deterministically with stable overlap behavior.
3. Chunk metadata is sufficient for provenance and reassembly.
4. Chunking regression tests pass across varied document structures.

Partial outcome:
1. Chunking works but collapses formatting semantics.
2. Overlap exists but causes heavy duplicate content.
3. Table/list chunks lose structural context.

Validation method:
1. Golden chunk snapshots for representative fixtures.
2. Token-budget simulations for long documents.
3. Retrieval dry-runs to verify chunk relevance distribution.

## 4. Potential Failure Modes
1. Oversized chunks exceed embedding limits.
2. Aggressive splitting separates key context from evidence spans.
3. Heading trail metadata drifts from true node ancestry.
4. Non-deterministic chunk IDs break dedupe/versioning.
