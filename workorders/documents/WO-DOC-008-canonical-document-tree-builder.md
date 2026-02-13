# WO-DOC-008: Canonical Document Tree and Information Graph Builder

Source item: User request (February 10, 2026) requiring hierarchical topic-based, formatting-based, and information-tree-based storage.

## 1. Verbose Description of Work Order
Implement a tree builder that converts parsed elements into a stable logical hierarchy and graph pointers for sections, subsections, lists, tables, and references.

This work order enables retrieval and context assembly beyond flat chunk lookup.

Scope includes:
1. Node taxonomy: `document`, `section`, `subsection`, `paragraph`, `list`, `table`, `code`, `figure`, `footnote`.
2. Parent-child and sibling links with deterministic ordering.
3. Stable node IDs and source span pointers.
4. Graph integrity checks and repair for malformed parse output.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_tree_builder.py`
2. New: `hypermindlabs/document_graph.py`
3. New migration: `db/migrations/099_document_node_edges.sql`
4. `hypermindlabs/document_ingestion_worker.py` (tree-build stage)

Secondary/API-surface files:
1. `hypermindlabs/document_models.py` (node schema contracts)
2. `web_ui.py` (tree preview endpoint)
3. `tests/test_document_tree_builder.py`
4. `tests/test_document_graph_integrity.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Parse outputs consistently produce valid trees.
2. Node-level provenance and order are retained.
3. Tree integrity checks catch orphan/cycle anomalies.
4. Retrieval can target nodes by depth/type.

Partial outcome:
1. Tree exists but ordering is inconsistent across re-ingests.
2. Some node types are flattened and lose structure.
3. Graph anomalies are logged but not repairable.

Validation method:
1. Tree-construction tests for mixed-format fixtures.
2. Integrity tests for edge and ordering constraints.
3. Re-ingest consistency tests for deterministic node IDs.

## 4. Potential Failure Modes
1. Non-deterministic node IDs break cache and version diff logic.
2. Heading detection errors create shallow or fragmented trees.
3. Table/list nodes lose row/item semantics.
4. Orphan nodes are accepted and degrade hierarchical retrieval.
