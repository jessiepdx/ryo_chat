# WO-DOC-014: Hierarchical Retrieval and Information-Tree Context Assembly

Source item: User request (February 10, 2026) requiring information-tree-based storage and retrieval.

## 1. Verbose Description of Work Order
Implement retrieval that selects evidence at multiple hierarchy levels and assembles context with parent/child/sibling awareness.

This work order enables coarse-to-fine recall: section-level relevance, then targeted leaf chunks.

Scope includes:
1. Multi-level candidate generation (node + chunk levels).
2. Neighbor expansion policies (`parent`, `sibling`, `child`) with budget constraints.
3. Context assembly preserving heading path and source offsets.
4. Redundancy pruning and coverage balancing.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_hierarchical_retrieval.py`
2. New: `hypermindlabs/document_context_assembly.py`
3. `hypermindlabs/document_hybrid_retrieval.py` (integration)
4. New migration: `db/migrations/107_document_retrieval_traces.sql`

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (context payload injection)
2. `hypermindlabs/history_recall.py` (pattern reuse for progressive rounds)
3. `tests/test_hierarchical_retrieval.py`
4. `tests/test_context_assembly.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Retrieval can return tree-aware evidence bundles.
2. Context assembly includes heading lineage and source anchors.
3. Redundancy is bounded while maintaining topical coverage.
4. Retrieval traces expose assembly decisions.

Partial outcome:
1. Hierarchical candidates exist but final context is still flat.
2. Tree expansion works but is not token-budget aware.
3. Assembly does not preserve provenance anchors.

Validation method:
1. End-to-end tests for section-to-leaf retrieval behavior.
2. Budget-pressure tests for context packing with long documents.
3. Trace inspection tests for expansion path correctness.

## 4. Potential Failure Modes
1. Neighbor expansion causes low-signal context inflation.
2. Heading lineage mismatches chunk ancestry.
3. Assembly favors one subtree and misses distributed evidence.
4. Token budgeting trims critical supporting context.
