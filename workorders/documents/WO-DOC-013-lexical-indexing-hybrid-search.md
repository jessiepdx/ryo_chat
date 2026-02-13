# WO-DOC-013: Lexical Indexing and Hybrid Retrieval Fusion

Source item: User request (February 10, 2026) requiring highly effective retrieval across varied formatting and document types.

## 1. Verbose Description of Work Order
Add lexical indexing and hybrid fusion so retrieval is robust when semantic vectors miss exact terms, identifiers, code snippets, or rare entities.

This work order introduces a two-lane retrieval stack: vector similarity plus lexical search with rank fusion.

Scope includes:
1. `tsvector` or equivalent lexical index materialization for chunk text.
2. Query rewrite for lexical terms and phrase preservation.
3. Hybrid scoring and reciprocal-rank-fusion policy.
4. Scoped filtering before ranking.

## 2. Expression of Affected Files
Primary files:
1. New migration: `db/migrations/106_document_chunk_lexical_index.sql`
2. New: `hypermindlabs/document_lexical_search.py`
3. New: `hypermindlabs/document_hybrid_retrieval.py`
4. `hypermindlabs/utils.py` (knowledge search path extension)

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (hybrid retrieval invocation path)
2. `hypermindlabs/runtime_settings.py` (fusion weights)
3. `tests/test_document_hybrid_retrieval.py`
4. `tests/test_document_lexical_search.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Hybrid search improves recall on exact-token and identifier-heavy queries.
2. All retrieval remains strictly scope-filtered.
3. Fusion metadata is emitted for debugging.
4. Query latency stays within defined SLO budgets.

Partial outcome:
1. Lexical and vector search both exist but without fusion.
2. Fusion works but ignores scope filters for one lane.
3. Hybrid scores are not explainable in logs/debug views.

Validation method:
1. Benchmark set comparing vector-only vs hybrid.
2. Query-class evaluations (exact IDs, policy text, code blocks).
3. Retrieval-debug traces for score decomposition.

## 4. Potential Failure Modes
1. Lexical lane overweights frequent terms and lowers precision.
2. Fused ranking duplicates near-identical chunks excessively.
3. Query rewrite strips critical quoted phrases.
4. Index bloat degrades write throughput.
