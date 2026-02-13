# WO-DOC-015: Reranking, Evidence Scoring, and Context Packing

Source item: User request (February 10, 2026) requiring highly effective retrieval quality and modern best-practice ranking.

## 1. Verbose Description of Work Order
Add reranking and context-packing layers that optimize which retrieved candidates are actually sent to generation under strict token budgets.

This work order separates retrieval candidate generation from final evidence selection.

Scope includes:
1. Reranker integration (cross-encoder or LLM-lite scorer).
2. Score blending with retrieval and taxonomy signals.
3. Context packer optimizing relevance/diversity/coverage under token budget.
4. Explainable ranking metadata persisted for debug.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_reranker.py`
2. New: `hypermindlabs/document_context_packer.py`
3. `hypermindlabs/document_hierarchical_retrieval.py` (reranker hook)
4. New migration: `db/migrations/108_document_rerank_events.sql`

Secondary/API-surface files:
1. `hypermindlabs/runtime_settings.py` (rerank toggles/limits)
2. `hypermindlabs/agents.py` (final context injection pipeline)
3. `tests/test_document_reranker.py`
4. `tests/test_document_context_packer.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Reranking improves answer-grounding quality on benchmark set.
2. Context packing respects token limits while preserving evidence diversity.
3. Ranking decisions are persisted for inspection.
4. Failure fallback exists when reranker is unavailable.

Partial outcome:
1. Reranker runs but decisions are not traceable.
2. Packer enforces limits but drops citation-critical chunks.
3. Fallback path is missing and retrieval fails closed.

Validation method:
1. Offline evaluation before/after reranker activation.
2. Token-budget stress tests.
3. Retrieval-debug assertions for rank and pack metadata.

## 4. Potential Failure Modes
1. Reranker latency dominates end-to-end response time.
2. Score blending weights drift and regress recall.
3. Packer over-optimizes for diversity and loses top evidence.
4. Reranker model changes break result stability.
