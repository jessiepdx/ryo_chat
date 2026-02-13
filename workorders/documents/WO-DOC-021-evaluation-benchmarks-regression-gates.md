# WO-DOC-021: Retrieval Evaluation Suite and Regression Gates

Source item: User request (February 10, 2026) requiring highly effective retrieval quality using modern best practices.

## 1. Verbose Description of Work Order
Create an evaluation framework for ingestion and retrieval quality so improvements are measurable and regressions are blocked.

Scope includes:
1. Labeled benchmark datasets (queries, expected evidence, expected citations).
2. Retrieval metrics (`Recall@K`, `MRR`, `nDCG`, citation precision).
3. Reranker and curiosity-policy ablation experiments.
4. CI regression gates for key quality thresholds.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_eval.py`
2. New: `hypermindlabs/document_eval_datasets.py`
3. New: `tests/test_document_eval_metrics.py`
4. New: `tests/test_document_retrieval_regressions.py`

Secondary/API-surface files:
1. `workorders/frontend/WO-FE-013-evaluation-datasets-evaluators-queues.md` (integration linkage)
2. `docs/master-engineering.md` (quality-gate status)
3. New: `docs/document-retrieval-evaluation.md`
4. CI config (test stage for retrieval quality)

## 3. Success and Validation Metrics
Valid outcome:
1. Benchmark suite runs reproducibly in CI.
2. Baseline metrics are defined and versioned.
3. Regressions fail CI with actionable deltas.
4. Evaluation supports segment breakdown (format, query class, scope type).

Partial outcome:
1. Metrics are tracked but not enforced in CI.
2. Dataset exists but lacks expected-evidence labels.
3. Results are non-deterministic and hard to compare.

Validation method:
1. CI execution on controlled fixtures.
2. Historical metric trend snapshots.
3. Manual spot checks for qualitative grounding quality.

## 4. Potential Failure Modes
1. Benchmark data is too narrow and overfits retrieval tuning.
2. Metric improvements do not correlate with real user quality.
3. CI runtime becomes too expensive for frequent execution.
4. Gold labels drift with document version changes.
