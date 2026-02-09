# WO-FE-013: Evaluation Datasets, Evaluators, and Review Queues

Source item: `docs/master-engineering.md` section 4.6, and TODO-FE-025 through TODO-FE-028.

## 1. Verbose Description of Work Order
Implement a complete evaluation workflow so teams can measure agent quality before promoting changes.

Scope:
1. Add dataset management:
- dataset CRUD
- versioning and immutable snapshots
- tag-based slicing and metadata filters
2. Add evaluator execution framework:
- rule-based evaluators (exact/fuzzy checks)
- model-judge evaluators with calibration options
- tool/citation correctness checks
- latency/step-budget checks
3. Add annotation/review queues:
- flag runs for review
- assign reviewers
- capture structured labels and notes
4. Implement regression gates to block prompt/tool/agent changes that degrade required slices.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/evaluation.py` (dataset + evaluator orchestration)
2. New: `hypermindlabs/annotation_queue.py` (review workflow engine)
3. `web_ui.py` (dataset/evaluator/annotation APIs)
4. New: `templates/evaluation-center.html`
5. New: `static/agent-playground/eval/datasets.js`
6. New: `static/agent-playground/eval/evaluators.js`
7. New: `static/agent-playground/eval/annotation-queue.js`

Secondary/API-surface files:
1. `hypermindlabs/run_manager.py` (batch run support for evaluation workloads)
2. `hypermindlabs/telemetry.py` (expose eval trend metrics)
3. New: `tests/test_evaluation.py`
4. New: `tests/test_annotation_queue.py`
5. `docs/master-engineering.md` and `readme.md` (operator guidance)

Data and schema surfaces:
1. Dataset tables with version + tag indexes.
2. Evaluator definitions and evaluator-run result tables.
3. Annotation records with reviewer/user references and status transitions.

## 3. Success and Validation Metrics
Valid outcome:
1. Users can run an evaluator suite against a dataset slice and persist results.
2. Review queue supports assignment, labeling, and status transitions.
3. Regression gates can fail a candidate config revision when target metrics degrade.
4. Eval history is queryable by dataset version, evaluator, model, and agent config.

Partial outcome:
1. Dataset CRUD exists but no versioning/tag slicing.
2. Evaluator runs exist but cannot be compared over time.
3. Annotation queue exists without structured labels.

Validation method:
1. Integration tests for dataset version/slice/evaluator execution.
2. Regression-gate tests using seeded pass/fail thresholds.
3. UI tests for annotation lifecycle transitions.

## 4. Potential Failure Modes
1. Evaluator drift causes unstable scoring between runs.
2. Queue assignments lose ownership/state under concurrent edits.
3. Regression gates are too coarse and block safe changes.
4. Large batch jobs starve interactive playground execution.
