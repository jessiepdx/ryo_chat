# WO-FE-011: Observability, Metrics, and Regression Dashboard

Source item: `docs/master-engineering.md` sections 4.5 and 9, and TODO-FE-021 through TODO-FE-022.

## 1. Verbose Description of Work Order
Implement trace-adjacent observability surfaces so operators and builders can measure run quality over time, detect regressions, and triage recurring failure clusters.

Scope:
1. Build a metrics aggregation pipeline for run/event telemetry:
- latency distributions (p50/p95/p99)
- token usage proxy and tool-call counts
- success/failure rates by scenario, model, and tool
- error rate breakdown by error class
2. Implement a regression dashboard in web UI with time slicing and tag filters.
3. Add failure triage clustering for recurring run failures:
- tool schema mismatches
- tool timeouts
- retrieval failures
- policy violations
4. Attach guided remediation hints from cluster labels (for example: tighten schema, add evaluator, adjust tool policy).
5. Ensure metrics are linked to immutable run IDs and event spans for drill-down.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/telemetry.py` (metric derivation from run events)
2. New: `hypermindlabs/failure_triage.py` (error clustering + remediation hints)
3. `web_ui.py` (metrics APIs + dashboard data routes)
4. New: `templates/observability-dashboard.html`
5. New: `static/agent-playground/observability/dashboard.js`
6. New: `static/agent-playground/observability/charts.js`

Secondary/API-surface files:
1. `hypermindlabs/run_events.py` (ensure event fields needed for aggregates)
2. `hypermindlabs/run_manager.py` (emit final run outcome metadata)
3. New: `tests/test_telemetry.py`
4. New: `tests/test_failure_triage.py`
5. `docs/master-engineering.md` (status + deltas)

Data and schema surfaces:
1. Metric materialization tables/views tied to run/event identifiers.
2. API response schemas for trend charts, table drill-down, and failure clusters.

## 3. Success and Validation Metrics
Valid outcome:
1. Dashboard renders run metrics with filters for project/model/tool/time.
2. Selecting a metric segment links back to relevant runs/traces.
3. Failure clusters are generated with stable labels and count trends.
4. Regression view clearly highlights degraded metric slices over baseline windows.

Partial outcome:
1. Metrics render but without drill-down to trace/run IDs.
2. Failure clusters exist but are inconsistent across repeated runs.
3. Dashboard shows aggregates only, without filter fidelity.

Validation method:
1. API tests for aggregation correctness on synthetic event datasets.
2. UI tests verifying filter + drill-down behavior.
3. Regression tests ensuring cluster labeling remains stable for known failures.

## 4. Potential Failure Modes
1. Metrics skew due to missing event fields or inconsistent timestamp units.
2. Over-aggregation hides important scenario-level regressions.
3. Failure clustering groups unrelated issues and gives poor remediation guidance.
4. High-cardinality labels degrade dashboard query performance.
