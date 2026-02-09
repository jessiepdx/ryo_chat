# WO-FE-008: Tool Test Harness, Golden Outputs, and Contract Checks

Source item: `docs/master-engineering.md` section 4.3 and TODO-FE-016.

## 1. Verbose Description of Work Order
Build an isolated tool test harness to validate tool behavior independently from full agent runs.

Scope:
1. Provide UI/API to execute a tool with controlled input fixtures.
2. Store and compare golden outputs for regression detection.
3. Add contract tests for schema stability over time.
4. Support mock mode and real mode test execution.
5. Surface regression deltas in a dedicated report UI.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/tool_test_harness.py`
2. `web_ui.py` (tool test execution/report endpoints)
3. New: `static/agent-playground/tools/tool-harness.js`
4. New: `static/agent-playground/tools/regression-report.js`
5. New: `templates/tool-harness.html` (or integrated panel)

Secondary/API-surface files:
1. `hypermindlabs/tool_registry.py` (fixture schema alignment)
2. `hypermindlabs/tool_runtime.py` (harness execution adapter)
3. New: `tests/test_tool_harness.py`
4. New: `tests/test_tool_contracts.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Tool tests can be run in isolation from web UI and API.
2. Golden output comparison reports pass/fail and field-level diffs.
3. Contract tests detect incompatible schema changes before deployment.
4. Harness artifacts are persisted and linked to tool version.

Partial outcome:
1. Harness executes tests but no historical regression comparison exists.
2. Regression report exists but missing structured diff details.
3. Contract checks do not block incompatible changes.

Validation method:
1. Automated harness tests for success/error/timeout cases.
2. Regression test suite with intentionally changed outputs.
3. CI gate integrating contract checks.

## 4. Potential Failure Modes
1. Flaky tests due to non-deterministic tool dependencies.
2. Golden outputs include sensitive data and violate privacy policy.
3. Tool version association is missing, making reports ambiguous.
4. Contract checks too coarse and miss backward-incompatible changes.
