# WO-FE-004: Run Modes (Chat, Workflow, Batch, Compare, Replay)

Source item: `docs/master-engineering.md` sections 4.1 and 4.5 plus TODO-FE-009.

## 1. Verbose Description of Work Order
Implement first-class run modes expected in a modern playground:
1. Chat mode
2. Workflow/graph mode
3. Batch mode
4. Compare mode
5. Replay mode

Scope:
1. Define mode-specific request schemas and execution orchestrators.
2. Implement mode switcher UI with persisted last-used mode per workspace/user.
3. Build mode-specific result views:
- batch summary table
- compare side-by-side diff view
- workflow graph step transitions
4. Ensure shared trace/event model still works across all modes.

## 2. Expression of Affected Files
Primary files:
1. `templates/agent-playground.html` (mode switcher + mode views)
2. New: `static/agent-playground/modes/chat-mode.js`
3. New: `static/agent-playground/modes/workflow-mode.js`
4. New: `static/agent-playground/modes/batch-mode.js`
5. New: `static/agent-playground/modes/compare-mode.js`
6. New: `static/agent-playground/modes/replay-mode.js`
7. `web_ui.py` (mode-aware API endpoints)
8. New: `hypermindlabs/run_mode_handlers.py`

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (workflow/handoff support contracts)
2. New: `tests/test_run_modes_api.py`
3. `docs/master-engineering.md` (status updates)

## 3. Success and Validation Metrics
Valid outcome:
1. All five run modes are selectable and executable from web UI.
2. Compare mode executes same input across selected models/configs and renders diffs.
3. Batch mode executes dataset rows and reports per-row outcomes.
4. Workflow mode displays graph/step transitions with trace correlation.

Partial outcome:
1. UI mode selector exists but backend only supports chat mode.
2. Compare results rendered without meaningful diff tooling.
3. Batch mode runs serially with no run-level observability.

Validation method:
1. API contract tests per mode.
2. E2E scenario per mode from start -> completion -> trace inspection.
3. Performance benchmark for compare/batch execution fanout.

## 4. Potential Failure Modes
1. Mode contracts diverge and break shared pane assumptions.
2. Compare mode produces misleading outputs due to config mismatch normalization issues.
3. Workflow graph definitions allow invalid transitions causing orphan runs.
4. Batch mode overloads event stream infrastructure.
