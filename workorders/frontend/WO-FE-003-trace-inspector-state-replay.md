# WO-FE-003: Trace, Inspector, State Diff, and Replay Controls

Source item: `docs/master-engineering.md` sections 4.5 and TODO-FE-005 through TODO-FE-008.

## 1. Verbose Description of Work Order
Deliver full debugging ergonomics for agent runs by implementing:
1. Nested timeline trace
2. Per-step inspector payloads
3. State-at-step snapshots + step-to-step diff
4. Replay from run start or selected step
5. What-if replay by editing selected state snapshot

Scope:
1. Build a span/step viewer with expand/collapse and searchable metadata.
2. Add step selection contract that synchronizes trace, inspector, state, and artifacts panes.
3. Implement replay actions with clear provenance:
- replay-original
- replay-from-step
- replay-with-edited-state
4. Persist replay lineage (which run/step produced new replay run).

## 2. Expression of Affected Files
Primary files:
1. New: `static/agent-playground/trace-store.js`
2. New: `static/agent-playground/state-diff.js`
3. New: `static/agent-playground/replay-controls.js`
4. `templates/agent-playground.html` (step controls and inspector sections)
5. `web_ui.py` (replay endpoints)
6. New: `hypermindlabs/replay_manager.py` (checkpoint and replay execution)

Secondary/API-surface files:
1. New: `hypermindlabs/state_snapshot_store.py`
2. `hypermindlabs/agents.py` (state snapshot emission hooks)
3. `hypermindlabs/run_events.py` (lineage metadata fields)
4. New: `tests/test_replay_manager.py`

## 3. Success and Validation Metrics
Valid outcome:
1. User can select any trace step and inspect payload, timing, errors, and state for that step.
2. State diff rendering is accurate and deterministic for adjacent steps.
3. Replay from selected step creates a new run with lineage reference.
4. Edited-state replay path applies edits and records mutation provenance.

Partial outcome:
1. Replay only from beginning; step replay unavailable.
2. State pane shows snapshots but no diff support.
3. Inspector lacks request/response details for tool/model events.

Validation method:
1. Integration tests with synthetic runs containing model/tool/error spans.
2. Replay verification test ensuring parent-run lineage and checkpoint IDs are stored.
3. UI test validating step selection updates all dependent panes.

## 4. Potential Failure Modes
1. Snapshot storage bloat from over-frequent state writes.
2. Non-deterministic replays due to incomplete captured context.
3. Incorrect state diff algorithm causing misleading diagnostics.
4. Replay permissions bypass for unauthorized users.
