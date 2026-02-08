# WO-FE-002: Multi-Pane Workspace Shell

Source item: `docs/master-engineering.md` sections 4.1 and TODO-FE-004.

## 1. Verbose Description of Work Order
Implement the core web playground workspace with five functional panes bound to real runtime data:
1. Chat/Run pane
2. Trace/Steps pane
3. State pane
4. Artifacts pane
5. Inspector pane

Scope:
1. Replace demo-only content in `templates/agent-playground.html` with a workspace layout that can load and hydrate live run data.
2. Introduce frontend state manager modules that subscribe to run event streams and distribute data to panes.
3. Implement pane persistence and layout preferences (resize/collapse/restore) without breaking mobile fallback behavior.
4. Ensure layout is render-safe when optional panes have no data yet (graceful empty states).

## 2. Expression of Affected Files
Primary files:
1. `templates/agent-playground.html` (full layout rewrite)
2. `static/base-style.css` (workspace-specific responsive layout styles)
3. New: `static/agent-playground/workspace.js`
4. New: `static/agent-playground/panes/chat-pane.js`
5. New: `static/agent-playground/panes/trace-pane.js`
6. New: `static/agent-playground/panes/state-pane.js`
7. New: `static/agent-playground/panes/artifacts-pane.js`
8. New: `static/agent-playground/panes/inspector-pane.js`

Secondary/API-surface files:
1. `web_ui.py` (page hydration payload and auth guards)
2. `static/ui-managers.js` (panel-manager integration or replacement path)
3. `tests/` frontend integration tests (new)

## 3. Success and Validation Metrics
Valid outcome:
1. All five panes render and can be toggled independently.
2. Chat pane can launch a run and stream tokens.
3. Trace pane updates in near-real-time from event stream.
4. State/artifact/inspector panes reflect selected step context.
5. Workspace remains functional on desktop and mobile viewports.

Partial outcome:
1. Pane layout is present but trace/state/artifacts remain static placeholders.
2. Pane data refresh requires page reload.
3. Mobile layout degrades into unusable overflow.

Validation method:
1. Frontend E2E tests covering initial load, run start, pane updates, and step selection.
2. Manual responsiveness checks on representative desktop/mobile widths.
3. Regression test ensuring base menu/account panels still function.

## 4. Potential Failure Modes
1. Pane coupling causes full-page rerenders on every token chunk.
2. Unbounded trace rendering causes memory/performance degradation.
3. Layout assumptions break existing base template panel mechanics.
4. Missing defensive handling for null/partial event payloads.
