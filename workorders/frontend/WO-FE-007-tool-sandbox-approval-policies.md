# WO-FE-007: Tool Sandbox and Human Approval Policies

Source item: `docs/master-engineering.md` sections 4.3 and 4.7 plus TODO-FE-015.

## 1. Verbose Description of Work Order
Introduce policy-governed sandbox controls and human-in-the-loop approval gates for risky tool calls.

Scope:
1. Define per-tool sandbox policy surface:
- network egress allowlist
- filesystem restrictions
- execution time/memory ceilings
- side-effect class (read-only/mutating)
2. Add approval workflow for mutating or sensitive tools.
3. Add dry-run mode with mock outputs for safe previews.
4. Emit approval and sandbox decision events into run timeline.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/tool_sandbox.py`
2. New: `hypermindlabs/approval_manager.py`
3. `hypermindlabs/tool_runtime.py` (sandbox + approval checks before execute)
4. `web_ui.py` (approval queue endpoints)
5. New: `static/agent-playground/tools/approval-queue.js`
6. New: `static/agent-playground/tools/sandbox-policy-editor.js`

Secondary/API-surface files:
1. `hypermindlabs/run_events.py` (approval/sandbox event classes)
2. `hypermindlabs/policy_manager.py` (default policy definitions)
3. New: `tests/test_tool_sandbox.py`
4. New: `tests/test_approval_manager.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Risky tool calls pause in pending-approval state until action is taken.
2. Approval decisions (approve/deny/timeout) are persisted and traced.
3. Sandbox restrictions enforce network/file/exec boundaries.
4. Dry-run mode executes with mock output and zero side effects.

Partial outcome:
1. Approval queue exists but runtime ignores pending state.
2. Sandbox policies are configurable but not enforced at execution.
3. Denied actions do not surface clear user-facing reasons.

Validation method:
1. Unit tests for policy enforcement decisions.
2. Integration tests for approval queue lifecycle.
3. Security tests for blocked filesystem/network operations.

## 4. Potential Failure Modes
1. Approval deadlocks where pending tool calls never resolve.
2. Sandbox bypass through unguarded execution path.
3. Policy defaults too strict and block legitimate workflows.
4. Dry-run returns unrealistic data, causing false confidence.
