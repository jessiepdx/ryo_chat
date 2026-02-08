# WO-FE-006: Tool Registry UI and Schema-Driven Forms

Source item: `docs/master-engineering.md` sections 4.3 and 4.9 plus TODO-FE-012 and TODO-FE-014.

## 1. Verbose Description of Work Order
Expose backend tool registry as a frontend-editable capability catalog with schema-driven argument form rendering.

Scope:
1. Build tool catalog UI from canonical tool metadata (name, description, arg schema, auth/risk metadata).
2. Generate argument forms directly from JSON schema and validation rules.
3. Allow user-defined tools and plugin-based tools to register through same contract.
4. Add role-sensitive controls for editing tool metadata/policies.
5. Ensure registry output remains compatible with `ToolRuntime` and model tool schema generation.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/tool_registry.py` (expose machine-readable tool manifest)
2. `web_ui.py` (tool registry read/update endpoints)
3. New: `static/agent-playground/tools/tool-registry-view.js`
4. New: `static/agent-playground/tools/tool-form-renderer.js`
5. New: `templates/tool-registry.html` (or integrated workspace pane)

Secondary/API-surface files:
1. `hypermindlabs/tool_runtime.py` (compatibility and validation surface)
2. `hypermindlabs/policy_manager.py` (policy controls for tool availability)
3. New: `tests/test_tool_registry_api.py`
4. New: `tests/test_schema_form_renderer.js` (frontend unit tests)

## 3. Success and Validation Metrics
Valid outcome:
1. Tool list renders from backend schemas without hardcoded frontend forms.
2. Argument editor validates required fields/type constraints before submission.
3. User-defined tool registration path is schema-validated and discoverable.
4. Registry changes are reflected in subsequent run tool availability.

Partial outcome:
1. Catalog is view-only with no create/update flows.
2. Schema renderer supports only subset of arg types.
3. Tool metadata edits do not propagate to runtime registration.

Validation method:
1. Contract tests validating manifest schema structure.
2. End-to-end test from tool definition update -> run invocation.
3. Negative tests for invalid schema/tool definitions.

## 4. Potential Failure Modes
1. Frontend schema renderer misinterprets backend schema keywords.
2. Tool metadata updates introduce runtime mismatch with function signatures.
3. Permission checks absent, allowing unauthorized tool mutation.
4. Registry cache invalidation issues expose stale tool definitions.
