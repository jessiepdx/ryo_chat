# WO-FE-005: Agent Definition Builder and Versioning

Source item: `docs/master-engineering.md` sections 4.1, 4.2, and TODO-FE-010.

## 1. Verbose Description of Work Order
Create a first-class agent definition system with versioning and shareable config artifacts.

Scope:
1. Define agent definition schema:
- identity metadata
- role/system prompt references
- model policy
- tool access policy
- memory strategy
- guardrail hooks
- orchestration pattern (single, delegated, hierarchical)
2. Build frontend builder/editor for agent definitions.
3. Add version history with changelog and rollback support.
4. Add import/export and copy-config UX (`JSON`/`YAML`).
5. Integrate generated definitions with run launcher in Chat/Workflow modes.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/agent_definitions.py`
2. `web_ui.py` (CRUD routes for agent definitions)
3. New: `static/agent-playground/agent-builder.js`
4. New: `templates/agent-builder.html` (or integrated section in playground)
5. New: `static/agent-playground/version-history.js`

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (consume persisted agent definition payload)
2. `hypermindlabs/policy_manager.py` (validation hooks for model/tool constraints)
3. New: `tests/test_agent_definitions_api.py`
4. `readme.md` (agent definition workflow docs)

Data/config surfaces:
1. Versioned agent definitions table(s) (linked with WO-FE-017 persistence design)
2. Serialized config export endpoint

## 3. Success and Validation Metrics
Valid outcome:
1. Users can create, edit, version, and launch runs from saved agent definitions.
2. Version history includes metadata (author/time/summary) and supports rollback.
3. Import/export roundtrip preserves schema-valid definitions.
4. Definition validation blocks unsupported model/tool/policy combinations.

Partial outcome:
1. CRUD exists but no version history.
2. Version history exists but cannot launch historical versions.
3. Export format omits key fields required for replay/portability.

Validation method:
1. API CRUD and versioning tests.
2. Integration test: create definition -> run -> replay using stored version.
3. Schema validation test coverage for invalid definitions.

## 4. Potential Failure Modes
1. Definition schema drift from actual runtime agent constructor contracts.
2. Rollback restores definition but not dependent prompt/tool version references.
3. Concurrent edits overwrite versions without optimistic locking.
4. Invalid imported configs bypass validation and break run execution.
