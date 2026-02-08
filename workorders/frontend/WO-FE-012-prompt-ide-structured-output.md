# WO-FE-012: Prompt IDE and Structured Output Controls

Source item: `docs/master-engineering.md` section 4.6, and TODO-FE-011, TODO-FE-023, and TODO-FE-024.

## 1. Verbose Description of Work Order
Implement a production-grade prompt authoring and experimentation surface with versioning, environment overlays, parameter sweeps, and structured-output validation loops.

Scope:
1. Build prompt IDE with:
- templates and named variables
- render preview with test bindings
- prompt version history and changelog
- environment overlays (dev/stage/prod)
2. Add prompt playground for rapid runs across model parameter variants (temperature, top_p, seed where supported).
3. Implement structured-output mode:
- JSON schema selection
- validation errors surfaced in UI
- automated repair-loop retries with bounded attempts
4. Persist prompt revisions and run bindings so experiments are reproducible.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/prompt_store.py` (versioned prompt persistence)
2. New: `hypermindlabs/structured_output.py` (schema validate + repair loop)
3. `web_ui.py` (prompt CRUD/version routes + playground execute route)
4. New: `templates/prompt-ide.html`
5. New: `static/agent-playground/prompt/prompt-ide.js`
6. New: `static/agent-playground/prompt/playground.js`

Secondary/API-surface files:
1. `hypermindlabs/agents.py` (consume resolved prompt/version references)
2. `hypermindlabs/model_router.py` (parameter pass-through consistency)
3. `hypermindlabs/run_events.py` (structured-output validation/repair events)
4. New: `tests/test_prompt_store.py`
5. New: `tests/test_structured_output.py`

Data and schema surfaces:
1. Versioned prompt table with environment-specific overlays.
2. JSON schema registry for output constraints.
3. Run-level capture of prompt version and rendered text hash.

## 3. Success and Validation Metrics
Valid outcome:
1. Users can create/edit/version prompts and compare versions.
2. Prompt playground runs side-by-side parameter sweeps and stores results.
3. Structured-output runs enforce schema validation and show repair traces.
4. Run replay resolves the original prompt version and rendered content.

Partial outcome:
1. Prompt editing works but no version diff/changelog.
2. Validation exists but no repair-loop controls.
3. Playground runs are not persisted for later comparison.

Validation method:
1. API tests for version creation, rollback, and environment overlays.
2. Validation tests for schema pass/fail and bounded repair behavior.
3. UI tests for side-by-side compare and prompt diff rendering.

## 4. Potential Failure Modes
1. Prompt variable rendering is non-deterministic across environments.
2. Repair loops mask root-cause schema issues and inflate latency.
3. Version references drift if runs store mutable prompt pointers.
4. Parameter sweep jobs overload local model runtime without concurrency limits.
