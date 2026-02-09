# WO-FE-016: Ollama Model Operations Cockpit

Source item: `docs/master-engineering.md` section 4.8, and TODO-FE-034 through TODO-FE-035.

## 1. Verbose Description of Work Order
Implement an Ollama-focused model operations cockpit inside the web playground so local model management and run reproducibility are first-class.

Scope:
1. Build model browser:
- installed models
- pullable models
- pull status/progress
- default model assignments by capability
2. Capture per-run model identity and params:
- exact tag
- temperature/top_p/seed where supported
- system prompt version
3. Add compare harness for running the same prompt across selected models and diffing outputs/tool behavior.
4. Add fallback-chain editor for escalation policies (for example schema-fail -> larger model).
5. Expose runtime knobs such as keepalive/concurrency where local backend supports them.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/model_router.py` (model metadata + compare/fallback policy support)
2. New: `hypermindlabs/ollama_ops.py` (model list/pull/metadata wrappers)
3. `web_ui.py` (model-ops endpoints and compare execution routes)
4. New: `templates/model-ops.html`
5. New: `static/agent-playground/models/model-cockpit.js`
6. New: `static/agent-playground/models/compare-harness.js`

Secondary/API-surface files:
1. `app.py` (launcher dashboard links/status integration)
2. `config.empty.json` and `config.json` (fallback chain + model defaults)
3. `.env.example` (Ollama host/model-related env references)
4. New: `tests/test_ollama_ops.py`
5. New: `tests/test_model_compare.py`

Data and schema surfaces:
1. Stored fallback-chain definitions and model profile metadata.
2. Run-level model capture schema with exact model tag and resolved params.

## 3. Success and Validation Metrics
Valid outcome:
1. Users can inspect/pull/select models from web cockpit.
2. Compare runs produce side-by-side outputs with model/tool behavior diffs.
3. Fallback chain policies are editable and executed during runtime failures.
4. Every run stores exact model tag and effective parameter set.

Partial outcome:
1. Model listing works but pull status/selection persistence is incomplete.
2. Compare harness exists without deterministic run metadata capture.
3. Fallback policy UI exists but runtime does not honor it.

Validation method:
1. Integration tests against local Ollama endpoints for list/pull/select flows.
2. Compare tests verifying identical input and per-model output capture.
3. Fallback tests for schema-failure escalation path.

## 4. Potential Failure Modes
1. Model metadata drift due to stale cache of Ollama model list.
2. Compare harness overloads local hardware with unconstrained parallel runs.
3. Fallback policies create loops or escalate to unavailable models.
4. Run records omit exact model tags, breaking replay reproducibility.
