# WO-FE-018: Capability Manifests and Schema-Driven UI Renderer

Source item: `docs/master-engineering.md` section 4.9, and TODO-FE-038 through TODO-FE-039.

## 1. Verbose Description of Work Order
Implement hydratable UI primitives so the frontend can discover runtime capabilities and render panels/forms directly from machine-readable schemas.

Scope:
1. Define capability-manifest contract for subsystems:
- models.list
- tools.list
- agents.list
- memory.strategies
- evals.list
- artifacts.types
2. Expose manifest endpoints with:
- capability ID
- permission requirements
- JSON schemas for config and input forms
- supported renderers/artifact types
3. Implement frontend schema renderer that generates:
- forms for tool args/model params/evaluator configs
- panel layouts from renderer manifests
- validation UX from schema constraints
4. Add manifest caching and invalidation strategy for runtime updates.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/capability_manifest.py` (manifest provider)
2. `web_ui.py` (manifest endpoints)
3. New: `static/agent-playground/schema-renderer/renderer.js`
4. New: `static/agent-playground/schema-renderer/validators.js`
5. `templates/agent-playground.html` (replace hardcoded forms with schema-driven mounts)

Secondary/API-surface files:
1. `hypermindlabs/tool_registry.py` (source tool schemas + metadata)
2. `hypermindlabs/model_router.py` (model param schemas)
3. `hypermindlabs/runtime_settings.py` (manifest-friendly runtime option schema)
4. New: `tests/test_capability_manifest.py`
5. New: `tests/test_schema_renderer_contract.py`

Data and schema surfaces:
1. Canonical manifest JSON schema with versioning.
2. Capability schema registry and compatibility policy.
3. Frontend renderer contract tests to detect breaking manifest changes.

## 3. Success and Validation Metrics
Valid outcome:
1. Frontend can render at least tools, model-parameter panels, and evaluator configs solely from manifest schemas.
2. Permission gates in manifests map to enforced backend authorization.
3. Manifest versioning supports backward-compatible UI rendering.
4. New capabilities can be surfaced without hardcoded frontend rewrites.

Partial outcome:
1. Manifest endpoints exist but frontend still hardcodes most forms.
2. Renderer works for one capability class only.
3. Schema changes break older clients without compatibility signaling.

Validation method:
1. Contract tests for manifest schema compliance.
2. UI tests generating forms from sample manifests.
3. Backward-compatibility tests between manifest versions.

## 4. Potential Failure Modes
1. Manifest schemas are underspecified and cannot drive full UI generation.
2. Permission metadata diverges from backend enforcement.
3. Renderer fails on complex nested schemas/arrays/oneOf unions.
4. Caching stale manifests hides newly deployed capabilities.
