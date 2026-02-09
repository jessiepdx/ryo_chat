# WO-FE-019: Agent-Assisted UI Composition and Quality-of-Life Tooling

Source item: `docs/master-engineering.md` section 4.9 and TODO-FE-040, plus north-star QoL features in the frontend requirements catalog.

## 1. Verbose Description of Work Order
Implement advanced UX helpers that make the playground feel like an agent IDE: trace explanations, bug-report packaging, scenario recording, and agent-suggested improvements.

Scope:
1. Add one-click bug report packaging that bundles:
- run config snapshot
- trace/event stream
- redacted logs
- environment/version metadata
2. Add "Explain this trace" assistant flow that summarizes what happened and why, grounded in run events.
3. Add scenario recorder to convert successful runs into reusable evaluation cases.
4. Add agent-assisted suggestions for:
- new evaluator proposals from recurring failure clusters
- prompt optimization hints tied to metric deltas
- candidate UI widgets based on detected capabilities
5. Add minimal reproduction generator for tool failures (input fixture + expected contract + failing trace).

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/qol_tools.py` (bug packager + scenario recorder + repro generator)
2. New: `hypermindlabs/trace_explainer.py` (trace summarization service)
3. `web_ui.py` (QOL endpoints and action routes)
4. New: `templates/assistant-lab.html`
5. New: `static/agent-playground/qol/trace-explainer.js`
6. New: `static/agent-playground/qol/bug-report.js`
7. New: `static/agent-playground/qol/scenario-recorder.js`

Secondary/API-surface files:
1. `hypermindlabs/failure_triage.py` (feed suggestion engine)
2. `hypermindlabs/evaluation.py` (scenario export into dataset format)
3. `hypermindlabs/security.py` (redaction pass for packaged reports)
4. New: `tests/test_qol_tools.py`
5. New: `tests/test_trace_explainer.py`

Data and schema surfaces:
1. Bug-report package manifest schema.
2. Scenario template schema compatible with evaluation dataset inputs.
3. Suggestion event schema for explainability/audit.

## 3. Success and Validation Metrics
Valid outcome:
1. Users can generate a redacted bug report package from any run.
2. Trace explanation summarizes spans/events with referenced evidence.
3. Scenario recorder exports reusable test cases that run in evaluation workflows.
4. Suggestion engine produces actionable recommendations linked to observed telemetry.

Partial outcome:
1. Bug packaging exists but omits critical replay metadata.
2. Trace explanation is generic and not grounded in actual events.
3. Scenario recording works but cannot be imported into evaluators.

Validation method:
1. End-to-end tests for bug package generation and redaction completeness.
2. Grounding tests verifying trace explanation references valid event IDs.
3. Integration tests for scenario export/import into dataset evaluation flows.

## 4. Potential Failure Modes
1. Bug reports leak sensitive context despite redaction policies.
2. Explanation assistant hallucinates causes not supported by trace data.
3. Suggestion engine generates noisy recommendations with low signal.
4. Scenario recorder captures non-deterministic fields and causes flaky regressions.
