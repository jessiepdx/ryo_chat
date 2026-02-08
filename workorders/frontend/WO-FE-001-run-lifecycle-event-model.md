# WO-FE-001: Run Lifecycle and Event Model

Source item: `docs/master-engineering.md` sections 4.1, 4.5, 4.9, and TODO-FE-001 through TODO-FE-003.

## 1. Verbose Description of Work Order
Implement the foundational run lifecycle API and append-only event model for web playground execution. This is the required backend contract that all frontend panes depend on.

Scope:
1. Create run lifecycle endpoints for:
- create run
- stream run events
- cancel run
- resume run
- replay run
2. Introduce immutable run records + append-only event records.
3. Add step/event taxonomy for:
- model generation steps
- tool-call steps
- handoff steps
- guardrail checks
- error events
- custom instrumentation events
4. Persist run configuration artifacts for deterministic replay:
- resolved prompt render
- selected model and params
- policy snapshot
- tool definitions/version references
5. Align web run pipeline with CLI orchestrator semantics (`ConversationOrchestrator`) so behavior is consistent across interfaces.

## 2. Expression of Affected Files
Primary files:
1. `web_ui.py` (new run lifecycle API routes)
2. New: `hypermindlabs/run_manager.py` (run orchestration facade for web)
3. New: `hypermindlabs/run_events.py` (event type definitions + serializers)
4. New: `hypermindlabs/replay_manager.py` (resume/replay orchestration)
5. `hypermindlabs/agents.py` (emit structured lifecycle events)
6. `hypermindlabs/model_router.py` (attach route metadata to event payloads)

Secondary/API-surface files:
1. `app.py` (optional route-link visibility in launcher summary)
2. `docs/master-engineering.md` (status updates)
3. `readme.md` (new operator endpoints and run semantics)
4. New: `tests/test_run_manager.py`
5. New: `tests/test_run_events.py`

Data and schema surfaces:
1. New DB tables for runs/events (detailed in WO-FE-017)
2. Shared event envelope schema consumed by frontend panels

## 3. Success and Validation Metrics
Valid outcome:
1. Web API supports create/stream/cancel/resume/replay for runs.
2. Every run produces an ordered append-only event stream with stable event schema.
3. Event payloads include sufficient metadata for trace rendering and replay.
4. Run config snapshot is persisted and retrievable for exact replay attempts.
5. CLI and web orchestrator runs generate comparable event classes for equivalent flows.

Partial outcome:
1. Create + stream exists but cancel/resume/replay is missing or unreliable.
2. Event model exists but lacks required step categories.
3. Replay works only from start and not from specific checkpoints.

Validation method:
1. API tests for all lifecycle endpoints and error paths.
2. Integration test running a prompt that triggers tool invocation and verifies event sequence.
3. Replay test verifying selected run config/model snapshot consistency.

## 4. Potential Failure Modes
1. Events emitted out of order under concurrent writes.
2. Replay behavior diverges because prompt render/model params were not captured immutably.
3. Cancel signal terminates process but leaves run status inconsistent.
4. Resume path duplicates event records or corrupts state progression.
5. Route metadata omitted, preventing reliable post-run diagnostics.
