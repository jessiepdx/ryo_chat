# WO-FE-017: Postgres Run/Event Persistence and Query Model

Source item: `docs/master-engineering.md` sections 4.8 and 4.9, and TODO-FE-036 through TODO-FE-037.

## 1. Verbose Description of Work Order
Implement the persistence foundation for a replayable agent playground: immutable runs, append-only events, state snapshots, artifacts, and versioned config entities.

Scope:
1. Design and migrate schema for:
- runs (immutable metadata + lifecycle state)
- run_events (append-only)
- run_state_snapshots (checkpointed JSONB)
- run_artifacts (typed payload pointers)
- versioned prompts/tools/agents/evaluators
2. Add index strategy for timeline/search UX:
- project/workspace, run_id, timestamps
- model/tool identifiers
- tag arrays
- full-text indexes over messages/tool outputs
3. Add retention and archival policy hooks for long-running deployments.
4. Ensure replay APIs resolve persisted run config + snapshot references deterministically.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/persistence.py` (repository interfaces)
2. `hypermindlabs/database_router.py` (routing and transactional boundaries)
3. New: `scripts/migrate_run_event_schema.py`
4. `hypermindlabs/run_manager.py` (write/read persistence integration)
5. `hypermindlabs/replay_manager.py` (snapshot restore reads)

Secondary/API-surface files:
1. `web_ui.py` (query/filter endpoints for runs/events/artifacts)
2. `hypermindlabs/utils.py` (existing DB helper extension points)
3. New: `tests/test_run_persistence.py`
4. New: `tests/test_replay_persistence.py`
5. `docs/master-engineering.md` (schema status and migration notes)

Data and schema surfaces:
1. SQL migrations for new run-centric tables and indexes.
2. JSONB shape contracts for event payloads and state snapshots.
3. Optional materialized views for dashboard workloads.

## 3. Success and Validation Metrics
Valid outcome:
1. Runs/events/snapshots/artifacts are persisted with immutable and append-only guarantees.
2. Query performance is acceptable for timeline and search use cases under expected load.
3. Replay can reconstruct state from persisted snapshots/config artifacts.
4. Versioned entities (prompt/tool/agent/evaluator) are referenceable by run records.

Partial outcome:
1. Basic run/event persistence exists but no snapshot/artifact model.
2. Schema exists without indexes, causing poor dashboard query latency.
3. Replay reads mutable config pointers instead of versioned references.

Validation method:
1. Migration tests and rollback safety checks.
2. Integration tests for write/read/replay of run event chains.
3. Query performance tests on synthetic large event datasets.

## 4. Potential Failure Modes
1. Inconsistent transaction boundaries produce orphaned run events.
2. JSONB payload drift breaks consumers expecting stable schemas.
3. Missing indexes cause severe latency under real workloads.
4. Retention jobs delete records needed for replay/compliance.
