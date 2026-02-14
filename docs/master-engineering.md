# RYO Chat Master Engineering Document

## 1. Purpose
This document is the engineering baseline for the current RYO Chat codebase and the master plan for the next upgrade cycle.

It serves three functions:
1. Capture the current architecture and operational reality.
2. Define the target behavior for automation, graceful degradation, and agent-playground evolution.
3. Drive implementation order, acceptance criteria, and operator documentation updates.

## 2. Scope
This document covers:
- Multi-platform runtime (`cli_ui.py`, `web_ui.py`, `telegram_ui.py`, `x_ui.py`, `app.py` launcher)
- Agent orchestration and tool-calling stack (`hypermindlabs/agents.py`, `hypermindlabs/tool_runtime.py`, `hypermindlabs/tool_registry.py`)
- Data/config runtime (`hypermindlabs/utils.py`, `hypermindlabs/runtime_settings.py`, `config.json`)
- Policy/system prompt stack (`policies/agent/*`)
- Existing upgrade items WO-001 through WO-010
- New frontend modernization scope for a full agent playground aligned with CLI capabilities and future multi-user tiers

### 2.1 Implementation Status Snapshot (February 8, 2026)
1. WO-001 Automatic model fallbacks: Implemented.
2. WO-002 Automatic PostgreSQL fallback generation + setup flow: Implemented.
3. WO-003 Documentation and master engineering maintenance: Implemented.
4. WO-004 Graceful degradation for tools with missing APIs: Implemented.
5. WO-005 Curses walkthrough for key ingress/model selection: Implemented.
6. WO-006 Policy walkthrough/setup: Implemented.
7. WO-007 Tool-calling stack hardening: Implemented.
8. WO-008 Automatic Telegram key ingress: Implemented.
9. WO-009 Prerequisite explainer and doc sync: Implemented.
10. WO-010 pgvector + PostgreSQL automation: Implemented.
11. Frontend agent-playground modernization: Not implemented yet (discovery complete, work-order set defined in `workorders/frontend/`).
12. Document RAG foundation (`workorders/documents/WO-DOC-001-document-data-contracts-scope.md`): In progress (contracts, schema migrations, and API validation scaffolding implemented on February 14, 2026).
13. Document scope isolation and RLS (`workorders/documents/WO-DOC-002-storage-tenancy-rls.md`): In progress (scope resolution/policy modules, SQL RLS policies, scope-aware document manager queries, and API scope guards implemented on February 14, 2026).
14. Document file ingress and object lifecycle (`workorders/documents/WO-DOC-003-file-ingress-object-lifecycle.md`): In progress (streaming upload storage backend, digest/mime/dedupe workflow, storage object schema, lifecycle endpoints, and soft-delete/restore APIs implemented on February 14, 2026).
15. Document ingestion queue/workers/idempotency (`workorders/documents/WO-DOC-004-ingestion-queue-workers-idempotency.md`): In progress (persistent ingestion jobs/attempt tables, idempotent enqueue wiring, worker lease/heartbeat/retry loop, and ingestion job operator APIs implemented on February 14, 2026).
16. Document parser adapter framework and fallback routing (`workorders/documents/WO-DOC-005-parser-adapter-framework.md`): In progress (adapter interface, canonical parse contracts, deterministic router/fallback chain, worker parser invocation, and parser routing tests implemented on February 14, 2026).

### 2.2 Frontend Discovery Snapshot (February 8, 2026)

#### 2.2.1 Confirmed Implemented
1. Route + auth gating for web pages exists in `web_ui.py` (`/agent-playground`, `/knowledge-tools`, `/admin-tools`, `/community-engagement`, `/hydra-network`).
2. Web login/signup/session plumbing exists (`/login`, `/signup`, `/logout`, `session["member_id"]`).
3. Base panel framework exists (`templates/base-html.html`, `static/ui-managers.js`, `static/base-javascript.js`, `static/base-style.css`).
4. Basic profile update endpoint exists for email (`/profile/email`) via `MemberData.storeData(...)`.
5. Telegram miniapp auth ingress exists (`/miniapp-login`, `static/miniapp/index.html`).

#### 2.2.2 Confirmed Partial / Stubbed
1. `templates/agent-playground.html` is UI-demo only:
- local message rendering in browser
- no backend run execution call
- no streaming transport
- no trace/state/artifact ingestion
2. Admin/community pages are placeholders with static copy.
3. Knowledge tools page has list/editor shell but no completed save/update API flow.
4. Frontend managers are panel/menu utilities, not agent-runtime/data-runtime clients.

#### 2.2.3 Confirmed Missing (Implied Requirements)
1. No run lifecycle API for web (create run, stream events, cancel, resume, replay).
2. No append-only run event model surfaced to web UI.
3. No trace timeline UI, state snapshots, artifact panel, inspector panel.
4. No versioned frontend agent configuration objects (agent/tool/prompt/memory policy).
5. No web tool registry UI, schema-driven forms, sandbox approvals, or test harness.
6. No web evaluation workflows (datasets/evaluators/annotation queue/regression gates).
7. No capability-manifest endpoints to hydrate UI from schemas.
8. No workspace/project RBAC model in web tier.
9. No frontend observability dashboards for run metrics/regressions.

## 3. Current State Review

### 3.1 Runtime Components
- `telegram_ui.py`: most feature-rich interface (commands, moderation, chat flows, media flow, community logic).
- `cli_ui.py`: conversational local interface with model commands and orchestrator integration.
- `web_ui.py`: authentication, route rendering, miniapp login, and profile/email update endpoint.
- `x_ui.py`: X/Twitter entry path.
- `app.py`: single-entry bootstrap + watchdog + curses setup orchestration.

### 3.2 Agent and Tooling Stack
- `ConversationOrchestrator` executes analysis -> tool-calling -> chat response.
- `ModelRouter` centralizes model candidate resolution and fallback.
- `ToolRuntime` enforces arg validation/coercion, timeout/retry, missing-key behavior.
- `tool_registry` is canonical source of tool metadata + model tool schemas.

### 3.3 Data Layer and Retrieval
- PostgreSQL primary with optional fallback router behavior.
- pgvector-backed tables for chat/knowledge/spam embeddings.
- `UsageManager` stores inference counters/latency per prompt-response pair.
- Data model is currently chat-centric, not run-event-centric.

### 3.4 Configuration and Runtime Contract
- Runtime is driven by `config.json` + runtime hydration from `runtime` and env overrides.
- Setup/bootstrap writes config/env templates and validates required values.
- Community score gates now surfaced as editable config values and hydrated runtime values.

### 3.5 Web/CLI Alignment Findings
1. CLI path uses full orchestrator/tool routing chain, while Web playground currently does not execute that chain.
2. CLI has model awareness and command controls; Web has no comparable runtime control surfaces.
3. Observability data generated in backend (routing metadata, usage records) is not yet exposed to Web in structured run timelines.
4. Tool-calling customization/policy surfaces exist in backend files but are not represented in a frontend builder/editor.

## 4. North-Star Capability Requirements (Frontend Modernization)
These are now codified as explicit engineering requirements for the next cycle.

### 4.1 Core UX Primitives
1. Multi-pane workspace:
- Chat/Run panel
- Trace/Steps panel
- State panel
- Artifacts panel
- Inspector panel
2. Run modes:
- Chat
- Workflow/Graph
- Batch
- Compare
- Replay
3. Config-as-objects:
- agent/tool/policy/memory/prompt definitions
- versioning
- copy/export JSON/YAML

### 4.2 Runtime Capabilities
1. Agent composition: single + multi-agent + handoffs + planner/executor/verifier patterns.
2. Reliability controls: stop/resume/cancel, retries, timeouts, budgets, determinism knobs.
3. Guardrail hooks: pre/mid/post with trace visibility.
4. Canonical `AgentState` with provenance.

### 4.3 Tooling and Capability Marketplace
1. Tool registry UX with schema metadata, auth requirements, side-effect class, sensitivity class.
2. Tool sandbox controls and human approvals for mutating/risky calls.
3. Tool isolation harness (fixtures/golden outputs/contract checks).

### 4.4 Memory and Retrieval
1. Short-term trimming/compression strategy controls.
2. Long-term memory patterns (episodic/semantic/procedural pointers).
3. RAG ingestion/versioning and retrieval debugging (chunks/scores/rerank/provenance).
4. Memory write observability (author/confidence/TTL/evidence).

### 4.5 Observability and Debugging
1. Trace-first design with nested spans and searchable metadata.
2. Replay/time-travel with restored state and what-if edits.
3. Metrics dashboards for latency/tool errors/token proxy/success rates/regression trends.
4. Failure triage clustering and guided remediation suggestions.

### 4.6 Prompt Management and Evaluation
1. Prompt IDE with templates/vars/versioning/environments.
2. Prompt playground for rapid param experimentation.
3. Structured output validation + repair loops.
4. Dataset/evaluator pipelines + annotation queue + regression gates.

### 4.7 Collaboration, Governance, and Security
1. Workspaces/projects/RBAC.
2. Shared run permalinks and threaded review comments.
3. Secrets management, audit logs, policy-based tool/data access.
4. PII redaction hooks and environment isolation.

### 4.8 Ollama + Postgres Ops UX
1. Ollama model cockpit (list/pull/pin/metadata/runtime controls).
2. Compare harness for model behavior/tool behavior deltas.
3. Postgres persistence for immutable runs + append-only events + snapshots + artifacts.
4. JSONB-first storage with targeted indexes and tenancy boundaries.

### 4.9 Hydratable UI Requirement
Every subsystem must expose machine-readable capability manifests and JSON schemas so UI can render without hardcoded forms.

## 5. Gap Analysis (What Exists vs Target)

### 5.1 Core UX
- Current: Base panel shell exists.
- Gap: No runtime-bound multi-pane agent workspace.

### 5.2 Run Lifecycle and Modes
- Current: Orchestrator exists in backend and CLI.
- Gap: No web run API/events, no workflow/batch/compare/replay modes.

### 5.3 Trace/State/Replay
- Current: Backend logs and usage table exist.
- Gap: No step-event store and no replay/time-travel UI.

### 5.4 Tooling UX
- Current: Tool runtime/registry exists.
- Gap: No frontend tool registry editor/sandbox/test harness.

### 5.5 Memory/RAG UX
- Current: Knowledge retrieval managers exist.
- Gap: No ingestion/retrieval-debug UX and no memory strategy controls.

### 5.6 Prompt/Eval UX
- Current: Prompt and policy files exist on disk.
- Gap: No prompt IDE/versioning UI, no dataset/evaluator annotation workflows.

### 5.7 Collaboration/Security
- Current: Role list and member model exist.
- Gap: No workspace/RBAC model, no review workflow, no frontend governance controls.

### 5.8 Hydratable UI
- Current: No capability-manifest API contracts.
- Gap: Full manifest + schema renderer stack required.

## 6. Codebase TODO Backlog (Frontend and Integration)
These TODOs are mandatory inputs for the new frontend work-order stream.

1. TODO-FE-001: Add run lifecycle API (`create`, `stream`, `cancel`, `resume`, `replay`) in `web_ui.py` and backend service layer.
2. TODO-FE-002: Define append-only run-event schema and persistence (run events, span events, custom events).
3. TODO-FE-003: Add run-state snapshot model and checkpointing API.
4. TODO-FE-004: Implement multi-pane workspace template + route in web UI.
5. TODO-FE-005: Build trace timeline panel with nested spans and searchable filters.
6. TODO-FE-006: Build inspector panel for payload/token/latency/retry metadata.
7. TODO-FE-007: Build state panel with JSON diff-by-step and state rewind entrypoint.
8. TODO-FE-008: Build artifacts panel (files/tables/images/markdown/diffs).
9. TODO-FE-009: Implement chat/workflow/batch/compare/replay mode switching.
10. TODO-FE-010: Define and store agent definitions as versioned objects (JSON schema + changelog).
11. TODO-FE-011: Define and store prompt definitions as versioned objects with environment overlays.
12. TODO-FE-012: Define and store tool definitions with schema/auth/risk metadata.
13. TODO-FE-013: Expose backend policy + tool-runtime settings as editable objects for authorized users.
14. TODO-FE-014: Implement tool registry UI with schema-driven argument editors.
15. TODO-FE-015: Add tool sandbox policy controls and human approval queue for mutating tools.
16. TODO-FE-016: Add isolated tool test harness with fixtures/golden outputs/contract tests.
17. TODO-FE-017: Add memory strategy controls (trim/compress/episodic/semantic/procedural pointers).
18. TODO-FE-018: Add RAG ingestion UI with chunking/metadata/versioning/dedupe controls.
19. TODO-FE-019: Add retrieval-debug panel showing chunks/scores/query rewrite/reranker decisions.
20. TODO-FE-020: Add citation/provenance-required response mode and evidence visualization.
21. TODO-FE-021: Add metrics dashboard (latency, token proxy, tool errors, success/failure, eval trends).
22. TODO-FE-022: Add failure triage clustering (schema mismatch, timeout, retrieval failure, policy violation).
23. TODO-FE-023: Add prompt playground with model parameter sweep and side-by-side output diff.
24. TODO-FE-024: Add structured-output validators and repair-loop UX.
25. TODO-FE-025: Add dataset CRUD/versioning/tag slicing for evaluation.
26. TODO-FE-026: Add evaluator execution APIs (rule-based + LLM-judge) with batch jobs.
27. TODO-FE-027: Add annotation queue and reviewer assignment workflow.
28. TODO-FE-028: Add regression gates to block degraded prompt/tool changes.
29. TODO-FE-029: Add workspace/project scoping and RBAC enforcement in web routes/APIs.
30. TODO-FE-030: Add run permalink sharing and trace commenting/review threads.
31. TODO-FE-031: Add secret management boundaries (never expose plaintext in frontend/API responses).
32. TODO-FE-032: Add PII detection/redaction hooks for logs/artifacts/exports.
33. TODO-FE-033: Add audit log stream for run/config/policy/tool changes.
34. TODO-FE-034: Add Ollama cockpit (list/pull/pin/metadata/runtime knobs/compare harness).
35. TODO-FE-035: Add fallback chain editor (for schema repair escalation across models).
36. TODO-FE-036: Evolve Postgres schema to immutable runs + append-only events + snapshots + artifacts + versioned entities.
37. TODO-FE-037: Add JSONB indexes/search over run messages/tool outputs/tags/timestamps/model/tool IDs.
38. TODO-FE-038: Introduce capability-manifest endpoints for models/tools/agents/memory/evals/renderers.
39. TODO-FE-039: Implement frontend schema renderer for form/panel generation from manifests.
40. TODO-FE-040: Add agent-assisted UI composition hooks (widget proposals, evaluator suggestions, bug report packaging).

## 7. Phased Implementation Plan (Next Cycle)

### Phase FE-0: Contracts and Foundations
1. Define run/event/state/artifact API contracts.
2. Define capability manifest schema contract.
3. Define versioned object contracts (agents/tools/prompts/evals).

### Phase FE-1: Run + Trace Vertical Slice
1. Implement run lifecycle API.
2. Implement append-only event capture.
3. Implement chat+trace+inspector minimal web UI.
4. Add cancel and replay-from-start.

### Phase FE-2: State/Artifacts/Replay
1. Add snapshot persistence and state panel.
2. Add artifact persistence and panel renderers.
3. Add replay-from-step and state-edit replay path.

### Phase FE-3: Builder Surfaces
1. Agent builder with versioning.
2. Tool registry builder and schema-driven forms.
3. Prompt IDE and parameter playground.

### Phase FE-4: Memory/RAG/Eval
1. RAG ingestion and retrieval debugging.
2. Dataset/evaluator execution pipeline.
3. Annotation queue and regression dashboards.

### Phase FE-5: Governance and Collaboration
1. Workspaces/projects/RBAC.
2. Audit logs and review workflows.
3. Secret and redaction hardening.

### Phase FE-6: Ollama and Ops Cockpit
1. Model browser/pull/pin controls.
2. Compare harness and fallback-chain editor.
3. Run/resource controls and policy guardrails.

## 8. Acceptance Criteria (Frontend Modernization)
1. A user can create an agent config, run it, see streaming response, inspect trace/state, and replay from a selected step.
2. Tool calls are visible as first-class trace spans with args/results/errors and retry metadata.
3. A user can define/edit/version prompt + tool + agent configs from web UI and rerun without restarting services.
4. Dataset-based evaluation can be launched from web UI and produces persisted, queryable results.
5. Capability manifests are consumed by frontend schema renderer for at least tools, model params, and evaluator forms.
6. Workspace RBAC gates editing actions while preserving view-only access for non-editors.
7. Ollama model metadata and selected model tag are captured on every run record.
8. Run/event/state/artifact records are persisted in Postgres with query indexes suitable for timeline and search UX.

## 9. Testing Strategy (Frontend + Integration)
1. Unit tests:
- manifest schema validation
- renderer generation from schema
- run state transition reducers
- replay checkpoint selection logic
2. API integration tests:
- run lifecycle endpoints
- event append/read APIs
- cancel/resume/replay behavior
- RBAC enforcement across endpoints
3. E2E tests:
- create config -> run -> trace inspect -> replay
- tool error path with graceful degradation
- dataset eval run -> annotation queue -> regression report
4. Load tests:
- concurrent run streaming
- timeline query performance on indexed event tables

## 10. Documentation Sync Policy
On every frontend modernization merge:
1. Update `docs/master-engineering.md` first.
2. Update related `workorders/frontend/*.md` status second.
3. Update `readme.md` operator-facing setup/runtime changes third.
4. Add a `docs/CHANGELOG_ENGINEERING.md` entry fourth.
5. Run `docs/DOC_UPDATE_CHECKLIST.md` before merge.

## 11. Linked Work Orders
- Existing reliability track: `workorders/WO-001-...` through `workorders/WO-010-...`
- Frontend modernization track: `workorders/frontend/WO-FE-001-...` through `workorders/frontend/WO-FE-019-...`

## 12. External References
- Ollama: https://ollama.com/
- Ollama GitHub: https://github.com/ollama/ollama
- PostgreSQL: https://www.postgresql.org/
- pgvector: https://github.com/pgvector/pgvector
