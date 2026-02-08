# RYO Chat Master Engineering Document

## 1. Purpose
This document is the engineering baseline for the current RYO Chat codebase and the master plan for the next upgrade cycle.

It serves three functions:
1. Capture the current architecture and operational reality.
2. Define the target behavior for automation and graceful degradation.
3. Drive implementation order, acceptance criteria, and operator documentation updates.

## 2. Scope
This document covers:
- Multi-platform runtime (`cli_ui.py`, `web_ui.py`, `telegram_ui.py`, `x_ui.py`)
- Agent orchestration and tool-calling stack (`hypermindlabs/agents.py`)
- Data and config runtime (`hypermindlabs/utils.py`, `config.json`)
- Policy and system prompt stack (`policies/agent/*`)
- Planned upgrades:
1. Automatic model fallbacks
2. Automatic PostgreSQL instance generation for remote fallback with curses setup
3. Complete documentation and master engineering document
4. Graceful degradation for tools with missing APIs
5. Curses walkthrough for key ingress and model selection
6. Policy walkthrough and setup
7. Tool-calling stack review and improvements
8. Automatic ingress of Telegram keys
9. Master engineering documentation and prerequisite explainer
10. Automated pgvector + PostgreSQL setup

### 2.1 Implementation Status Snapshot (February 8, 2026)
1. Automatic model fallbacks: Partially implemented (`ModelRouter` integrated into core multi-agent chain; additional coverage and telemetry pending).
2. Automatic PostgreSQL fallback generation with setup flow: Partially implemented (`DatabaseRouter` integrated; fallback config support and curses setup flow added; sync strategy and orchestration coverage pending).
3. Complete documentation and master engineering documentation maintenance: Implemented in this cycle (checklist, changelog, and sync workflow artifacts added).
4. Graceful degradation for tools with missing APIs: Implemented (structured tool runtime added with missing-key handling, arg validation, timeout/retry bounds, and error envelopes).
5. Curses walkthrough for key ingress and model selection: Implemented (curses wizard with endpoint probing, model mapping, validation, optional `.env` updates, and resumable partial-state flow).
6. Policy walkthrough and setup: Planned.
7. Tool-calling stack review and improvements: Planned.
8. Automatic ingress of Telegram keys: Planned.
9. Master engineering documentation and prerequisite explainer: In progress.
10. Automated pgvector + PostgreSQL setup: Partially implemented (bootstrap script exists; wider orchestration integration pending).

## 3. Current State Review

### 3.1 Runtime Components
- `telegram_ui.py`: primary production-style interface (commands, moderation, chat flows, knowledge ingestion, tweet workflow, newsletter workflow).
- `web_ui.py`: Flask web interface and Telegram miniapp login validation.
- `cli_ui.py`: local conversational CLI with model switch/list commands.
- `x_ui.py`: X/Twitter integration entrypoint.

### 3.2 Agent and Tooling Stack
- Orchestrator (`ConversationOrchestrator`) runs:
1. Message analysis agent
2. Tool-calling agent
3. Chat-conversation agent
- Tool set currently wired:
1. `braveSearch`
2. `chatHistorySearch`
3. `knowledgeSearch`
4. `skipTools` behavior prompting
- Policies and system prompts are loaded from disk, with no centralized runtime manager.

### 3.3 Data Layer and Vector Retrieval
- PostgreSQL is the primary store.
- Managers auto-create tables on startup.
- `vector(768)` columns are used in chat history, knowledge, and spam tables.
- Vector search uses `<->` distance operations through pgvector-compatible types.

### 3.4 Configuration and Secrets
- Runtime config is loaded from `config.json` only (hard requirement).
- Keys and endpoints are mostly file-based (no environment merge at runtime).
- API credentials exist in config placeholders (`api_keys`, `twitter_keys`).

### 3.5 Current Gaps/Risks (Observed)
1. No startup migration/version system; table creation is distributed across managers.
2. No explicit `CREATE EXTENSION IF NOT EXISTS vector` path in runtime bootstrap.
3. Missing/failing external services (Ollama, Brave API, DB) are not consistently converted to graceful user-facing fallbacks.
4. No standard health-check gate for model availability before agent chain execution.
5. No central fallback routing for model/provider selection.
6. Policy loading assumes files exist and expected keys are valid.
7. Config template did not fully match runtime expectations (now corrected in `config.empty.json`).
8. `.env` contract was missing, making upcoming bootstrap automation unclear (now added as `.env.example` for planning).
9. Some feature paths are inconsistent with current agent signatures (notably parts of tweet flow), indicating need for integration hardening before expansion.

## 4. Target Architecture (Upgrade Direction)

### 4.1 Model Fallback Layer
Add a `ModelRouter` abstraction:
- Inputs: capability (`analysis`, `tool`, `chat`, `embedding`, `multimodal`), policy constraints, user override.
- Behavior:
1. Attempt primary model for capability.
2. On error/timeout/unavailable model, retry according to policy.
3. Automatically select next compatible fallback model.
4. Return structured metadata about fallback events for logs and usage tracking.
- Output contract:
1. Chosen model
2. Host/provider
3. Fallback reason
4. Retry count

### 4.2 PostgreSQL Remote-to-Local Fallback
Add `DatabaseRouter` with prioritized endpoints:
1. Remote primary (`config.database`)
2. Local fallback instance (auto-provisioned or pre-existing)

Behavior:
- Pre-flight connection test at startup.
- On remote failure:
1. switch to local fallback
2. surface warning state in UI/ops logs
3. continue service where possible
- Optionally enqueue sync jobs for eventual reconciliation (future phase).

### 4.3 pgvector/Postgres Automated Bootstrap
Provide automation scripts for:
1. Local Postgres provisioning
2. pgvector extension enablement
3. schema initialization
4. connectivity verification

Expected script outcomes:
- Idempotent execution.
- Clear terminal output for success/failure.
- Generated config suggestions for `config.json` and `.env`.

### 4.4 Curses Setup Wizard
Create a guided terminal wizard for first-run setup:
1. Collect Telegram values (bot name/id/token, owner metadata)
2. Collect DB values (remote primary + optional local fallback)
3. Collect API keys (Brave, optional X/Twitter)
4. Probe Ollama host
5. List available Ollama models and let user map models to capabilities
6. Write `config.json`
7. Optionally write `.env`

Graceful wizard behavior:
- Validate each step before continue.
- Allow skip for optional keys.
- Persist partial progress on exit.

### 4.5 Tool-Calling Degradation and Safety
Introduce a `ToolRuntime` layer:
- Schema validation for tool arguments.
- Per-tool timeout and retry policy.
- API-key presence checks before invocation.
- Circuit breaker behavior for repeated failures.
- Safe fallback response injected into agent context when tool unavailable.

Required user-facing behavior:
- No hard crash if Brave key missing.
- Model can proceed without tools when tool execution fails.

### 4.6 Policy Walkthrough and Setup
Implement `PolicyManager`:
- Validate policy JSON schema on load.
- Validate referenced model names against available models (warn vs fail by mode).
- Provide CLI/curses policy walkthrough:
1. select allowed models
2. set prompt override allowance
3. preview system prompt
4. save with validation report

## 5. Configuration and Environment Contract

### 5.1 Current Runtime Contract
- `config.json` is mandatory.
- Policy files in `policies/agent/` and `policies/agent/system_prompt/` are mandatory.
- `logs/` directory must exist before app start.

### 5.2 User-Editable Files
1. `config.json`
2. `.env` (for operator automation, planned bootstrap use)
3. `policies/agent/*.json`
4. `policies/agent/system_prompt/*.txt`

### 5.3 Required User Editing Rules
1. Never commit real secrets.
2. Keep model names valid for the target Ollama host.
3. Keep role lists consistent (`roles_list`) for Telegram flows.
4. If Brave tool is enabled, supply API key; otherwise disable tool at policy/runtime level.
5. Ensure DB endpoint points to pgvector-enabled PostgreSQL.

## 6. Implementation Plan

### Phase 0: Documentation + Config Contract (done in this cycle)
1. Master engineering doc created.
2. README expanded with setup, prerequisites, and references.
3. `config.empty.json` aligned with runtime keys.
4. `.env.example` added for bootstrap contract.

### Phase 1: Reliability Foundations
1. Add centralized health checks:
- Ollama host/model checks
- DB availability checks
- API-key presence checks
2. Add structured startup report.
3. Add unified error classes for degradation paths.

### Phase 2: Fallback Engines
1. Implement `ModelRouter`.
2. Implement `DatabaseRouter`.
3. Implement tool-level graceful fallback behavior.

### Phase 3: Setup Automation
1. Implement curses setup wizard.
2. Implement Postgres+pgvector local provisioning workflow.
3. Add config writer with backup and validation.

### Phase 4: Policy and Tooling Hardening
1. Implement `PolicyManager`.
2. Implement `ToolRuntime` schema enforcement and retries.
3. Add telemetry for fallback rates, tool failure rates, and model routing outcomes.

## 7. Acceptance Criteria

### 7.1 Automatic Model Fallbacks
- If primary model is unavailable, response still completes via fallback model.
- Response metadata records fallback event.
- No unhandled exception reaches UI handler.

### 7.2 Database Fallback
- If remote DB is unavailable at startup, system can switch to local fallback and continue serving at least core chat operations.
- Fallback state is visible in logs and operator status output.

### 7.3 Tool Degradation
- Missing Brave API key does not crash the conversation.
- Tool failure injects structured fallback context and conversation continues.

### 7.4 Curses Walkthrough
- First-time user can finish setup without manual file editing.
- Wizard validates required fields before save.
- Saved config starts at least one interface successfully.

### 7.5 Policy Walkthrough
- Invalid policy/model references are surfaced before runtime execution.
- Policy updates can be applied through guided flow and persisted correctly.

## 8. Testing Strategy
1. Unit tests:
- model routing decisions
- DB routing decisions
- tool runtime validation and fallback behavior
2. Integration tests:
- startup with healthy dependencies
- startup with missing Brave key
- startup with remote DB down + local DB up
- startup with missing model in policy
3. Smoke tests:
- `telegram_ui.py` start path
- `web_ui.py` start path
- `cli_ui.py` login + one prompt round trip

## 9. Documentation Sync Policy
On every major upgrade:
1. Update `docs/master-engineering.md` first.
2. Update `readme.md` setup sections second.
3. Update templates (`config.empty.json`, `.env.example`) third.
4. Run startup smoke checks before merge.
5. Add an entry to `docs/CHANGELOG_ENGINEERING.md`.
6. Run and complete `docs/DOC_UPDATE_CHECKLIST.md`.

## 10. External References
- Ollama: https://ollama.com/
- Ollama GitHub: https://github.com/ollama/ollama
- PostgreSQL: https://www.postgresql.org/
- pgvector: https://github.com/pgvector/pgvector
