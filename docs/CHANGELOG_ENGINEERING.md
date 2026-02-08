# Engineering Changelog

This changelog tracks implementation deltas against `docs/master-engineering.md` and linked work orders.

## 2026-02-08

### WO-001: Automatic Model Fallbacks
Status: Partially implemented

Implemented:
1. Added centralized model routing/fallback in `hypermindlabs/model_router.py`.
2. Integrated routing into core multi-agent classes in `hypermindlabs/agents.py`.
3. Added endpoint-aware setup support via `scripts/setup_wizard.py`.
4. Added tests in `tests/test_model_router.py`.

Remaining:
1. Extend fallback/degradation behavior to legacy agent paths and tool-runtime hardening.
2. Add broader telemetry and integration tests across all interfaces.

### WO-002: Automatic PostgreSQL Fallback Instance Generation
Status: Partially implemented

Implemented:
1. Added DB primary/fallback routing in `hypermindlabs/database_router.py`.
2. Integrated route resolution in `ConfigManager` in `hypermindlabs/utils.py`.
3. Added startup route logging to `telegram_ui.py`, `web_ui.py`, `cli_ui.py`, and `x_ui.py`.
4. Extended setup workflow for DB fallback settings in `scripts/setup_wizard.py`.
5. Added pg bootstrap helper in `scripts/bootstrap_postgres.py`.
6. Added tests in `tests/test_database_router.py`.

Remaining:
1. Add runtime reconciliation/sync policy for fallback-write scenarios.
2. Add broader integration tests against live DB failover conditions.

### WO-003: Complete Documentation and Master Engineering Maintenance
Status: Implemented

Implemented:
1. Added per-work-order status snapshot in `docs/master-engineering.md`.
2. Added sync and process artifacts:
   - `docs/DOC_UPDATE_CHECKLIST.md`
   - `docs/CHANGELOG_ENGINEERING.md`
3. Added PR checklist template in `.github/PULL_REQUEST_TEMPLATE.md`.
4. Updated README links and doc maintenance guidance in `readme.md`.

### WO-004: Graceful Degradation for Tools with Missing APIs
Status: Implemented

Implemented:
1. Added structured tool runtime in `hypermindlabs/tool_runtime.py`.
2. Integrated runtime in `ToolCallingAgent` in `hypermindlabs/agents.py`.
3. Added pre-invocation API key checks (including Brave key handling).
4. Added argument validation and normalization for model-generated tool args.
5. Added bounded timeout/retry behavior for tool execution.
6. Added structured error envelopes so tool failures degrade without breaking orchestration.
7. Added tests in `tests/test_tool_runtime.py`.
8. Added policy-configurable tool runtime settings in `policies/agent/tool_calling_policy.json`.

Remaining:
1. Extend this runtime with policy-driven timeout/retry values.
2. Add integration tests that exercise real API/network failure scenarios in end-to-end chat flows.

### WO-005: Curses Walkthrough for Key Ingress and Model Selection
Status: Implemented

Implemented:
1. Expanded `scripts/setup_wizard.py` with a curses-first interactive setup flow for required Telegram, DB, API key, Ollama endpoint, and model mapping values.
2. Added graceful dependency degradation in the wizard when `curses` or `ollama` dependencies are unavailable.
3. Added resumable partial-progress persistence via `.setup_wizard_state.json` (configurable with `--state-path`) and automatic cleanup on successful save.
4. Added cancellation-safe behavior so user cancellation exits cleanly without writing `config.json`.
5. Added wizard validation/unit tests in `tests/test_setup_wizard_validation.py`.
6. Updated setup documentation in `readme.md` and engineering status snapshot in `docs/master-engineering.md`.

### WO-006: Policy Walkthrough and Setup
Status: Implemented

Implemented:
1. Added centralized policy/prompt validation and safe update management in `hypermindlabs/policy_manager.py`.
2. Added schema checks for `allow_custom_system_prompt` and `allowed_models` (plus optional `tool_runtime` shape validation).
3. Added prompt-file existence/readability validation before runtime execution.
4. Added endpoint-aware model compatibility validation with deterministic host precedence:
   - explicit override
   - configured `inference.*.url`
   - default local host
5. Added guided policy editing/validation workflow in `scripts/policy_wizard.py`.
6. Integrated runtime policy loading in `hypermindlabs/agents.py` through `PolicyManager`, including graceful defaults on invalid policy artifacts.
7. Added tests in `tests/test_policy_manager.py`.
8. Added operator-facing policy documentation in `docs/policy-guide.md` and updated README policy instructions.

### WO-007: Tool-Calling Stack Review and Improvements
Status: Implemented

Implemented:
1. Added canonical tool metadata/registration layer in `hypermindlabs/tool_registry.py`.
2. Updated `ToolCallingAgent` in `hypermindlabs/agents.py` to use the canonical registry for both model tool schemas and runtime registration.
3. Hardened tool-call parsing in `hypermindlabs/tool_runtime.py`:
   - supports nested/malformed argument payloads and JSON-string arguments
   - supports argument coercion hooks
   - supports strict unknown-argument rejection
4. Added structured parsing/execution path for raw model tool calls via `ToolRuntime.execute_tool_call(...)`.
5. Added policy behavior flags in `policies/agent/tool_calling_policy.json` (`reject_unknown_args`, `unknown_tool_behavior`).
6. Updated `policies/agent/system_prompt/tool_calling_sp.txt` to align prompt constraints with runtime schema rules.
7. Added stack-level tests in `tests/test_tool_calling_agent.py` for malformed args, nested args, unknown tools, and strict argument rejection.
8. Updated operator notes in `readme.md` and status snapshot in `docs/master-engineering.md`.

### WO-008: Automatic Ingress of Telegram Bot Keys
Status: Implemented

Implemented:
1. Added Telegram-only setup/rotation mode in `scripts/setup_wizard.py` (`--telegram-only`) for guided credential ingress without rewriting inference/database sections.
2. Added Telegram-specific config validation (`validate_required_telegram_config`) and merge path (`apply_telegram_state`) to support safe key rotation.
3. Updated env-write behavior so Telegram-only setup updates Telegram env keys while preserving existing `OLLAMA_HOST` and PostgreSQL env values.
4. Added runtime Telegram config validation helpers in `hypermindlabs/utils.py` (`ConfigManager.getTelegramConfigIssues`).
5. Hardened Telegram miniapp validation in `MemberManager.validateMiniappData` to gracefully handle missing payload fields and missing bot token.
6. Added startup guardrails:
   - `telegram_ui.py` now blocks startup with actionable error messaging when Telegram config is incomplete.
   - `web_ui.py` now reports configuration errors for miniapp login with explicit 4xx/5xx responses instead of unsafe failures.
7. Added tests in `tests/test_telegram_config_ingress.py` for Telegram ingress merge behavior and env preservation.
8. Updated operator documentation in `readme.md` and status snapshot in `docs/master-engineering.md`.
