# WO-004: Graceful Degradation for Tools with Missing APIs

Source item: `docs/master-engineering.md` section 2 (planned upgrade #4), section 4.5, section 6 phase 1-2, section 7.3.

## 1. Verbose Description of Work Order
Implement standardized tool failure handling so missing credentials, API outages, malformed tool arguments, or timeout conditions do not crash the orchestration flow.

The immediate priority is `braveSearch`, but the design must generalize for all tools. Tool failures should produce structured fallback results that can be passed to downstream agents as context while preserving conversation continuity.

Behavior requirements:
1. Pre-invocation check for required secrets and config.
2. Argument schema validation before invocation.
3. Timeout and bounded retry policy per tool.
4. Structured error response object with safe message and diagnostics.
5. Orchestrator continues to response generation when tools fail.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/agents.py` (tool invocation path in `ToolCallingAgent.generateResponse`)
2. New: `hypermindlabs/tool_runtime.py` (validation, timeout, retry, fallback envelope)
3. New: `tests/test_tool_runtime.py`

Secondary/API-surface files:
1. `hypermindlabs/utils.py` (error classes/log helpers)
2. `policies/agent/tool_calling_policy.json` (optional tool behavior flags)
3. `config.empty.json` (`api_keys.brave_search` semantics)
4. `.env.example` (`BRAVE_SEARCH_API_KEY`)
5. `readme.md` (tool degradation behavior and key requirements)

Runtime variables and configuration surfaces:
1. `config.api_keys.brave_search`
2. `.env` `BRAVE_SEARCH_API_KEY`
3. Tool-call schema (`queryString`, `count`)

## 3. Success and Validation Metrics
Valid outcome:
1. Missing Brave key no longer produces unhandled exception.
2. Tool failure produces structured fallback payload consumed by orchestrator.
3. Chat response still completes when tools fail.
4. Unit tests cover missing key, timeout, bad arguments, and successful invocation.

Partial outcome:
1. Some failures handled, but key-missing or timeout still crashes path.
2. Errors handled but agent chain stops early instead of continuing.
3. Tool fallback payload exists but format is inconsistent across tools.

Validation method:
1. Run conversation with empty Brave key.
2. Inject timeout against Brave endpoint.
3. Confirm response continuity and log diagnostics.

## 4. Potential Failure Modes
1. Tool exceptions swallowed without observable logs.
2. Retry storm against failing external APIs.
3. Tool fallback payload too verbose and leaks secrets.
4. Tool argument coercion bypasses validation and causes unexpected request behavior.
