# WO-001: Automatic Model Fallbacks

Source item: `docs/master-engineering.md` section 2 (planned upgrade #1), section 4.1, section 6 phase 2, section 7.1.

## 1. Verbose Description of Work Order
Implement a deterministic model-routing and fallback subsystem that can choose an Ollama model by capability (`analysis`, `tool`, `chat`, `embedding`, `multimodal`), validate that model availability against policy and runtime health, and gracefully fail over to secondary models when the primary model is unavailable, times out, or returns recoverable errors.

This work order must remove direct, static model selection in agents where possible and replace it with a centralized route request. The routing decision must be observable, logged, and traceable, so each conversation response can be audited for which model was selected and why fallback occurred.

This work order must also include setup-time endpoint selection behavior. The setup path must allow operators to provide a custom Ollama endpoint or accept the default local endpoint (`http://127.0.0.1:11434`). Endpoint selection rules must be explicit and shared with the model router so fallback is evaluated against the selected endpoint, not an implicit hardcoded host.

Expected behavior includes:
1. First-choice model is selected from policy/config.
2. If the model fails health or request execution, fallback candidates are attempted in priority order.
3. Fallback attempts stop on first success.
4. If all candidates fail, a controlled error object is returned and UI-facing handlers can degrade gracefully.
5. Setup flow supports custom Ollama endpoint entry and default-local endpoint fallback.
6. Endpoint precedence is deterministic (for example: setup-specified host -> config host -> default local host).

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/agents.py` (replace direct model assumptions in agent constructors and request calls)
2. `hypermindlabs/utils.py` (shared health-check helpers and structured error primitives)
3. New: `hypermindlabs/model_router.py` (route policy, retries, fallback order, decision metadata)
4. New: `tests/test_model_router.py` (unit tests for route and fallback decisions)
5. `scripts/setup_wizard.py` (or setup workflow entrypoint) to collect/validate custom Ollama endpoint and default-local behavior

Secondary/API-surface files:
1. `policies/agent/tool_calling_policy.json`
2. `policies/agent/message_analysis_policy.json`
3. `policies/agent/chat_conversation_policy.json`
4. `policies/agent/dev_test_policy.json`
5. `config.empty.json` (document fallback model arrays if schema changes)
6. `.env.example` (`OLLAMA_TOOL_MODEL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_MULTIMODAL_MODEL`, optional fallback vars)
7. `readme.md` (operator docs for fallback behavior)
8. `docs/master-engineering.md` (record implementation status)

Runtime variables and configuration surfaces:
1. `config.inference.embedding.model`
2. `config.inference.generate.model`
3. `config.inference.chat.model`
4. `config.inference.tool.model`
5. `config.inference.multimodal.model`
6. Policy `allowed_models`
7. `config.inference.embedding.url`
8. `config.inference.generate.url`
9. `config.inference.chat.url`
10. `config.inference.tool.url`
11. `config.inference.multimodal.url`
12. `.env`/`.env.example` `OLLAMA_HOST`, `OLLAMA_*` values

## 3. Success and Validation Metrics
Valid outcome:
1. At least one integration path demonstrates fallback from unavailable primary model to available secondary model.
2. Agent response completes without unhandled exception when first model fails.
3. Logs include: requested capability, primary model, fallback model (if used), failure reason class, retry count.
4. Unit tests cover:
   - Primary success
   - Primary failure + fallback success
   - All fallbacks exhausted
5. Existing chat/tool flows still work when primary model is healthy.
6. Setup validates and persists a custom Ollama endpoint when provided.
7. If no custom endpoint is provided, setup persists or uses default local endpoint (`http://127.0.0.1:11434`) consistently.

Partial outcome:
1. Fallback works only for some agent classes or capabilities.
2. Fallback works but lacks consistent metadata/logging.
3. Route decision is split between multiple files rather than centralized.
4. Endpoint behavior works in runtime code but is not integrated into setup flow.

Validation method:
1. Simulate nonexistent model in policy and verify fallback model is used.
2. Run smoke flow through `telegram_ui.py` or `cli_ui.py` with forced primary failure.
3. Confirm structured fallback events in logs.
4. Run setup flow once with custom endpoint and once with blank endpoint; verify both produce deterministic endpoint selection.

## 4. Potential Failure Modes
1. Infinite fallback loop due to cyclic candidate list.
2. Policy-allowed model list mismatched to available local models causing repeated startup degradation.
3. Inconsistent fallback behavior between streaming and non-streaming calls.
4. Silent fallback with no audit trail, making debugging impossible.
5. Capability-to-model mapping drift (tool agent accidentally routed through chat model without policy checks).
6. Setup accepts malformed custom endpoint and runtime repeatedly fails health checks.
7. Endpoint precedence ambiguity causes one capability to use a different host than others unintentionally.
8. Default local fallback endpoint is applied in some codepaths but ignored in others.
