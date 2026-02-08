# WO-006: Policy Walkthrough and Setup

Source item: `docs/master-engineering.md` section 2 (planned upgrade #6), section 4.6, section 6 phase 4, section 7.5.

## 1. Verbose Description of Work Order
Build a policy-management workflow that validates policy files and system prompts before runtime execution, then provides a guided setup/edit experience for operators.

Current policy loading is file-based and assumes valid structure. This work order introduces strict validation and user-friendly editing of `allowed_models` and `allow_custom_system_prompt`, with compatibility checks against available models.

Policy validation and walkthrough must consume the same endpoint decision from setup (`WO-001`/`WO-005`), meaning model inventory checks should run against the selected custom endpoint or fallback default local endpoint, not a hidden hardcoded host.

Required capabilities:
1. Schema validation for each policy JSON.
2. Prompt file existence and readability checks.
3. Allowed-model validation against Ollama model list.
4. Guided editing flow (CLI/curses).
5. Save and rollback behavior on invalid edits.
6. Endpoint-aware model discovery with deterministic host precedence.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/agents.py` (`loadAgentPolicy`, `loadAgentSystemPrompt` call path hardening)
2. New: `hypermindlabs/policy_manager.py`
3. `policies/agent/*.json` (validated schema)
4. `policies/agent/system_prompt/*.txt`
5. New: `tests/test_policy_manager.py`

Secondary/API-surface files:
1. `readme.md` (policy setup instructions)
2. `docs/master-engineering.md` (status update)
3. New: `docs/policy-guide.md`
4. `scripts/setup_wizard.py` (optional policy setup integration)

Runtime variables and configuration surfaces:
1. Policy `allowed_models`
2. Policy `allow_custom_system_prompt`
3. `OLLAMA_HOST` / `inference.*.url` for model discovery
4. Endpoint precedence metadata from setup flow (`custom` vs `default local`)

## 3. Success and Validation Metrics
Valid outcome:
1. Invalid policy JSON is detected before agent execution.
2. Missing prompt files are surfaced with clear startup diagnostics.
3. Guided policy editor can update and save policy safely.
4. Model names in policy validated against live model inventory from the selected endpoint.
5. Policy walkthrough correctly handles both custom endpoint and default-local endpoint paths.

Partial outcome:
1. Validation exists but only runs for one policy type.
2. Validation warnings emitted but invalid policy still used silently.
3. Guided setup exists but cannot persist changes safely.
4. Policy validation probes a different host than setup/runtime host.

Validation method:
1. Add invalid model to policy and verify startup validation failure/warning mode.
2. Remove a prompt file and verify clear error path.
3. Edit policy through guided flow and run agent smoke test.
4. Run validation with custom endpoint and default endpoint paths; verify consistent host usage.

## 4. Potential Failure Modes
1. Strict validation blocks startup in environments that need warning-mode behavior.
2. Policy auto-fixes mutate operator intent without visibility.
3. Prompt-policy mismatches not detected.
4. Race conditions when multiple processes edit policy files.
5. Policy editor caches model inventory from a stale or wrong Ollama endpoint.
