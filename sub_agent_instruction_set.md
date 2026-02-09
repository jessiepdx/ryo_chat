# Sub-Agent Instruction Set - Prompt/Orchestration Hardening

Date: 2026-02-08
Owner: Core orchestration maintainer
Primary objective: Eliminate prompt/meta leakage and enforce strict stage contracts for Telegram flows.

## Global Rules For All Sub-Agents

1. Do not introduce new hardcoded model names or endpoint URLs.
2. Preserve existing runtime settings hydration behavior (`.env` + `config.json` + runtime defaults).
3. Keep changes backward-compatible with existing API surfaces unless explicitly noted.
4. Every sub-agent must provide:
   - changed files list
   - risk notes
   - validation output summary

---

## Sub-Agent A - Prompt Contract Refactor

Goal:
Rewrite stage prompt files so each stage has a strict contract and no internal self-reference.

Primary files:
- `policies/agent/system_prompt/message_analysis_sp.txt`
- `policies/agent/system_prompt/tool_calling_sp.txt`
- `policies/agent/system_prompt/chat_conversation_sp.txt`
- `policies/agent/system_prompt/dev_test_sp.txt`

Tasks:
1. Remove all references to "future agents", hidden reasoning, and orchestration internals.
2. Enforce strict JSON output contract for analysis stage.
3. Keep tool stage prompt policy-oriented and schema-driven (no stale static API docs).
4. Keep response stage natural-language only and explicitly ban internal disclosure by default.

Acceptance criteria:
1. All prompt files exist and are non-empty.
2. Analysis prompt schema is syntactically valid JSON example.
3. No prompt includes phrases like "one agent of many agents" or "future agents."

Validation commands:
```bash
rg -n "future agents|one agent of many agents|thoughts|chain-of-thought" policies/agent/system_prompt/*.txt
```

---

## Sub-Agent B - Orchestration Safety and Handoff

Goal:
Enforce instance-local state and typed/sanitized stage handoff.

Primary file:
- `hypermindlabs/agents.py`

Tasks:
1. Ensure mutable orchestrator state is instance scoped.
2. Parse and normalize analysis output to a typed dict before downstream use.
3. Pass sanitized analysis payload into tool stage and response stage.
4. Add final response sanitizer to remove internal meta artifacts when diagnostics were not requested.

Acceptance criteria:
1. No class-level mutable message buffers in orchestrator.
2. Tool stage no longer consumes raw analysis text blob.
3. Final response path includes sanitization guard.

Validation commands:
```bash
python3 -m py_compile hypermindlabs/agents.py
rg -n "_messages = list\\(\\)|response\\.sanitized|_normalize_analysis_payload|_sanitize_final_response" hypermindlabs/agents.py
```

---

## Sub-Agent C - Telegram Prompt Path Unification

Goal:
Reduce prompt drift by routing `/generate` through policy-managed defaults.

Primary file:
- `telegram_ui.py`

Tasks:
1. Add helper that resolves default generate system prompt from policy prompt files.
2. Stop concatenating legacy `config.defaults` prompt strings in `/generate`.
3. Keep support for optional user-supplied extra instructions.
4. Add stage label mapping for response sanitization event.

Acceptance criteria:
1. `/generate` uses policy-managed base system prompt.
2. Manual custom prompt remains supported as additive instruction.
3. Stage status display supports `response.sanitized`.

Validation commands:
```bash
python3 -m py_compile telegram_ui.py
rg -n "_default_generate_system_prompt|response\\.sanitized|Additional user instruction" telegram_ui.py
```

---

## Sub-Agent D - Regression Guard and Smoke Validation

Goal:
Perform targeted non-destructive validation.

Primary files:
- `hypermindlabs/agents.py`
- `telegram_ui.py`
- prompt files in `policies/agent/system_prompt/`

Tasks:
1. Run compile checks for modified Python modules.
2. Run focused unit/smoke tests that do not depend on external services when possible.
3. Capture and report any compatibility or import issues.

Acceptance criteria:
1. `py_compile` passes for touched Python files.
2. No syntax regressions in prompt/system files.

Validation commands:
```bash
python3 -m py_compile hypermindlabs/agents.py telegram_ui.py
```

---

## Merge Order

1. Sub-Agent A
2. Sub-Agent B
3. Sub-Agent C
4. Sub-Agent D

Rationale:
- Contracts first, runtime handoff second, interface wiring third, then verification.

