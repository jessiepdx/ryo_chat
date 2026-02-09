# System Prompt Review - Telegram Agent Pipeline

Date: 2026-02-08
Scope: Stage prompts and orchestration used by Telegram flows, plus legacy prompt paths still active in Telegram commands.

## Executive Summary

The current Telegram multi-stage pipeline has a solid foundation (separate analysis/tool/response stages), but it is currently vulnerable to self-referential artifacts, prompt drift, and context leakage. The largest risks are:

1. Shared mutable class state in orchestrator (`_messages`) causing cross-run contamination risk.
2. Analysis prompt design that explicitly asks for internal "thoughts" and "future agents" framing, then feeds that text into final generation.
3. Mixed prompt sources (`policies/agent/system_prompt/*.txt` and `config.defaults`) causing inconsistent behavior between normal chat and `/generate`/legacy flows.
4. No strict schema validation/sanitization between stages, despite requiring JSON in analysis.

These issues explain the observed self-referential and meta-agent artifacts.

---

## Current Prompt Injection Surfaces

### 1) Stage Prompt Files (Policy-Managed)

- `policies/agent/system_prompt/message_analysis_sp.txt`
- `policies/agent/system_prompt/tool_calling_sp.txt`
- `policies/agent/system_prompt/chat_conversation_sp.txt`
- `policies/agent/system_prompt/dev_test_sp.txt`

Loaded via:
- `hypermindlabs/agents.py:1247`
- `hypermindlabs/policy_manager.py:392`
- fallback behavior: `hypermindlabs/policy_manager.py:29`

### 2) Legacy/Default Prompt Paths Still Active in Telegram

- `/generate` composes system prompt from config defaults:
  - `telegram_ui.py:625`
  - `telegram_ui.py:649`
  - defaults source: `config.json:42`
- legacy agent classes still hardcode defaults:
  - `hypermindlabs/agents.py:832`
  - `hypermindlabs/agents.py:961`
- tweet system prompt is class-level hardcoded:
  - `hypermindlabs/agents.py:1010`

### 3) Orchestration Wiring

- Stage chain:
  - analysis -> tools -> response (`hypermindlabs/agents.py:200`)
- analysis output is appended into final response context as a `tool` message:
  - `hypermindlabs/agents.py:259`

---

## Stage-by-Stage Critique

## Stage A: Message Analysis

Source:
- `policies/agent/system_prompt/message_analysis_sp.txt`
- invocation: `hypermindlabs/agents.py:665`

### Issues

1. Prompt requests internal "thoughts" and references "future agents" (`message_analysis_sp.txt:17`, `message_analysis_sp.txt:22`-`message_analysis_sp.txt:27`).
2. Output contract is malformed JSON (`message_analysis_sp.txt:35` has broken key quoting).
3. Runtime uses `format="json"` but does not parse and validate the object before downstream use (`hypermindlabs/agents.py:671`, `hypermindlabs/agents.py:230`).

### Why this is problematic (best practice)

- Best practice is to separate internal planning metadata from user-facing narrative and avoid exposing chain-of-thought style traces.
- Structured outputs should use strict machine-validated schemas between stages, not free-form JSON-like text.

### Recommended upgrade

Replace with strict, minimal schema:

```json
{
  "topic": "string",
  "intent": "string",
  "needs_tools": true,
  "tool_hints": ["knowledgeSearch"],
  "risk_flags": [],
  "response_style": {
    "tone": "friendly",
    "length": "short"
  },
  "context_summary": "string"
}
```

Remove `thoughts` entirely from the model contract.

---

## Stage B: Tool Calling

Source:
- `policies/agent/system_prompt/tool_calling_sp.txt`
- tool schema source of truth: `hypermindlabs/tool_registry.py:120`
- model tool payload generation: `hypermindlabs/tool_registry.py:197`

### Issues

1. Prompt hardcodes tool list text (`tool_calling_sp.txt:4`-`tool_calling_sp.txt:8`) while runtime already has canonical schema definitions.
2. This creates documentation/runtime drift risk as tools evolve.
3. Tool planner currently receives conversation messages, but not a strongly typed analysis object as planning input.

### Why this is problematic (best practice)

- Best practice is a single source of truth for tool interfaces (schema/registry driven).
- Tool-calling reliability improves when tool choice is driven by validated intent fields, not loosely interpreted free text.

### Recommended upgrade

1. Keep concise system prompt focused on policy, not enumerated tool docs.
2. Inject tool definitions only through runtime function schema.
3. Feed parsed analysis object (`needs_tools`, `tool_hints`) into tool stage as typed context.

---

## Stage C: Final Response

Source:
- `policies/agent/system_prompt/chat_conversation_sp.txt`
- invocation: `hypermindlabs/agents.py:790`

### Issues

1. Prompt says "one agent of many agents" (`chat_conversation_sp.txt:2`), which primes meta-role leakage.
2. Final stage receives raw analysis/tool artifacts as `tool` messages (`hypermindlabs/agents.py:259`, `hypermindlabs/agents.py:263`).
3. No explicit "do not reveal internals unless asked" constraint in final prompt.

### Why this is problematic (best practice)

- User-facing agent prompts should be persona/task focused and suppress internal orchestration narration by default.
- Multi-agent internals should be observable in logs/traces, not in default user reply content.

### Recommended upgrade

1. Rewrite prompt to remove multi-agent self-reference.
2. Add hard instruction: do not mention system prompts, stages, tools, or internal metadata unless user explicitly requests diagnostics.
3. Pass only sanitized structured summaries from prior stages.

---

## Cross-Cutting Orchestration Critique

Source:
- `hypermindlabs/agents.py:113` onward

### Issue 1: Shared class state for `_messages`

- `ConversationOrchestrator` declares `_messages` at class scope (`hypermindlabs/agents.py:114`).
- This risks cross-conversation contamination under concurrency and long-running process reuse.

### Recommendation

- Move `_messages` to instance scope (`self._messages = []` in `__init__`), same for any mutable stage state.

### Issue 2: Unvalidated stage handoff

- analysis output is concatenated string and directly appended (`hypermindlabs/agents.py:230`, `hypermindlabs/agents.py:259`).

### Recommendation

- Parse analysis JSON into typed object, validate required keys, reject/fallback on malformed payload.
- Provide explicit handoff DTO structure between stages.

### Issue 3: Internal content and user content not cleanly separated

- analysis/tool artifacts are directly in same message stream as user/assistant content.

### Recommendation

- Maintain separate channels:
  - `conversation_messages` (user/assistant)
  - `planner_state` (analysis/tool metadata)
- Only project a compact, sanitized subset into final generation.

---

## Telegram-Specific Prompt Path Risks

## `/generate` command path

Source:
- `telegram_ui.py:552`-`telegram_ui.py:679`

### Issues

1. Legacy direct prompt composition bypasses stage policies.
2. User-provided "system prompt" plus defaults can conflict with policy-managed behavior.
3. Prompt quality and safety become inconsistent with normal chat orchestration.

### Recommendation

1. Either:
   - route `/generate` through the same policy-managed pipeline with explicit mode flags, or
   - keep `/generate` but clearly mark as "raw mode" and apply strict guard rails and caps.
2. Prefer one canonical prompt loading path for Telegram.

---

## Concrete Upgrade Plan (Priority)

## P0 (Immediate)

1. Make orchestrator mutable state instance-scoped.
2. Replace analysis prompt contract with strict JSON schema and remove `thoughts`.
3. Parse/validate analysis output before passing to tools/response.
4. Rewrite chat response prompt to ban internal self-reference by default.
5. Add an explicit final-response sanitizer pass (strip obvious internal meta phrases if leaked).

## P1 (Short Term)

1. Unify Telegram `/generate` with policy-managed prompt system.
2. Move image/tweet prompt templates into policy-managed files (versioned like other stages).
3. Add prompt version IDs to run events for traceability.
4. Add unit tests for:
   - malformed analysis JSON fallback
   - no internal-stage leakage in final response
   - prompt loading fallback behavior.

## P2 (Stabilization)

1. Add prompt linting pipeline:
   - schema validity
   - anti-leak checks ("future agents", "I analyzed", etc.)
   - forbidden phrasing checks per stage.
2. Add "prompt diff + approval" workflow for production prompt changes.
3. Add eval set specifically for meta-leakage and style consistency across Telegram commands.

---

## Proposed Prompt Design Pattern (Recommended)

For each stage, enforce this template:

1. Role and objective (single sentence).
2. Allowed inputs (explicit fields).
3. Required output schema (strict JSON if machine-consumed).
4. Disallowed behaviors (especially internal disclosure).
5. Failure mode behavior (e.g., return `{"status":"unavailable"}`).

This is the most reliable pattern for multi-stage systems because it reduces ambiguity, keeps contracts machine-checkable, and prevents hidden planning text from leaking into user-facing replies.

---

## Suggested Success Metrics

1. Meta-leak rate:
   - `% of Telegram responses containing internal orchestration terms` (target near 0).
2. Schema conformance:
   - `% analysis outputs that parse and validate` (target >= 99%).
3. Cross-flow consistency:
   - `/generate` and normal chat style alignment score.
4. Incident rate:
   - count of malformed tool call args per 1k requests.
5. User-visible quality:
   - reduced reports of "self-referential" or "agent talking about itself."

---

## Notes on Best-Practice Basis

The recommendations above follow common production patterns for agentic systems:

1. Structured, validated inter-stage contracts over free-form text.
2. Strict separation of planner/internal state from user-facing response generation.
3. Single source of truth for tool schemas and prompt governance.
4. Explicit guard rails against internal implementation disclosure.
5. Versioned and testable prompt artifacts with replay/eval support.

