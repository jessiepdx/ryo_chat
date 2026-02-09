# Personality + Narrative Adaptation Implementation Pack

## 1. Objective
Build a dedicated, production-safe personality adaptation subsystem for RYO Chat that supports:

1. User-defined personality/style directives (explicit preferences).
2. Adaptive per-user tuning of style and verbosity over time (implicit behavior signals).
3. Persistent narrative continuity (chunked summaries + rolling narrative memory).
4. Deterministic prompt injection into orchestration stages for Telegram/CLI/Web.

This system must improve writing alignment per user while remaining bounded, explainable, reversible, and safe.

## 2. Non-Negotiable Requirements

1. User explicit preferences always override inferred adaptations.
2. Adaptation must be gradual and bounded (no large style swings in one turn).
3. Narrative memory must be chunked and summarized; raw history must not be injected unbounded.
4. Injection payload must be compact, structured, and stage-visible for debugging.
5. No chain-of-thought exposure in user-facing responses.
6. Guest/transient sessions must not persist adaptation state.
7. All behavior must be runtime-configurable from `config.json` + `.env`.
8. All persistence must use SQL migrations in `db/migrations/` (no hardcoded schema strings).

## 3. Existing Integration Points (Current Repo)

1. Orchestration: `hypermindlabs/agents.py` (`ConversationOrchestrator` stage pipeline).
2. Runtime config hydration: `hypermindlabs/runtime_settings.py`.
3. Prompt policy loading: `hypermindlabs/policy_manager.py` and `policies/agent/system_prompt/*`.
4. DB migration framework: `hypermindlabs/utils.py` + `db/migrations/*.sql`.
5. Telegram stage UI: `telegram_ui.py` (`TelegramStageStatus` and stage metadata rendering).
6. Curses route config editor: `app.py` (`ROUTE_CONFIG_SPECS`, `_curses_route_config_workspace`).

## 4. Target Architecture

### 4.1 New Modules

1. `hypermindlabs/personality_store.py`
- Persistence manager for personality profiles, narrative chunks, and adaptation events.

2. `hypermindlabs/personality_engine.py`
- Core adaptation logic:
  - signal extraction
  - bounded parameter updates
  - merge of explicit directives + inferred adjustments
  - style profile materialization

3. `hypermindlabs/personality_injector.py`
- Creates compact prompt-safe injection payloads for analysis/tool/response stages.

4. `hypermindlabs/personality_rollup.py`
- Chunking + summarization logic for narrative continuity.

### 4.2 DB Migrations

Add migrations:

1. `086_member_personality_profile.sql`
- Per-member canonical profile (explicit + adaptive state).

2. `087_member_narrative_chunks.sql`
- Append-only narrative chunks with summary text and token/char budget metadata.

3. `088_member_personality_events.sql`
- Adaptation/audit events (why and how profile changed).

Add these migration filenames to startup migration lists in `hypermindlabs/utils.py`.

## 5. Data Model Specification

### 5.1 `member_personality_profile`

Required fields:

1. `profile_id` (PK)
2. `member_id` (FK -> `member_data.member_id`, unique)
3. `explicit_directive_json` (JSONB)
4. `adaptive_state_json` (JSONB)
5. `effective_profile_json` (JSONB)
6. `profile_version` (int)
7. `locked_fields_json` (JSONB)  // fields user hard-locked
8. `updated_at` (timestamp)
9. `created_at` (timestamp)

### 5.2 `member_narrative_chunks`

Required fields:

1. `chunk_id` (PK)
2. `member_id` (FK)
3. `chunk_index` (int)
4. `source_turn_start_id` (nullable)
5. `source_turn_end_id` (nullable)
6. `summary_text` (text)
7. `summary_json` (JSONB)
8. `compression_ratio` (numeric)
9. `updated_at` (timestamp)
10. `created_at` (timestamp)

### 5.3 `member_personality_events`

Required fields:

1. `event_id` (PK)
2. `member_id` (FK)
3. `event_type` (`explicit_update`, `adaptive_update`, `rollup`, `reset`)
4. `before_json` (JSONB)
5. `after_json` (JSONB)
6. `reason_code` (text)
7. `reason_detail` (text)
8. `created_at` (timestamp)

## 6. Effective Personality Schema

Use this JSON contract across modules:

```json
{
  "schema": "ryo.personality_profile.v1",
  "member_id": 0,
  "explicit": {
    "tone": "friendly|professional|neutral|energetic|direct",
    "verbosity": "brief|standard|detailed",
    "format": "plain|markdown_light",
    "reading_level": "simple|moderate|advanced",
    "humor": "low|medium|high",
    "emoji": "off|minimal|normal",
    "language": "en",
    "hard_constraints": [],
    "locked_fields": []
  },
  "adaptive": {
    "tone": "friendly",
    "verbosity": "brief",
    "reading_level": "moderate",
    "confidence": 0.0,
    "turns_observed": 0,
    "last_reason": ""
  },
  "narrative": {
    "active_summary": "",
    "chunk_count": 0,
    "last_chunk_index": 0,
    "last_rollup_at": ""
  },
  "effective": {
    "tone": "friendly",
    "verbosity": "brief",
    "reading_level": "moderate",
    "format_rules": [],
    "behavior_rules": []
  }
}
```

## 7. Adaptation Algorithm Directives

### 7.1 Signal Extraction (Per Turn)

Extract from user turn + short context:

1. `message_length_chars`
2. `message_length_tokens` (approx)
3. `question_density`
4. `instruction_density`
5. `complexity_hint` (simple/moderate/advanced heuristic)
6. `verbosity_preference_hint` (explicit short/long ask signals)
7. `repair_signal` (user says too long, too short, confusing, etc.)
8. `topic_shift_intensity` (reuse existing topic transition output)

### 7.2 Adaptation Rules

1. Compute weighted adaptation deltas only when:
- explicit lock for that field is absent
- confidence above threshold
- minimum observation window reached

2. Clamp deltas:
- max one level shift per `N` turns (runtime setting).

3. Prefer stability:
- decay to previous effective values when contradictory signals are sparse.

4. Treat explicit user commands as hard writes:
- `/style`, `/verbosity`, etc. update explicit directive + lock optional fields.

### 7.3 Narrative Chunking Rules

1. Maintain rolling per-user narrative buffer from recent turns.
2. Roll up when thresholds reached:
- turns >= `rollup_turn_threshold`
- chars >= `rollup_char_threshold`
- or forced on process completion.
3. Summarize buffer into one chunk:
- preserve key commitments, preferences, unresolved threads.
4. Keep bounded number of chunks in active window; older chunks compressed to macro-summary.

## 8. Orchestrator Integration (Mandatory)

### 8.1 New Stages in `ConversationOrchestrator`

Insert stages in `hypermindlabs/agents.py`:

1. `persona.load`
- Load/resolve profile + active narrative summary.

2. `persona.inject`
- Inject compact directive payload into stage messages (analysis/tool/response).

3. `persona.adapt`
- After response generation, compute adaptation delta and persist.

4. `persona.rollup`
- If threshold reached, create/refresh narrative chunk summary.

### 8.2 Injection Placement

Inject as tool/system-side structured payload after known context and before analysis + response model calls.

Use compact block:

```json
{
  "tool_name": "Personality Context",
  "tool_results": {
    "schema": "ryo.personality_injection.v1",
    "effective_style": { "...": "..." },
    "narrative_summary": "...",
    "directive_rules": ["..."],
    "safety_rules": ["never expose internal reasoning"]
  }
}
```

### 8.3 Runtime Guards

1. Skip adaptation for guest sessions.
2. Skip adaptation if DB unavailable, but continue response generation.
3. Hard cap injection chars/tokens using runtime settings.
4. Emit stage metadata only (do not expose full internal payload by default).

## 9. Prompt/Policy Updates

### 9.1 `policies/agent/system_prompt/message_analysis_sp.txt`

Add required output fields:

1. `persona_adjustment_request`:
- optional explicit signal classification from user intent.

2. `response_style` must honor injected `effective_style`.

### 9.2 `policies/agent/system_prompt/chat_conversation_sp.txt`

Add explicit constraints:

1. Follow `Personality Context` style and narrative continuity.
2. Keep response aligned to selected verbosity and reading-level.
3. Never expose orchestration internals.

### 9.3 `policies/agent/system_prompt/tool_calling_sp.txt`

Add guard:

1. Tool planning must not overwrite personality; only explicit personality tools can modify profile.

## 10. New Tools (Optional but Recommended)

Register tools in `hypermindlabs/tool_registry.py`:

1. `getPersonalityProfile`
2. `updatePersonalityDirective`
3. `resetPersonalityAdaptiveState`
4. `listNarrativeChunks`

Use strict argument schemas and per-member authorization.

## 11. Runtime Settings + Config Surface

### 11.1 `hypermindlabs/runtime_settings.py`

Add `runtime.personality` section:

1. `enabled` (bool)
2. `adaptive_enabled` (bool)
3. `narrative_enabled` (bool)
4. `max_injection_chars` (int)
5. `rollup_turn_threshold` (int)
6. `rollup_char_threshold` (int)
7. `max_active_chunks` (int)
8. `adaptation_min_turns` (int)
9. `adaptation_max_step_per_window` (int)
10. `adaptation_window_turns` (int)
11. `default_tone` (string)
12. `default_verbosity` (string)
13. `default_reading_level` (string)

Hydrate into:

1. `config.empty.json`
2. `.env.example` with `RYO_PERSONALITY_*` keys
3. `app.py` curses route configuration category.

### 11.2 Curses Configuration in `app.py`

Add category under Telegram route:

1. Personality engine toggles.
2. Default style values.
3. Adaptation thresholds.
4. Narrative rollup thresholds.
5. Injection budget limits.

All edits must remain in curses workflows (arrow/enter controls where choices apply).

## 12. Telegram/Web/CLI UX Integration

### 12.1 Telegram Commands

Add command handlers:

1. `/style`
2. `/verbosity`
3. `/tone`
4. `/persona_reset`
5. `/persona_show`

Behavior:

1. Write explicit profile directives.
2. Optional lock/unlock fields.
3. Confirm effective profile summary to user.

### 12.2 Web UI

Add profile settings panel:

1. explicit style controls
2. adaptation toggle
3. narrative summary preview
4. reset adaptive state

### 12.3 CLI

Add text commands:

1. `/style ...`
2. `/persona show`
3. `/persona reset`

## 13. Observability and Debugging

### 13.1 Stage Events

Emit events with compact metadata:

1. `persona.load` (`profile_version`, `adaptive_enabled`)
2. `persona.inject` (`injection_chars`, `narrative_chunk_count`)
3. `persona.adapt` (`fields_changed`, `confidence`, `reason_code`)
4. `persona.rollup` (`chunks_added`, `compression_ratio`)

### 13.2 Logging

Log reasons and deltas, never raw sensitive user content beyond bounded excerpts.

### 13.3 Failure Handling

If personality subsystem fails:

1. Keep response pipeline alive.
2. Emit `persona.error` stage.
3. Fallback to default style profile.

## 14. Sub-Agent Work Breakdown

## Sub-Agent A: Persistence + Migrations

Deliverables:

1. `086/087/088` SQL migrations.
2. `PersonalityStoreManager` in `hypermindlabs/personality_store.py`.
3. Startup migration registration in `hypermindlabs/utils.py`.

Validation:

1. Fresh DB bootstrap applies migrations.
2. Existing DB upgrade is idempotent.
3. CRUD tests pass.

## Sub-Agent B: Runtime + Config Plumbing

Deliverables:

1. Runtime defaults and env overrides in `hypermindlabs/runtime_settings.py`.
2. Add config template keys in `config.empty.json` + `.env.example`.
3. Curses settings integration in `app.py`.

Validation:

1. `build_runtime_settings()` covers all keys.
2. Curses saves/restores values without crashing.

## Sub-Agent C: Engine + Narrative Logic

Deliverables:

1. `hypermindlabs/personality_engine.py`
2. `hypermindlabs/personality_rollup.py`
3. deterministic adaptation + bounded rollup logic.

Validation:

1. Unit tests for adaptation clamps and lock behavior.
2. Unit tests for narrative chunking and compression thresholds.

## Sub-Agent D: Orchestrator Injection

Deliverables:

1. Integrate load/inject/adapt/rollup stages in `hypermindlabs/agents.py`.
2. Wire profile loading by `member_id`.
3. Ensure guest-mode bypass.

Validation:

1. Stage events appear in Telegram status debug.
2. Injection stays under configured size.
3. No regression in tool calling or response generation.

## Sub-Agent E: Prompt/Policy Contract

Deliverables:

1. Update `message_analysis_sp.txt`, `chat_conversation_sp.txt`, `tool_calling_sp.txt`.
2. Keep strict JSON schema validity for analysis outputs.

Validation:

1. Analysis payload parser accepts new fields.
2. No malformed JSON / schema break regressions.

## Sub-Agent F: UX + Commands

Deliverables:

1. Telegram persona commands.
2. CLI persona commands.
3. Web profile controls (minimal first pass).

Validation:

1. Explicit preference updates reflect in subsequent responses.
2. Reset clears adaptive but preserves explicit profile unless requested.

## Sub-Agent G: Test + QA + Rollout

Deliverables:

1. Unit tests for each new module.
2. Integration tests for stage pipeline.
3. Rollout flags and fallback behavior checklist.

Validation:

1. p95 latency regression under configured budget.
2. No hard crash path when DB/personality tables unavailable.
3. Deterministic fallback to defaults.

## 15. Definition of Done

Feature is complete only when all are true:

1. Personality directives can be explicitly set per user and persist.
2. Adaptive tuning updates style gradually from behavior signals.
3. Narrative chunks roll up and inject as compact continuity context.
4. Orchestrator stages show persona load/inject/adapt/rollup lifecycle.
5. Telegram/CLI/Web can inspect and modify profile settings.
6. Runtime and env knobs control all critical behavior.
7. Full test suite additions pass and no stage regression is introduced.

## 16. High-Risk Failure Modes to Prevent

1. Prompt bloat from unbounded narrative injection.
2. Thrashing style changes turn-to-turn.
3. Adaptive overrides ignoring explicit user locks.
4. Crashes on missing `member_id`/guest mode.
5. DB write failures blocking response generation.
6. Exposure of internal reasoning in user-facing text.

## 17. Recommended Implementation Order

1. Sub-Agent A (schema/store)
2. Sub-Agent B (runtime/config)
3. Sub-Agent C (engine/rollup)
4. Sub-Agent D (orchestrator integration)
5. Sub-Agent E (prompt contract)
6. Sub-Agent F (UX commands)
7. Sub-Agent G (hardening, tests, rollout)

