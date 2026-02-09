# Earth Tool-Calling Reference (For `ryo_chat`)

## Purpose
This document explains how tool-calling is implemented in:

- `/home/robit/Documents/repositories/noclip-unified/earth`

The goal is to give future `ryo_chat` agents an implementation-ready, model-agnostic pattern they can reuse.

## Scope
This analysis covers:

- In-browser agent contract (system prompt + JSON command protocol)
- Function calling/tool execution loop in frontend runtime
- Context injection from live page/runtime state
- Streaming transport integration and command application timing

It does **not** include backend route implementations for `chat.begin` / `/api/chat/*`, because those server handlers are not present in this repo snapshot. Only client-side contracts are visible.

## Executive Architecture
Earth uses **application-level structured command execution**, not provider-native function calling:

1. Model is instructed to return strict JSON (`reply`, `command`, `tags`).
2. Frontend parses JSON and normalizes command shape.
3. Frontend executes commands against exposed runtime APIs (`sceneApi`).
4. Tool outputs are fed back as special user messages with a reserved prefix.
5. Model uses tool result message in the next turn to continue reasoning.

This pattern is model-agnostic and works even if the model has no native tool-call API.

## Primary Files
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentPrompt.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentCommandProcessor.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/scene/PlanetScene.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatApiService.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/transport/types.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/modules/geocoding-service.js`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/ui/ConfigSidebar.js`

---

## 1) Prompt Contract and Command Schema

### What Earth does
- Defines a reserved tool-result prefix:
  - `AGENT_TOOL_MESSAGE_PREFIX = '__noclip_tool_result__'`
  - Source: `AgentPrompt.js:12`
- Defines allowed intents and slot semantics:
  - Source: `AgentPrompt.js:14-90`
- Builds a strict system prompt requiring JSON-only output:
  - Source: `AgentPrompt.js:139-186`

### Required output format
The model is instructed to produce:

- `reply` (user-facing text)
- `command` (or `null`) with:
  - `intent`
  - `slots`
  - `confidence`
  - `requiresConfirmation`
  - `followUpQuestion`
- `tags`
- `followUpQuestion`

Reference: `AgentPrompt.js:142-154`

### Tool-loop instruction strategy
Prompt tells model:

- Tool results will come back as user messages prefixed with `__noclip_tool_result__`
- Tool message format is `<prefix> <tool_name> <json>`

Reference: `AgentPrompt.js:161-164`

### Why it matters for `ryo_chat`
This creates deterministic machine-parsable control without requiring Ollama-native tool calls.

---

## 2) Runtime Action Execution (Function Calling Layer)

### What Earth does
`applyAgentCommand(...)` is the executor for structured commands.

Reference: `AgentCommandProcessor.js:247-324`

It:

- Validates command shape
- Switches by `intent`
- Applies mutations via runtime APIs (`sceneApi`)
- Returns normalized execution result:
  - `applied`
  - `actionTags`
  - `tuneableTags`
  - `errors`
- Emits `agent:action` when applied

Reference: `AgentCommandProcessor.js:248-253`, `AgentCommandProcessor.js:319-321`

### Supported intents and behavior
- `set_tuneables` -> patch tuneables tree
  - `AgentCommandProcessor.js:265-279`
- `set_location` -> camera relocation
  - `AgentCommandProcessor.js:113-177`, `:281-289`
- `set_altitude`
  - `AgentCommandProcessor.js:179-211`, `:290-295`
- `set_camera_mode`
  - `AgentCommandProcessor.js:213-221`, `:296-302`
- `set_overlay_center`
  - `AgentCommandProcessor.js:223-245`, `:303-309`

### Robustness details worth copying
- Slot alias handling (e.g., lat/lon variants)
- Unit coercion for altitude/delta (m/km/ft)
- Overlay key normalization (`flight` -> `flights`)

Reference: `AgentCommandProcessor.js:35-46`, `:65-101`, `:114-131`

---

## 3) Tool Calling Loop (Model-Agnostic)

### Core mechanism
Tool results are represented as synthetic user messages.

References:
- Prefix detection: `ChatPanel.js:3565-3568`
- Tool message construction: `ChatPanel.js:3570-3578`
- Send tool message as new `chat.begin`: `ChatPanel.js:3384-3415`

### Place-search as concrete example
Earth uses one built-in tool loop pattern (`place_search`):

1. Model emits `command.intent = "search_places"`.
   - Parsing/applier path: `ChatPanel.js:4944-4953`
2. Frontend runs tool (`_runPlaceSearch`) via:
   - `configSidebar.searchPlacesFromAgent(...)` or `GeocodingService.search(...)`
   - `ChatPanel.js:3673-3679`
3. Tool output payload is serialized and sent back:
   - `_sendAgentToolMessage(PLACE_SEARCH_TOOL, payload)`
   - `ChatPanel.js:3707-3713`
4. Next model turn sees prefixed tool result and can issue `select_place`.
5. `select_place` is applied by resolving result and navigating:
   - `ChatPanel.js:3759-3800`

### Why this pattern is strong
- Works for any model that can output JSON.
- Decouples tool execution from LLM provider-specific APIs.
- Keeps full trace in transcript via deterministic tool-result messages.

---

## 4) Streaming + Parsing + Apply Timing

### Transport event flow
- Client sends `chat.begin`
- Receives streamed `chat.chunk`
- Finalizes on `chat.done`

References:
- Event constants: `transport/types.js:119-121`
- Send begin: `ChatPanel.js:3333-3338`
- Chunk handler: `ChatPanel.js:4744-4839`
- Done handler: `ChatPanel.js:4841-4978`

### Parsing strategy
On `chat.done`, client:

- Takes final message content
- Tries parse JSON payload (`_tryParseAgentPayload`)
- Falls back to brace-sliced parse when raw text has wrappers
- Extracts `reply`, `command`, `tags`

Reference: `ChatPanel.js:3802-3834`, `ChatPanel.js:4927-4942`

### Apply strategy
If command exists and `requiresConfirmation` is false:

- Auto-apply command
- Add tags
- Run tool loop if `search_places`

Reference: `ChatPanel.js:4944-4969`

### Important note
Tool messages are filtered from visible history so users only see meaningful conversational messages:

- On load: `ChatPanel.js:3193`
- On render: `ChatPanel.js:3482`
- On ingestion: `ChatPanel.js:4501`

---

## 5) Context Injection and UI-State Exposure

### System message composition
`_buildSystemMessage()` layers:

- Base agent prompt from `buildAgentSystemPrompt(...)`
- User-configured system template with token replacement
- Selected HUD context key-values
- Scene-selected objects
- Active place-search context

Reference: `ChatPanel.js:2857-2902`

### Where context values come from
- HUD store value/data channels:
  - `ChatPanel.js:2825-2831`
- Available context keys come from `HUD_ITEMS`:
  - `ChatPanel.js:2510-2550`
  - `modules/hud-info-store.js:16-149`

### Scene selection bridge
- Chat panel emits selected scene keys:
  - `ChatPanel.js:2647-2656`
- Scene listens and applies selection to geojson layers:
  - listener: `PlanetScene.js:453-455`
  - application: `PlanetScene.js:3350-3359`

### Why this matters for `ryo_chat`
Earth demonstrates a clean way to expose page/runtime state to agents without hard-coding all prompt text.

---

## 6) Exposed Runtime Function Surface (Equivalent of “tools”)

Earth exposes runtime capabilities via `scene.api` and then executes commands against it.

References:
- API facade comments: `PlanetScene.js:5-14`
- Overlay/domain methods: `PlanetScene.js:1248-1338`
- Camera facade: `PlanetScene.js:1445-1453`

Key domains callable by agent executor:

- `tuneables.set/get`
- `camera` and `surfaceController`
- `buildings/flights/satellites/maritime/incidents/infrastructure`
- `events.emit/on/off`

This is effectively “tool exposure to page components” via controlled API wrappers.

---

## 7) Session/Model/Channel Lifecycle

### Session and model management
- REST session CRUD via `ChatApiService`
  - `ChatApiService.js:28-122`
- Model discovery:
  - `ChatApiService.js:201-208`
- Session model updates debounced:
  - `ChatPanel.js:2448-2475`

### Realtime channel model
- Session-scoped channel subscription:
  - subscribe/unsubscribe to `agent.session.<id>`
  - `ChatPanel.js:4722-4741`

This keeps push messages isolated by agent session.

---

## 8) Safety and Confirmation Pattern

Earth protocol has first-class confirmation controls:

- `command.requiresConfirmation`
- `command.followUpQuestion`

Normalization:
- `ChatPanel.js:3532-3555`

Execution gate:
- Auto-apply only if `!requiresConfirmation`
- `ChatPanel.js:4946-4969`

This is a useful pattern for sensitive tool actions in `ryo_chat`.

---

## 9) What To Reuse in `ryo_chat`

### Recommended direct ports
1. JSON command envelope in system prompt (model-agnostic).
2. Tool-result injection protocol (`__tool_result__ <tool> <json>` equivalent).
3. Parser + normalizer + executor split:
   - parse model output
   - normalize command
   - execute via explicit capability registry
4. Session-scoped streaming with begin/chunk/done.
5. `requiresConfirmation` gate before mutating actions.
6. Tag emission for observability (`actionTags`, `toolTags`, `memoryTags` in `ryo_chat`).

### Architecture mapping suggestion
For `ryo_chat` Python stack:

- `PromptBuilder`:
  - emits strict JSON contract + available intents/tools schema
- `CommandParser`:
  - tolerant parse + schema validation
- `CommandExecutor`:
  - dispatch by intent, side-effect policy hooks
- `ToolBus`:
  - executes tool
  - serializes tool result to synthetic context message
- `StageOrchestrator`:
  - streaming lifecycle + progress events for Telegram/UI

---

## 10) Gaps / Risks in Earth Pattern (and how to improve in `ryo_chat`)

1. JSON parsing is tolerant but not schema-validated.
   - Add strict JSON Schema validation in `ryo_chat`.
2. Tool protocol uses plain text prefix.
   - Keep prefix for compatibility, but store structured metadata alongside.
3. Server route behavior is external to this repo.
   - Define explicit contract tests in `ryo_chat` for begin/chunk/done payloads.
4. Command dedupe/idempotency is limited.
   - Add command IDs and idempotency keys for retries.

---

## 11) Reference Index

### Prompt + contract
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentPrompt.js:12`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentPrompt.js:14`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentPrompt.js:139`

### Command execution
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentCommandProcessor.js:247`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentCommandProcessor.js:264`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/agent/AgentCommandProcessor.js:319`

### Chat orchestration and tool loop
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3313`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3384`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3565`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3681`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3759`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:3802`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:4744`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:4841`

### Runtime API exposure
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/scene/PlanetScene.js:1248`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/scene/PlanetScene.js:1265`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/scene/PlanetScene.js:1445`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/scene/PlanetScene.js:3350`

### Context system
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:2857`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatPanel.js:2503`
- `/home/robit/Documents/repositories/noclip-unified/earth/modules/hud-info-store.js:16`

### Geocoding tool implementation
- `/home/robit/Documents/repositories/noclip-unified/earth/modules/geocoding-service.js:90`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/ui/ConfigSidebar.js:9959`

### Transport and events
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/transport/types.js:119`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/transport/BaseTransport.js:235`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/transport/CentralizedTransport.js:17`

### API client contract (server not in repo)
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatApiService.js:28`
- `/home/robit/Documents/repositories/noclip-unified/earth/refactor/chat/ChatApiService.js:201`

