# Inter-User Messaging + Process Workspace

## Goal
Provide model-callable tools that let a member:

- Discover known users by username/member id.
- Send/queue a direct message to another known user.
- Create and maintain long-running multi-turn processes with explicit steps and progress.
- Resume process state in later turns without losing context.

## Dedicated Manager Class
`hypermindlabs/utils.py` now includes `CollaborationWorkspaceManager`, responsible for:

- User directory lookup and target resolution.
- Process workspace create/update/list/read operations.
- Step-level process updates with automatic completion/progress recalculation.
- Outbox queue + optional immediate Telegram delivery.

## Persistence Layer
New migrations:

- `db/migrations/084_agent_process_workspace.sql`
  - `agent_processes`
  - `agent_process_steps`
- `db/migrations/085_member_outbox.sql`
  - `member_outbox`

These are included in startup core migrations so `app.py` auto-migrate applies them.

## Tool Surface Added
Built-in tools wired into `ToolCallingAgent`:

- `knownUsersList`
- `messageKnownUser`
- `upsertProcessWorkspace`
- `listProcessWorkspace`
- `updateProcessWorkspaceStep`
- `listOutboxMessages`

These are registered in `hypermindlabs/tool_registry.py`, exposed to runtime, and included in tool catalogs/capability manifest/web harness.

## Multi-Turn Process Model
Each process has:

- `process_label`
- `process_status`
- `steps_total`
- `steps_completed`
- `completion_percent`
- `process_payload` JSON for arbitrary long/complex plan data

Each step has:

- order + label + optional details
- status (`pending`, `in_progress`, `blocked`, `completed`, `skipped`, `cancelled`)
- optional structured payload

Progress auto-refreshes on every step update.

## Runtime Context Requirements
Process and outbox tools require caller identity (`member_id`) from tool runtime context.
`ConversationOrchestrator` now injects member id into `run_context`/`tool_runtime_context` before tool execution.

## Delivery Behavior
`messageKnownUser`:

1. Resolves target by username/member id.
2. Persists row to `member_outbox`.
3. If `sendNow=true` and Telegram prerequisites exist (`bot_token`, target `user_id`), tries Telegram `sendMessage`.
4. Updates outbox status to `sent` or `failed` with failure reason and metadata.
