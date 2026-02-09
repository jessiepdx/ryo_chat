# RYO Launcher Route Config Dashboard Spec (app.py)

Date: 2026-02-08
Scope: `app.py` only (primary), with optional tests/docs follow-up.
Purpose: Define a complete implementation contract for sub-agents to build a nested curses configuration UX per route (telegram/web/cli/x), with reliable editing and persistence of tuneables.

## 1. Product Intent

The launcher dashboard currently supports route lifecycle actions (toggle/start/stop/log/open), but route configuration and runtime tuneables are edited elsewhere. This update makes tuning first-class in launcher.

Required user outcome:
1. Select route in dashboard (`telegram`, `web`, `cli`, `x`).
2. Press `Enter`.
3. Open route configuration workspace (not immediate open-action).
4. Navigate categories.
5. Enter category.
6. Navigate settings.
7. Enter setting.
8. Edit value safely with type validation and defaults visible.
9. Persist changes reliably to `config.json` (and optional `.env` mirror where applicable).
10. Return to dashboard and optionally apply/restart route.

Critical immediate requirement:
- Telegram stage-progress visibility toggle must be easy in launcher.
- Setting path: `runtime.telegram.show_stage_progress` (boolean).

## 2. UX Requirements

## 2.1 Dashboard behavior changes

Current behavior:
- `Enter`/`r` opens interface action for selected route.

New behavior:
1. `Enter` opens route configuration workspace.
2. `r` keeps existing “open route interface” behavior.
3. `Space` keeps existing desired-state toggle behavior.
4. `l` keeps existing log view.
5. Add footer hints reflecting new keymap.

Target keymap:
- `Up/Down`: move selection
- `Enter`: route config workspace
- `r`: open route interface (existing)
- `Space`: toggle desired on/off (existing)
- `s`: start selected route
- `x`: stop selected route
- `a/o`: start all / stop all (existing)
- `l`: log tail (existing)
- `q`: back/quit (existing)

## 2.2 Route configuration workspace

Each selected route opens a dedicated page with:
1. Header: route name + script + running status + current endpoint/link summary.
2. Category list.
3. Breadcrumb (e.g., `Dashboard > Route: telegram > Category: Pipeline`).
4. Explicit actions.

Workspace actions:
1. Enter category.
2. Quick route actions.
3. Save pending changes.
4. Discard pending changes.
5. Restart route to apply.
6. Back to dashboard.

## 2.3 Category page

Shows settings table with columns:
1. Key
2. Current value
3. Source (`config`, `runtime-default`, `env-override-detected`)
4. Type
5. Pending change marker

Actions:
1. Enter on setting opens setting editor.
2. `d` reset setting to default.
3. `u` reset category to defaults (confirm required).
4. `s` save.
5. `b` back.

## 2.4 Setting editor page

Must include:
1. Key/path.
2. Description/help text.
3. Current value.
4. Default value.
5. Allowed type/range/options.
6. Validation errors inline.

Edit behavior by type:
1. `bool`: toggle widget (`True`/`False`).
2. `int`: numeric input with range validation.
3. `float`: numeric input with range validation.
4. `string`: text input with optional non-empty constraint.
5. `enum`: select from options.
6. `port`: integer 1-65535.
7. `url`: valid http/https.
8. `secret`: masked preview, unmasked edit prompt only.

## 3. Route -> Category -> Setting Model

Implement a declarative schema in `app.py` using dataclasses.

Recommended structures:
1. `SettingSpec`
2. `CategorySpec`
3. `RouteConfigSpec`

Required `SettingSpec` fields:
1. `id`
2. `label`
3. `path` (dot path within `config_data`)
4. `value_type`
5. `description`
6. `default_path` or static default fallback
7. `required`
8. `min_value`/`max_value` (numeric types)
9. `choices` (enum types)
10. `sensitive` (secret masking)
11. `restart_required` (route apply metadata)
12. `env_override_keys` (for visibility only)

Required route coverage:

### telegram
Categories:
1. `Pipeline`
2. `Access/Score Gates`
3. `Identity`
4. `Runtime`

Must-have settings:
1. `runtime.telegram.show_stage_progress` (bool)  <- primary requirement
2. `runtime.telegram.get_updates_write_timeout` (int >= 1)
3. `runtime.telegram.minimum_community_score_private_chat` (int >= 0)
4. `runtime.telegram.minimum_community_score_private_image` (int >= 0)
5. `runtime.telegram.minimum_community_score_group_image` (int >= 0)
6. `runtime.telegram.minimum_community_score_other_group` (int >= 0)
7. `runtime.telegram.minimum_community_score_link` (int >= 0)
8. `runtime.telegram.minimum_community_score_forward` (int >= 0)
9. `bot_name`
10. `bot_token` (secret)
11. `bot_id` (int)

### web
Categories:
1. `Bind/Port`
2. `Runtime`
3. `External URL`

Must-have settings:
1. `runtime.web.host`
2. `runtime.web.port`
3. `runtime.web.port_scan_limit`
4. `runtime.web.debug`
5. `runtime.web.use_reloader`
6. `web_ui_url`

### cli
Categories:
1. `Conversation`
2. `Retrieval`
3. `Tool Runtime`

Must-have settings:
1. `runtime.conversation.knowledge_lookup_word_threshold`
2. `runtime.conversation.knowledge_lookup_result_limit`
3. `runtime.retrieval.conversation_short_history_limit`
4. `runtime.tool_runtime.default_timeout_seconds`
5. `runtime.tool_runtime.default_max_retries`

### x
Categories:
1. `Twitter Keys`
2. `Runtime`

Must-have settings:
1. `twitter_keys.consumer_key` (secret)
2. `twitter_keys.consumer_secret` (secret)
3. `twitter_keys.access_token` (secret)
4. `twitter_keys.access_token_secret` (secret)

## 4. Persistence and Source-of-Truth Rules

Primary source:
1. `config.json` runtime tree and route keys.

Read flow:
1. Load `config_data`.
2. Compute `runtime_settings = build_runtime_settings(config_data=config_data)`.
3. Use config path value when present.
4. If missing, show runtime resolved default.
5. Detect and display matching env override key presence but do not silently mutate `.env` unless explicitly requested.

Write flow:
1. Stage edits in memory (`pending_changes` dict keyed by dot path).
2. Validate each setting before commit.
3. On save:
   - backup `config.json` with timestamp
   - apply all changes
   - write atomically via existing helper
4. Rebuild runtime settings after save.
5. Show save result summary and list of changed keys.

Optional `.env` mirror:
1. Add explicit action `Mirror selected settings to .env`.
2. Never overwrite unrelated keys.
3. Preserve comments/order best-effort.
4. Display conflict notice when env override differs from saved config value.

## 5. Reliability Requirements

1. No partial writes on failed validation.
2. Atomic config write only.
3. Backup before write.
4. Input validation per type/range.
5. Secret masking in list/detail views.
6. Route restart prompt when any changed setting has `restart_required=true`.
7. Recoverable error dialogs (no curses crash to shell).
8. ESC/back always returns to prior menu safely.

## 6. App.py Implementation Plan (Sub-Agent Work Breakdown)

## Sub-Agent A - Config Schema and Validation Engine

Files:
1. `app.py`

Tasks:
1. Add declarative route/category/setting schema.
2. Add path get/set helpers for nested dict updates.
3. Add typed validators and coercion helpers.
4. Add source-resolution helpers (`config`, `runtime-default`, `env-override`).

Acceptance:
1. Unit-callable helpers validate and coerce settings correctly.
2. Invalid values never commit.

## Sub-Agent B - Curses Navigation Stack

Files:
1. `app.py`

Tasks:
1. Add `route_config_workspace_curses(...)`.
2. Add `route_category_curses(...)`.
3. Add `setting_editor_curses(...)`.
4. Integrate breadcrumbs and key hints.

Acceptance:
1. Enter on dashboard route opens workspace.
2. Nested nav works to setting depth and back without state loss.

## Sub-Agent C - Save/Apply/Restart Flow

Files:
1. `app.py`

Tasks:
1. Add pending change buffer.
2. Add save/discard actions.
3. Add restart prompt and route apply actions.
4. Recompute runtime settings post-save.

Acceptance:
1. Save updates `config.json` reliably.
2. Changed route can be restarted from same workflow.
3. Dashboard reflects new values after return.

## Sub-Agent D - Route-Specific Settings Coverage

Files:
1. `app.py`

Tasks:
1. Implement route-specific categories/settings listed in Section 3.
2. Ensure Telegram `show_stage_progress` has obvious quick access.
3. Add dashboard badge for Telegram stage-progress status (`Stages: on/off`).

Acceptance:
1. Telegram stage progress is togglable from launcher config flow in <= 4 keypresses from selected route.
2. Web/CLI/X categories present and editable.

## Sub-Agent E - Regression and Operator UX

Files:
1. `app.py`
2. `readme.md` (optional follow-up)

Tasks:
1. Verify old non-curses/non-interactive route menu still works.
2. Ensure `r` still opens route interface action.
3. Add concise operator help text in dashboard footer.

Acceptance:
1. Existing launch/watchdog behavior preserved.
2. New config workflows do not block route lifecycle controls.

## 7. Explicit Acceptance Criteria

1. `Enter` on route opens route config workspace.
2. Telegram route exposes `runtime.telegram.show_stage_progress` and toggles it reliably.
3. Edits are validated and saved atomically to `config.json`.
4. Dashboard can restart selected route after save.
5. Stage progress status is visible in dashboard summary/details.
6. `r` still opens route interface (browser/telegram link/transient terminal).
7. No crash when invalid input is entered repeatedly.
8. `python3 -m py_compile app.py` passes.

## 8. Test Matrix (Manual)

1. Telegram stage progress toggle:
   - Set `true`, save, restart telegram route, confirm stage status messages appear.
   - Set `false`, save, restart telegram route, confirm stage status messages do not appear.
2. Web port edit:
   - Change `runtime.web.port`, save, restart web route, confirm new endpoint in dashboard.
3. Invalid integer:
   - Enter negative score gate; verify rejection and no commit.
4. Secret edit:
   - Edit `bot_token`; verify masked list display and persisted value.
5. Discard:
   - Stage multiple edits, discard, confirm no config write.
6. Non-interactive compatibility:
   - `python3 app.py --non-interactive --bootstrap-only` still succeeds.

## 9. Out-of-Scope (for this update)

1. Full schema-driven generic editor for all config keys.
2. Live hot-reload across running processes without restart.
3. Editing policy files (`policies/agent/*`) from launcher.
4. Web-based dashboard config editor (this spec is curses launcher only).

## 10. Delivery Notes

1. Keep implementation isolated to `app.py` first.
2. Reuse existing helpers (`build_runtime_settings`, `get_runtime_setting`, atomic config writes, backup helper).
3. Preserve existing launcher controls and route-open actions.
4. Prefer adding compact reusable curses primitives instead of route-specific one-off loops.

