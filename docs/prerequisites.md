# RYO Prerequisites

This document is the operator-facing prerequisite layer for RYO startup, setup, and upgrade workflows.

## 1. Runtime Matrix

| Runtime | Hard Requirements | Optional Features | Graceful/Degraded Behavior |
| --- | --- | --- | --- |
| `scripts/setup_wizard.py` | Python deps installed, writable `config.json` path, `config.empty.json` available for first run | `curses` UI, `.env` write (`--write-env`), Ollama model probe, immediate DB bootstrap (`--bootstrap-postgres`) | Falls back to plain prompts if curses is unavailable; model probe errors do not block save |
| `cli_ui.py` | Valid `config.json`, PostgreSQL reachable through `database` or configured fallback route, Ollama tool host reachable (`inference.tool.url`) | Brave search API key | Conversation continues when Brave key is missing; explicit `/search` command may fail without key |
| `web_ui.py` (core site) | Valid `config.json`, PostgreSQL route available, `logs/` exists | Telegram miniapp login | Core routes continue even when miniapp login is unavailable |
| `web_ui.py` (`/miniapp-login`) | Valid Telegram bot token and miniapp payload | N/A | Returns `503` when Telegram config is incomplete; returns `400` when payload is missing |
| `telegram_ui.py` | Valid Telegram bot config (`bot_name`, `bot_id`, `bot_token`, `owner_info.*`, `web_ui_url`), PostgreSQL route available, `logs/` exists | Brave search and X/Twitter flows | Startup is blocked with actionable error if Telegram config is invalid |
| `x_ui.py` | Valid `twitter_keys.*` in `config.json`, `logs/` exists | N/A | No dedicated fallback yet; invalid keys fail X client operations |
| `scripts/policy_wizard.py` | Policy files and prompt files present, Ollama host reachable for strict model checks | Strict model checks | Validation can still run in non-strict mode with warnings |
| `scripts/bootstrap_postgres.py` | PostgreSQL reachable for manual/config mode (and Docker installed when `--docker` is used) | SQL check customization | Script is idempotent and exits non-zero on failures; supports primary/fallback targets from config |

## 2. Required vs Optional Config Map

`ConfigManager` runtime currently reads `config.json` directly. `.env` is used by setup/automation flows.

| Key Surface | Required For | Required/Optional | Notes |
| --- | --- | --- | --- |
| `bot_name` | Telegram runtime, miniapp docs consistency | Required for Telegram stack | Validated at Telegram startup |
| `bot_id` | Telegram runtime metadata | Required for Telegram stack | Must be positive numeric |
| `bot_token` | Telegram bot auth, miniapp HMAC validation | Required for Telegram stack | Missing token blocks Telegram startup and miniapp login |
| `owner_info.first_name` | Telegram owner workflows | Required for Telegram stack | |
| `owner_info.last_name` | Telegram owner workflows | Required for Telegram stack | |
| `owner_info.user_id` | Telegram owner workflows | Required for Telegram stack | Must be positive numeric |
| `owner_info.username` | Telegram owner workflows | Required for Telegram stack | |
| `web_ui_url` | Telegram/web linking | Required for Telegram stack | Must be valid `http(s)` URL |
| `database.*` | All runtimes that read/write app data | Required | Must point to pgvector-capable PostgreSQL |
| `database_fallback.*` | DB fallback routing | Optional but recommended | Enables fallback route when primary is down |
| `inference.embedding|generate|chat|tool|multimodal.url` | Ollama calls | Required | Host should be reachable from runtime host |
| `inference.*.model` | Ollama calls | Required | Must exist on selected Ollama host |
| `api_keys.brave_search` | Live `braveSearch` tool results | Optional | Tool runtime degrades gracefully when missing |
| `twitter_keys.*` | X/Twitter posting workflows | Optional unless `x_ui.py` or tweet flows are used | Missing keys break X operations |
| `roles_list` | Telegram role-aware flows | Required | Must include expected roles used by handlers |

### 2.1 `.env` Keys (automation/setup layer)

Primary keys mirrored by setup tooling:
1. `OLLAMA_HOST`
2. `OLLAMA_*_MODEL`
3. `POSTGRES_*`
4. `POSTGRES_FALLBACK_*`
5. `TELEGRAM_BOT_*`
6. `TELEGRAM_OWNER_*`
7. `WEB_UI_URL`
8. `BRAVE_SEARCH_API_KEY`
9. `TWITTER_*`

Important:
1. Runtime reads `config.json` directly today.
2. Telegram-only setup (`--telegram-only`) updates only Telegram fields and preserves existing inference/database values, including `.env` `OLLAMA_HOST`.

## 3. Operator Decision Path

1. Decide target interface: Telegram, Web, CLI, and/or X/Twitter workflows.
2. Run setup:
`python3 scripts/setup_wizard.py` for full setup, or `python3 scripts/setup_wizard.py --telegram-only` for Telegram key rotation only.
3. Validate required services: Ollama host reachability, PostgreSQL reachability (or fallback), and `logs/` presence.
4. Validate optional integrations only if needed: Brave API key for live web search and Twitter keys for X workflows.

## 4. Ollama Endpoint Selection and Precedence

Setup host precedence is:
1. Explicit custom host provided by user (`--ollama-host` or prompt entry)
2. Existing host already present in `config.json` inference sections
3. Default local host `http://127.0.0.1:11434`

Operator checks:
1. Verify host value in `config.json` under `inference.*.url`.
2. Verify host is reachable from runtime machine.
3. Verify required models exist on that host.

Troubleshooting commands:
```bash
curl -sS http://127.0.0.1:11434/api/tags | head
ollama list
```

Host mismatch failure mode:
1. Setup writes custom host, but runtime machine cannot reach it.
2. Result is model invocation failures (`connection refused`, timeout, missing model).
3. Fix by rerunning setup with reachable host or repairing network/routing to remote host.

## 5. Documentation Sync Policy

When setup/config/runtime prerequisites change:
1. Update `docs/master-engineering.md` status snapshot and impacted sections.
2. Update this file (`docs/prerequisites.md`) and `docs/troubleshooting-startup.md`.
3. Update `readme.md` quick-start and prerequisite links.
4. Update template contracts: `config.empty.json` and `.env.example`.
5. Add an entry to `docs/CHANGELOG_ENGINEERING.md`.
6. Complete `docs/DOC_UPDATE_CHECKLIST.md` before merge.
