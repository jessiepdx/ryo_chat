# WO-002: Automatic PostgreSQL Fallback Instance Generation (with Curses Flow)

Source item: `docs/master-engineering.md` section 2 (planned upgrade #2), section 4.2, section 4.4, section 6 phase 2-3, section 7.2.

## 1. Verbose Description of Work Order
Implement database failover orchestration so the application can use a remote PostgreSQL instance as primary, then automatically fail over to a local fallback PostgreSQL instance when remote availability checks fail.

This work order includes generation/provisioning hooks for a local fallback DB path and integrating those hooks into a setup flow (curses) so operators can configure fallback without manual database bootstrapping.

The failover behavior must be explicit and safe:
1. Perform startup health checks on remote DB.
2. If remote fails and fallback is enabled, switch to fallback connection settings.
3. Record degraded mode status in logs/startup report.
4. Continue serving core operations (chat history, member operations, knowledge reads/writes as feasible).

Because this flow uses the same setup surface as model configuration, it must preserve and respect the Ollama endpoint decision introduced in `WO-001`. If setup captures a custom Ollama endpoint, DB setup actions must not reset host values to implicit defaults.

## 2. Expression of Affected Files
Primary files:
1. `hypermindlabs/utils.py` (refactor `ConfigManager` and DB connection factory behavior)
2. New: `hypermindlabs/database_router.py` (primary/fallback selection and health checks)
3. New: `scripts/bootstrap_postgres.sh` or `scripts/bootstrap_postgres.py` (local fallback provisioning path)
4. New: `tests/test_database_router.py`
5. `scripts/setup_wizard.py` (shared curses flow must preserve selected `OLLAMA_HOST`/`config.inference.*.url`)

Secondary/API-surface files:
1. `telegram_ui.py`, `web_ui.py`, `cli_ui.py`, `x_ui.py` (consume startup DB status report; avoid hard crash paths)
2. `config.empty.json` (primary DB + fallback DB structure if schema extends)
3. `.env.example` (`POSTGRES_*`, `POSTGRES_FALLBACK_*`, `POSTGRES_FALLBACK_ENABLED`, `POSTGRES_FALLBACK_MODE`)
4. `readme.md` (fallback setup instructions)
5. `docs/master-engineering.md` (status update)

Runtime variables and configuration surfaces:
1. `database.db_name`
2. `database.user`
3. `database.password`
4. `database.host`
5. `database.port`
6. `.env` `POSTGRES_*`
7. `.env` `POSTGRES_FALLBACK_*`
8. `config.inference.*.url` (must remain unchanged unless explicitly edited)
9. `.env` `OLLAMA_HOST` (must remain unchanged unless explicitly edited)

## 3. Success and Validation Metrics
Valid outcome:
1. Remote DB healthy: application connects to primary, reports normal mode.
2. Remote DB unavailable + fallback enabled: application switches to fallback and starts successfully.
3. Startup output/log identifies selected DB endpoint and failover reason.
4. Core DB operations pass in fallback mode: table creation, member lookup, chat insert.
5. Integration test covers primary-down scenario.
6. Editing DB settings via setup does not overwrite previously selected custom Ollama endpoint.

Partial outcome:
1. Fallback works only at startup, not on runtime reconnect events.
2. Fallback works for some managers but not all managers in `utils.py`.
3. Fallback mode starts but without clear operator visibility.
4. DB setup changes unintentionally alter model-host settings.

Validation method:
1. Bring remote host down intentionally.
2. Verify process starts with fallback endpoint.
3. Execute smoke DB calls from CLI path.

## 4. Potential Failure Modes
1. Split-brain writes when remote recovers but app keeps writing to fallback without sync policy.
2. Connection strings assembled inconsistently across managers.
3. Provision script succeeds partially and leaves unusable database state.
4. Failover path introduces sensitive credential leakage in logs.
5. Fallback enabled but pgvector extension missing, causing vector table creation failures.
6. Shared setup merge logic clobbers `config.inference.*.url` or `.env` `OLLAMA_HOST`.
