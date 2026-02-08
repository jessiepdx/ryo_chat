# Startup Troubleshooting

Use this guide when any RYO entrypoint fails to start or degrades unexpectedly.

## 1. `config.json` Missing or Invalid

Symptoms:
1. Startup crash with file-not-found or JSON parse errors.

Actions:
1. Create config from template:
```bash
cp config.empty.json config.json
```
2. Re-run setup:
```bash
python3 scripts/setup_wizard.py
```
3. Validate JSON syntax if you edited manually.

## 2. `logs/` Directory Missing

Symptoms:
1. `telegram_ui.py`, `web_ui.py`, or `x_ui.py` fails before serving/connecting.

Actions:
1. Create logs directory:
```bash
mkdir -p logs
```
2. Restart target runtime.

## 3. Telegram Startup Blocked by Config Validation

Symptoms:
1. `telegram_ui.py` logs: `Telegram startup blocked: missing/invalid config values`.

Likely cause:
1. Missing or invalid Telegram keys in `config.json`.

Actions:
1. Run Telegram-only ingress:
```bash
python3 scripts/setup_wizard.py --telegram-only
```
2. Ensure `bot_id` and `owner_info.user_id` are positive numeric values.
3. Ensure `web_ui_url` is a valid `http(s)` URL.

## 4. Web Miniapp Login Returns `503`

Symptoms:
1. `POST /miniapp-login` returns `503`.
2. Response includes missing Telegram config fields.

Likely cause:
1. Miniapp validation token/config is not complete.

Actions:
1. Populate Telegram fields via setup wizard (`--telegram-only`).
2. Restart `web_ui.py`.
3. Retry miniapp login.

Expected degraded behavior:
1. Core web UI remains available.
2. Only miniapp login path is blocked.

## 5. Web Miniapp Login Returns `400`

Symptoms:
1. `POST /miniapp-login` returns `400` with missing payload message.

Likely cause:
1. Client did not send `query-string` form field.

Actions:
1. Verify miniapp client submits Telegram init payload.
2. Confirm proxy/gateway does not strip form body.

## 6. PostgreSQL Route Issues (`fallback` or `failed_all`)

Symptoms:
1. Startup logs include database route status with warnings or errors.
2. Runtime cannot load/save chat/member/knowledge data.

Actions:
1. Check primary DB credentials in `config.json` `database.*`.
2. If fallback is expected, verify `database_fallback.enabled=true` and valid fallback credentials.
3. Confirm PostgreSQL is reachable:
```bash
psql "host=<host> dbname=<db> user=<user> password=<password>"
```
4. Ensure pgvector extension is installed:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
5. Run automated bootstrap verification:
```bash
python3 scripts/bootstrap_postgres.py --config config.json --target both
```

Expected degraded behavior:
1. If primary fails and fallback is healthy, route status becomes `fallback`.
2. If both fail, status becomes `failed_all` and data operations fail.

## 7. Ollama Host or Model Failures

Symptoms:
1. Connection refused/timeouts to Ollama host.
2. Model-not-found errors.

Likely causes:
1. Host mismatch (configured host not reachable from runtime machine).
2. Model not pulled on target Ollama instance.

Actions:
1. Check configured host in `config.json` `inference.*.url`.
2. Validate host reachability:
```bash
curl -sS <OLLAMA_HOST>/api/tags | head
```
3. Validate model availability:
```bash
ollama list
```
4. Re-run setup with reachable custom host or default local host:
```bash
python3 scripts/setup_wizard.py --non-interactive --ollama-host http://127.0.0.1:11434
```

## 8. Brave Search Key Missing

Symptoms:
1. Tool runtime returns structured missing-key error for `braveSearch`.

Actions:
1. Add `api_keys.brave_search` in `config.json`, or
2. Continue without Brave results; orchestration should continue in degraded mode.

## 9. X/Twitter Key Problems

Symptoms:
1. `x_ui.py` or tweet flows fail during X API calls.

Likely cause:
1. Missing/invalid `twitter_keys.*`.

Actions:
1. Populate `twitter_keys.consumer_key`, `consumer_secret`, `access_token`, `access_token_secret`.
2. Restart runtime.

## 10. Setup Wizard Interrupted Mid-Run

Symptoms:
1. Setup exits early and no config updates appear.

Likely cause:
1. Wizard cancelled/interrupted; partial state saved.

Actions:
1. Re-run setup to resume using `.setup_wizard_state.json`.
2. Remove stale state file only if you want a clean restart:
```bash
rm -f .setup_wizard_state.json
```
