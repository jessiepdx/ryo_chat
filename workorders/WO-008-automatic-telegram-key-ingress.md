# WO-008: Automatic Ingress of Telegram Bot Keys

Source item: `docs/master-engineering.md` section 2 (planned upgrade #8), section 4.4, section 6 phase 3.

## 1. Verbose Description of Work Order
Implement secure, guided intake of Telegram bot configuration values and owner identity metadata, then persist them to configuration artifacts with validation.

This work order focuses specifically on Telegram credentials and identity values consumed by `telegram_ui.py` and miniapp login validation paths. The implementation should support first-run setup and update/rotation workflows.

Because Telegram key ingress runs through the shared setup surface, this flow must preserve existing Ollama endpoint/model host settings unless the user explicitly edits them. Telegram-only updates must not reset custom endpoint selections back to defaults.

Minimum requirements:
1. Collect Telegram bot values interactively.
2. Validate required fields and basic types.
3. Write to `config.json` keys expected by runtime.
4. Support key rotation without deleting unrelated config sections.
5. Avoid writing secrets to logs.
6. Preserve `config.inference.*.url` and `.env` `OLLAMA_HOST` unless explicitly changed by user.

## 2. Expression of Affected Files
Primary files:
1. `config.empty.json` (Telegram key schema reference)
2. New: `scripts/setup_wizard.py` (Telegram section)
3. `hypermindlabs/utils.py` (`ConfigManager` compatibility checks and clearer errors)
4. New: `tests/test_telegram_config_ingress.py`

Secondary/API-surface files:
1. `telegram_ui.py` (startup validation and user-facing error messages if keys missing)
2. `web_ui.py` (miniapp flow relies on `bot_token` for HMAC validation)
3. `.env.example` (`TELEGRAM_BOT_*`, `TELEGRAM_OWNER_*`)
4. `readme.md` (Telegram setup and rotation instructions)
5. `docs/master-engineering.md` (status update)

Runtime variables and configuration surfaces:
1. `bot_name`
2. `bot_id`
3. `bot_token`
4. `owner_info.first_name`
5. `owner_info.last_name`
6. `owner_info.user_id`
7. `owner_info.username`
8. `web_ui_url`
9. `config.inference.*.url` (read/merge/preserve behavior)
10. `.env` `OLLAMA_HOST` (read/merge/preserve behavior)

## 3. Success and Validation Metrics
Valid outcome:
1. Telegram config ingress flow produces valid keys in `config.json`.
2. `telegram_ui.py` starts and builds `Application` with provided token.
3. Miniapp auth flow can validate payload signature using configured token.
4. Key rotation path updates values without removing DB/model sections.
5. Telegram-only setup update does not alter existing custom Ollama endpoint values.

Partial outcome:
1. Setup collects values but writes incomplete owner_info.
2. Telegram runtime starts but miniapp auth fails due to token mismatch or missing fields.
3. Rotation workflow requires manual file edits to complete.
4. Telegram update flow unexpectedly rewrites inference host URLs.

Validation method:
1. Run setup ingress flow.
2. Start bot process and verify no missing-token error.
3. Execute miniapp login validation test path.

## 4. Potential Failure Modes
1. Bot token leaked into logs or shell history.
2. Wrong numeric/string coercion for `bot_id` and `owner_info.user_id`.
3. Partial writes corrupt `config.json`.
4. Config merge logic overwrites unrelated secrets or inference settings.
5. Shared setup flow applies default local endpoint over previously configured custom endpoint during Telegram-only edits.
