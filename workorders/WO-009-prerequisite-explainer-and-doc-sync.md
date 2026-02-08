# WO-009: Master Engineering Documentation and Prerequisite Explainer

Source item: `docs/master-engineering.md` section 2 (planned upgrade #9), sections 5 and 9.

## 1. Verbose Description of Work Order
Create and maintain a dedicated prerequisite explainer layer that translates engineering constraints into operator-ready setup checks. This work order ensures new users and agents can quickly identify required services, required keys, optional features, and degradation behavior.

The deliverable is not only static text; it must include checklists and decision paths so users can determine what to configure depending on target interface and features (Telegram-only, CLI-only, with or without Brave/X integrations, local vs remote DB).

The prerequisite layer must explicitly document Ollama endpoint behavior, including how to choose a custom host versus relying on the default local host (`http://127.0.0.1:11434`), and how endpoint precedence is resolved during setup.

Required outputs:
1. Prerequisite matrix by runtime component.
2. Required vs optional key map.
3. Startup troubleshooting path by failure symptom.
4. Sync policy ensuring prerequisites stay updated as features change.
5. Endpoint selection/troubleshooting flow for custom and default-local Ollama hosts.

## 2. Expression of Affected Files
Primary files:
1. `readme.md` (quick start and prerequisite matrix)
2. `docs/master-engineering.md` (authoritative engineering status)
3. New: `docs/prerequisites.md`
4. New: `docs/troubleshooting-startup.md`

Secondary/API-surface files:
1. `.env.example` and `config.empty.json` (must reflect prerequisite docs)
2. `telegram_ui.py`, `web_ui.py`, `cli_ui.py`, `x_ui.py` (startup checks should align with docs)
3. `requirements.txt` (dependency docs alignment)

Runtime variables and configuration surfaces:
1. `database.*`
2. `inference.*`
3. `api_keys.brave_search`
4. `twitter_keys.*`
5. Telegram owner and bot fields
6. `.env` fallback and setup values
7. `.env` `OLLAMA_HOST`
8. `config.inference.*.url`

## 3. Success and Validation Metrics
Valid outcome:
1. Prerequisite docs exist and are linked from README.
2. Each runtime entrypoint has a documented prerequisite set.
3. Required/optional status is explicit for each key/service.
4. Startup troubleshooting includes degraded-mode expectations.
5. Ollama endpoint setup supports both custom and default-local paths with explicit precedence and troubleshooting guidance.

Partial outcome:
1. Docs exist but do not cover all interfaces.
2. Required/optional markers are ambiguous.
3. Troubleshooting guide lacks concrete actions for common failures.
4. Endpoint guidance exists but does not explain custom-vs-default behavior or precedence.

Validation method:
1. Run a new-user setup simulation using only docs.
2. Verify no undocumented required input appears during startup.
3. Confirm docs updated whenever template files change.

## 4. Potential Failure Modes
1. Docs claim optional features are required (or inverse), causing misconfiguration.
2. Prerequisite docs lag behind code changes.
3. Troubleshooting steps are generic and not mapped to real log/error patterns.
4. Missing linkage between README and deeper prerequisite docs.
5. Endpoint troubleshooting omits host mismatch cases (setup host differs from runtime host).
