# WO-005: Curses Walkthrough for Key Ingress and Model Selection

Source item: `docs/master-engineering.md` section 2 (planned upgrade #5), section 4.4, section 6 phase 3, section 7.4.

## 1. Verbose Description of Work Order
Implement an interactive curses-based first-run setup wizard that collects required operator inputs and writes validated configuration artifacts.

The wizard should reduce manual setup friction and provide immediate validation feedback for inputs. It should guide users through Telegram values, DB values, API keys, and Ollama model selection. It must allow optional values to be skipped with clear defaults while enforcing required keys.

Target workflow:
1. Start setup wizard command.
2. Detect existing `config.json` and offer backup/update paths.
3. Collect required fields and validate format.
4. Collect Ollama endpoint choice: custom host or default local (`http://127.0.0.1:11434`).
5. Validate/probe the selected Ollama endpoint and list available models from that endpoint.
6. Map selected models to capability keys using the selected endpoint.
7. Write `config.json`; optionally write `.env`.
8. Print startup command hints.

Endpoint precedence must be explicit and deterministic in setup output (for example: user-entered custom host -> existing config host -> default local host).

## 2. Expression of Affected Files
Primary files:
1. New: `scripts/setup_wizard.py` (or `scripts/setup_wizard.sh` + Python backend)
2. `config.empty.json` (template source)
3. `.env.example` (source for optional env output)
4. New: `tests/test_setup_wizard_validation.py`

Secondary/API-surface files:
1. `readme.md` (setup wizard usage instructions)
2. `docs/master-engineering.md` (status and workflow updates)
3. `hypermindlabs/utils.py` (`ConfigManager` compatibility with generated files)

Runtime variables and configuration surfaces:
1. `TELEGRAM_BOT_NAME`, `TELEGRAM_BOT_ID`, `TELEGRAM_BOT_TOKEN`
2. `TELEGRAM_OWNER_*`
3. `POSTGRES_*`
4. `BRAVE_SEARCH_API_KEY`
5. `TWITTER_*`
6. `OLLAMA_HOST`, `OLLAMA_*_MODEL`
7. `config.inference.embedding.url`
8. `config.inference.generate.url`
9. `config.inference.chat.url`
10. `config.inference.tool.url`
11. `config.inference.multimodal.url`
12. `roles_list` and `knowledge.domains`

## 3. Success and Validation Metrics
Valid outcome:
1. Wizard can produce a complete, valid `config.json` from blank setup.
2. Required fields cannot be skipped silently.
3. Model selection list comes from live probe of the selected Ollama endpoint.
4. Generated config successfully starts at least one interface.
5. Wizard supports both custom endpoint entry and default-local endpoint fallback.
6. Generated endpoint values are consistent across `config.inference.*.url` (and `.env` `OLLAMA_HOST` when written).

Partial outcome:
1. Wizard writes config but does not validate required sections.
2. Wizard validates fields but cannot probe models.
3. Wizard works only for new installs and fails update/edit path.
4. Wizard probes one host but writes another host to config/env.

Validation method:
1. Dry-run validation with mock inputs.
2. Real run against local Ollama + Postgres.
3. Start `cli_ui.py` with generated config.
4. Run setup twice (custom host path and blank/default path) and verify deterministic endpoint selection.

## 4. Potential Failure Modes
1. Terminal compatibility issues for curses UI.
2. Wizard writes malformed JSON when interrupted.
3. Existing config overwritten without backup.
4. Model selection data stale or mismatched to runtime policy constraints.
5. Setup accepts malformed/custom Ollama URL and cannot recover to local default cleanly.
6. Endpoint precedence is undocumented or implemented inconsistently across wizard steps.
