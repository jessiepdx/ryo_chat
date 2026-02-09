# WO-003: Complete Documentation and Master Engineering Documentation Maintenance

Source item: `docs/master-engineering.md` section 2 (planned upgrade #3), sections 6 and 9.

## 1. Verbose Description of Work Order
Define and implement a documentation maintenance workflow that keeps engineering plans, setup guides, and operator prerequisites synchronized with real implementation state.

This work order operationalizes documentation as a release artifact, not an afterthought. It requires adding maintenance rules, checklists, and update triggers for every feature change that impacts configuration, APIs, startup behavior, or fallback behavior.

Deliverables include:
1. Work-order-to-implementation traceability.
2. Document update sequence enforced in contribution workflow.
3. Clear ownership of docs touching setup, policy, fallback behavior, and operational troubleshooting.
4. Explicit setup documentation for Ollama endpoint selection (custom host vs default local `http://127.0.0.1:11434`) and precedence rules.

## 2. Expression of Affected Files
Primary files:
1. `docs/master-engineering.md` (status-tracked source-of-truth updates)
2. `readme.md` (installation/runtime instructions)
3. New: `docs/CHANGELOG_ENGINEERING.md` (implementation vs plan delta log)
4. New: `docs/DOC_UPDATE_CHECKLIST.md`

Secondary/API-surface files:
1. `.env.example` and `config.empty.json` (must stay aligned with runtime expectations)
2. `policies/agent/*.json` and `policies/agent/system_prompt/*.txt` (policy documentation alignment)
3. New: `.github/PULL_REQUEST_TEMPLATE.md` (if repo workflow supports it)

Runtime variables and configuration surfaces to track:
1. `bot_*` values
2. `database.*`
3. `roles_list`
4. `inference.*`
5. `api_keys.brave_search`
6. `twitter_keys.*`
7. `.env` fallback and setup variables
8. `.env` `OLLAMA_HOST`
9. `config.inference.*.url`

## 3. Success and Validation Metrics
Valid outcome:
1. Documentation checklist exists and is actionable.
2. Master doc has a clear status section per planned upgrade (not just future intent).
3. README reflects exact runtime prerequisites and setup commands.
4. Config/env templates match current code-required keys.
5. Setup docs explicitly describe custom-vs-default Ollama endpoint behavior and precedence.

Partial outcome:
1. Docs updated but no process to keep them updated.
2. Master doc and README diverge on setup requirements.
3. Template files include keys not used or omit required keys.
4. Ollama endpoint behavior implemented in code but missing or ambiguous in docs.

Validation method:
1. Run a doc audit against startup path (`telegram_ui.py`, `web_ui.py`, `cli_ui.py`).
2. Verify each required config key appears in README and template files.
3. Confirm doc update checklist is used in at least one merged implementation PR.

## 4. Potential Failure Modes
1. Documentation drift after implementation merges.
2. Incomplete update coverage for secondary files (`.env.example`, policy files).
3. Ambiguous instructions causing operators to misconfigure DB or model endpoints.
4. Overly generic docs that do not identify concrete files or variables.
5. Conflicting docs on whether endpoint defaults to local or must always be manually supplied.
