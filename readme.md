# RYO Chat (Run Your Own)

Multi-platform agent playground with:
- Telegram bot + command workflows
- Web interface + Telegram miniapp
- CLI interface
- Multi-agent orchestration with tool-calling
- PostgreSQL + pgvector-backed chat/knowledge retrieval

For the full engineering baseline and upgrade roadmap, see:
- `docs/master-engineering.md`

## References
- Ollama: https://ollama.com/
- Ollama GitHub: https://github.com/ollama/ollama
- PostgreSQL: https://www.postgresql.org/
- pgvector: https://github.com/pgvector/pgvector

## Quick Start
1. Create a virtual environment and install dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create runtime directories/files.
```bash
mkdir -p logs
cp config.empty.json config.json
cp .env.example .env
```

3. Set or update Ollama endpoint defaults (custom or local default).
```bash
python3 scripts/setup_wizard.py
```

4. Edit remaining `config.json` fields (required).
5. Start one or more interfaces.
```bash
python3 telegram_ui.py
python3 web_ui.py
python3 cli_ui.py
```

## Required Configuration (`config.json`)
`ConfigManager` currently reads only `config.json` at runtime.

Required top-level keys:
- `bot_name`
- `bot_id`
- `bot_token`
- `web_ui_url`
- `owner_info`
- `database`
- `roles_list`
- `defaults`
- `inference`
- `api_keys`

Conditionally required keys:
- `twitter_keys` if using `/tweet` flows or `x_ui.py`.

Critical notes:
- `roles_list` must be present or Telegram command chains that build role selectors will fail.
- `database` must point to PostgreSQL with pgvector enabled (vector columns are used by chat history, knowledge, and spam stores).
- `inference.*.url` values should point to reachable Ollama hosts.

## PostgreSQL + pgvector Setup
The app auto-creates tables, but it expects pgvector types to be available.

### Option A: Docker (simple local dev)
```bash
docker run --name ryo-pg \
  -e POSTGRES_DB=ryo_chat \
  -e POSTGRES_USER=ryo \
  -e POSTGRES_PASSWORD=ryo_pass \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

Then ensure the extension exists:
```bash
docker exec -it ryo-pg psql -U ryo -d ryo_chat -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Option B: Existing PostgreSQL server
1. Install pgvector on that server.
2. Run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
3. Use those connection details in `config.json`.

## Environment File (`.env`)
`.env` is included for setup automation and upcoming fallback/bootstrap workflows documented in `docs/master-engineering.md`.

Current runtime behavior:
- Python runtime currently reads `config.json` directly.
- `.env` values are primarily for operator workflows and planned bootstrap tooling.

## Ollama Endpoint Selection
Setup precedence for Ollama host is:
1. Explicit user-provided host in setup (`--ollama-host` or prompt value)
2. Existing host already present in `config.json` inference sections
3. Default local host `http://127.0.0.1:11434`

To run non-interactive endpoint setup:
```bash
python3 scripts/setup_wizard.py --non-interactive --ollama-host http://127.0.0.1:11434
```

## Policies
Agent policies and system prompts live in:
- `policies/agent/*.json`
- `policies/agent/system_prompt/*.txt`

Current behavior:
- Policies and prompts are loaded from disk at runtime.
- Missing policy/prompt files are not yet gracefully handled; keep these files present.

## Known Operational Prerequisites
- `logs/` directory must exist before starting `telegram_ui.py`, `web_ui.py`, or `x_ui.py`.
- Ollama endpoints in config must be reachable for embeddings/chat/generate/tool/multimodal flows.
- Brave API key is required for `braveSearch` tool.
