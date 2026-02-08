# WO-010: Automated pgvector and PostgreSQL Setup

Source item: `docs/master-engineering.md` section 2 (planned upgrade #10), section 4.3, section 6 phase 3.

## 1. Verbose Description of Work Order
Implement repeatable automation for provisioning PostgreSQL with pgvector and initializing the schema required by the application.

This work order should provide an operator-friendly bootstrap path that can run on a clean machine and prepare all database prerequisites with minimal manual SQL operations. It should explicitly verify pgvector extension availability and produce clear pass/fail output.

Scope includes:
1. Local Postgres provisioning path (containerized or local service binding).
2. `CREATE EXTENSION IF NOT EXISTS vector` execution.
3. Schema initialization sanity checks.
4. Optional integration into setup wizard and DB router fallback flows.

## 2. Expression of Affected Files
Primary files:
1. New: `scripts/bootstrap_postgres.sh` or `scripts/bootstrap_postgres.py`
2. New: `scripts/verify_pgvector.sql`
3. `hypermindlabs/utils.py` (optional startup-time extension check and diagnostics)
4. New: `tests/test_pg_bootstrap.py` (or integration smoke script)

Secondary/API-surface files:
1. `readme.md` (automated bootstrap usage)
2. `docs/master-engineering.md` (status update)
3. `.env.example` (`POSTGRES_*` and fallback vars)
4. `config.empty.json` (DB field alignment)
5. `requirements.txt` (if automation adds runtime dependency)

Runtime variables and configuration surfaces:
1. `POSTGRES_DB`
2. `POSTGRES_USER`
3. `POSTGRES_PASSWORD`
4. `POSTGRES_HOST`
5. `POSTGRES_PORT`
6. `POSTGRES_FALLBACK_*`
7. `database.*` config values

## 3. Success and Validation Metrics
Valid outcome:
1. Bootstrap script creates/targets DB and enables pgvector extension idempotently.
2. Vector-dependent schema creation succeeds without manual intervention.
3. Running bootstrap multiple times does not break existing data or schema.
4. Script returns clear non-zero exit status on failure and actionable output.

Partial outcome:
1. Postgres provisioning works, but pgvector enablement is manual.
2. Extension enabled but schema verification absent.
3. Script is not idempotent and fails on second run.

Validation method:
1. Run bootstrap on clean environment.
2. Confirm `vector` extension exists in target DB.
3. Start app and verify vector table creation paths in managers.
4. Re-run bootstrap and confirm no destructive behavior.

## 4. Potential Failure Modes
1. Bootstrap script assumes Docker/local tooling unavailable in operator environment.
2. Extension check performed in wrong database context.
3. Script logs sensitive DB credentials.
4. Schema bootstrap order causes foreign-key or extension-type failures.
5. Inconsistent behavior across PostgreSQL versions.
