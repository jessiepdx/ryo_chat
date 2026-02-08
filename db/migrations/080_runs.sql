CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    parent_run_id UUID,
    member_id INTEGER,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    request_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolved_config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_json JSONB,
    error_text TEXT,
    lineage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_member_created
    ON runs (member_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_status_created
    ON runs (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_mode_created
    ON runs (mode, created_at DESC);
