CREATE TABLE IF NOT EXISTS run_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    step_seq INTEGER,
    stage TEXT,
    snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    snapshot_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT run_snapshots_run_fk
        FOREIGN KEY(run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_snapshots_run_step
    ON run_state_snapshots (run_id, step_seq);

CREATE INDEX IF NOT EXISTS idx_run_snapshots_timestamp
    ON run_state_snapshots (snapshot_timestamp DESC);
