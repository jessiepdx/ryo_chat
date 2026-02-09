CREATE TABLE IF NOT EXISTS run_artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    step_seq INTEGER,
    artifact_type TEXT NOT NULL,
    artifact_name TEXT NOT NULL,
    artifact_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT run_artifacts_run_fk
        FOREIGN KEY(run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_run_type
    ON run_artifacts (run_id, artifact_type);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_timestamp
    ON run_artifacts (artifact_timestamp DESC);
