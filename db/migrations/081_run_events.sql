CREATE TABLE IF NOT EXISTS run_events (
    event_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    event_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT run_events_run_fk
        FOREIGN KEY(run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE,
    CONSTRAINT run_events_run_seq_unique
        UNIQUE (run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_timestamp
    ON run_events (run_id, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_run_events_type_timestamp
    ON run_events (event_type, event_timestamp DESC);
