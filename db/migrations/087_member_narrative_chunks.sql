CREATE TABLE IF NOT EXISTS member_narrative_chunks (
    chunk_id SERIAL PRIMARY KEY,
    member_id INT NOT NULL,
    chunk_index INT NOT NULL,
    source_turn_start_id INT,
    source_turn_end_id INT,
    summary_text TEXT NOT NULL,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    compression_ratio REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT narrative_chunk_member_link
        FOREIGN KEY(member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_member_narrative_chunks_member_index
    ON member_narrative_chunks (member_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_member_narrative_chunks_member_created
    ON member_narrative_chunks (member_id, created_at DESC);
