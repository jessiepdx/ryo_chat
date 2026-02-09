CREATE TABLE IF NOT EXISTS member_personality_events (
    event_id SERIAL PRIMARY KEY,
    member_id INT NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    before_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    after_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason_code VARCHAR(96) NOT NULL DEFAULT '',
    reason_detail TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT personality_event_member_link
        FOREIGN KEY(member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_member_personality_events_member_created
    ON member_personality_events (member_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_member_personality_events_type_created
    ON member_personality_events (event_type, created_at DESC);
