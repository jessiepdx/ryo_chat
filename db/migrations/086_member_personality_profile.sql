CREATE TABLE IF NOT EXISTS member_personality_profile (
    profile_id SERIAL PRIMARY KEY,
    member_id INT NOT NULL UNIQUE,
    explicit_directive_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    adaptive_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    effective_profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    profile_version INT NOT NULL DEFAULT 1,
    locked_fields_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT personality_profile_member_link
        FOREIGN KEY(member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_member_personality_profile_member
    ON member_personality_profile (member_id, updated_at DESC);
