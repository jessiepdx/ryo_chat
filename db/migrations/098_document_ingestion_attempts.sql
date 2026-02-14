CREATE TABLE IF NOT EXISTS document_ingestion_attempts (
    ingestion_attempt_id BIGSERIAL PRIMARY KEY,
    ingestion_job_id BIGINT NOT NULL,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    attempt_number INT NOT NULL,
    attempt_status VARCHAR(24) NOT NULL DEFAULT 'running',
    worker_id VARCHAR(96),
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP,
    duration_ms INT,
    error_message TEXT,
    error_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    stage_events JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_ingestion_attempts_job_link
        FOREIGN KEY(ingestion_job_id)
        REFERENCES document_ingestion_jobs(ingestion_job_id)
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_attempts_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_attempts_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_ingestion_attempts_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_ingestion_attempts_number_check
        CHECK(attempt_number > 0),
    CONSTRAINT document_ingestion_attempts_status_check
        CHECK(attempt_status IN ('leased', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT document_ingestion_attempts_duration_check
        CHECK(duration_ms IS NULL OR duration_ms >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_ingestion_attempts_job_number
    ON document_ingestion_attempts (ingestion_job_id, attempt_number);

CREATE INDEX IF NOT EXISTS idx_document_ingestion_attempts_scope_created
    ON document_ingestion_attempts (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

ALTER TABLE document_ingestion_attempts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_ingestion_attempts_scope_policy ON document_ingestion_attempts;
CREATE POLICY document_ingestion_attempts_scope_policy ON document_ingestion_attempts
    FOR ALL
    USING (
        (
            COALESCE(NULLIF(current_setting('app.scope_bypass', true), ''), '0') = '1'
        )
        OR
        (
            owner_member_id = NULLIF(current_setting('app.owner_member_id', true), '')::INT
            AND chat_host_id = NULLIF(current_setting('app.chat_host_id', true), '')::BIGINT
            AND chat_type = NULLIF(current_setting('app.chat_type', true), '')
            AND platform = NULLIF(current_setting('app.platform', true), '')
            AND (
                NULLIF(current_setting('app.community_id', true), '') IS NULL
                OR community_id = NULLIF(current_setting('app.community_id', true), '')::INT
            )
            AND (
                NULLIF(current_setting('app.topic_id', true), '') IS NULL
                OR topic_id = NULLIF(current_setting('app.topic_id', true), '')::INT
            )
        )
    )
    WITH CHECK (
        (
            COALESCE(NULLIF(current_setting('app.scope_bypass', true), ''), '0') = '1'
        )
        OR
        (
            owner_member_id = NULLIF(current_setting('app.owner_member_id', true), '')::INT
            AND chat_host_id = NULLIF(current_setting('app.chat_host_id', true), '')::BIGINT
            AND chat_type = NULLIF(current_setting('app.chat_type', true), '')
            AND platform = NULLIF(current_setting('app.platform', true), '')
            AND (
                NULLIF(current_setting('app.community_id', true), '') IS NULL
                OR community_id = NULLIF(current_setting('app.community_id', true), '')::INT
            )
            AND (
                NULLIF(current_setting('app.topic_id', true), '') IS NULL
                OR topic_id = NULLIF(current_setting('app.topic_id', true), '')::INT
            )
        )
    );
