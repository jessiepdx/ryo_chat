CREATE TABLE IF NOT EXISTS document_ingestion_jobs (
    ingestion_job_id BIGSERIAL PRIMARY KEY,
    document_source_id BIGINT NOT NULL,
    document_version_id BIGINT NOT NULL,
    storage_object_id BIGINT,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    pipeline_version VARCHAR(48) NOT NULL DEFAULT 'v1',
    idempotency_key VARCHAR(196) NOT NULL,
    job_status VARCHAR(24) NOT NULL DEFAULT 'queued',
    priority INT NOT NULL DEFAULT 100,
    attempt_count INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    available_at TIMESTAMP NOT NULL DEFAULT NOW(),
    scheduled_at TIMESTAMP NOT NULL DEFAULT NOW(),
    lease_owner VARCHAR(96),
    lease_expires_at TIMESTAMP,
    heartbeat_at TIMESTAMP,
    last_error TEXT,
    last_error_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    dead_letter_reason TEXT,
    cancel_requested BOOL NOT NULL DEFAULT FALSE,
    cancelled_at TIMESTAMP,
    completed_at TIMESTAMP,
    record_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_ingestion_jobs_source_link
        FOREIGN KEY(document_source_id)
        REFERENCES document_sources(document_source_id)
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_jobs_version_link
        FOREIGN KEY(document_version_id)
        REFERENCES document_versions(document_version_id)
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_jobs_storage_link
        FOREIGN KEY(storage_object_id)
        REFERENCES document_storage_objects(storage_object_id)
        ON DELETE SET NULL,
    CONSTRAINT document_ingestion_jobs_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_ingestion_jobs_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_ingestion_jobs_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_ingestion_jobs_status_check
        CHECK(job_status IN ('queued', 'leased', 'running', 'retry_wait', 'completed', 'cancelled', 'failed', 'dead_letter')),
    CONSTRAINT document_ingestion_jobs_priority_check
        CHECK(priority >= 0),
    CONSTRAINT document_ingestion_jobs_attempt_count_check
        CHECK(attempt_count >= 0),
    CONSTRAINT document_ingestion_jobs_max_attempts_check
        CHECK(max_attempts > 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_ingestion_jobs_idempotent
    ON document_ingestion_jobs (document_source_id, document_version_id, pipeline_version);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_ingestion_jobs_key
    ON document_ingestion_jobs (idempotency_key);

CREATE INDEX IF NOT EXISTS idx_document_ingestion_jobs_ready
    ON document_ingestion_jobs (job_status, available_at, priority, lease_expires_at);

CREATE INDEX IF NOT EXISTS idx_document_ingestion_jobs_scope_created
    ON document_ingestion_jobs (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

ALTER TABLE document_ingestion_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_ingestion_jobs_scope_policy ON document_ingestion_jobs;
CREATE POLICY document_ingestion_jobs_scope_policy ON document_ingestion_jobs
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
