CREATE TABLE IF NOT EXISTS document_storage_objects (
    storage_object_id BIGSERIAL PRIMARY KEY,
    document_source_id BIGINT NOT NULL,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    storage_backend VARCHAR(32) NOT NULL DEFAULT 'local_fs',
    storage_key TEXT NOT NULL,
    storage_path TEXT,
    object_state VARCHAR(24) NOT NULL DEFAULT 'received',
    file_name VARCHAR(512) NOT NULL,
    file_mime VARCHAR(128),
    file_sha256 VARCHAR(128) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    dedupe_status VARCHAR(32) NOT NULL DEFAULT 'new',
    retention_until TIMESTAMP,
    deleted_at TIMESTAMP,
    record_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_storage_objects_source_link
        FOREIGN KEY(document_source_id)
        REFERENCES document_sources(document_source_id)
        ON DELETE CASCADE,
    CONSTRAINT document_storage_objects_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_storage_objects_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_storage_objects_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_storage_objects_state_check
        CHECK(object_state IN ('received', 'queued', 'parsed', 'failed', 'archived', 'deleted')),
    CONSTRAINT document_storage_objects_size_check
        CHECK(file_size_bytes >= 0)
);

CREATE INDEX IF NOT EXISTS idx_document_storage_objects_storage_key
    ON document_storage_objects (storage_key);

CREATE INDEX IF NOT EXISTS idx_document_storage_objects_scope_created
    ON document_storage_objects (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_storage_objects_scope_digest
    ON document_storage_objects (owner_member_id, chat_host_id, platform, file_sha256, object_state);

ALTER TABLE document_storage_objects ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_storage_objects_scope_policy ON document_storage_objects;
CREATE POLICY document_storage_objects_scope_policy ON document_storage_objects
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
