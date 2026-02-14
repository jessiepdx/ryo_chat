CREATE TABLE IF NOT EXISTS document_sources (
    document_source_id BIGSERIAL PRIMARY KEY,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    source_external_id VARCHAR(96),
    source_name VARCHAR(512) NOT NULL,
    source_mime VARCHAR(128),
    source_sha256 VARCHAR(128),
    source_size_bytes BIGINT,
    source_uri TEXT,
    source_state VARCHAR(24) NOT NULL DEFAULT 'received',
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_sources_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_sources_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_sources_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_sources_source_state_check
        CHECK(source_state IN ('received', 'queued', 'parsed', 'failed', 'archived', 'deleted')),
    CONSTRAINT document_sources_size_check
        CHECK(source_size_bytes IS NULL OR source_size_bytes >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_sources_external_scope
    ON document_sources (owner_member_id, chat_host_id, platform, source_external_id)
    WHERE source_external_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_document_sources_scope_created
    ON document_sources (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_sources_scope_sha
    ON document_sources (owner_member_id, chat_host_id, platform, source_sha256)
    WHERE source_sha256 IS NOT NULL;
