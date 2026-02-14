CREATE TABLE IF NOT EXISTS document_versions (
    document_version_id BIGSERIAL PRIMARY KEY,
    document_source_id BIGINT NOT NULL,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    version_number INT NOT NULL,
    source_sha256 VARCHAR(128),
    parser_name VARCHAR(96),
    parser_version VARCHAR(96),
    parser_status VARCHAR(24) NOT NULL DEFAULT 'queued',
    parse_artifact JSONB NOT NULL DEFAULT '{}'::jsonb,
    record_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_versions_source_link
        FOREIGN KEY(document_source_id)
        REFERENCES document_sources(document_source_id)
        ON DELETE CASCADE,
    CONSTRAINT document_versions_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_versions_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_versions_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_versions_version_number_check
        CHECK(version_number > 0),
    CONSTRAINT document_versions_parser_status_check
        CHECK(parser_status IN ('received', 'queued', 'parsed', 'failed', 'archived', 'deleted'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_versions_source_version
    ON document_versions (document_source_id, version_number);

CREATE INDEX IF NOT EXISTS idx_document_versions_scope_created
    ON document_versions (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_versions_scope_status
    ON document_versions (owner_member_id, chat_host_id, platform, parser_status, created_at DESC);
