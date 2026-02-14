CREATE TABLE IF NOT EXISTS document_retrieval_events (
    retrieval_event_id BIGSERIAL PRIMARY KEY,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    request_id VARCHAR(96) NOT NULL,
    query_text TEXT NOT NULL,
    document_source_id BIGINT,
    document_version_id BIGINT,
    result_count INT NOT NULL DEFAULT 0,
    max_distance REAL,
    query_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_retrieval_events_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_retrieval_events_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_retrieval_events_source_link
        FOREIGN KEY(document_source_id)
        REFERENCES document_sources(document_source_id)
        ON DELETE SET NULL,
    CONSTRAINT document_retrieval_events_version_link
        FOREIGN KEY(document_version_id)
        REFERENCES document_versions(document_version_id)
        ON DELETE SET NULL,
    CONSTRAINT document_retrieval_events_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_retrieval_events_result_count_check
        CHECK(result_count >= 0),
    CONSTRAINT document_retrieval_events_distance_check
        CHECK(max_distance IS NULL OR max_distance >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_retrieval_events_request_scope
    ON document_retrieval_events (owner_member_id, chat_host_id, platform, request_id);

CREATE INDEX IF NOT EXISTS idx_document_retrieval_events_scope_created
    ON document_retrieval_events (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_retrieval_events_version
    ON document_retrieval_events (document_version_id, created_at DESC);
