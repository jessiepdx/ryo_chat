CREATE TABLE IF NOT EXISTS document_chunks (
    document_chunk_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL,
    document_node_id BIGINT,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    chunk_key VARCHAR(128) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INT NOT NULL DEFAULT 0,
    start_char INT,
    end_char INT,
    start_page INT,
    end_page INT,
    chunk_digest VARCHAR(128),
    chunk_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_chunks_version_link
        FOREIGN KEY(document_version_id)
        REFERENCES document_versions(document_version_id)
        ON DELETE CASCADE,
    CONSTRAINT document_chunks_node_link
        FOREIGN KEY(document_node_id)
        REFERENCES document_nodes(document_node_id)
        ON DELETE SET NULL,
    CONSTRAINT document_chunks_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_chunks_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_chunks_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_chunks_chunk_index_check
        CHECK(chunk_index >= 0),
    CONSTRAINT document_chunks_token_count_check
        CHECK(token_count >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunks_version_chunk_key
    ON document_chunks (document_version_id, chunk_key);

CREATE INDEX IF NOT EXISTS idx_document_chunks_scope_created
    ON document_chunks (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_chunks_node_index
    ON document_chunks (document_node_id, chunk_index);
