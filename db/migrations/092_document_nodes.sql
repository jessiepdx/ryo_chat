CREATE TABLE IF NOT EXISTS document_nodes (
    document_node_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL,
    parent_node_id BIGINT,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    node_key VARCHAR(128) NOT NULL,
    node_type VARCHAR(32) NOT NULL,
    node_title TEXT,
    ordinal INT NOT NULL DEFAULT 0,
    token_count INT NOT NULL DEFAULT 0,
    page_start INT,
    page_end INT,
    char_start INT,
    char_end INT,
    node_path TEXT,
    node_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_nodes_version_link
        FOREIGN KEY(document_version_id)
        REFERENCES document_versions(document_version_id)
        ON DELETE CASCADE,
    CONSTRAINT document_nodes_parent_link
        FOREIGN KEY(parent_node_id)
        REFERENCES document_nodes(document_node_id)
        ON DELETE CASCADE,
    CONSTRAINT document_nodes_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_nodes_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_nodes_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_nodes_ordinal_check
        CHECK(ordinal >= 0),
    CONSTRAINT document_nodes_token_count_check
        CHECK(token_count >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_nodes_version_node_key
    ON document_nodes (document_version_id, node_key);

CREATE INDEX IF NOT EXISTS idx_document_nodes_parent
    ON document_nodes (parent_node_id);

CREATE INDEX IF NOT EXISTS idx_document_nodes_scope_type
    ON document_nodes (owner_member_id, chat_host_id, platform, node_type, created_at DESC);
