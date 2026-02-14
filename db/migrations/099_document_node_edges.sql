CREATE TABLE IF NOT EXISTS document_node_edges (
    document_node_edge_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL,
    source_node_id BIGINT NOT NULL,
    target_node_id BIGINT NOT NULL,
    schema_version INT NOT NULL,
    owner_member_id INT NOT NULL,
    chat_host_id BIGINT NOT NULL,
    chat_type VARCHAR(32) NOT NULL,
    community_id INT,
    topic_id INT,
    platform VARCHAR(24) NOT NULL,
    edge_type VARCHAR(32) NOT NULL,
    ordinal INT NOT NULL DEFAULT 0,
    edge_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT document_node_edges_version_link
        FOREIGN KEY(document_version_id)
        REFERENCES document_versions(document_version_id)
        ON DELETE CASCADE,
    CONSTRAINT document_node_edges_source_link
        FOREIGN KEY(source_node_id)
        REFERENCES document_nodes(document_node_id)
        ON DELETE CASCADE,
    CONSTRAINT document_node_edges_target_link
        FOREIGN KEY(target_node_id)
        REFERENCES document_nodes(document_node_id)
        ON DELETE CASCADE,
    CONSTRAINT document_node_edges_member_link
        FOREIGN KEY(owner_member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE,
    CONSTRAINT document_node_edges_community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE SET NULL,
    CONSTRAINT document_node_edges_schema_version_check
        CHECK(schema_version > 0),
    CONSTRAINT document_node_edges_ordinal_check
        CHECK(ordinal >= 0),
    CONSTRAINT document_node_edges_type_check
        CHECK(edge_type IN ('parent_child', 'next_sibling', 'reference'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_node_edges_unique
    ON document_node_edges (document_version_id, source_node_id, target_node_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_document_node_edges_target
    ON document_node_edges (target_node_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_document_node_edges_scope_created
    ON document_node_edges (owner_member_id, chat_host_id, platform, community_id, topic_id, created_at DESC);

ALTER TABLE document_node_edges ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_node_edges_scope_policy ON document_node_edges;
CREATE POLICY document_node_edges_scope_policy ON document_node_edges
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
