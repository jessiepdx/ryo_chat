ALTER TABLE document_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_retrieval_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_sources_scope_policy ON document_sources;
CREATE POLICY document_sources_scope_policy ON document_sources
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

DROP POLICY IF EXISTS document_versions_scope_policy ON document_versions;
CREATE POLICY document_versions_scope_policy ON document_versions
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

DROP POLICY IF EXISTS document_nodes_scope_policy ON document_nodes;
CREATE POLICY document_nodes_scope_policy ON document_nodes
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

DROP POLICY IF EXISTS document_chunks_scope_policy ON document_chunks;
CREATE POLICY document_chunks_scope_policy ON document_chunks
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

DROP POLICY IF EXISTS document_retrieval_events_scope_policy ON document_retrieval_events;
CREATE POLICY document_retrieval_events_scope_policy ON document_retrieval_events
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
