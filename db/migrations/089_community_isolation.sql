CREATE TABLE IF NOT EXISTS community_isolation (
    isolation_id SERIAL PRIMARY KEY,
    community_id INT UNIQUE NOT NULL,
    platform VARCHAR(24) NOT NULL DEFAULT 'telegram',
    chat_id BIGINT NOT NULL,
    storage_mode VARCHAR(24) NOT NULL DEFAULT 'shared_pg',
    storage_key VARCHAR(128) NOT NULL UNIQUE,
    storage_schema VARCHAR(96),
    storage_database VARCHAR(96),
    context_isolation_enabled BOOL NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT community_isolation_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_community_isolation_chat_platform
    ON community_isolation (chat_id, platform);

CREATE INDEX IF NOT EXISTS idx_community_isolation_storage_key
    ON community_isolation (storage_key);
