CREATE TABLE IF NOT EXISTS community_telegram (
    record_id SERIAL PRIMARY KEY,
    community_id INT UNIQUE NOT NULL,
    chat_id BIGINT UNIQUE NOT NULL,
    chat_title VARCHAR(96),
    has_topics BOOL,
    CONSTRAINT community_link
        FOREIGN KEY(community_id)
        REFERENCES community_data(community_id)
        ON DELETE CASCADE
);
