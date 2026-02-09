CREATE TABLE IF NOT EXISTS chat_history (
    history_id SERIAL PRIMARY KEY,
    member_id INT,
    community_id INT,
    chat_host_id INT,
    topic_id INT,
    chat_type VARCHAR(16),
    platform VARCHAR(24),
    message_id INT NOT NULL,
    message_text TEXT,
    message_timestamp TIMESTAMP
);
