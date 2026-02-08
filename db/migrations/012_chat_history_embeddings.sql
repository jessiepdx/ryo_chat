CREATE TABLE IF NOT EXISTS chat_history_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    history_id INT NOT NULL,
    embeddings vector({{vector_dimensions}}),
    CONSTRAINT message_link
        FOREIGN KEY(history_id)
        REFERENCES chat_history(history_id)
        ON DELETE CASCADE
);
