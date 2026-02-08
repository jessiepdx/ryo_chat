CREATE TABLE IF NOT EXISTS knowledge_retrievals (
    retrieval_id SERIAL PRIMARY KEY,
    prompt_id INT,
    response_id INT,
    knowledge_id INT,
    distance DOUBLE PRECISION,
    retrieval_timestamp TIMESTAMP,
    CONSTRAINT prompt_link
        FOREIGN KEY(prompt_id)
        REFERENCES chat_history(history_id)
        ON DELETE SET NULL,
    CONSTRAINT response_link
        FOREIGN KEY(response_id)
        REFERENCES chat_history(history_id)
        ON DELETE SET NULL,
    CONSTRAINT knowledge_link
        FOREIGN KEY(knowledge_id)
        REFERENCES knowledge(knowledge_id)
        ON DELETE CASCADE
);
