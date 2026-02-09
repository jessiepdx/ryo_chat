CREATE TABLE IF NOT EXISTS inference_usage (
    usage_id SERIAL PRIMARY KEY,
    prompt_history_id INT NOT NULL,
    response_history_id INT NOT NULL,
    load_duration BIGINT NOT NULL,
    prompt_eval_count INTEGER NOT NULL,
    prompt_eval_duration BIGINT NOT NULL,
    eval_count INTEGER NOT NULL,
    eval_duration BIGINT NOT NULL,
    total_duration BIGINT NOT NULL,
    CONSTRAINT prompt_history
        FOREIGN KEY(prompt_history_id)
        REFERENCES chat_history(history_id)
        ON DELETE CASCADE,
    CONSTRAINT response_history
        FOREIGN KEY(response_history_id)
        REFERENCES chat_history(history_id)
        ON DELETE CASCADE
);
