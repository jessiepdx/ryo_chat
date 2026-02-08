CREATE TABLE IF NOT EXISTS community_score (
    score_id SERIAL PRIMARY KEY,
    history_id INTEGER NOT NULL,
    event TEXT NOT NULL,
    read_score REAL,
    points_awarded REAL NOT NULL,
    awarded_from_id INTEGER NOT NULL,
    multiplier REAL NOT NULL,
    CONSTRAINT history_link
        FOREIGN KEY(history_id)
        REFERENCES chat_history(history_id)
        ON DELETE SET NULL
);
