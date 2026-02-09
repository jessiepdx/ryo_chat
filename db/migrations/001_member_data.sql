CREATE TABLE IF NOT EXISTS member_data (
    member_id SERIAL PRIMARY KEY,
    first_name VARCHAR(96),
    last_name VARCHAR(96),
    email VARCHAR(72),
    roles VARCHAR(32)[],
    register_date TIMESTAMP NOT NULL,
    community_score REAL NOT NULL DEFAULT 0
);
