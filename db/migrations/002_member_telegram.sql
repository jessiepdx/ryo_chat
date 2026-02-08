CREATE TABLE IF NOT EXISTS member_telegram (
    record_id SERIAL PRIMARY KEY,
    member_id INT UNIQUE NOT NULL,
    first_name VARCHAR(96),
    last_name VARCHAR(96),
    username VARCHAR(96),
    user_id BIGINT UNIQUE NOT NULL,
    CONSTRAINT member_link
        FOREIGN KEY(member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);
