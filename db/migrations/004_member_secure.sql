CREATE TABLE IF NOT EXISTS member_secure (
    secure_id SERIAL PRIMARY KEY,
    member_id INTEGER UNIQUE NOT NULL,
    secure_hash BYTEA,
    CONSTRAINT member_link
        FOREIGN KEY(member_id)
        REFERENCES member_data(member_id)
        ON DELETE CASCADE
);
