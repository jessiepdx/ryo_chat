CREATE TABLE IF NOT EXISTS community_data (
    community_id SERIAL PRIMARY KEY,
    community_name VARCHAR(96),
    community_link VARCHAR(256),
    roles VARCHAR(32)[],
    created_by INT,
    register_date TIMESTAMP NOT NULL,
    CONSTRAINT member_link
        FOREIGN KEY(created_by)
        REFERENCES member_data(member_id)
        ON DELETE SET NULL
);
