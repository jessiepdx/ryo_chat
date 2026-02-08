CREATE TABLE IF NOT EXISTS spam (
    spam_id SERIAL PRIMARY KEY,
    spam_text TEXT,
    embeddings vector({{vector_dimensions}}),
    record_timestamp TIMESTAMP,
    added_by INT,
    CONSTRAINT member_link
        FOREIGN KEY(added_by)
        REFERENCES member_data(member_id)
        ON DELETE SET NULL
);
