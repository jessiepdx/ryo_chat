CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id SERIAL PRIMARY KEY,
    domains TEXT[],
    roles TEXT[],
    categories TEXT[],
    knowledge_document TEXT,
    document_metadata JSON,
    embeddings vector({{vector_dimensions}}),
    record_timestamp TIMESTAMP,
    record_metadata JSON
);
