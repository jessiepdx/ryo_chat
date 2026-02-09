CREATE EXTENSION IF NOT EXISTS vector;

CREATE TEMP TABLE IF NOT EXISTS __pgvector_bootstrap_probe (
    embedding vector(3)
);

TRUNCATE __pgvector_bootstrap_probe;
INSERT INTO __pgvector_bootstrap_probe (embedding) VALUES ('[1,2,3]');

CREATE TEMP TABLE IF NOT EXISTS __pgvector_schema_probe_history (
    history_id SERIAL PRIMARY KEY,
    message_text TEXT
);

CREATE TEMP TABLE IF NOT EXISTS __pgvector_schema_probe_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    history_id INT NOT NULL REFERENCES __pgvector_schema_probe_history(history_id) ON DELETE CASCADE,
    embedding vector(768)
);

TRUNCATE __pgvector_schema_probe_history RESTART IDENTITY CASCADE;
INSERT INTO __pgvector_schema_probe_history (message_text) VALUES ('schema sanity check');
INSERT INTO __pgvector_schema_probe_embeddings (history_id, embedding)
VALUES (
    (SELECT history_id FROM __pgvector_schema_probe_history LIMIT 1),
    ('[' || array_to_string(array_fill(0::int, ARRAY[768]), ',') || ']')::vector
);

SELECT embedding <-> ('[' || array_to_string(array_fill(0::int, ARRAY[768]), ',') || ']')::vector AS distance
FROM __pgvector_schema_probe_embeddings
LIMIT 1;

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
