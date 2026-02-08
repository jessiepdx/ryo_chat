CREATE EXTENSION IF NOT EXISTS vector;

CREATE TEMP TABLE IF NOT EXISTS __pgvector_bootstrap_probe (
    embedding vector(3)
);

TRUNCATE __pgvector_bootstrap_probe;
INSERT INTO __pgvector_bootstrap_probe (embedding) VALUES ('[1,2,3]');

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
