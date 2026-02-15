ALTER TABLE document_chunks
    DROP CONSTRAINT IF EXISTS document_chunks_metadata_object_check;

ALTER TABLE document_chunks
    ADD CONSTRAINT document_chunks_metadata_object_check
    CHECK (jsonb_typeof(chunk_metadata) = 'object');

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_node_path
    ON document_chunks ((chunk_metadata->>'node_path'))
    WHERE chunk_metadata ? 'node_path';

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_node_type
    ON document_chunks ((chunk_metadata->>'node_type'))
    WHERE chunk_metadata ? 'node_type';

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_heading_trail
    ON document_chunks
    USING GIN ((chunk_metadata->'heading_trail'))
    WHERE chunk_metadata ? 'heading_trail';
