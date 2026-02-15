CREATE INDEX IF NOT EXISTS idx_document_sources_taxonomy_topics
    ON document_sources
    USING GIN ((COALESCE(source_metadata->'taxonomy'->'topic_tags', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_sources_taxonomy_domains
    ON document_sources
    USING GIN ((COALESCE(source_metadata->'taxonomy'->'domain_labels', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_versions_taxonomy_topics
    ON document_versions
    USING GIN ((COALESCE(record_metadata->'taxonomy'->'topic_tags', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_versions_taxonomy_domains
    ON document_versions
    USING GIN ((COALESCE(record_metadata->'taxonomy'->'domain_labels', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_nodes_taxonomy_topics
    ON document_nodes
    USING GIN ((COALESCE(node_metadata->'taxonomy'->'topic_tags', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_nodes_taxonomy_formats
    ON document_nodes
    USING GIN ((COALESCE(node_metadata->'taxonomy'->'format_labels', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_chunks_taxonomy_topics
    ON document_chunks
    USING GIN ((COALESCE(chunk_metadata->'topic_tags', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_chunks_taxonomy_domains
    ON document_chunks
    USING GIN ((COALESCE(chunk_metadata->'domain_labels', '[]'::jsonb)));

CREATE INDEX IF NOT EXISTS idx_document_chunks_taxonomy_formats
    ON document_chunks
    USING GIN ((COALESCE(chunk_metadata->'format_labels', '[]'::jsonb)));
