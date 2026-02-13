# WO-DOC-010: Metadata Taxonomy, Topic Signals, and Structural Labels

Source item: User request (February 10, 2026) requiring hierarchical topic-based storage and retrieval.

## 1. Verbose Description of Work Order
Define and implement metadata enrichment for documents, nodes, and chunks to improve topic routing, filtering, and ranking.

This work order provides a normalized taxonomy layer that retrieval can use for deterministic pre-filtering and ranking.

Scope includes:
1. Metadata taxonomy for source, version, node, and chunk levels.
2. Topic tags and domain labels with confidence scores.
3. Formatting labels (table, code, policy, procedure, FAQ, etc.).
4. Controlled vocabularies and synonym maps.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_metadata.py`
2. New: `hypermindlabs/document_taxonomy.py`
3. New migration: `db/migrations/101_document_metadata_taxonomy.sql`
4. `hypermindlabs/document_ingestion_worker.py` (metadata stage)

Secondary/API-surface files:
1. `hypermindlabs/runtime_settings.py` (taxonomy toggles and thresholds)
2. `web_ui.py` (metadata filter exposure)
3. `tests/test_document_taxonomy.py`
4. `tests/test_topic_labeling.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Metadata enrichment executes consistently across formats.
2. Topic and format labels improve retrieval precision.
3. Taxonomy values remain bounded to controlled sets.
4. Metadata filters are available in retrieval APIs.

Partial outcome:
1. Labels are generated but not persisted in queryable fields.
2. Topic extraction exists without confidence scoring.
3. Controlled vocabularies are ignored in parser outputs.

Validation method:
1. Precision/recall checks on labeled fixture datasets.
2. Schema tests for metadata shape and value constraints.
3. Retrieval filter tests proving scoped metadata filtering.

## 4. Potential Failure Modes
1. Label explosion creates noisy metadata and weak filters.
2. Topic assignment drifts across document versions.
3. Metadata defaults hide low-confidence extraction quality.
4. Format labels are inconsistent across parser adapters.
