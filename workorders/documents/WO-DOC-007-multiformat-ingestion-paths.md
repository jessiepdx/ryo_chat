# WO-DOC-007: Multi-Format Ingestion Paths Beyond PDF

Source item: User request (February 10, 2026) requiring support for "pdf and other text documents".

## 1. Verbose Description of Work Order
Add ingestion support for non-PDF text-bearing formats and normalize them into the same canonical document representation used by PDF parsing.

Scope includes:
1. Plain text and markdown.
2. HTML and web-captured documents.
3. Office formats (`docx`, `pptx`, `xlsx`) via extractors.
4. Structured plain-data formats (`csv`, `json`, `xml`) with controlled flattening.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_parser/adapters/text_markdown.py`
2. New: `hypermindlabs/document_parser/adapters/html.py`
3. New: `hypermindlabs/document_parser/adapters/office.py`
4. New: `hypermindlabs/document_parser/adapters/structured_data.py`

Secondary/API-surface files:
1. `hypermindlabs/document_parser/registry.py`
2. `hypermindlabs/document_parser/router.py`
3. `hypermindlabs/runtime_settings.py` (format allowlist/denylist)
4. `tests/test_multiformat_ingestion.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Supported formats parse to canonical element records.
2. Format-specific metadata is preserved without breaking downstream interfaces.
3. Unsupported formats fail with actionable error details.
4. Routing and parser selection are covered by tests.

Partial outcome:
1. Ingestion supports formats but output contracts differ by adapter.
2. Structured data conversion is lossy without source reference mapping.
3. Parser errors are untyped and not operator-actionable.

Validation method:
1. Fixture-driven adapter tests per format.
2. Contract validation across adapters.
3. End-to-end ingest -> chunk -> retrieve smoke tests.

## 4. Potential Failure Modes
1. HTML boilerplate dominates extracted content.
2. Office parser dependencies differ by environment and break portability.
3. JSON/XML flattening loses hierarchy needed for retrieval.
4. CSV ingestion accidentally indexes low-signal columns as text blocks.
