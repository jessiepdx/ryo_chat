# WO-DOC-017: Citation, Provenance, and Evidence-Gated Responses

Source item: User request (February 10, 2026) requiring reliable information-tree-based retrieval and evidence-grounded recall.

## 1. Verbose Description of Work Order
Implement citation and provenance primitives that tie generated claims to specific document nodes/chunks and optionally enforce evidence-gated response modes.

This work order ensures traceable grounding for retrieved knowledge.

Scope includes:
1. Citation span schema linked to chunk/node/source/version IDs.
2. Claim-to-evidence mapping in model output post-processing.
3. Citation-required mode for selected prompts or policies.
4. Evidence confidence checks and fallback behavior.

## 2. Expression of Affected Files
Primary files:
1. New: `hypermindlabs/document_citations.py`
2. New migration: `db/migrations/109_document_citation_spans.sql`
3. `hypermindlabs/agents.py` (citation envelope in response composition)
4. `hypermindlabs/run_events.py` (citation-required violations)

Secondary/API-surface files:
1. `web_ui.py` (citation render payloads)
2. `templates/knowledge-tools.html` (provenance visualization)
3. `tests/test_document_citations.py`
4. `tests/test_evidence_gated_responses.py`

## 3. Success and Validation Metrics
Valid outcome:
1. Responses can include source-linked citations to stored chunk spans.
2. Citation-required mode blocks unsupported claims.
3. Citation payloads are consistent and machine-readable.
4. Provenance links are resolvable to stored source versions.

Partial outcome:
1. Citation text is present but not linked to stored IDs.
2. Citation-required mode is advisory only.
3. Evidence checks do not account for scope or version pinning.

Validation method:
1. Integration tests across retrieve -> respond -> citation resolve.
2. Negative tests for unsupported claims under citation-required mode.
3. Provenance integrity checks for deleted/replaced versions.

## 4. Potential Failure Modes
1. Citation references stale chunk IDs after re-indexing.
2. Strict citation mode over-blocks conversational responses.
3. Citation spans are too coarse to support claim verification.
4. UI leaks restricted source content across tenancy boundaries.
